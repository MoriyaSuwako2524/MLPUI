"""TensorNet for excited-state transition charges and transition dipoles.

For each of ``num_states`` transitions 0 -> n the model predicts atom-centred
transition charges q_i^(n) and the transition dipole

    mu^(n) = sum_i q_i^(n) r_i  [+ sum_i p_i^(n)]

* Sum_i q_i^(n) = 0 exactly (projection per structure and state), because the
  transition density integrates to <Psi_0|Psi_n> = 0. This also makes mu
  independent of the coordinate origin.
* Optional atomic transition dipoles p_i^(n) = sum_j w_ij^(n) (r_j - r_i), with
  scalar edge weights from TensorNet features and radial basis, restore the
  part of mu that point charges cannot carry. Their last layer starts at zero.
* q and mu come from the same per-state head, so they share one phase: the
  arbitrary sign s_n = +-1 of state n flips both together. Training resolves
  that sign per (structure, state) in the loss; see mlpui.models.transition.phase.

* Optional atom-index embedding (``atom_index_embedding: N``): a learned vector per
  within-structure atom index (0..N-1) is added to the TensorNet features before the heads.
  Symmetry-equivalent atoms of a symmetric molecule have identical invariant features, so
  without it q_P(i) = q_i is forced and antisymmetric transition charges collapse to 0.
  Requires a fixed atom ordering across structures; zero-initialised.

* Optional global basis head (``global_basis: M``, requires ``atom_index_embedding``):
  q_i^(n) += sum_a C_na(R) b_a(i), with M learned per-atom-index basis vectors b_a (plus a
  zero-initialised local correction from TensorNet features) and molecule-level mixing
  coefficients C(R) [num_states, M] read out from attention-pooled atom features. The
  transition densities of a chromophore are dominated by a few configurations whose mixing is a
  global property of the geometry; a purely local per-atom head cannot make all atoms agree on it.
  The basis can be initialised from a PCA of training labels (``init_global_basis``).

* Optional diabatic head (``diabatic_states: M`` >= num_states, requires ``atom_index_embedding``;
  excludes ``global_basis``): a symmetric M x M diabatic Hamiltonian built from atomic
  contributions, H_ab(R) = eps0_a delta_ab + sum_i h_iab(x_i), is diagonalised,
  H = U diag(E) U^T. The lowest num_states eigenvalues are the excitation energies
  (target ``excitation_energies``) and the adiabatic transition charges are
  q_i^(n) = sum_a U_an b_a(i) with per-atom-index diabatic basis vectors b_a (+ zero-initialised
  local correction). Mixing near avoided crossings then follows the energies, eigenvector
  signs are arbitrary (absorbed by the phase-less loss), and M > num_states leaves room for
  higher states. Eigen-decomposition gradients use Lorentzian broadening of 1/(E_j - E_i)
  (``EIG_BROADENING``, energy units) so that (near-)degenerate spectra stay finite.
  ``init_diabatic(basis, energies)`` sets b and eps0 (e.g. state-averaged templates, mean energies).

Representation: QM-only O(3) TensorNet (scalar features, invariant under
reflection). Units: angstrom, e, e*angstrom. Nonperiodic. Gas phase only.
"""
import copy

import numpy as np
import torch
from torch import nn

from .phase import segment_sum


class _BroadenedEigh(torch.autograd.Function):
    """torch.linalg.eigh with F_ij = 1/(E_j - E_i) replaced by d / (d^2 + eps^2) in the backward."""

    @staticmethod
    def forward(ctx, A, eps):
        E, U = torch.linalg.eigh(A)
        ctx.save_for_backward(E, U)
        ctx.eps = eps
        return E, U

    @staticmethod
    def backward(ctx, gE, gU):
        E, U = ctx.saved_tensors
        inner = torch.zeros_like(U)
        if gE is not None:
            inner = inner + torch.diag_embed(gE)
        if gU is not None:
            d = E[..., None, :] - E[..., :, None]                         # d_ij = E_j - E_i
            F = d / (d * d + ctx.eps ** 2)
            inner = inner + F * (U.transpose(-1, -2) @ gU)
        gA = U @ inner @ U.transpose(-1, -2)
        return 0.5 * (gA + gA.transpose(-1, -2)), None


def broadened_eigh(A, eps):
    return _BroadenedEigh.apply(A, eps)


class TransitionTensorNet(nn.Module):
    ARCHITECTURE = 'tensornet_transition_v1'
    EIG_BROADENING = 1e-2

    def __init__(self, config):
        super().__init__()
        from mlpui.models.torchmdnet.models.tensornet import TensorNet
        if config.get('schema_version') != 1 or config.get('architecture') != self.ARCHITECTURE:
            raise ValueError('Incompatible transition-property architecture')
        self.config = copy.deepcopy(config)
        self.mlpui_family = 'tensornet_transition'
        dtype = torch.float64 if config['precision'] == 64 else torch.float32
        self.register_buffer('transition_schema', torch.tensor(1, dtype=torch.long))
        width, states = config['embedding_dimension'], config['num_states']
        self.num_states = states
        self.representation = TensorNet(
            hidden_channels=width, num_layers=config['num_layers'], num_rbf=config['num_rbf'],
            max_num_neighbors=config['max_num_neighbors'], cutoff_upper=config['cutoff_upper'],
            static_shapes=False, equivariance_invariance_group='O(3)', dtype=dtype)
        self.index_embedding = None
        if config.get('atom_index_embedding', 0):
            self.index_embedding = nn.Embedding(config['atom_index_embedding'], width, dtype=dtype)
            # Zero init: training starts from the permutation-invariant model.
            nn.init.zeros_(self.index_embedding.weight)
        self.global_basis = int(config.get('global_basis', 0))
        if self.global_basis:
            if not self.index_embedding:
                raise ValueError('global_basis requires atom_index_embedding (fixed atom ordering)')
            M = self.global_basis
            self.basis = nn.Parameter(torch.zeros(config['atom_index_embedding'], M, dtype=dtype))
            nn.init.normal_(self.basis, std=0.05)
            self.basis_local = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                             nn.Linear(width, M, dtype=dtype))
            nn.init.zeros_(self.basis_local[2].weight)
            nn.init.zeros_(self.basis_local[2].bias)
            self.pool_score = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                            nn.Linear(width, 4, dtype=dtype))       # 4 attention heads
            self.pool_value = nn.Linear(width, width, dtype=dtype)
            self.coeff_head = nn.Sequential(nn.Linear(5 * width, 2 * width, dtype=dtype), nn.SiLU(),
                                            nn.Linear(2 * width, width, dtype=dtype), nn.SiLU(),
                                            nn.Linear(width, states * M, dtype=dtype))
        self.diabatic = int(config.get('diabatic_states', 0))
        if self.diabatic:
            if not self.index_embedding:
                raise ValueError('diabatic_states requires atom_index_embedding (fixed atom ordering)')
            if self.diabatic < states:
                raise ValueError('diabatic_states must be >= num_states')
            M = self.diabatic
            self.dia_basis = nn.Parameter(torch.zeros(config['atom_index_embedding'], M, dtype=dtype))
            nn.init.normal_(self.dia_basis, std=0.05)
            self.dia_basis_local = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                                 nn.Linear(width, M, dtype=dtype))
            nn.init.zeros_(self.dia_basis_local[2].weight)
            nn.init.zeros_(self.dia_basis_local[2].bias)
            self.dia_energy0 = nn.Parameter(torch.arange(M, dtype=dtype) * 0.3 + 2.0)
            self.ham_head = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                          nn.Linear(width, M * (M + 1) // 2, dtype=dtype))
            nn.init.zeros_(self.ham_head[2].weight)
            nn.init.zeros_(self.ham_head[2].bias)
            self.register_buffer('dia_triu', torch.triu_indices(M, M), persistent=False)
        self.charge_head = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                         nn.Linear(width, states, dtype=dtype))
        if config['atomic_dipoles']:
            self.edge_i = nn.Linear(width, width, dtype=dtype)
            self.edge_j = nn.Linear(width, width, dtype=dtype)
            self.edge_r = nn.Linear(config['num_rbf'], width, dtype=dtype)
            self.edge_out = nn.Sequential(nn.SiLU(), nn.Linear(width, states, dtype=dtype))
            # Zero last layer: training starts from the pure point-charge dipole.
            nn.init.zeros_(self.edge_out[1].weight)
            nn.init.zeros_(self.edge_out[1].bias)
        self.to(dtype=dtype)

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'index_embedding' not in self._modules:   # serialized before atom_index_embedding existed
            self.index_embedding = None
            self.config.setdefault('atom_index_embedding', 0)
        if 'diabatic' not in self.__dict__:          # serialized before diabatic_states existed
            self.diabatic = 0
            self.config.setdefault('diabatic_states', 0)
        if 'global_basis' not in self.__dict__:      # serialized before global_basis existed
            self.global_basis = 0
            self.config.setdefault('global_basis', 0)
        if self.config.get('architecture') != self.ARCHITECTURE or self.transition_schema.item() != 1:
            raise ValueError('Incompatible transition-property serialized architecture')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        schema = state_dict.get('transition_schema')
        if schema is None or schema.numel() != 1 or schema.item() != 1:
            raise ValueError('Incompatible transition-property checkpoint: requires transition_schema 1')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _atomic_dipoles(self, x, pos, batch):
        """p_i [A, S, 3] = sum_j w_ij (r_j - r_i) over the representation's neighbour graph."""
        rep = self.representation
        edge_index, edge_weight, _ = rep.distance(pos, batch, None)
        keep = edge_index[0] != edge_index[1]
        i, j, d = edge_index[0][keep], edge_index[1][keep], edge_weight[keep]
        cutoff = 0.5 * (torch.cos(d * torch.pi / rep.cutoff_upper) + 1.0) * (d < rep.cutoff_upper)
        h = self.edge_i(x)[i] * self.edge_j(x)[j] * self.edge_r(rep.distance_expansion(d))
        w = self.edge_out(h) * cutoff[:, None]                            # [E, S]
        vec = pos[j] - pos[i]                                             # [E, 3]
        return segment_sum(w[:, :, None] * vec[:, None, :], i, len(pos))  # [A, S, 3]

    def _atom_index(self, batch, counts):
        """Index of each atom within its own structure (atoms of a structure are contiguous)."""
        if len(batch) > 1 and bool((batch[1:] < batch[:-1]).any()):
            raise ValueError('atom_index_embedding requires the atoms of each structure to be contiguous')
        start = torch.cumsum(counts, 0) - counts
        index = torch.arange(len(batch), device=batch.device) - start[batch]
        limit = self.index_embedding.num_embeddings
        if len(index) and int(index.max()) >= limit:
            raise ValueError(f'Structure has {int(counts.max())} atoms but atom_index_embedding is {limit}')
        return index

    @torch.no_grad()
    def init_global_basis(self, basis):
        """Set b [N, M] (e.g. top-M sign-invariant PCA vectors of training transition charges)."""
        if not isinstance(basis, torch.Tensor):
            basis = torch.tensor(np.array(basis, dtype=np.float64))            # copy: accepts any strides
        basis = basis.to(dtype=self.basis.dtype, device=self.basis.device)
        if basis.shape != self.basis.shape:
            raise ValueError(f'basis must have shape {tuple(self.basis.shape)}')
        self.basis.copy_(basis)

    @torch.no_grad()
    def init_diabatic(self, basis, energies):
        """Set diabatic basis b [N, M] and diabatic energy offsets eps0 [M]."""
        out = []
        for value, ref in ((basis, self.dia_basis), (energies, self.dia_energy0)):
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(np.array(value, dtype=np.float64))
            value = value.to(dtype=ref.dtype, device=ref.device)
            if value.shape != ref.shape:
                raise ValueError(f'expected shape {tuple(ref.shape)}, got {tuple(value.shape)}')
            out.append(value)
        self.dia_basis.copy_(out[0])
        self.dia_energy0.copy_(out[1])

    def diabatic_hamiltonian(self, x, batch, count):
        """H [B, M, M] = diag(eps0) + sum over atoms of symmetric atomic contributions."""
        M = self.diabatic
        h = segment_sum(self.ham_head(x), batch, count)                   # [B, M(M+1)/2]
        H = x.new_zeros(count, M, M)
        i, j = self.dia_triu
        H[:, i, j] = h
        H = H + H.transpose(1, 2) - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
        return H + torch.diag(self.dia_energy0)

    def _diabatic(self, x, index, batch, count):
        S = self.num_states
        H = self.diabatic_hamiltonian(x, batch, count)
        E, U = broadened_eigh(H, self.EIG_BROADENING)                    # ascending
        b = self.dia_basis[index] + self.dia_basis_local(x)               # [A, M]
        raw = torch.einsum('am,ams->as', b, U[batch][:, :, :S])          # [A, S]
        return raw, E[:, :S]

    def _global_charges(self, x, index, batch, count):
        """sum_a C_na(R) b_a(i) -> [A, S]."""
        M, S = self.global_basis, self.num_states
        b = self.basis[index] + self.basis_local(x)                       # [A, M]
        score = self.pool_score(x)                                         # [A, H]
        top = torch.zeros(count, score.shape[1], dtype=score.dtype, device=score.device).scatter_reduce(
            0, batch[:, None].expand_as(score), score.detach(), 'amax', include_self=False)
        score = score - top[batch]                                         # stable softmax
        w = score.exp()
        w = w / segment_sum(w, batch, count)[batch]                        # softmax over atoms per structure
        v = self.pool_value(x)                                             # [A, F]
        heads = [segment_sum(w[:, h:h + 1] * v, batch, count) for h in range(w.shape[1])]
        mean = segment_sum(x, batch, count) / torch.bincount(batch, minlength=count).to(x.dtype)[:, None]
        g = torch.cat(heads + [mean], dim=1)                               # [B, 5F]
        C = self.coeff_head(g).reshape(count, S, M)                        # [B, S, M]
        return torch.einsum('am,asm->as', b, C[batch])

    def forward(self, z, pos, charge=None, *, batch=None,
                properties=('transition_charges', 'transition_dipole')):
        single = batch is None
        if single:
            batch = torch.zeros_like(z)
        count = int(batch.max()) + 1 if len(batch) else 0
        if charge is None:
            charge = pos.new_zeros(count)
        charge = charge.reshape(-1)
        if charge.numel() != count:
            raise ValueError('Supply one total QM charge per structure')
        x, _, _, _, _ = self.representation(z, pos, batch, q=charge)
        counts = torch.bincount(batch, minlength=count)
        index = None
        if self.index_embedding is not None:
            index = self._atom_index(batch, counts)
            x = x + self.index_embedding(index)
        extra = {}
        if self.diabatic:
            raw, energies = self._diabatic(x, index, batch, count)
            extra['excitation_energies'] = energies[0] if single else energies
        else:
            if 'excitation_energies' in set(properties):
                raise ValueError('excitation_energies requires diabatic_states > 0')
            raw = self.charge_head(x)                                      # [A, S]
            if self.global_basis:
                raw = raw + self._global_charges(x, index, batch, count)
        atoms = counts.to(raw.dtype)
        mean = segment_sum(raw, batch, count) / atoms[:, None]
        q = raw - mean[batch]                                              # sum_i q_i = 0 per structure/state
        result = {'transition_charges': q, **extra}
        if 'transition_dipole' in set(properties):
            local = q[:, :, None] * pos[:, None, :]                       # [A, S, 3]
            if self.config['atomic_dipoles']:
                p = self._atomic_dipoles(x, pos, batch)
                result['transition_atomic_dipoles'] = p
                local = local + p
            mu = segment_sum(local, batch, count)                          # [B, S, 3]
            result['transition_dipole'] = mu[0] if single else mu
        return result
