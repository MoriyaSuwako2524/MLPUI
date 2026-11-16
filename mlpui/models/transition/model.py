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

Representation: QM-only O(3) TensorNet (scalar features, invariant under
reflection). Units: angstrom, e, e*angstrom. Nonperiodic. Gas phase only.
"""
import copy

import torch
from torch import nn

from .phase import segment_sum


class TransitionTensorNet(nn.Module):
    ARCHITECTURE = 'tensornet_transition_v1'

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
        if self.index_embedding is not None:
            x = x + self.index_embedding(self._atom_index(batch, counts))
        raw = self.charge_head(x)                                          # [A, S]
        atoms = counts.to(raw.dtype)
        mean = segment_sum(raw, batch, count) / atoms[:, None]
        q = raw - mean[batch]                                              # sum_i q_i = 0 per structure/state
        result = {'transition_charges': q}
        if 'transition_dipole' in set(properties):
            local = q[:, :, None] * pos[:, None, :]                       # [A, S, 3]
            if self.config['atomic_dipoles']:
                p = self._atomic_dipoles(x, pos, batch)
                result['transition_atomic_dipoles'] = p
                local = local + p
            mu = segment_sum(local, batch, count)                          # [B, S, 3]
            result['transition_dipole'] = mu[0] if single else mu
        return result
