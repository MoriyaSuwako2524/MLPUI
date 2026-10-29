"""Variational charge equilibration (QEq) electrostatics for QM/MM: E_charge.

Per structure the QM charges minimise

    E_QEq(q; phi) = sum_i chi_i q_i + 1/2 q^T H q + sum_i q_i phi_i,   sum_i q_i = Q

with H_ii = J_i + k/(sigma_i sqrt(pi)) (Gaussian self-interaction) and
H_ij = k erf(r_ij / sqrt(2(sigma_i^2 + sigma_j^2))) / r_ij. phi_i is the potential
of the fixed MM point charges at QM atom i, screened by the Gaussian of atom i.
chi_i(R) and J_i(R) come from a QM-only TensorNet; sigma is per element.

The model energy is the environment part only:

    E_charge = E_QEq(q*; phi) - E_QEq(q0; 0),

so E_charge = 0 without MM. q0 (gas) and q* (polarized) are both stationary
points, hence first derivatives need no dq/dR or dq/dr_MM terms (Hellmann-
Feynman); autograd through the linear solve reproduces exactly that. Outputs:

* energy, forces = -dE/dR, mm_forces = -dE/dr_MM
* mm_potential = dE/dQ_MM = ESP of q* at MM sites (eV/e)
* mm_potential_gradient = spatial gradient of that ESP with q* fixed
  (= -field; the same fixed-density definition as the Q-Chem labels)
* gas_mm_potential / gas_mm_potential_gradient: the same for q0
* charges = q* (polarized), gas_charges = q0

Units: angstrom, eV, e. Nonperiodic.
"""
import copy
import math

import torch
from torch import nn

COULOMB_EV_ANGSTROM = 14.3996454784255


def _pad(values, batch, count, fill=0.):
    """Scatter [A, ...] rows into [B, Nmax, ...] plus a boolean mask and the gather index."""
    counts = torch.bincount(batch, minlength=count)
    nmax = int(counts.max()) if len(batch) else 0
    starts = torch.cumsum(counts, 0) - counts
    perm = torch.argsort(batch, stable=True)
    rows = batch[perm]
    cols = torch.arange(len(batch), device=batch.device) - starts[rows]
    out = values.new_full((count, nmax) + tuple(values.shape[1:]), fill)
    out = out.index_put((rows, cols), values[perm])
    mask = torch.zeros(count, nmax, dtype=torch.bool, device=batch.device)
    mask[rows, cols] = True
    return out, mask, (perm, rows, cols)


def _unpad(padded, index):
    perm, rows, cols = index
    values = padded[rows, cols]
    return values[torch.argsort(perm)]


def _gauss_kernel(r, width):
    """erf(r/width)/r and its radial derivative (both finite for r > 0)."""
    erf = torch.erf(r / width)
    f = erf / r
    df = (2. / (math.sqrt(math.pi) * width)) * torch.exp(-(r / width) ** 2) / r - erf / r ** 2
    return f, df


class QEqCharge(nn.Module):
    ARCHITECTURE = 'qeq_tensornet_v1'

    def __init__(self, config):
        super().__init__()
        from mlpui.models.torchmdnet.models.tensornet import TensorNet
        if config.get('schema_version') != 1 or config.get('architecture') != self.ARCHITECTURE:
            raise ValueError('Incompatible QEq architecture')
        self.config = copy.deepcopy(config)
        self.mlpui_family = 'qeq_charge'
        dtype = torch.float64 if config['precision'] == 64 else torch.float32
        self.register_buffer('qeq_schema', torch.tensor(1, dtype=torch.long))
        width = config['embedding_dimension']
        self.representation = TensorNet(
            hidden_channels=width, num_layers=config['num_layers'], num_rbf=config['num_rbf'],
            max_num_neighbors=config['max_num_neighbors'], cutoff_upper=config['cutoff_upper'],
            static_shapes=False, equivariance_invariance_group='O(3)', dtype=dtype)
        self.head = nn.Sequential(nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                  nn.Linear(width, 2, dtype=dtype))
        # Zero last layer: training starts from the per-element (geometry-independent) model.
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)
        self.chi_element = nn.Embedding(119, 1, dtype=dtype)
        self.hardness_element = nn.Embedding(119, 1, dtype=dtype)
        self.width_element = nn.Embedding(119, 1, dtype=dtype)
        for table in (self.chi_element, self.hardness_element, self.width_element):
            nn.init.zeros_(table.weight)
        self.to(dtype=dtype)

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.config.get('architecture') != self.ARCHITECTURE or self.qeq_schema.item() != 1:
            raise ValueError('Incompatible QEq serialized architecture')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        schema = state_dict.get('qeq_schema')
        if schema is None or schema.numel() != 1 or schema.item() != 1:
            raise ValueError('Incompatible QEq checkpoint: requires qeq_schema 1')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _bounded(raw, bounds):
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    def atomic_parameters(self, z, pos, batch, charge):
        """chi [A] (eV/e), J [A] (eV/e^2), sigma [A] (angstrom)."""
        x, _, _, _, _ = self.representation(z, pos, batch, q=charge)
        delta = self.head(x)
        chi = self.chi_element(z).squeeze(-1) + delta[:, 0]
        hardness = self._bounded(self.hardness_element(z).squeeze(-1) + delta[:, 1], self.config['hardness_bounds'])
        width = self._bounded(self.width_element(z).squeeze(-1), self.config['sigma_bounds'])
        return chi, hardness, width

    def _solve(self, chi, hardness, width, pos, mask, charge, phi):
        """Return (q*, q0, E_QEq(q*;phi) - E_QEq(q0;0)) for padded [B, N] tensors."""
        count, n = mask.shape
        pair = mask[:, :, None] & mask[:, None, :]
        offdiag = pair & ~torch.eye(n, dtype=torch.bool, device=pos.device)[None]
        d2 = (pos[:, :, None] - pos[:, None]).square().sum(-1)
        r = torch.sqrt(d2 + (~offdiag).to(pos.dtype))          # safe: 1 where not a real pair
        sw = torch.sqrt(2. * (width[:, :, None] ** 2 + width[:, None] ** 2))
        gamma = COULOMB_EV_ANGSTROM * torch.erf(r / sw) / r * offdiag
        self_term = hardness + COULOMB_EV_ANGSTROM / (width * math.sqrt(math.pi))
        diag = torch.where(mask, self_term, torch.ones_like(self_term))   # padded atoms: q = 0
        h = gamma + torch.diag_embed(diag)
        ones = mask.to(pos.dtype)
        border = torch.cat([ones, pos.new_zeros(count, 1)], 1)             # constraint row/column
        a = torch.cat([torch.cat([h, ones[:, :, None]], 2), border[:, None, :]], 1)

        def solve(rhs_phi):
            rhs = torch.cat([-(chi + rhs_phi) * ones, charge[:, None]], 1)
            q = torch.linalg.solve(a, rhs)[:, :n] * ones
            energy = ((chi + rhs_phi) * q * ones).sum(1) + 0.5 * torch.einsum('bi,bij,bj->b', q, h * pair, q)
            return q, energy
        q1, e1 = solve(phi)
        q0, e0 = solve(torch.zeros_like(phi))
        return q1, q0, e1 - e0

    def forward(self, z, pos, mm_pos, mm_charges, charge, *, properties=('energy',),
                baseline=None, batch=None, mm_batch=None, mm_offsets=None):
        from mlpui.models.tensornet_qmmm.model import TensorNetQMMM
        properties = set(properties)
        qb, mb, single, count = TensorNetQMMM.region_batches(z, mm_pos, batch, mm_batch, mm_offsets)
        charge = charge.reshape(-1)
        if charge.numel() != count:
            raise ValueError('Supply one total QM charge per structure')
        with torch.enable_grad():
            if properties & {'forces', 'mm_forces'}:
                pos.requires_grad_(True)
                mm_pos.requires_grad_(True)
            chi, hardness, width = self.atomic_parameters(z, pos, qb, charge)
            P, mask, qidx = _pad(pos, qb, count)
            C, _, _ = _pad(chi, qb, count)
            J, _, _ = _pad(hardness, qb, count)
            W, _, _ = _pad(width, qb, count, fill=1.)
            if len(mm_pos):
                R, mmask, midx = _pad(mm_pos, mb, count)
                Q, _, _ = _pad(mm_charges, mb, count)
            else:
                R, Q = pos.new_zeros(count, 0, 3), pos.new_zeros(count, 0)
                mmask, midx = torch.zeros(count, 0, dtype=torch.bool, device=pos.device), None
            pairm = mask[:, :, None] & mmask[:, None, :]                       # [B, N, M]
            dvec = P[:, :, None] - R[:, None]                                  # R_i - r_M
            r = torch.sqrt(dvec.square().sum(-1) + (~pairm).to(pos.dtype))
            if torch.any(pairm & (r < 1e-6)):
                raise ValueError('Coincident QM/MM sites are not supported')
            f, df = _gauss_kernel(r, math.sqrt(2.) * W[:, :, None])
            f, df = f * pairm, df * pairm
            phi = COULOMB_EV_ANGSTROM * (f * Q[:, None, :]).sum(-1)            # MM potential at QM atoms
            q1, q0, energy = self._solve(C, J, W, P, mask, charge, phi)
            energy = energy + 0 * (pos.sum() + mm_pos.sum() + mm_charges.sum())
            result = {'energy': energy[0] if single else energy,
                      'components': {'charge_electrostatics': energy[0] if single else energy}}
            unit = dvec / r[..., None]                                         # (R_i - r_M)/|R_i - r_M|

            def esp(q):
                pot = COULOMB_EV_ANGSTROM * (q[:, :, None] * f).sum(1)            # [B, M]
                # d/dr_M of erf(|R_i - r_M|/a)/|R_i - r_M| = -df * unit
                grad = -COULOMB_EV_ANGSTROM * (q[:, :, None, None] * df[..., None] * unit).sum(1)
                return pot, grad
            wanted = {}
            if properties & {'mm_potential', 'mm_potential_gradient'}:
                wanted['mm_potential'], wanted['mm_potential_gradient'] = esp(q1)
            if properties & {'gas_mm_potential', 'gas_mm_potential_gradient'}:
                wanted['gas_mm_potential'], wanted['gas_mm_potential_gradient'] = esp(q0)
            for key, value in wanted.items():
                if key in properties:
                    result[key] = (_unpad(value, midx) if midx is not None
                                   else value.reshape((0,) + tuple(value.shape[2:])))
            result['charges'] = _unpad(q1, qidx)
            result['gas_charges'] = _unpad(q0, qidx)
            if properties & {'forces', 'mm_forces'}:
                grads = torch.autograd.grad(energy.sum(), (pos, mm_pos), create_graph=self.training,
                                            retain_graph=True, allow_unused=True)
                anchor = energy.sum() * 0
                result['forces'] = -(grads[0] if grads[0] is not None else torch.zeros_like(pos)) + anchor
                result['mm_forces'] = -(grads[1] if grads[1] is not None else torch.zeros_like(mm_pos)) + anchor
            return result
