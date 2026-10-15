"""Finite, fixed-environment TensorNet adaptation (not the Pan/DeepPot-SE model).

Units: angstrom, eV, elementary charge. MM points are selected upstream within
10 A of a reference QM geometry and kept fixed; there is no dynamic truncation.
"""
import copy
import torch
from torch import nn

COULOMB_EV_ANGSTROM = 14.3996454784255


def environment(pos, mm_pos, mm_charges):
    """Differentiable electrostatic potential and polar electric field at QM sites."""
    delta = pos[:, None, :] - mm_pos[None, :, :]
    distance2 = delta.square().sum(-1)
    if torch.any(distance2 < 1.e-12):
        raise ValueError('Coincident QM/MM sites are not supported')
    inv_r = distance2.rsqrt()
    phi = COULOMB_EV_ANGSTROM * (inv_r * mm_charges).sum(-1)
    field = COULOMB_EV_ANGSTROM * (delta * (inv_r.pow(3) * mm_charges)[..., None]).sum(1)
    return phi, field


def fixed_response_mm_properties(pos, mm_pos, potential_response, field_response):
    """Potential and spatial GESP of the model's instantaneous QM response.

    With r = R_QM - R_MM, hold dE/dphi and dE/dfield fixed when taking
    the spatial derivative of the Coulomb kernels. This excludes response
    derivatives (the Hessian terms in d(dE/dq_MM)/dR_MM). Do NOT detach the
    coefficients here: training must still differentiate them w.r.t. weights.
    No division by MM charges is used, including at q_MM=0.
    """
    r = pos[:, None, :] - mm_pos[None, :, :]
    r2 = r.square().sum(-1)
    if torch.any(r2 < 1.e-12):
        raise ValueError('Coincident QM/MM sites are not supported')
    inv = r2.rsqrt()
    a = potential_response[:, None]
    b = field_response[:, None, :]
    dot = (b * r).sum(-1)
    potential = COULOMB_EV_ANGSTROM * (a * inv + dot * inv.pow(3)).sum(0)
    gradient = COULOMB_EV_ANGSTROM * (
        a[..., None] * r * inv.pow(3)[..., None]
        + 3 * dot[..., None] * r * inv.pow(5)[..., None]
        - b * inv.pow(3)[..., None]).sum(0)
    return potential, gradient


class GasCoulombBaseline(nn.Module):
    """Composition hook for real differentiable gas providers; no fake UMA loader.

    gas_energy(z, pos, charge) -> scalar eV; gas_charge(z, pos, charge) -> [N] e.
    Providers must be deterministic differentiable Torch modules. Parameters
    remain frozen, but dq/dR and dEgas/dR remain in the graph. This Python hook
    is not yet serialized by the CLI backend; do not replace providers at load.
    """
    def __init__(self, gas_energy, gas_charge):
        super().__init__()
        self.gas_energy, self.gas_charge = gas_energy, gas_charge
        for module in (gas_energy, gas_charge):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.gas_energy.eval()
        self.gas_charge.eval()
        return self

    def projected_charges(self, z, pos, charge):
        """Conserve total QM charge including its coordinate Jacobian.

        This explicit uniform projection is part of the online baseline definition.
        It must also be used when generating matching offline baselines.
        """
        if z.ndim != 1 or len(z) == 0 or charge.numel() != 1 or not torch.isfinite(charge).all():
            raise ValueError('Gas baseline requires atoms and one finite total QM charge')
        q = self.gas_charge(z, pos, charge)
        if q.shape != z.shape or not torch.isfinite(q).all():
            raise ValueError('Gas charge provider must return one finite charge per QM atom')
        return q - q.mean() + charge.reshape(()) / len(z)

    def forward(self, z, pos, mm_pos, mm_charges, charge):
        e = self.gas_energy(z, pos, charge)
        q = self.projected_charges(z, pos, charge)
        if e.numel() != 1 or not torch.isfinite(e).all():
            raise ValueError('Gas energy provider must return one finite energy')
        phi, _ = environment(pos, mm_pos, mm_charges)
        return {'gas': e.reshape(()), 'coulomb': (q * phi).sum()}

    def compose(self, residual_model, inputs):
        """Return total energy/components for an explicitly supplied online provider pair.

        Inputs are single-structure Torch tensors. Use autograd on energy for
        all derivatives. Online provider serialization is deliberately not part
        of the CLI checkpoint format yet.
        """
        values = {k: inputs[k] for k in ('z', 'pos', 'mm_pos', 'mm_charges', 'charge')}
        parts = self(**values)
        parts.update(residual_model.energy_components(**values))
        return {'energy': sum(parts.values()), 'components': parts}


class TensorNetQMMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        from mlpui.models.torchmdnet.models.tensornet import TensorNet
        self.config = copy.deepcopy(config)
        self.mlpui_family = 'tensornet_qmmm'
        dtype = torch.float64 if config['precision'] == 64 else torch.float32
        self.register_buffer('qmmm_schema', torch.tensor(1, dtype=torch.long))
        self.qm_representation = TensorNet(
            hidden_channels=config['embedding_dimension'], num_layers=config['num_layers'],
            num_rbf=config['num_rbf'], max_num_neighbors=config['max_num_neighbors'],
            cutoff_upper=config['cutoff_upper'], static_shapes=False,
            equivariance_invariance_group='O(3)', dtype=dtype)
        width = config['embedding_dimension']
        # x, phi, |E|^2 and E.g are O(3) scalars. g is a QM geometric polar vector.
        self.residual_head = nn.Sequential(nn.Linear(width + 3, width, dtype=dtype),
            nn.SiLU(), nn.Linear(width, 1, dtype=dtype))
        # Some vendored TensorNet sublayers initialize in the default dtype.
        # Convert before load_state_dict to avoid rounding float64 checkpoints.
        self.to(dtype=dtype)

    def _energy_with_environment(self, z, pos, mm_pos, mm_charges, charge):
        batch = torch.zeros_like(z)
        x, _, _, _, _ = self.qm_representation(z, pos, batch, q=charge.reshape(1))
        phi, field = environment(pos, mm_pos, mm_charges)
        displacement = pos[:, None, :] - pos[None, :, :]
        # No coordinate-dependent discrete selection in this small first-release descriptor.
        geometry = (displacement * torch.exp(-displacement.square().sum(-1))[..., None]).sum(1)
        scaled_phi = phi / self.config['potential_scale']
        scaled_field = field / self.config['field_scale']
        invariant = torch.stack((scaled_phi, scaled_field.square().sum(-1), (scaled_field * geometry).sum(-1)), -1)
        residual = self.residual_head(torch.cat((x, invariant), -1)).sum()
        return {'residual': residual}, phi, field

    def energy_components(self, z, pos, mm_pos, mm_charges, charge):
        return self._energy_with_environment(z, pos, mm_pos, mm_charges, charge)[0]

    def forward(self, z, pos, mm_pos, mm_charges, charge, *, properties=('energy',), baseline=None):
        properties = set(properties)
        derivatives = properties & {'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient'}
        with torch.enable_grad():
            if derivatives:
                pos.requires_grad_(True)
                mm_pos.requires_grad_(True)
                mm_charges.requires_grad_(True)
            components, phi, field = self._energy_with_environment(z, pos, mm_pos, mm_charges, charge)
            residual = components['residual']
            # Explicit zero connections keep zero-MM derivatives well defined.
            residual = residual + 0 * (pos.sum() + mm_pos.sum() + mm_charges.sum())
            result = {'energy': residual, 'components': components}
            if derivatives:
                wants_gradient = 'mm_potential_gradient' in properties
                variables = (pos, mm_pos, mm_charges) + ((phi, field) if wants_gradient else ())
                grads = torch.autograd.grad(residual, variables,
                    create_graph=self.training, retain_graph=self.training)
                gq, gm, gp = grads[:3]
                anchor = residual * 0  # permits force-only batches with zero MM points
                values = dict(forces=-gq + anchor, mm_forces=-gm + anchor, mm_potential=gp + anchor)
                if wants_gradient:
                    _, gradient = fixed_response_mm_properties(pos, mm_pos, *grads[3:])
                    values['mm_potential_gradient'] = gradient + anchor
                result.update({key: value for key, value in values.items() if key in properties})
            if self.config['baseline']['mode'] == 'offline':
                baseline = baseline or {}
                for prop in sorted(properties | {'energy'}):
                    key = 'baseline_' + prop
                    if key not in baseline:
                        raise ValueError(f'Offline baseline requires {key}')
                    result[prop] = result[prop] + baseline[key]
                components['offline_gas_plus_coulomb'] = baseline['baseline_energy']
                result['derivative_contract'] = 'offline_component_values_not_total_energy_autograd'
            return result
