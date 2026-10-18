"""TensorNet-ext residual plus explicitly supplied UMA/gas-charge Coulomb baseline.

Finite reference-selected 10 A environment; nonperiodic. The external neural
message has its own configurable cosine cutoff, independent of data selection.
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
    """Analytic fixed-response Coulomb helper (not the ext neural descriptor).

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
    ARCHITECTURE = 'tensornet_ext_v1'

    def __init__(self, config):
        super().__init__()
        from .ext import TensorNetExt
        if config.get('schema_version') != 2 or config.get('architecture') != self.ARCHITECTURE:
            raise ValueError('Incompatible QMMM architecture: legacy H+3 checkpoints require the old implementation')
        self.config = copy.deepcopy(config)
        self.mlpui_family = 'tensornet_qmmm'
        dtype = torch.float64 if config['precision'] == 64 else torch.float32
        self.register_buffer('qmmm_schema', torch.tensor(2, dtype=torch.long))
        self.qm_representation = TensorNetExt(config, dtype)
        width = config['embedding_dimension']
        self.residual_head = nn.Sequential(nn.Linear(width, width // 2, dtype=dtype),
                                          nn.SiLU(), nn.Linear(width // 2, 1, dtype=dtype))
        for layer in (self.residual_head[0], self.residual_head[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.to(dtype=dtype)

    def __setstate__(self, state):
        # Trusted full-module pickle loading bypasses __init__/load_state_dict.
        super().__setstate__(state)
        if (self.config.get('architecture') != self.ARCHITECTURE
                or self.qmmm_schema.numel() != 1 or self.qmmm_schema.item() != 2):
            raise ValueError('Incompatible QMMM architecture: legacy H+3 serialized model is unsupported')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        schema = state_dict.get('qmmm_schema')
        if schema is None or schema.numel() != 1 or schema.item() != 2:
            raise ValueError('Incompatible QMMM architecture: legacy H+3 weights cannot load into TensorNet-ext')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def region_batches(z, mm_pos, batch=None, mm_batch=None, mm_offsets=None):
        single = batch is None
        batch = torch.zeros_like(z) if single else batch
        if (batch.shape != z.shape or batch.dtype != torch.long or not len(z)
                or torch.any(batch < 0)):
            raise ValueError('batch must contain one integer structure ID per QM atom')
        count = int(batch.max()) + 1
        if not torch.equal(torch.unique(batch), torch.arange(count, device=z.device)):
            raise ValueError('QM batch IDs must be contiguous from zero')
        if mm_offsets is not None:
            if mm_batch is not None:
                raise ValueError('Specify mm_batch or mm_offsets, not both')
            if (mm_offsets.dtype != torch.long or mm_offsets.shape != (count + 1,)
                    or mm_offsets[0] != 0 or mm_offsets[-1] != len(mm_pos)
                    or torch.any(mm_offsets[1:] < mm_offsets[:-1])):
                raise ValueError('Invalid MM offsets')
            mm_batch = torch.repeat_interleave(torch.arange(count, device=z.device),
                                               mm_offsets[1:] - mm_offsets[:-1])
        if mm_batch is None:
            if count != 1:
                raise ValueError('Ragged batches require mm_batch or mm_offsets')
            mm_batch = torch.zeros(len(mm_pos), device=z.device, dtype=torch.long)
        if (mm_batch.dtype != torch.long or mm_batch.shape != (len(mm_pos),)
                or torch.any((mm_batch < 0) | (mm_batch >= count))):
            raise ValueError('Invalid MM batch IDs')
        return batch, mm_batch, single, count

    def _energy_with_environment(self, z, pos, mm_pos, mm_charges, charge,
                                 batch=None, mm_batch=None, mm_offsets=None):
        batch, mm_batch, single, count = self.region_batches(z, mm_pos, batch, mm_batch, mm_offsets)
        if charge.numel() != count:
            raise ValueError('Supply one total QM charge per structure')
        message = self.qm_representation.ext_layer(pos, mm_pos, mm_charges, batch, mm_batch)
        x = self.qm_representation(z, pos, batch, message)
        energy = pos.new_zeros(count).index_add(0, batch, self.residual_head(x).flatten())
        return {'residual': energy[0] if single else energy}, message, batch, mm_batch

    def energy_components(self, z, pos, mm_pos, mm_charges, charge, **kwargs):
        return self._energy_with_environment(z, pos, mm_pos, mm_charges, charge, **kwargs)[0]

    def forward(self, z, pos, mm_pos, mm_charges, charge, *, properties=('energy',),
                baseline=None, batch=None, mm_batch=None, mm_offsets=None):
        properties = set(properties)
        derivatives = properties & {'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient'}
        with torch.enable_grad():
            if derivatives:
                pos.requires_grad_(True)
                mm_pos.requires_grad_(True)
                mm_charges.requires_grad_(True)
            components, message, batch, mm_batch = self._energy_with_environment(
                z, pos, mm_pos, mm_charges, charge, batch, mm_batch, mm_offsets)
            residual = components['residual'] + 0 * (pos.sum() + mm_pos.sum() + mm_charges.sum())
            result = {'energy': residual, 'components': components}
            if derivatives:
                wants_gradient = 'mm_potential_gradient' in properties
                variables = (pos, mm_pos, mm_charges) + ((message,) if wants_gradient else ())
                grads = torch.autograd.grad(residual.sum(), variables,
                    create_graph=self.training, retain_graph=self.training or wants_gradient)
                gq, gm, gp = grads[:3]
                anchor = residual.sum() * 0
                values = dict(forces=-gq + anchor, mm_forces=-gm + anchor, mm_potential=gp + anchor)
                if wants_gradient:
                    # c=dE/dm includes all layers reusing m. Rprobe is created AFTER c:
                    # d/dRprobe holds c fixed spatially, without detaching dc/dtheta.
                    # Hence G_a=sum_i c_i:dK_ia/dR_a = (dE/dR_a)/q_a for q_a!=0.
                    # This also defines the finite continuous q_a=0 limit.
                    probe = mm_pos.clone()
                    unit_message = self.qm_representation.ext_layer(
                        pos, probe, torch.ones_like(mm_charges), batch, mm_batch)
                    gradient = torch.autograd.grad((grads[3] * unit_message).sum(), probe,
                        create_graph=self.training, retain_graph=self.training)[0]
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
