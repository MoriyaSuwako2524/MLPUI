"""Concrete potential terms (angstrom / eV / e)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import (MMDependence, PotentialTerm, QMMMInput, TorchPotentialTerm,
                   checkpoint_metadata)

COULOMB_EV_ANGSTROM = 14.3996454784255   # same constant as the MLPUI QM/MM models


class ConstantEnergy(PotentialTerm):
    """Constant energy shift, e.g. the train-mean (E_ref - E_UMA) offset."""
    properties = frozenset({'energy', 'forces'})
    mm_dependence = MMDependence.NONE

    def __init__(self, energy_eV, name='offset', source=None):
        super().__init__(name)
        self.energy_eV = float(energy_eV)
        if not np.isfinite(self.energy_eV):
            raise ValueError('offset must be finite')
        self.source = source

    @classmethod
    def from_manifest(cls, path, key='energy_offset_eV', name='offset'):
        value = json.loads(Path(path).read_text())[key]
        return cls(value, name=name, source={**checkpoint_metadata(path), 'key': key})

    def _compute(self, inputs, properties):
        return {'energy': self.energy_eV, 'forces': np.zeros((inputs.n_qm, 3))}

    def metadata(self):
        return {**super().metadata(), 'energy_eV': self.energy_eV, 'source': self.source}


class FixedChargeCoulomb(PotentialTerm):
    """E = k sum_iM q_i Q_M / |R_i - r_M| with fixed QM point charges q_i.

    A simple electrostatic baseline and an exact analytic oracle for tests.
    """
    properties = frozenset({'energy', 'forces', 'mm_potential', 'mm_potential_gradient', 'mm_forces'})
    mm_dependence = MMDependence.LINEAR_CHARGE

    def __init__(self, qm_charges, name='fixed_charge_coulomb'):
        super().__init__(name)
        self.qm_charges = np.asarray(qm_charges, dtype=np.float64).reshape(-1)

    def _compute(self, inputs, properties):
        if len(self.qm_charges) != inputs.n_qm:
            raise ValueError(f'{self.name}: {len(self.qm_charges)} charges for {inputs.n_qm} QM atoms')
        d = inputs.positions[:, None, :] - inputs.mm_positions[None, :, :]       # R_i - r_M
        r = np.linalg.norm(d, axis=-1)
        if np.any(r < 1e-6):
            raise ValueError('coincident QM/MM sites')
        k, q, Q = COULOMB_EV_ANGSTROM, self.qm_charges, inputs.mm_charges
        inv, inv3 = 1 / r, 1 / r ** 3
        phi = k * (q[:, None] * inv).sum(0)                                     # [M]
        grad_phi = k * (q[:, None, None] * d * inv3[..., None]).sum(0)          # d phi_M / d r_M
        return {'energy': float(phi @ Q),
                'forces': k * q[:, None] * (Q[None, :, None] * d * inv3[..., None]).sum(1),
                'mm_potential': phi, 'mm_potential_gradient': grad_phi,
                'mm_forces': -Q[:, None] * grad_phi}

    def metadata(self):
        return {**super().metadata(), 'qm_charges': self.qm_charges.tolist()}


class ASECalculatorTerm(PotentialTerm):
    """Gas-phase (MM-independent) term from any ASE calculator.

    ``charge``/``spin`` are written to ``atoms.info`` (FAIR-Chem convention).
    Calculators that fix charge/spin at construction (MLPUI native UMA) should
    be given ``expected_charge``/``expected_spin`` so a mismatch with the
    driver's QM charge fails loudly instead of silently.
    """
    properties = frozenset({'energy', 'forces'})
    mm_dependence = MMDependence.NONE

    def __init__(self, calculator, name, *, expected_charge=None, expected_spin=None, info=None):
        super().__init__(name)
        self.calculator = calculator
        self.expected_charge, self.expected_spin = expected_charge, expected_spin
        self.info = dict(info or {})
        self._atoms = None

    def _compute(self, inputs, properties):
        from ase import Atoms
        if self.expected_charge is not None and inputs.charge != self.expected_charge:
            raise ValueError(f'{self.name}: calculator built for charge {self.expected_charge}, got {inputs.charge}')
        if self.expected_spin is not None and inputs.spin != self.expected_spin:
            raise ValueError(f'{self.name}: calculator built for spin {self.expected_spin}, got {inputs.spin}')
        if self._atoms is None or not np.array_equal(self._atoms.numbers, inputs.numbers):
            self._atoms = Atoms(numbers=inputs.numbers, positions=inputs.positions)
            self._atoms.calc = self.calculator
        else:
            self._atoms.set_positions(inputs.positions)
        self._atoms.info.update(charge=inputs.charge, spin=inputs.spin)
        out = {'energy': self._atoms.get_potential_energy()}
        if 'forces' in properties:
            out['forces'] = self._atoms.get_forces()
        return out

    def metadata(self):
        return {**super().metadata(), 'calculator': type(self.calculator).__name__, **self.info}


def uma_term(checkpoint, *, charge, spin=1, task='omol', device='cpu', implementation='native',
             name='uma', hash_checkpoint=True, dtype=None):
    """UMA gas-phase term. implementation: 'native' (MLPUI predictor) or 'fairchem'.

    dtype ('float32'/'float64'/torch dtype) applies to the native predictor; None keeps
    the checkpoint precision. FAIR-Chem manages its own precision.
    """
    if charge is None:
        raise ValueError('uma_term needs the total QM charge')
    if implementation == 'native':
        import torch
        from mlpui.calculator import CalculatorBuilder
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        calc = CalculatorBuilder.from_checkpoint(str(checkpoint), task=task, charge=charge, spin=spin,
                                                 device=device, dtype=torch_dtype).build()
        fixed = dict(expected_charge=charge, expected_spin=spin)
    elif implementation == 'fairchem':
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(str(checkpoint), device=device),
                                  task_name=task)
        fixed = {}
    else:
        raise ValueError("implementation must be 'native' or 'fairchem'")
    info = {'implementation': implementation, 'task': task, 'device': device}
    info.update(checkpoint_metadata(checkpoint) if hash_checkpoint else {'checkpoint': str(checkpoint)})
    return ASECalculatorTerm(calc, name, info=info, **fixed)


class MLPUICheckpointTerm(TorchPotentialTerm):
    """An MLPUI QM/MM checkpoint (qeq_charge or tensornet_qmmm, baseline residual_only).

    Returns E, F_QM, phi = dE/dQ_M and GESP from the model's own autograd /
    Hellmann-Feynman definitions. Both families are linear in each MM charge.
    """
    properties = frozenset({'energy', 'forces', 'mm_potential', 'mm_potential_gradient'})
    mm_dependence = MMDependence.LINEAR_CHARGE
    FAMILIES = ('qeq_charge', 'tensornet_qmmm')

    def __init__(self, checkpoint, name, *, family=None, device='cpu', dtype='float64'):
        super().__init__(name, device=device, dtype=dtype)
        torch = self.torch
        from mlpui.backends import get_backend
        from mlpui.external_models import load_external_checkpoint
        self.checkpoint = Path(checkpoint)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        family = family or state.get('backend')
        if family is None:
            raise ValueError(f'{self.name}: checkpoint has no backend; pass family=')
        self.backend = get_backend(family)
        self.family = self.backend.name
        if self.family not in self.FAMILIES:
            raise ValueError(f'{self.name}: family {self.family!r} is not a QM/MM family {self.FAMILIES}')
        config = self.backend.training_config(dict(state['model_config']), ['energy'], self.dtype)
        if config.get('baseline', {}).get('mode') != 'residual_only':
            raise ValueError(f'{self.name}: only residual_only checkpoints can be composed')
        patcher = load_external_checkpoint(self.checkpoint, family=self.family, model_config=config,
                                           device=self.device, dtype=self.dtype)
        self.model = patcher.model.to(device=self.device, dtype=self.dtype).eval()
        self.model_config = config
        self._provenance = checkpoint_metadata(self.checkpoint)

    def _compute(self, inputs: QMMMInput, properties):
        if inputs.spin != 1:
            raise ValueError(f'{self.name}: MLPUI QM/MM families support singlets only (spin=1)')
        sample = {'z': inputs.numbers, 'pos': inputs.positions, 'mm_pos': inputs.mm_positions,
                  'mm_charges': inputs.mm_charges, 'charge': float(inputs.charge)}
        model_inputs = self.backend.prepare_sample(sample, self.device, self.dtype, model=self.model)
        out = self.backend.forward(self.model, model_inputs, sorted(properties | {'energy'}))
        result = {k: self.numpy(out[k]) for k in properties}
        for key in ('charges', 'gas_charges'):
            if key in out:
                self.extras[key] = self.numpy(out[key])
        return result

    def metadata(self):
        return {**super().metadata(), 'family': self.family, **self._provenance,
                'model_config': self.model_config}
