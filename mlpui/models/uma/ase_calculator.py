"""ASE boundary for MLPUI's native UMA predictor."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch
import yaml
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError

from mlpui.models.common.atomic_data import AtomicData


class NativeUMACalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, builder, device):
        super().__init__()
        if set(builder.properties) - set(self.implemented_properties):
            raise PropertyNotImplementedError('Native UMA currently supports energy and forces only')
        if builder.energy_to_ev != 1 or builder.length_to_angstrom != 1:
            raise ValueError('UMA outputs are already in eV and Angstrom')
        self.patcher = builder.model_patcher
        self.model = self.patcher.model
        self.device = torch.device(device)
        self.dtype = builder.dtype or next(self.model.parameters()).dtype
        self.task = builder.task or 'omol'
        if self.task != 'omol':
            raise ValueError('Native UMA currently supports the omol molecular task only')
        self.charge = builder.charge if builder.charge is not None else 0
        self.spin = builder.spin if builder.spin is not None else 1
        if isinstance(self.charge, bool) or not isinstance(self.charge, (int, np.integer)) or not -100 <= self.charge <= 100:
            raise ValueError('UMA charge must be an integer in [-100, 100]')
        if isinstance(self.spin, bool) or not isinstance(self.spin, (int, np.integer)) or not 1 <= self.spin <= 100:
            raise ValueError('omol spin multiplicity must be an integer in [1, 100]')
        self.keep_on_device = builder.keep_on_device
        refs = builder.atom_refs
        if isinstance(refs, (str, Path)):
            with open(refs, encoding='utf-8') as stream:
                refs = yaml.safe_load(stream)
        self.atom_refs = refs
        for prop in self.implemented_properties:
            self.model.task_config(self.task, prop)
        energy_norm = self.model.task_config(self.task, 'energy')['normalizer']
        force_norm = self.model.task_config(self.task, 'forces')['normalizer']
        if not np.isclose(float(energy_norm['rmsd']), float(force_norm['rmsd'])) or float(force_norm['mean']) != 0:
            raise ValueError('Checkpoint energy/force normalization is not conservative')

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if len(atoms) == 0 or not np.isfinite(atoms.positions).all():
            raise ValueError('UMA requires nonempty finite atomic coordinates')
        if np.any(atoms.pbc):
            raise ValueError('Native UMA currently supports nonperiodic molecules only')
        if np.any(atoms.numbers < 1) or np.any(atoms.numbers >= self.model.backbone.max_num_elements):
            raise ValueError('Atomic number outside checkpoint embedding range')
        if len(atoms) == 1:
            refs = self.atom_refs or {}
            refs = refs.get(self.task, refs.get(f'{self.task}_elem_refs', {}))
            entry = refs.get(int(atoms.numbers[0])) if isinstance(refs, dict) else refs[int(atoms.numbers[0])]
            energy = entry.get(self.charge) if isinstance(entry, dict) else entry
            if energy is None or not np.isfinite(energy):
                raise ValueError('Single atoms require an isolated atom reference for the element and charge')
            warnings.warn('Single-atom energy uses supplied DFT reference; spin is ignored.', UserWarning)
            self.results = {'energy': float(energy), 'forces': np.zeros((1, 3))}
            return
        inputs = atoms.copy()
        inputs.info = {'charge': int(self.charge), 'spin': int(self.spin)}
        data = AtomicData.from_ase(inputs, task_name=self.task, r_edges=False,
            r_data_keys=['charge', 'spin'], target_dtype=self.dtype)
        data['batch'] = torch.zeros(len(atoms), dtype=torch.long)
        data = data.to(self.device)
        try:
            self.model.to(device=self.device, dtype=self.dtype).eval()
            if self.patcher.num_patches:
                self.patcher.patch_model(device=self.device)
            output = self.model(data, self.task)
            energy = float(output['energy'].cpu().reshape(()))
            forces = output['forces'].cpu().numpy().reshape(len(atoms), 3)
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError('UMA returned non-finite energy or forces')
            self.results = {'energy': energy, 'forces': forces}
        finally:
            if self.patcher.is_patched:
                self.patcher.unpatch_model()
            if not self.keep_on_device:
                self.patcher.offload()
