"""ASE Calculator view of any PotentialTerm (QM atoms = Atoms, MM = set_mm)."""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .base import ADDITIVE_PROPERTIES, PotentialTerm, QMMMInput


class QMMMCalculator(Calculator):
    """ASE calculator for the QM region in a fixed MM point-charge embedding.

    MM sites are not ASE atoms: pass them with ``set_mm`` (which invalidates
    cached results). Extra results: ``mm_potential``, ``mm_potential_gradient``,
    ``mm_forces`` (via ``calc.get_property(name, atoms)``), per-term
    ``components`` and ``extras`` in ``calc.results``.
    """
    implemented_properties = list(ADDITIVE_PROPERTIES)

    def __init__(self, potential: PotentialTerm, *, charge=0, spin=1, mm_positions=None, mm_charges=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.potential = potential
        self.charge, self.spin = int(charge), int(spin)
        self.mm_positions = np.zeros((0, 3))
        self.mm_charges = np.zeros(0)
        if mm_positions is not None:
            self.set_mm(mm_positions, mm_charges)

    def set_mm(self, positions, charges):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        charges = np.asarray(charges, dtype=np.float64).reshape(-1)
        if len(positions) != len(charges):
            raise ValueError('MM positions and charges differ in length')
        self.mm_positions, self.mm_charges = positions.copy(), charges.copy()
        self.reset()

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if np.any(self.atoms.pbc):
            raise ValueError('QMMMCalculator expects a nonperiodic QM region; PBC is the driver\'s job')
        inputs = QMMMInput(self.atoms.numbers, self.atoms.positions, self.charge, self.spin,
                           self.mm_positions, self.mm_charges)
        wanted = {p for p in properties if p in ADDITIVE_PROPERTIES} | {'energy', 'forces'}
        result = self.potential.evaluate(inputs, wanted)      # unsupported requests raise
        self.results = dict(result)
        self.results['free_energy'] = result['energy']
        self.results['extras'] = dict(self.potential.extras)
        if hasattr(self.potential, 'components'):
            self.results['components'] = {k: dict(v) for k, v in self.potential.components.items()}
