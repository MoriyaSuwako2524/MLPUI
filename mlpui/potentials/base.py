"""Composable QM/MM potential terms.

Every term maps one QM/MM configuration to additive physical properties in a
single unit system (angstrom, eV, elementary charge):

    energy                  eV              total-energy contribution
    forces                  eV/A     [N,3]  -dE/dR_QM
    mm_potential            eV/e     [M]    phi_M = +dE/dQ_M
    mm_potential_gradient   eV/(e A) [M,3]  GESP: spatial gradient of the potential of the
                                            (fixed, already polarized) QM response at r_M,
                                            = -field (Q-Chem ESP_EFIELD convention). Equals
                                            d(phi_M)/d(r_M) only for a probe charge Q_M -> 0.
    mm_forces               eV/A     [M,3]  -dE/dr_M  (= -Q_M * GESP_M for LINEAR_CHARGE terms)

The convention matches the MLPUI NpyDataset labels and, after unit conversion,
the QMHub engine contract (energy, QM gradient, ESP and ESP gradient at the
embedding MM sites). Terms are combined by summation (``CompositePotential``),
so only additive properties appear here. Term-specific diagnostics (atomic
charges, energy decompositions, uncertainties) go in ``extras``; they are kept
per term and never summed.
"""
from __future__ import annotations

import abc
import enum
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping

import numpy as np

ADDITIVE_PROPERTIES = ('energy', 'forces', 'mm_potential', 'mm_potential_gradient', 'mm_forces')
MM_PROPERTIES = ('mm_potential', 'mm_potential_gradient', 'mm_forces')
UNITS = {'length': 'angstrom', 'energy': 'eV', 'charge': 'e'}


class MMDependence(enum.IntEnum):
    """How a term depends on the MM environment (ordered: NONE < LINEAR_CHARGE < GENERAL).

    NONE           E does not depend on MM sites; all MM properties are zero.
    LINEAR_CHARGE  E depends on site M only through Q_M * K(r_M, ...), so
                   dE/dr_M = Q_M * d(phi_M)/d(r_M) holds exactly. QMHub's
                   MM-gradient chain rule assumes this (true for an SCF QM
                   engine); QEq and TensorNet-QMMM satisfy it.
    GENERAL        Any other dependence; mm_forces must be computed explicitly.
    """
    NONE = 0
    LINEAR_CHARGE = 1
    GENERAL = 2


def property_shape(prop, n_qm, n_mm):
    return {'energy': (), 'forces': (n_qm, 3), 'mm_potential': (n_mm,),
            'mm_potential_gradient': (n_mm, 3), 'mm_forces': (n_mm, 3)}[prop]


@dataclass(frozen=True)
class QMMMInput:
    """One configuration: QM atoms plus (optionally) MM point charges.

    ``spin`` is the multiplicity 2S+1. MM sites are whatever embedding the
    driver supplies (e.g. QMHub's lrec-scaled + projected near-field charges);
    terms must not re-select or rescale them.
    """
    numbers: np.ndarray
    positions: np.ndarray
    charge: int = 0
    spin: int = 1
    mm_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    mm_charges: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        numbers = np.asarray(self.numbers)
        if numbers.dtype.kind not in 'iu':
            raise ValueError('numbers must be integer atomic numbers')
        numbers = numbers.astype(np.int64).reshape(-1)
        positions = np.array(self.positions, dtype=np.float64).reshape(-1, 3)
        mm_positions = np.array(self.mm_positions, dtype=np.float64).reshape(-1, 3)
        mm_charges = np.array(self.mm_charges, dtype=np.float64).reshape(-1)
        if len(numbers) == 0 or len(positions) != len(numbers):
            raise ValueError('numbers and positions must describe the same nonempty QM region')
        if np.any((numbers < 1) | (numbers > 118)):
            raise ValueError('atomic numbers must be in 1..118')
        if len(mm_positions) != len(mm_charges):
            raise ValueError('mm_positions and mm_charges must have the same length')
        if not all(np.isfinite(a).all() for a in (positions, mm_positions, mm_charges)):
            raise ValueError('non-finite QM/MM input')
        if int(self.charge) != self.charge or int(self.spin) != self.spin or self.spin < 1:
            raise ValueError('charge must be an integer and spin a multiplicity >= 1')
        for name, value in (('numbers', numbers), ('positions', positions),
                            ('mm_positions', mm_positions), ('mm_charges', mm_charges)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'charge', int(self.charge))
        object.__setattr__(self, 'spin', int(self.spin))

    @property
    def n_qm(self):
        return len(self.numbers)

    @property
    def n_mm(self):
        return len(self.mm_charges)

    def replace(self, **changes):
        values = dict(numbers=self.numbers, positions=self.positions, charge=self.charge, spin=self.spin,
                      mm_positions=self.mm_positions, mm_charges=self.mm_charges)
        values.update(changes)
        return QMMMInput(**values)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path):
    path = Path(path)
    return {'checkpoint': str(path.resolve()), 'checkpoint_sha256': file_sha256(path)}


class PotentialTerm(abc.ABC):
    """Abstract additive QM/MM potential term.

    Subclasses implement ``_compute`` and declare ``properties`` (what
    ``_compute`` can return directly) and ``mm_dependence``. Callers use
    ``evaluate``, which

    * returns zeros for MM properties of MM-independent terms,
    * derives ``mm_forces = -Q * mm_potential_gradient`` for LINEAR_CHARGE terms,
    * checks shapes and finiteness and returns float64 numpy arrays
      (energy as a Python float).

    ``_compute`` may also fill ``self.extras`` with non-additive diagnostics.
    """

    #: Additive properties ``_compute`` can return directly.
    properties: frozenset = frozenset({'energy', 'forces'})
    mm_dependence: MMDependence = MMDependence.NONE

    def __init__(self, name):
        if not isinstance(name, str) or not name or ':' in name:
            raise ValueError('term name must be a nonempty string without ":"')
        self.name = name
        self.extras: dict[str, Any] = {}

    # --- subclass API -------------------------------------------------
    @abc.abstractmethod
    def _compute(self, inputs: QMMMInput, properties: frozenset) -> Mapping[str, Any]:
        """Return a mapping holding at least the requested directly-provided properties."""

    def metadata(self) -> dict:
        """Provenance (checkpoint hashes, offsets, options). Extend in subclasses."""
        return {'name': self.name, 'class': f'{type(self).__module__}.{type(self).__qualname__}',
                'mm_dependence': self.mm_dependence.name, 'properties': sorted(self.properties)}

    # --- public API ---------------------------------------------------
    def available_properties(self) -> frozenset:
        available = set(self.properties)
        if self.mm_dependence is MMDependence.NONE:
            available |= set(MM_PROPERTIES)
        elif self.mm_dependence is MMDependence.LINEAR_CHARGE and 'mm_potential_gradient' in available:
            available.add('mm_forces')
        return frozenset(available)

    def evaluate(self, inputs: QMMMInput, properties: Collection[str] = ('energy', 'forces')) -> dict:
        wanted = frozenset(properties)
        unknown = wanted - set(ADDITIVE_PROPERTIES)
        if unknown:
            raise ValueError(f'{self.name}: not additive properties: {sorted(unknown)}')
        missing = wanted - self.available_properties()
        if missing:
            raise ValueError(f'{self.name}: cannot provide {sorted(missing)}')
        direct = wanted & self.properties
        derive_mm_forces = 'mm_forces' in wanted and 'mm_forces' not in self.properties \
            and self.mm_dependence is MMDependence.LINEAR_CHARGE
        if derive_mm_forces:
            direct |= {'mm_potential_gradient'}
        self.extras = {}
        raw = dict(self._compute(inputs, frozenset(direct))) if direct else {}
        out = {}
        for prop in sorted(wanted | direct, key=ADDITIVE_PROPERTIES.index):   # mm_forces last
            shape = property_shape(prop, inputs.n_qm, inputs.n_mm)
            if prop in direct:
                if prop not in raw:
                    raise ValueError(f'{self.name}: _compute did not return {prop}')
                value = np.asarray(raw[prop], dtype=np.float64)
                if value.size != int(np.prod(shape, dtype=int)):
                    raise ValueError(f'{self.name}: {prop} has shape {value.shape}, expected {shape}')
                value = value.reshape(shape)
            elif prop in MM_PROPERTIES and self.mm_dependence is MMDependence.NONE:
                value = np.zeros(shape)
            else:                                   # derived mm_forces
                value = -inputs.mm_charges[:, None] * out['mm_potential_gradient']
            if not np.isfinite(value).all():
                raise ValueError(f'{self.name}: non-finite {prop}')
            out[prop] = float(value) if prop == 'energy' else value
        return {k: out[k] for k in wanted}

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r})'


class TorchPotentialTerm(PotentialTerm):
    """Base for PyTorch terms: owns device/dtype and tensor conversion."""

    def __init__(self, name, *, device='cpu', dtype='float64'):
        super().__init__(name)
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError('dtype must be float32 or float64')

    def tensor(self, array, dtype=None):
        return self.torch.as_tensor(np.asarray(array), dtype=dtype or self.dtype, device=self.device)

    @staticmethod
    def numpy(value):
        return value.detach().to('cpu').double().numpy() if hasattr(value, 'detach') else np.asarray(value, float)

    def metadata(self):
        return {**super().metadata(), 'device': str(self.device), 'dtype': str(self.dtype).replace('torch.', '')}
