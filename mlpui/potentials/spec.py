"""Build a PotentialTerm from a JSON-able specification.

Example (paths relative to the spec file)::

    {"name": "cm_full", "device": "cpu", "dtype": "float32", "charge": -2, "spin": 1,
     "terms": [
       {"type": "uma", "checkpoint": "models/uma-s-1p2.pt", "implementation": "native"},
       {"type": "constant", "name": "offset", "manifest": "models/uma_residual_manifest.json"},
       {"type": "mlpui", "name": "qeq", "checkpoint": "models/best.pt"},
       {"type": "mlpui", "name": "dmlp", "checkpoint": "models/dmlp_best.pt"}]}

Term types: uma, constant, mlpui, fixed_charge_coulomb, composite (nested "terms"),
and python ({"type": "python", "target": "package.module:factory", "kwargs": {...}}),
which calls ``factory(name=..., **kwargs)`` and must return a PotentialTerm. Top-level
"device"/"dtype" are defaults for terms that accept them. "charge"/"spin", if given,
are checked by drivers (e.g. the QMHub engine) against the system they receive.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from .base import PotentialTerm
from .composite import CompositePotential
from .terms import ConstantEnergy, FixedChargeCoulomb, MLPUICheckpointTerm, uma_term


def load_spec(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    spec.setdefault('_base_dir', str(path.resolve().parent))
    return spec


def _path(value, base):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def _build_term(entry, defaults, base):
    entry = dict(entry)
    kind = entry.pop('type')
    device = entry.pop('device', defaults.get('device', 'cpu'))
    dtype = entry.pop('dtype', defaults.get('dtype', 'float64'))
    if kind == 'uma':
        return uma_term(_path(entry.pop('checkpoint'), base), charge=entry.pop('charge', defaults.get('charge')),
                        spin=entry.pop('spin', defaults.get('spin', 1)), device=device, dtype=dtype, **entry)
    if kind == 'constant':
        name = entry.pop('name', 'offset')
        if 'manifest' in entry:
            return ConstantEnergy.from_manifest(_path(entry.pop('manifest'), base), name=name, **entry)
        return ConstantEnergy(entry.pop('energy_eV'), name=name, **entry)
    if kind == 'mlpui':
        return MLPUICheckpointTerm(_path(entry.pop('checkpoint'), base), entry.pop('name'),
                                   device=device, dtype=dtype, **entry)
    if kind == 'fixed_charge_coulomb':
        return FixedChargeCoulomb(entry.pop('qm_charges'), name=entry.pop('name', 'fixed_charge_coulomb'))
    if kind == 'composite':
        return build_potential({**defaults, **entry, '_base_dir': base})
    if kind == 'python':
        module, _, attr = entry.pop('target').partition(':')
        factory = getattr(importlib.import_module(module), attr)
        term = factory(name=entry.pop('name'), **entry.pop('kwargs', {}))
        if not isinstance(term, PotentialTerm):
            raise TypeError(f'{module}:{attr} did not return a PotentialTerm')
        return term
    raise ValueError(f'unknown term type {kind!r}')


def build_potential(spec, *, overrides=None):
    """spec: dict or path to a JSON file. overrides: top-level defaults (e.g. device) to replace."""
    if not isinstance(spec, dict):
        spec = load_spec(spec)
    spec = {**spec, **(overrides or {})}
    base = spec.get('_base_dir', '.')
    defaults = {k: spec[k] for k in ('device', 'dtype', 'charge', 'spin') if k in spec}
    terms = [_build_term(t, defaults, base) for t in spec['terms']]
    return CompositePotential(terms, name=spec.get('name', 'composite'))
