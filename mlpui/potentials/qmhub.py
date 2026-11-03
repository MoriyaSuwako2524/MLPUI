"""QMHub engine backed by any mlpui.potentials PotentialTerm.

QMHub builds the electrostatic embedding (lrec-scaled near-field MM charges +
projected charges, PBC) and asks the engine for the QM energy, QM gradient and
the ESP / ESP gradient at the embedding sites. This module provides that engine.

Recommended: the QMHub fork github.com/MoriyaSuwako2524/qmhub (branch ``mlpui``),
which registers the engine natively (``qmhub/qmtools/mlpui.py``) and fixes
numpy>=2 compatibility, so sander's EXTERN/QMHub interface can call ``qmhub`` as
usual. With upstream panxl/qmhub (numpy<2 only) use the launcher instead, via a
script named ``qmhub`` first on PATH::

    python -m mlpui.potentials.qmhub qmhub.ini --fifo qmmm.inp --driver sander

qmhub.ini::

    [simulation]
    protocol = md
    [model]
    switching_function = lrec
    cutoff = 10.
    pbc = True
    [engine]
    high = mlpui
    [high]
    spec = /path/cm_model.json      ; mlpui.potentials.spec JSON
    device = cpu                    ; optional override of the spec default
    threads = 16                    ; optional torch threads (default OMP_NUM_THREADS)
    log = /path/engine_log.jsonl    ; optional: per-call energies/timings
    dump = /path/dump_dir           ; optional: per-call npz of inputs and outputs

Engine units: QMHub expects hartree, hartree/bohr, ESP in hartree/e and its
gradient in hartree/(e bohr) at the embedding sites (same as qmtools/qchem.py
and qmtools/torch.py). The model requirement is that the term is LINEAR_CHARGE
(or NONE): QMHub's MM gradient uses dE/dr_M = Q_M * GESP_M.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from .base import MMDependence, QMMMInput
from .spec import build_potential, load_spec

ENGINE_NAME = 'mlpui'
_POTENTIALS = {}            # one model instance per process and spec (FIFO mode keeps it resident)


def _potential(spec_path, overrides):
    key = (str(Path(spec_path).resolve()), json.dumps(overrides, sort_keys=True))
    if key not in _POTENTIALS:
        spec = load_spec(spec_path)
        potential = build_potential(spec, overrides=overrides)
        if potential.mm_dependence is MMDependence.GENERAL:
            raise ValueError('QMHub requires LINEAR_CHARGE (or MM-independent) terms; '
                             'GENERAL MM dependence breaks its dE/dr_M = Q_M*GESP_M chain rule')
        _POTENTIALS[key] = (spec, potential)
    return _POTENTIALS[key]


def make_engine_class():
    from qmhub.qmtools.qmbase import QMBase
    from qmhub.units import CODATA08_BOHR_TO_A as BOHR_TO_A, CODATA08_HARTREE_TO_EV as HARTREE_TO_EV

    grad_unit = HARTREE_TO_EV / BOHR_TO_A          # eV/A per hartree/bohr

    class MLPUIEngine(QMBase):
        OUTPUT = None
        default_options = {'spec': '', 'device': '', 'dtype': '', 'threads': '', 'log': '', 'dump': ''}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not self.options['spec']:
                raise ValueError('[high] spec = <potential JSON> is required for the mlpui engine')
            import torch
            torch.set_num_threads(int(self.options['threads'] or self.nproc))
            overrides = {k: self.options[k] for k in ('device', 'dtype') if self.options[k]}
            spec, self.potential = _potential(self.options['spec'], overrides)
            for key, value in (('charge', int(self.charge)), ('spin', int(self.mult))):
                if key in spec and int(spec[key]) != value:
                    raise ValueError(f'spec {key}={spec[key]} but QMHub system has {key}={value}')
            self.calls = 0

        def gen_input(self):            # no input files: the model runs in-process
            pass

        def _get_qm_cache(self, *args, output=None):
            t0 = time.perf_counter()
            inputs = QMMMInput(np.asarray(self.qm_elements[:], dtype=np.int64),
                               np.array(self.qm_positions[:], dtype=np.float64).T,
                               int(self.charge), int(self.mult),
                               np.array(self.mm_positions[:], dtype=np.float64).T,
                               np.array(self.mm_charges[:], dtype=np.float64))
            out = self.potential.evaluate(inputs, ['energy', 'forces', 'mm_potential', 'mm_potential_gradient'])
            elapsed = time.perf_counter() - t0
            self._record(inputs, out, elapsed)
            self.calls += 1
            return (out['energy'], out['forces'], out['mm_potential'], out['mm_potential_gradient'])

        def _record(self, inputs, out, elapsed):
            if self.options['log']:
                row = {'call': self.calls, 'time_s': elapsed, 'n_mm': inputs.n_mm, 'energy_eV': out['energy'],
                       'components_eV': {k: v['energy'] for k, v in getattr(self.potential, 'components', {}).items()}}
                with open(self.options['log'], 'a') as stream:
                    stream.write(json.dumps(row) + '\n')
            if self.options['dump']:
                path = Path(self.options['dump'])
                path.mkdir(parents=True, exist_ok=True)
                comps = getattr(self.potential, 'components', {})
                np.savez(path / f'call_{self.calls:06d}.npz', numbers=inputs.numbers, positions=inputs.positions,
                         mm_positions=inputs.mm_positions, mm_charges=inputs.mm_charges,
                         **{k: np.asarray(v) for k, v in out.items()},
                         **{f'component_energy__{k}': v['energy'] for k, v in comps.items()},
                         time_s=elapsed)

        # QMHub unit conventions (see qmtools/qchem.py, qmtools/torch.py)
        def _get_qm_energy(self, qm_cache=None):
            return float(qm_cache[0]) / HARTREE_TO_EV

        def _get_qm_energy_gradient(self, qm_cache=None):
            return -np.asarray(qm_cache[1]).T / grad_unit

        def _get_mm_esp(self, qm_cache=None):
            esp = np.zeros((4, len(qm_cache[2])))
            esp[0] = np.asarray(qm_cache[2]) / HARTREE_TO_EV
            esp[1:] = np.asarray(qm_cache[3]).T / grad_unit
            return esp

    return MLPUIEngine


def register_qmhub_engine(name=ENGINE_NAME):
    """Make ``[engine] high = mlpui`` resolvable by qmhub.qmtools.QM.create."""
    try:
        import qmhub.qmtools as qmtools
    except AttributeError as exc:          # qmhub.utils.darray uses numpy.lib.user_array.container.__div__
        if '__div__' in str(exc):
            raise RuntimeError('upstream QMHub (06fdbe7) is incompatible with numpy>=2: use the fork '
                               'github.com/MoriyaSuwako2524/qmhub@mlpui or numpy<2') from exc
        raise
    module_name = f'qmhub.qmtools.{name}'
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.MLPUIEngine = make_engine_class()
        sys.modules[module_name] = module
    qmtools.QM_TO_CLASS_MAP[name] = 'MLPUIEngine'
    return sys.modules[module_name].MLPUIEngine


def main(argv=None):
    register_qmhub_engine()
    from qmhub.__main__ import main as qmhub_main
    if argv is not None:
        sys.argv = ['qmhub', *argv]
    qmhub_main()


if __name__ == '__main__':
    main()
