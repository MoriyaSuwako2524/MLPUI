"""QMHub engine adapter: end-to-end through the real qmhub CLI (text exchange).

Checks that QMHub + the mlpui engine give a total energy whose finite
differences match the gradient QMHub returns for both QM and MM atoms, with and
without PBC. qmmm.out units: energy hartree, gradient hartree/bohr (positions
in the exchange file are angstrom).
"""
import json

import numpy as np
import pytest

try:
    qmhub = pytest.importorskip('qmhub')
except AttributeError:                      # upstream QMHub 06fdbe7 needs numpy<2
    pytest.skip('upstream QMHub needs numpy<2; install github.com/MoriyaSuwako2524/qmhub@mlpui',
                allow_module_level=True)
from qmhub.units import CODATA08_BOHR_TO_A as BOHR_TO_A

from mlpui.potentials.qmhub import _POTENTIALS, main as qmhub_main

QM_Z = np.array([6, 8, 8, 1, 1])
QM_Q = np.array([0.5, -0.8, -0.7, 0.25, -0.25])        # sums to total charge -1


def system(seed=0, n_mm=150, box=24.0):
    rng = np.random.default_rng(seed)
    qm = np.array([[0, 0, 0], [1.2, 0, 0], [-0.7, 1.0, 0], [-0.6, -0.9, 0.3], [1.8, 0.8, 0.2]], float)
    mm = []
    while len(mm) < n_mm:
        x = rng.uniform(-box / 2, box / 2, size=3)
        if np.linalg.norm(qm - x, axis=1).min() > 2.0:
            mm.append(x)
    q = rng.normal(scale=0.4, size=n_mm)
    q -= q.mean()
    return qm, np.array(mm), q


def write_input(path, qm, mm, q, cell):
    with open(path, 'w') as f:
        f.write(f'{len(qm)} {len(mm)} -1 1 0\n')
        for (x, y, z), c, e in zip(qm, QM_Q, QM_Z):
            f.write(f'{x:.12f} {y:.12f} {z:.12f} {c:.8f} {e}\n')
        for (x, y, z), c in zip(mm, q):
            f.write(f'{x:.12f} {y:.12f} {z:.12f} {c:.8f}\n')
        for row in cell:
            f.write(' '.join(f'{v:.10f}' for v in row) + '\n')


@pytest.fixture()
def workdir(tmp_path):
    spec = {'name': 'toy', 'charge': -1, 'spin': 1, 'terms': [
        {'type': 'constant', 'energy_eV': -3.0},
        {'type': 'fixed_charge_coulomb', 'qm_charges': QM_Q.tolist()}]}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    _POTENTIALS.clear()
    return tmp_path


def run(workdir, pbc, qm, mm, q, cell, tag):
    ini = workdir / f'qmhub_{pbc}.ini'
    ini.write_text('[simulation]\nprotocol = md\n[model]\nswitching_function = lrec\ncutoff = 8.\n'
                   f'pbc = {pbc}\n[engine]\nhigh = mlpui\n[high]\nspec = {workdir / "spec.json"}\n'
                   f'log = {workdir / "engine.jsonl"}\n')
    inp = workdir / f'qmmm_{tag}.inp'
    write_input(inp, qm, mm, q, cell)
    qmhub_main([str(ini), '--text', str(inp), '--driver', 'sander'])
    lines = inp.with_suffix('.out').read_text().splitlines()
    return float(lines[0]), np.loadtxt(lines[1:]).reshape(-1, 3)


@pytest.mark.parametrize('pbc', [False, True])
def test_qmhub_energy_gradient_consistency(workdir, pbc):
    qm, mm, q = system()
    cell = np.diag([24.0] * 3) if pbc else np.zeros((3, 3))
    energy, grad = run(workdir, pbc, qm, mm, q, cell, 'ref')
    assert grad.shape == (len(qm) + len(mm), 3) and np.isfinite(grad).all()
    h = 1e-4
    near = int(np.argmin(np.linalg.norm(mm - qm.mean(0), axis=1)))        # inside the near field
    for atom, label in ((0, 'qm'), (3, 'qm'), (len(qm) + near, 'mm_near')):
        for axis in range(3):
            plus, minus = [np.vstack([qm, mm]) for _ in range(2)]
            plus[atom, axis] += h
            minus[atom, axis] -= h
            ep, _ = run(workdir, pbc, plus[:len(qm)], plus[len(qm):], q, cell, 'p')
            em, _ = run(workdir, pbc, minus[:len(qm)], minus[len(qm):], q, cell, 'm')
            fd = (ep - em) / (2 * h) * BOHR_TO_A                    # hartree/A -> hartree/bohr
            assert abs(fd - grad[atom, axis]) < 1e-7 + 1e-5 * abs(fd), (label, axis, fd, grad[atom, axis])


def test_engine_matches_direct_potential_and_logs(workdir):
    qm, mm, q = system(1)
    energy, _ = run(workdir, False, qm, mm, q, np.zeros((3, 3)), 'log')
    row = json.loads((workdir / 'engine.jsonl').read_text().splitlines()[-1])
    assert row['components_eV']['offset'] == -3.0 and row['n_mm'] > 0
    assert np.isfinite(row['energy_eV']) and np.isfinite(energy)


def test_engine_rejects_charge_mismatch(workdir):
    spec = json.loads((workdir / 'spec.json').read_text())
    spec['charge'] = 0
    (workdir / 'spec.json').write_text(json.dumps(spec))
    qm, mm, q = system(2)
    with pytest.raises(ValueError, match='charge'):
        run(workdir, False, qm, mm, q, np.zeros((3, 3)), 'bad')
