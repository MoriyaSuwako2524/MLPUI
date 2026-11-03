"""Composable QM/MM potentials: contract, composition, ASE view, MLPUI checkpoints."""
import os

import numpy as np
import pytest

os.environ.setdefault('MLPUI_NEIGHBORS', 'python')

from mlpui.potentials import (ASECalculatorTerm, CompositePotential, ConstantEnergy, FixedChargeCoulomb,
                              MLPUICheckpointTerm, MMDependence, PotentialTerm, QMMMInput)
from mlpui.potentials.ase_calculator import QMMMCalculator
from mlpui.potentials.testing import check_derivatives

Z = np.array([6, 6, 8, 8, 1, 1, 6, 8])


def configuration(seed=0, n_mm=12, charge=-1):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.8, size=(len(Z), 3))
    pos[:, 0] += np.arange(len(Z)) * 0.9
    mm = []
    while len(mm) < n_mm:
        x = pos.mean(0) + rng.normal(scale=3.5, size=3)
        if np.linalg.norm(pos - x, axis=1).min() > 1.8:
            mm.append(x)
    return QMMMInput(Z, pos, charge, 1, np.array(mm), rng.normal(scale=0.5, size=n_mm))


class Harmonic(PotentialTerm):
    """MM-independent toy term E = k/2 sum |R - R0|^2."""
    properties = frozenset({'energy', 'forces'})
    mm_dependence = MMDependence.NONE

    def __init__(self, ref, k=0.7, name='harmonic'):
        super().__init__(name)
        self.ref, self.k = np.asarray(ref), k

    def _compute(self, x, props):
        d = x.positions - self.ref
        self.extras['rmsd'] = float(np.sqrt((d ** 2).sum(1).mean()))
        return {'energy': 0.5 * self.k * (d ** 2).sum(), 'forces': -self.k * d}


def test_input_validation():
    x = configuration()
    with pytest.raises(ValueError):
        QMMMInput(Z.astype(float), x.positions)
    with pytest.raises(ValueError):
        QMMMInput(Z, x.positions, mm_positions=np.zeros((2, 3)), mm_charges=np.zeros(3))
    with pytest.raises(ValueError):
        QMMMInput(Z, x.positions, spin=0)
    with pytest.raises(ValueError):
        x.positions[0, 0] = 1.0          # inputs are immutable


def test_fixed_charge_coulomb_is_exact_and_consistent():
    x = configuration(1)
    term = FixedChargeCoulomb(np.random.default_rng(2).normal(scale=0.3, size=len(Z)))
    errors = check_derivatives(term, x, step=1e-5)
    assert max(errors.values()) < 1e-6, errors
    out = term.evaluate(x, ['energy', 'mm_potential', 'mm_forces', 'mm_potential_gradient'])
    assert np.isclose(out['energy'], out['mm_potential'] @ x.mm_charges)
    assert np.allclose(out['mm_forces'], -x.mm_charges[:, None] * out['mm_potential_gradient'])


def test_composite_sums_components_and_fills_mm_zeros():
    x = configuration(3)
    harmonic = Harmonic(x.positions + 0.05)
    coulomb = FixedChargeCoulomb(np.linspace(-0.4, 0.3, len(Z)))
    pot = CompositePotential([ConstantEnergy(-1.5), harmonic, coulomb])
    assert pot.mm_dependence is MMDependence.LINEAR_CHARGE
    props = ['energy', 'forces', 'mm_potential', 'mm_potential_gradient', 'mm_forces']
    total = pot.evaluate(x, props)
    for p in props:
        parts = sum(np.asarray(pot.components[t][p]) for t in ('offset', 'harmonic', 'fixed_charge_coulomb'))
        assert np.allclose(total[p], parts)
    assert np.all(pot.components['harmonic']['mm_potential'] == 0)
    assert 'harmonic:rmsd' in pot.extras
    assert max(check_derivatives(pot, x, step=1e-5).values()) < 1e-6
    nested = CompositePotential([pot, ConstantEnergy(2.0, name='shift')], name='outer')
    assert np.isclose(nested.evaluate(x, ['energy'])['energy'], total['energy'] + 2.0)
    with pytest.raises(ValueError):
        CompositePotential([ConstantEnergy(1.0), ConstantEnergy(2.0)])


def test_general_term_without_mm_forces_restricts_composite():
    class General(Harmonic):
        properties = frozenset({'energy', 'forces', 'mm_potential'})
        mm_dependence = MMDependence.GENERAL

        def _compute(self, x, props):
            out = super()._compute(x, props)
            out['mm_potential'] = np.zeros(x.n_mm)
            return out

    x = configuration(4)
    pot = CompositePotential([General(x.positions), FixedChargeCoulomb(np.zeros(len(Z)))])
    assert 'mm_forces' not in pot.available_properties()
    with pytest.raises(ValueError):
        pot.evaluate(x, ['mm_forces'])


def test_bad_term_output_is_rejected():
    class Broken(Harmonic):
        def _compute(self, x, props):
            return {'energy': 0.0, 'forces': np.zeros((x.n_qm + 1, 3))}

    with pytest.raises(ValueError, match='shape'):
        Broken(np.zeros((len(Z), 3))).evaluate(configuration(), ['energy', 'forces'])


def test_ase_calculator_view_and_set_mm():
    from ase import Atoms
    x = configuration(5)
    pot = CompositePotential([Harmonic(x.positions + 0.1), FixedChargeCoulomb(np.full(len(Z), -1 / len(Z)))])
    calc = QMMMCalculator(pot, charge=x.charge, mm_positions=x.mm_positions, mm_charges=x.mm_charges)
    atoms = Atoms(numbers=Z, positions=x.positions)
    atoms.calc = calc
    ref = pot.evaluate(x, ['energy', 'forces', 'mm_potential'])
    assert np.isclose(atoms.get_potential_energy(), ref['energy'])
    assert np.allclose(atoms.get_forces(), ref['forces'])
    assert np.allclose(calc.get_property('mm_potential', atoms), ref['mm_potential'])
    calc.set_mm(x.mm_positions, 2 * x.mm_charges)       # environment change must invalidate cache
    assert not np.isclose(atoms.get_potential_energy(), ref['energy'])


def test_ase_calculator_term_wraps_gas_phase_calculator():
    from ase.calculators.lj import LennardJones
    x = configuration(6)
    # sigma below the closest contact keeps LJ out of its hard core (FD error ~ step^2 * F''').
    term = ASECalculatorTerm(LennardJones(sigma=0.4, epsilon=0.01, rc=6.0, smooth=True), 'lj')
    assert term.evaluate(x, ['mm_potential_gradient'])['mm_potential_gradient'].shape == (x.n_mm, 3)
    assert check_derivatives(term, x, step=1e-5)['forces_vs_-dE/dR'] < 1e-6
    strict = ASECalculatorTerm(LennardJones(), 'lj2', expected_charge=0)
    with pytest.raises(ValueError, match='charge'):
        strict.evaluate(x, ['energy'])


# --- MLPUI checkpoints ------------------------------------------------------------------------

torch = pytest.importorskip('torch')

SMALL = {
    'qeq_charge': dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=4.0),
    'tensornet_qmmm': dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, ext_num_rbf=8,
                           cutoff_upper=4.0),
}


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    from mlpui.training import Trainer, TrainingConfig
    paths = {}
    for i, (family, config) in enumerate(SMALL.items()):
        torch.manual_seed(i)
        trainer = Trainer(family, config, TrainingConfig(dtype=torch.float64, loss_weights={'energy': 1.0}))
        with torch.no_grad():     # non-trivial weights, incl. the zero-initialised QEq head
            for p in trainer.model.parameters():
                p.normal_(0, 0.2)
        paths[family] = trainer.save(tmp_path_factory.mktemp(family) / 'best.pt')
    return paths


@pytest.mark.parametrize('family', list(SMALL))
def test_mlpui_checkpoint_term_derivatives(checkpoints, family):
    term = MLPUICheckpointTerm(checkpoints[family], family)
    assert term.family == family and term.mm_dependence is MMDependence.LINEAR_CHARGE
    x = configuration(7, n_mm=10)
    errors = check_derivatives(term, x, step=1e-5, n_qm=3, n_mm=3)
    assert max(errors.values()) < 1e-6, errors
    meta = term.metadata()
    assert len(meta['checkpoint_sha256']) == 64
    with pytest.raises(ValueError, match='singlet'):
        term.evaluate(x.replace(spin=3), ['energy'])


def test_mlpui_composite_matches_direct_backend_calls(checkpoints):
    from mlpui.backends import get_backend
    qeq = MLPUICheckpointTerm(checkpoints['qeq_charge'], 'qeq')
    dmlp = MLPUICheckpointTerm(checkpoints['tensornet_qmmm'], 'dmlp')
    pot = CompositePotential([ConstantEnergy(9.32, name='offset'), qeq, dmlp], name='cm_model')
    x = configuration(8, n_mm=15)
    props = ['energy', 'forces', 'mm_potential', 'mm_potential_gradient', 'mm_forces']
    total = pot.evaluate(x, props)
    expected = {p: 0 for p in props}
    expected['energy'] = 9.32
    for term in (qeq, dmlp):
        be = get_backend(term.family)
        sample = dict(z=x.numbers, pos=x.positions, mm_pos=x.mm_positions, mm_charges=x.mm_charges,
                      charge=float(x.charge))
        out = be.forward(term.model, be.prepare_sample(sample, 'cpu', torch.float64, model=term.model), props)
        for p in props:
            expected[p] = expected[p] + out[p].detach().numpy()
    for p in props:
        assert np.allclose(total[p], expected[p], atol=1e-10), p
    assert 'qeq:charges' in pot.extras and abs(pot.extras['qeq:charges'].sum() - x.charge) < 1e-8
