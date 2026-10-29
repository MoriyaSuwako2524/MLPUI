"""QEq charge-electrostatics backend: physics identities, derivatives, batching, training."""
import json
import os

import numpy as np
import pytest
import torch

os.environ.setdefault('MLPUI_NEIGHBORS', 'python')

from mlpui.backends import get_backend, find_backend
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig

K = 14.3996454784255
Z = np.array([6, 6, 8, 8, 1, 1, 6, 8])


def structure(seed, n_mm=30):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=1.0, size=(len(Z), 3))
    pos[:, 0] += np.arange(len(Z)) * 0.9
    mm = []
    while len(mm) < n_mm:
        x = pos.mean(0) + rng.normal(scale=4.0, size=3)
        if np.linalg.norm(pos - x, axis=1).min() > 1.8:
            mm.append(x)
    return dict(z=Z.copy(), pos=pos, mm_pos=np.array(mm), mm_charges=rng.normal(scale=0.5, size=n_mm), charge=np.array(-1.0))


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    be = get_backend('qeq')
    m = be.build_model(dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=4.0))
    with torch.no_grad():   # non-trivial parameters, including a geometry-dependent head
        for p in m.parameters():
            p.normal_(0, 0.3)
    m.eval()
    return m


def run(model, s, props, **override):
    be = get_backend('qeq_charge')
    sample = dict(s, **override)
    inputs = be.prepare_sample(sample, 'cpu', torch.float64, model=model)
    return be.forward(model, inputs, props)


ALL = ['energy', 'forces', 'mm_forces', 'mm_potential', 'mm_potential_gradient',
       'gas_mm_potential', 'gas_mm_potential_gradient', 'charges']


def test_charge_conservation_and_gas_limit(model):
    s = structure(1)
    out = run(model, s, ALL)
    assert abs(float(out['charges'].sum()) + 1.0) < 1e-10
    assert abs(float(out['gas_charges'].sum()) + 1.0) < 1e-10
    gas = run(model, s, ['energy', 'charges'], mm_charges=np.zeros_like(s['mm_charges']))
    assert abs(float(gas['energy'])) < 1e-10
    assert torch.allclose(gas['charges'], gas['gas_charges'], atol=1e-12)
    # gas ESP output is independent of the MM charges
    assert torch.allclose(out['gas_mm_potential'], run(model, s, ['gas_mm_potential'], mm_charges=2 * s['mm_charges'])['gas_mm_potential'], atol=1e-10)


def test_finite_difference_forces(model):
    s = structure(2)
    out = run(model, s, ['energy', 'forces', 'mm_forces'])
    h = 1e-5
    for key, field, idx in (('forces', 'pos', (3, 1)), ('forces', 'pos', (0, 2)), ('mm_forces', 'mm_pos', (5, 0))):
        plus, minus = s[field].copy(), s[field].copy()
        plus[idx] += h
        minus[idx] -= h
        fd = -(float(run(model, s, ['energy'], **{field: plus})['energy']) - float(run(model, s, ['energy'], **{field: minus})['energy'])) / (2 * h)
        assert abs(fd - float(out[key][idx])) < 1e-6


def test_potential_is_dE_dQ_and_gradient_fixed_charge(model):
    s = structure(3)
    out = run(model, s, ['energy', 'mm_potential', 'mm_potential_gradient', 'mm_forces'])
    be = get_backend('qeq')
    inputs = be.prepare_sample(s, 'cpu', torch.float64, model=model)
    inputs['mm_charges'].requires_grad_(True)
    e = be.forward(model, inputs, ['energy'])['energy']
    dEdQ = torch.autograd.grad(e, inputs['mm_charges'])[0]
    assert torch.allclose(dEdQ, out['mm_potential'], atol=1e-10)          # Hellmann-Feynman
    # GESP = gradient of the ESP of the fixed polarized charges
    q = out['charges'].detach().numpy()
    def esp(point):
        sig = model._bounded(model.width_element(torch.tensor(Z)).squeeze(-1), model.config['sigma_bounds']).detach().numpy()
        r = np.linalg.norm(s['pos'] - point, axis=1)
        from scipy.special import erf
        return K * (q * erf(r / (np.sqrt(2) * sig)) / r).sum()
    h = 1e-5
    for m in (0, 7):
        g = [(esp(s['mm_pos'][m] + h * e_) - esp(s['mm_pos'][m] - h * e_)) / (2 * h) for e_ in np.eye(3)]
        assert np.allclose(g, out['mm_potential_gradient'][m].detach().numpy(), atol=1e-7)
        assert abs(esp(s['mm_pos'][m]) - float(out['mm_potential'][m])) < 1e-10
    # variational: MM force = -Q * GESP exactly
    expected = -torch.as_tensor(s['mm_charges'])[:, None] * out['mm_potential_gradient']
    assert torch.allclose(out['mm_forces'], expected, atol=1e-9)


def test_no_dq_dR_term_needed(model):
    """Forces equal the derivative at fixed (detached) charges: B term vanishes."""
    s = structure(4)
    out = run(model, s, ['forces', 'charges'])
    be = get_backend('qeq')
    inputs = be.prepare_sample(s, 'cpu', torch.float64, model=model)
    pos = inputs['pos'].requires_grad_(True)
    assert out['forces'].shape == (len(Z), 3)
    # perturb geometry, keep charges fixed: first-order energy change equals -F.dR
    d = torch.zeros_like(pos); d[2, 1] = 1e-6
    e0 = float(run(model, s, ['energy'])['energy'])
    e1 = float(run(model, s, ['energy'], pos=s['pos'] + d.numpy())['energy'])
    assert abs((e1 - e0) / 1e-6 + float(out['forces'][2, 1])) < 1e-4


def test_batched_matches_single(model):
    be = get_backend('qeq')
    samples = [structure(10), structure(11, n_mm=17), structure(12, n_mm=40)]
    single = [run(model, s, ALL) for s in samples]
    inputs, layout = be.prepare_batch(samples, 'cpu', torch.float64, model=model)
    out = be.forward(model, inputs, ALL)
    assert torch.allclose(out['energy'], torch.stack([o['energy'] for o in single]), atol=1e-10)
    for key, kind in (('forces', 'atoms'), ('charges', 'atoms'), ('mm_potential', 'mm'), ('mm_potential_gradient', 'mm'),
                      ('gas_mm_potential', 'mm'), ('mm_forces', 'mm')):
        parts = torch.split(out[key], layout[kind])
        for p, o in zip(parts, single):
            assert torch.allclose(p, o[key], atol=1e-9), key


def test_backend_detection_and_dataset_fields(tmp_path):
    be = get_backend('qeq_charge')
    m = be.build_model(dict(precision=64, embedding_dimension=8, num_layers=1, num_rbf=4))
    assert find_backend(m.state_dict()) is be
    s = [structure(20 + i) for i in range(3)]
    d = tmp_path / 'ds'
    d.mkdir()
    np.save(d / 'z.npy', Z)
    np.save(d / 'pos.npy', np.stack([x['pos'] for x in s]))
    np.save(d / 'charge.npy', np.full(3, -1.0))
    np.save(d / 'mm_pos.npy', np.concatenate([x['mm_pos'] for x in s]))
    np.save(d / 'mm_charges.npy', np.concatenate([x['mm_charges'] for x in s]))
    np.save(d / 'mm_offsets.npy', np.array([0, 30, 60, 90]))
    np.save(d / 'gas_mm_potential.npy', np.zeros(90))
    np.save(d / 'gas_mm_potential_gradient.npy', np.zeros((90, 3)))
    ds = NpyDataset(d)
    assert ds[1]['gas_mm_potential'].shape == (30,) and ds[2]['gas_mm_potential_gradient'].shape == (30, 3)
    del ds
    bad = tmp_path / 'bad'
    bad.mkdir()
    for f in d.glob('*.npy'):
        np.save(bad / f.name, np.load(f))
    np.save(bad / 'gas_mm_potential.npy', np.zeros(89))   # wrong MM layout
    with pytest.raises(ValueError):
        NpyDataset(bad)


def test_trainer_fits_and_reloads(tmp_path, model):
    rng = np.random.default_rng(0)
    teacher = model
    samples = [structure(30 + i) for i in range(6)]
    d = tmp_path / 'train'
    d.mkdir()
    labels = [run(teacher, s, ALL) for s in samples]
    np.save(d / 'z.npy', Z)
    np.save(d / 'pos.npy', np.stack([x['pos'] for x in samples]))
    np.save(d / 'charge.npy', np.full(6, -1.0))
    np.save(d / 'mm_pos.npy', np.concatenate([x['mm_pos'] for x in samples]))
    np.save(d / 'mm_charges.npy', np.concatenate([x['mm_charges'] for x in samples]))
    np.save(d / 'mm_offsets.npy', np.arange(7) * 30)
    np.save(d / 'energy.npy', np.array([float(o['energy']) for o in labels]))
    np.save(d / 'forces.npy', np.stack([o['forces'].detach().numpy() for o in labels]))
    for key in ('mm_potential', 'mm_potential_gradient', 'gas_mm_potential'):
        np.save(d / f'{key}.npy', np.concatenate([o[key].detach().numpy() for o in labels]))
    cfg = TrainingConfig(epochs=3, batch_size=3, learning_rate=1e-3, dtype=torch.float64,
                         loss_weights={'energy': 1., 'forces': 1., 'mm_potential': 1., 'mm_potential_gradient': 1., 'gas_mm_potential': 1.})
    student = dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=4.0)
    tr = Trainer('qeq', student, cfg)
    before = tr.evaluate(NpyDataset(d))['metrics']['mm_potential']['rmse']
    tr.fit(NpyDataset(d), NpyDataset(d), output_dir=tmp_path / 'run')
    after = tr.evaluate(NpyDataset(d))['metrics']['mm_potential']['rmse']
    assert np.isfinite(after) and after < before
    ckpts = sorted((tmp_path / 'run').glob('*.pt'))
    assert ckpts
    tr2 = Trainer('qeq', student, cfg, checkpoint=ckpts[-1])
    assert np.isfinite(tr2.evaluate(NpyDataset(d))['metrics']['energy']['rmse'])


def test_residual_script_zero_for_own_labels(tmp_path, model, monkeypatch):
    import sys
    from scripts import qeq_residual
    samples = [structure(40 + i, n_mm=12 + i) for i in range(4)]
    labels = [run(model, s, ALL) for s in samples]
    d = tmp_path / 'src'
    d.mkdir()
    np.save(d / 'z.npy', Z)
    np.save(d / 'pos.npy', np.stack([x['pos'] for x in samples]))
    np.save(d / 'charge.npy', np.full(4, -1.0))
    np.save(d / 'mm_pos.npy', np.concatenate([x['mm_pos'] for x in samples]))
    np.save(d / 'mm_charges.npy', np.concatenate([x['mm_charges'] for x in samples]))
    np.save(d / 'mm_offsets.npy', np.concatenate([[0], np.cumsum([12 + i for i in range(4)])]))
    np.save(d / 'energy.npy', np.array([float(o['energy']) for o in labels]))
    np.save(d / 'forces.npy', np.stack([o['forces'].detach().numpy() for o in labels]))
    for key in ('mm_potential', 'mm_potential_gradient'):
        np.save(d / f'{key}.npy', np.concatenate([o[key].detach().numpy() for o in labels]))
    cfg = TrainingConfig(dtype=torch.float64, loss_weights={'energy': 1.})
    tr = Trainer('qeq', dict(model.config), cfg)
    tr.model.load_state_dict(model.state_dict())
    ckpt = tr.save(tmp_path / 'qeq.pt')
    monkeypatch.setattr(sys, 'argv', ['qeq_residual', str(ckpt), str(d), str(tmp_path / 'out'), '--batch-size', '3'])
    qeq_residual.main()
    manifest = json.loads((tmp_path / 'out' / 'residual_manifest.json').read_text())
    assert set(manifest['subtracted']) == {'energy', 'forces', 'mm_potential', 'mm_potential_gradient'}
    for key in manifest['subtracted']:
        assert manifest['summary'][key]['residual_rms'] < 1e-8 * max(1., manifest['summary'][key]['label_rms'])
    ds = NpyDataset(tmp_path / 'out')
    assert len(ds) == 4 and ds[3]['mm_potential'].shape == (15,)
