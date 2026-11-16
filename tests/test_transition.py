"""TensorNet transition charges / dipoles: constraints, symmetry, phase-less loss, training."""
import itertools
import os

import numpy as np
import pytest
import torch

os.environ.setdefault('MLPUI_NEIGHBORS', 'python')

from mlpui.backends import get_backend, find_backend
from mlpui.data import NpyDataset
from mlpui.models.transition.phase import optimal_signs, align_transition_targets
from mlpui.training import Trainer, TrainingConfig

Z = np.array([6, 6, 8, 7, 1, 1, 6, 8])
S = 3
SMALL = dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=4.0, num_states=S)


def geometry(seed):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.8, size=(len(Z), 3))
    pos[:, 0] += np.arange(len(Z)) * 1.0
    return pos


def build(atomic_dipoles, seed=0):
    torch.manual_seed(seed)
    m = get_backend('transition').build_model(dict(SMALL, atomic_dipoles=atomic_dipoles))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m.eval()


def run(model, pos, z=Z, charge=0.):
    be = get_backend('tensornet_transition')
    inputs = be.prepare_sample(dict(z=z, pos=pos, charge=np.array(charge)), 'cpu', torch.float64, model=model)
    return be.forward(model, inputs, ('transition_charges', 'transition_dipole'))


def rotation(seed, reflect=False):
    """Proper rotation (det +1), optionally times a mirror; built in torch to avoid numpy BLAS."""
    a, b, c = (torch.tensor(v, dtype=torch.float64) for v in np.random.default_rng(seed).uniform(0, 2 * np.pi, size=3))
    one, zero = torch.ones((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    rz = lambda t: torch.stack([torch.stack([t.cos(), -t.sin(), zero]), torch.stack([t.sin(), t.cos(), zero]),
                                torch.stack([zero, zero, one])])
    rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, b.cos(), -b.sin()]),
                      torch.stack([zero, b.sin(), b.cos()])])
    r = rz(a) @ rx @ rz(c)
    if reflect:
        r = r @ torch.diag(torch.tensor([1., 1., -1.], dtype=torch.float64))
    return r


def test_registry():
    be = get_backend('transition')
    assert be is get_backend('tensornet_transition')
    m = build(False)
    assert find_backend(m) is be and find_backend(m.state_dict()) is be


@pytest.mark.parametrize('atomic_dipoles', [False, True])
def test_neutral_origin_free_and_o3_equivariant(atomic_dipoles):
    m = build(atomic_dipoles)
    pos = geometry(1)
    out = run(m, pos)
    q, mu = out['transition_charges'], out['transition_dipole']
    assert q.shape == (len(Z), S) and mu.shape == (S, 3)
    assert torch.allclose(q.sum(0), torch.zeros(S, dtype=q.dtype), atol=1e-12)
    if not atomic_dipoles:
        assert torch.allclose(mu, (q[:, :, None] * torch.tensor(pos)[:, None]).sum(0), atol=1e-12)
    shifted = run(m, pos + np.array([3.0, -7.0, 11.0]))
    assert torch.allclose(shifted['transition_dipole'], mu, atol=1e-9)
    assert torch.allclose(shifted['transition_charges'], q, atol=1e-9)
    for reflect in (False, True):
        R = rotation(2, reflect)
        rot = run(m, (torch.tensor(pos) @ R.T).numpy())
        assert torch.allclose(rot['transition_charges'], q, atol=1e-9)
        assert torch.allclose(rot['transition_dipole'], mu @ R.T, atol=1e-9)


@pytest.mark.parametrize('atomic_dipoles', [False, True])
def test_batched_matches_single(atomic_dipoles):
    m = build(atomic_dipoles)
    be = get_backend('transition')
    samples = [dict(z=Z, pos=geometry(k), charge=np.array(0.)) for k in range(3)]
    samples[1] = dict(z=Z[:6], pos=geometry(9)[:6], charge=np.array(1.))
    inputs, layout = be.prepare_batch(samples, 'cpu', torch.float64, model=m)
    out = be.forward(m, inputs, ('transition_charges', 'transition_dipole'))
    qs = torch.split(out['transition_charges'], layout['atoms'])
    for k, s in enumerate(samples):
        ref = run(m, s['pos'], s['z'], float(s['charge']))
        assert torch.allclose(qs[k], ref['transition_charges'], atol=1e-10)
        assert torch.allclose(out['transition_dipole'][k], ref['transition_dipole'], atol=1e-10)


def test_optimal_signs_is_exact_minimiser():
    rng = np.random.default_rng(3)
    for trial in range(20):
        n = 7
        qp, qr = torch.tensor(rng.normal(size=(n, S))), torch.tensor(rng.normal(size=(n, S)))
        mp, mr = torch.tensor(rng.normal(size=(S, 3))), torch.tensor(rng.normal(size=(S, 3)))
        wq, wm = rng.uniform(0.1, 10), rng.uniform(0.1, 10)

        def loss(s):
            s = torch.tensor(s, dtype=qp.dtype)
            return wq * ((qp - s * qr) ** 2).mean() + wm * ((mp - s[:, None] * mr) ** 2).mean()
        brute = min(itertools.product((1., -1.), repeat=S), key=loss)
        s = optimal_signs(S, q_pred=qp, q_ref=qr, mu_pred=mp, mu_ref=mr, w_q=wq, w_mu=wm)
        assert abs(loss(s.tolist()) - loss(brute)) < 1e-12


def write_dataset(directory, model, seeds, flip_seed=None, ragged=False):
    """Teacher labels from `model`, each (frame, state) sign optionally flipped at random."""
    pos = np.stack([geometry(k) for k in seeds])
    q, mu = [], []
    for p in pos:
        out = run(model, p)
        q.append(out['transition_charges'].detach().numpy())
        mu.append(out['transition_dipole'].detach().numpy())
    q, mu = np.stack(q), np.stack(mu)
    if flip_seed is not None:
        s = np.random.default_rng(flip_seed).choice([-1.0, 1.0], size=(len(seeds), S))
        q, mu = q * s[:, None, :], mu * s[:, :, None]
    directory.mkdir(parents=True, exist_ok=True)
    if ragged:
        np.save(directory / 'z.npy', np.tile(Z, len(seeds)))
        np.save(directory / 'pos.npy', pos.reshape(-1, 3))
        np.save(directory / 'offsets.npy', np.arange(len(seeds) + 1) * len(Z))
        np.save(directory / 'transition_charges.npy', q.reshape(-1, S))
    else:
        np.save(directory / 'z.npy', Z)
        np.save(directory / 'pos.npy', pos)
        np.save(directory / 'transition_charges.npy', q)
    np.save(directory / 'transition_dipole.npy', mu)
    return directory


def test_dataset_layouts_and_validation(tmp_path):
    teacher = build(True, seed=1)
    dense = NpyDataset(write_dataset(tmp_path / 'dense', teacher, range(4)))
    ragged = NpyDataset(write_dataset(tmp_path / 'ragged', teacher, range(4), ragged=True))
    assert dense.num_states == ragged.num_states == S
    assert np.allclose(dense[2]['transition_charges'], ragged[2]['transition_charges'])
    assert dense[2]['transition_charges'].shape == (len(Z), S) and dense[2]['transition_dipole'].shape == (S, 3)
    bad = tmp_path / 'bad'
    write_dataset(bad, teacher, range(4))
    np.save(bad / 'transition_dipole.npy', np.zeros((4, S + 1, 3)))
    with pytest.raises(ValueError, match='disagree'):
        NpyDataset(bad)


def test_metrics_invariant_to_label_signs(tmp_path):
    teacher, student = build(True, seed=1), build(True, seed=2)
    a = write_dataset(tmp_path / 'a', teacher, range(6))
    b = write_dataset(tmp_path / 'b', teacher, range(6), flip_seed=5)
    cfg = TrainingConfig(loss_weights={'transition_charges': 1.0, 'transition_dipole': 0.5},
                         dtype=torch.float64, batch_size=3)
    tr = Trainer('transition', dict(student.config), cfg)
    tr.model.load_state_dict(student.state_dict())
    ma, mb = tr.evaluate(NpyDataset(a))['metrics'], tr.evaluate(NpyDataset(b))['metrics']
    for key in ('transition_charges', 'transition_dipole'):
        assert abs(ma[key]['rmse'] - mb[key]['rmse']) < 1e-12
    # Self-consistency: the teacher reproduces sign-scrambled labels exactly.
    tr.model.load_state_dict(teacher.state_dict())
    mt = tr.evaluate(NpyDataset(b))['metrics']
    assert mt['transition_charges']['rmse'] < 1e-10 and mt['transition_dipole']['rmse'] < 1e-10


def test_batched_and_unbatched_losses_agree(tmp_path):
    teacher, student = build(False, seed=1), build(False, seed=2)
    data = NpyDataset(write_dataset(tmp_path / 'd', teacher, range(5), flip_seed=7))
    weights = {'transition_charges': 1.0, 'transition_dipole': 0.3}
    results = []
    for batched in (True, False):
        tr = Trainer('transition', dict(student.config),
                     TrainingConfig(loss_weights=weights, dtype=torch.float64, batched=batched, batch_size=5))
        tr.model.load_state_dict(student.state_dict())
        results.append(tr.evaluate(data)['metrics'])
    for key in weights:
        assert abs(results[0][key]['rmse'] - results[1][key]['rmse']) < 1e-10


def test_training_with_random_signs_learns_and_reloads(tmp_path):
    teacher = build(False, seed=1)
    train = NpyDataset(write_dataset(tmp_path / 'train', teacher, range(24), flip_seed=11))
    val = NpyDataset(write_dataset(tmp_path / 'val', teacher, range(100, 108), flip_seed=12))
    cfg = TrainingConfig(epochs=30, batch_size=8, learning_rate=5e-3, dtype=torch.float64, seed=0,
                         loss_weights={'transition_charges': 1.0, 'transition_dipole': 1.0},
                         early_stopping=True, early_stopping_patience=30)
    model_config = dict(SMALL, atomic_dipoles=False)
    tr = Trainer('transition', model_config, cfg)
    before = tr.evaluate(val)['metrics']['transition_dipole']['rmse']
    history = tr.fit(train, val, output_dir=tmp_path / 'run')
    after = tr.evaluate(val)['metrics']['transition_dipole']['rmse']
    assert after < 0.5 * before, (before, after)
    assert history[-1]['train']['transition_charges'] < history[0]['train']['transition_charges']
    reloaded = Trainer('transition', model_config, cfg, checkpoint=tmp_path / 'run' / 'model.pt',
                       trusted_checkpoint=True)
    again = reloaded.evaluate(val)['metrics']['transition_dipole']['rmse']
    assert abs(again - after) < 1e-10


def test_rejects_environment_and_unknown_targets():
    be = get_backend('transition')
    m = build(False)
    with pytest.raises(ValueError, match='gas phase'):
        be.prepare_sample(dict(z=Z, pos=geometry(0), mm_pos=np.zeros((1, 3)), mm_charges=np.zeros(1)),
                          'cpu', torch.float64, model=m)
    with pytest.raises(ValueError, match='Unsupported transition target'):
        be.configure_targets(dict(SMALL), {'energy': 1.0})
    with pytest.raises(ValueError, match='Unknown transition config'):
        be.build_model(dict(SMALL, foo=1))


# ---- atom_index_embedding: symmetric molecules with antisymmetric transition charges ----------
ETHYLENE_Z = np.array([6, 6, 1, 1, 1, 1])
ETHYLENE = np.array([[-0.667, 0.0, 0.0], [0.667, 0.0, 0.0],
                     [-1.232, 0.923, 0.0], [-1.232, -0.923, 0.0],
                     [1.232, 0.923, 0.0], [1.232, -0.923, 0.0]])
MIRROR = np.array([1, 0, 4, 5, 2, 3])       # x -> -x maps atom i onto atom MIRROR[i]
SYM = dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=4.0, num_states=1)


def antisymmetric_labels(pos):
    """q_i = 0.3 (r_i - centre) . u with u along C0->C1: antisymmetric under the mirror, sum q = 0."""
    u = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
    return 0.3 * (pos - pos.mean(0)) @ u


def ethylene_dataset(directory, seeds, noise=0.005):
    # Small distortions: with large ones an invariant model can partly fake the antisymmetry
    # through the phase-less loss (it picks "which side is positive" from the distortion).
    pos, q = [], []
    for k in seeds:
        rng = np.random.default_rng(k)
        p = ETHYLENE + rng.normal(scale=noise, size=ETHYLENE.shape)
        p = (torch.tensor(p) @ rotation(1000 + k).T).numpy()
        pos.append(p)
        q.append(antisymmetric_labels(p))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'z.npy', ETHYLENE_Z)
    np.save(directory / 'pos.npy', np.stack(pos))
    np.save(directory / 'transition_charges.npy', np.stack(q))
    return NpyDataset(directory)


def random_model(config, seed=0):
    torch.manual_seed(seed)
    m = get_backend('transition').build_model(config)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m.eval()


def test_invariant_model_cannot_separate_mirror_atoms_but_index_embedding_can():
    plain = random_model(dict(SYM))
    q = run(plain, ETHYLENE, ETHYLENE_Z)['transition_charges'][:, 0]
    assert torch.allclose(q, q[MIRROR], atol=1e-10)          # q_P(i) = q_i is forced
    indexed = random_model(dict(SYM, atom_index_embedding=6))
    q = run(indexed, ETHYLENE, ETHYLENE_Z)['transition_charges'][:, 0]
    assert (q - q[MIRROR]).abs().max() > 1e-3
    assert abs(float(q.sum())) < 1e-12


def test_index_embedding_starts_as_plain_model_and_keeps_o3_invariance():
    plain = random_model(dict(SYM, atomic_dipoles=True), seed=4)
    indexed = get_backend('transition').build_model(dict(SYM, atomic_dipoles=True, atom_index_embedding=8))
    state = dict(plain.state_dict(), **{'index_embedding.weight': indexed.state_dict()['index_embedding.weight']})
    indexed.load_state_dict(state)
    assert torch.count_nonzero(indexed.index_embedding.weight) == 0
    pos = geometry(3)
    a, b = run(plain, pos), run(indexed.eval(), pos)
    assert torch.allclose(a['transition_charges'], b['transition_charges'], atol=1e-12)
    m = random_model(dict(SYM, atomic_dipoles=True, atom_index_embedding=8), seed=5)
    out = run(m, pos)
    for reflect in (False, True):
        R = rotation(6, reflect)
        rot = run(m, (torch.tensor(pos) @ R.T).numpy())
        assert torch.allclose(rot['transition_charges'], out['transition_charges'], atol=1e-9)
        assert torch.allclose(rot['transition_dipole'], out['transition_dipole'] @ R.T, atol=1e-9)


def test_index_embedding_batched_matches_single():
    m = random_model(dict(SYM, num_states=S, atom_index_embedding=8), seed=7)
    be = get_backend('transition')
    samples = [dict(z=Z, pos=geometry(k), charge=np.array(0.)) for k in range(3)]
    samples[1] = dict(z=Z[:5], pos=geometry(9)[:5], charge=np.array(0.))
    inputs, layout = be.prepare_batch(samples, 'cpu', torch.float64, model=m)
    out = be.forward(m, inputs, ('transition_charges', 'transition_dipole'))
    qs = torch.split(out['transition_charges'], layout['atoms'])
    for k, s in enumerate(samples):
        ref = run(m, s['pos'], s['z'])
        assert torch.allclose(qs[k], ref['transition_charges'], atol=1e-10)
        assert torch.allclose(out['transition_dipole'][k], ref['transition_dipole'], atol=1e-10)


def test_index_embedding_validation():
    be = get_backend('transition')
    for bad in (-1, 2.0, True):
        with pytest.raises(ValueError, match='atom_index_embedding'):
            be.build_model(dict(SYM, atom_index_embedding=bad))
    m = random_model(dict(SYM, atom_index_embedding=4))
    with pytest.raises(ValueError, match='atom_index_embedding is 4'):
        run(m, ETHYLENE, ETHYLENE_Z)
    assert be.build_model(dict(SYM)).index_embedding is None    # default off: old configs unchanged
    assert 'index_embedding.weight' not in be.build_model(dict(SYM)).state_dict()


def test_index_embedding_learns_antisymmetric_charges(tmp_path):
    train = ethylene_dataset(tmp_path / 'train', range(32))
    val = ethylene_dataset(tmp_path / 'val', range(100, 116))
    label_rms = float(np.sqrt(np.mean(np.load(tmp_path / 'val' / 'transition_charges.npy') ** 2)))
    rmse = {}
    for n in (0, 6):
        cfg = TrainingConfig(epochs=40, batch_size=8, learning_rate=5e-3, dtype=torch.float64, seed=0,
                             loss_weights={'transition_charges': 1.0}, early_stopping=False)
        tr = Trainer('transition', dict(SYM, atom_index_embedding=n), cfg)
        tr.fit(train, val, output_dir=tmp_path / f'run{n}')
        rmse[n] = tr.evaluate(val)['metrics']['transition_charges']['rmse']
    assert rmse[0] > 0.8 * label_rms, (rmse, label_rms)     # invariant model stuck near q = 0
    assert rmse[6] < 0.3 * label_rms, (rmse, label_rms)
