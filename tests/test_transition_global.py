"""Global basis head of the transition model (global_basis: M)."""
import os

import numpy as np
import pytest
import torch

os.environ.setdefault('MLPUI_NEIGHBORS', 'python')

from mlpui.backends import get_backend
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig

Z = np.array([6, 6, 8, 7, 1, 1, 6, 8])
S, M = 3, 4
CFG = dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=3.0, num_states=S,
           atom_index_embedding=len(Z))


def line(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.1, size=(len(Z), 3))
    pos[:, 0] += np.arange(len(Z)) * 1.3
    return pos


def build(global_basis=M, seed=0):
    torch.manual_seed(seed)
    m = get_backend('transition').build_model(dict(CFG, global_basis=global_basis))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m.eval()


def run(model, pos):
    be = get_backend('transition')
    inputs = be.prepare_sample(dict(z=Z, pos=pos, charge=np.array(0.)), 'cpu', torch.float64, model=model)
    return be.forward(model, inputs, ('transition_charges', 'transition_dipole'))


def test_config_validation():
    be = get_backend('transition')
    with pytest.raises(ValueError, match='requires atom_index_embedding'):
        be.build_model(dict(CFG, atom_index_embedding=0, global_basis=2))
    with pytest.raises(ValueError, match='non-negative'):
        be.build_model(dict(CFG, global_basis=-1))
    assert be.default_config()['global_basis'] == 0


def test_constraints_and_invariance():
    m = build()
    pos = line(1)
    out = run(m, pos)
    q, mu = out['transition_charges'], out['transition_dipole']
    assert q.shape == (len(Z), S) and mu.shape == (S, 3)
    assert torch.allclose(q.sum(0), torch.zeros(S, dtype=q.dtype), atol=1e-12)
    assert torch.allclose(mu, (q[:, :, None] * torch.tensor(pos)[:, None]).sum(0), atol=1e-12)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, -1.]])   # improper
    rot = run(m, pos @ R.T + 5.0)
    assert torch.allclose(rot['transition_charges'], q, atol=1e-9)
    assert torch.allclose(rot['transition_dipole'], mu @ torch.tensor(R).T, atol=1e-9)


def test_batched_matches_single():
    m = build()
    be = get_backend('transition')
    samples = [dict(z=Z, pos=line(s), charge=np.array(0.)) for s in range(3)]
    inputs, _ = be.prepare_batch(samples, 'cpu', torch.float64, model=m)
    batched = be.forward(m, inputs, ('transition_charges', 'transition_dipole'))
    for k, s in enumerate(samples):
        single = run(m, s['pos'])
        assert torch.allclose(batched['transition_charges'][k * len(Z):(k + 1) * len(Z)],
                              single['transition_charges'], atol=1e-10)
        assert torch.allclose(batched['transition_dipole'][k], single['transition_dipole'], atol=1e-10)


@pytest.mark.parametrize('global_basis', [0, M])
def test_far_atom_reaches_distant_charges(global_basis):
    """Moving the last atom changes q0 - q1 (mean projection cancels) only with the global head."""
    m = build(global_basis)
    pos = line(2)
    moved = pos.copy()
    moved[-1] += np.array([0.3, 0.2, -0.1])
    assert np.linalg.norm(pos[-1] - pos[:2], axis=1).min() > 2 * CFG['cutoff_upper']
    a, b = run(m, pos)['transition_charges'], run(m, moved)['transition_charges']
    diff = ((a[0] - a[1]) - (b[0] - b[1])).abs().max().item()
    assert (diff > 1e-6) if global_basis else (diff < 1e-12)


def test_init_basis_and_checkpoint_roundtrip(tmp_path):
    be = get_backend('transition')
    m = build()
    basis = np.random.default_rng(0).normal(size=(len(Z), M))
    m.init_global_basis(basis)
    assert np.allclose(m.basis.detach().numpy(), basis)
    m.init_global_basis(basis[:, ::-1])                          # negative strides (e.g. reversed eigh output)
    assert np.allclose(m.basis.detach().numpy(), basis[:, ::-1])
    m.init_global_basis(basis)
    with pytest.raises(ValueError, match='shape'):
        m.init_global_basis(basis[:, :2])
    m2 = be.load_model(m.state_dict(), m.config)
    pos = line(3)
    assert torch.allclose(run(m, pos)['transition_charges'], run(m2.eval(), pos)['transition_charges'])


def test_training_reduces_loss(tmp_path):
    rng = np.random.default_rng(0)
    F = 12
    pos = np.stack([line(s) for s in range(F)])
    base = rng.normal(size=(len(Z), S))
    base -= base.mean(0)
    q = base[None] * (1 + 0.3 * pos[:, -1:, :1])                # depends on a distant coordinate
    q -= q.mean(1, keepdims=True)
    mu = np.einsum('fns,fnx->fsx', q, pos)
    for split in ('train', 'val'):
        d = tmp_path / split
        d.mkdir()
        np.save(d / 'z.npy', np.repeat(Z[None], F, 0)); np.save(d / 'pos.npy', pos)
        np.save(d / 'transition_charges.npy', q); np.save(d / 'transition_dipole.npy', mu)
    cfg = TrainingConfig(epochs=15, batch_size=4, learning_rate=3e-3, dtype=torch.float64, seed=0,
                         loss_weights={'transition_charges': 1.0, 'transition_dipole': 0.1}, test_interval=10**6)
    tr = Trainer('tensornet_transition', dict(CFG, global_basis=M), cfg)
    hist = tr.fit(NpyDataset(tmp_path / 'train'), NpyDataset(tmp_path / 'val'))
    first, last = hist[0]['train']['transition_charges'], hist[-1]['train']['transition_charges']
    assert np.isfinite(last) and last < 0.5 * first
