"""Diabatic Hamiltonian head of the transition model (diabatic_states: M) and excitation_energies."""
import os

import numpy as np
import pytest
import torch

os.environ.setdefault('MLPUI_NEIGHBORS', 'python')

from mlpui.backends import get_backend
from mlpui.data import NpyDataset
from mlpui.models.transition.model import broadened_eigh
from mlpui.training import Trainer, TrainingConfig

Z = np.array([6, 6, 8, 7, 1, 1, 6, 8])
S, M = 3, 5
CFG = dict(precision=64, embedding_dimension=16, num_layers=1, num_rbf=8, cutoff_upper=3.0, num_states=S,
           atom_index_embedding=len(Z), diabatic_states=M)
PROPS = ('transition_charges', 'transition_dipole', 'excitation_energies')


def line(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.1, size=(len(Z), 3))
    pos[:, 0] += np.arange(len(Z)) * 1.3
    return pos


def build(seed=0, **kw):
    torch.manual_seed(seed)
    m = get_backend('transition').build_model(dict(CFG, **kw))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m.eval()


def run(model, pos):
    be = get_backend('transition')
    inputs = be.prepare_sample(dict(z=Z, pos=pos, charge=np.array(0.)), 'cpu', torch.float64, model=model)
    return be.forward(model, inputs, PROPS)


def sym(seed, n=5, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return 0.5 * (a + a.T) + torch.diag(torch.arange(n, dtype=torch.float64) * spread)


def test_config_validation():
    be = get_backend('transition')
    with pytest.raises(ValueError, match='requires atom_index_embedding'):
        be.build_model(dict(CFG, atom_index_embedding=0))
    with pytest.raises(ValueError, match='>= num_states'):
        be.build_model(dict(CFG, diabatic_states=S - 1))
    with pytest.raises(ValueError, match='mutually exclusive'):
        be.build_model(dict(CFG, global_basis=2))
    with pytest.raises(ValueError, match='diabatic head'):
        be.configure_targets(dict(CFG, diabatic_states=0), {'excitation_energies': 1.0})
    assert be.default_config()['diabatic_states'] == 0


def test_broadened_eigh_matches_exact_gradient_and_is_finite_when_degenerate():
    w = torch.randn(5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    P = torch.randn(5, 5, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    def f(A, eig):
        E, U = eig(A)
        V = U[:, :2]
        return (E * w).sum() + (V @ V.T * P).sum()          # gauge-invariant (projector)

    S0 = sym(0, spread=2.0).requires_grad_()
    S1 = S0.detach().clone().requires_grad_()
    f(S0, lambda A: broadened_eigh(A, 1e-9)).backward()
    f(S1, torch.linalg.eigh).backward()
    assert torch.allclose(0.5 * (S1.grad + S1.grad.T), S0.grad, atol=1e-8)
    D = torch.diag(torch.tensor([1., 1., 2., 3., 3.], dtype=torch.float64)).requires_grad_()
    f(D, lambda A: broadened_eigh(A, 1e-2)).backward()
    assert torch.isfinite(D.grad).all()


def test_constraints_invariance_and_sorted_energies():
    m = build()
    pos = line(1)
    out = run(m, pos)
    q, mu, E = out['transition_charges'], out['transition_dipole'], out['excitation_energies']
    assert q.shape == (len(Z), S) and mu.shape == (S, 3) and E.shape == (S,)
    assert torch.all(E[1:] >= E[:-1])
    assert torch.allclose(q.sum(0), torch.zeros(S, dtype=q.dtype), atol=1e-12)
    assert torch.allclose(mu, (q[:, :, None] * torch.tensor(pos)[:, None]).sum(0), atol=1e-12)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, -1.]])
    rot = run(m, pos @ R.T + 5.0)
    assert torch.allclose(rot['excitation_energies'], E, atol=1e-9)
    # eigenvector signs may flip under the transformation; compare per state up to sign
    for k in range(S):
        a, b = rot['transition_charges'][:, k], q[:, k]
        assert min((a - b).abs().max(), (a + b).abs().max()) < 1e-8


def test_batched_matches_single():
    m = build()
    be = get_backend('transition')
    samples = [dict(z=Z, pos=line(s), charge=np.array(0.)) for s in range(3)]
    inputs, _ = be.prepare_batch(samples, 'cpu', torch.float64, model=m)
    batched = be.forward(m, inputs, PROPS)
    for k, smp in enumerate(samples):
        single = run(m, smp['pos'])
        assert torch.allclose(batched['excitation_energies'][k], single['excitation_energies'], atol=1e-10)
        for n in range(S):
            a = batched['transition_charges'][k * len(Z):(k + 1) * len(Z), n]
            b = single['transition_charges'][:, n]
            assert min((a - b).abs().max(), (a + b).abs().max()) < 1e-10


def test_init_diabatic_starts_from_templates():
    torch.manual_seed(0)
    m = get_backend('transition').build_model(CFG).eval()      # zero-initialised Hamiltonian/basis heads
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(len(Z), M))
    energies = np.array([2.0, 2.5, 3.1, 3.4, 4.0])
    m.init_diabatic(basis, energies)
    out = run(m, line(4))
    assert np.allclose(out['excitation_energies'].detach().numpy(), energies[:S])
    centred = basis - basis.mean(0)
    for k in range(S):
        a = out['transition_charges'][:, k].detach().numpy()
        assert min(np.abs(a - centred[:, k]).max(), np.abs(a + centred[:, k]).max()) < 1e-10
    with pytest.raises(ValueError, match='expected shape'):
        m.init_diabatic(basis[:, :2], energies)


def write_split(d, pos, q, mu, E):
    d.mkdir()
    np.save(d / 'z.npy', np.repeat(Z[None], len(pos), 0)); np.save(d / 'pos.npy', pos)
    np.save(d / 'transition_charges.npy', q); np.save(d / 'transition_dipole.npy', mu)
    np.save(d / 'excitation_energies.npy', E)


def test_dataset_state_axis_checks(tmp_path):
    F = 4
    pos = np.stack([line(s) for s in range(F)])
    q = np.zeros((F, len(Z), S)); mu = np.zeros((F, S, 3))
    write_split(tmp_path / 'ok', pos, q, mu, np.ones((F, S)))
    sample = NpyDataset(tmp_path / 'ok')[0]
    assert sample['excitation_energies'].shape == (S,)
    write_split(tmp_path / 'bad', pos, q, mu, np.ones((F, S + 1)))
    with pytest.raises(ValueError, match='disagree'):
        NpyDataset(tmp_path / 'bad')


def test_training_reduces_loss(tmp_path):
    rng = np.random.default_rng(0)
    F = 12
    pos = np.stack([line(s) for s in range(F)])
    t = pos[:, -1, 0] - pos[:, -1, 0].mean()                  # a distant coordinate drives the mixing
    base = rng.normal(size=(len(Z), 2)); base -= base.mean(0)
    c, s_ = np.cos(t), np.sin(t)
    q = np.zeros((F, len(Z), S))
    q[:, :, 0] = c[:, None] * base[:, 0] + s_[:, None] * base[:, 1]
    q[:, :, 1] = -s_[:, None] * base[:, 0] + c[:, None] * base[:, 1]
    q[:, :, 2] = 0.5 * base[:, 0]
    mu = np.einsum('fns,fnx->fsx', q, pos)
    E = np.stack([2.0 + 0.2 * t, 2.4 - 0.1 * t, 3.0 + 0 * t], 1)
    write_split(tmp_path / 'train', pos, q, mu, E)
    write_split(tmp_path / 'val', pos, q, mu, E)
    cfg = TrainingConfig(epochs=15, batch_size=4, learning_rate=3e-3, dtype=torch.float64, seed=0,
                         loss_weights={'transition_charges': 1.0, 'transition_dipole': 0.1,
                                       'excitation_energies': 1.0}, test_interval=10**6)
    tr = Trainer('tensornet_transition', CFG, cfg)
    hist = tr.fit(NpyDataset(tmp_path / 'train'), NpyDataset(tmp_path / 'val'))
    first, last = hist[0]['train'], hist[-1]['train']
    assert all(np.isfinite(v) for v in last.values())
    assert last['excitation_energies'] < 0.5 * first['excitation_energies']
    assert last['transition_charges'] < first['transition_charges']
