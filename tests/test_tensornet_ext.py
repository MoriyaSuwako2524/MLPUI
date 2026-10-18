"""Pinned upstream parity and ext-specific scientific contracts (CPU float64)."""
import ast
import hashlib
import copy
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from mlpui.backends import get_backend
from mlpui.models.torchmdnet.models import utils
from test_qmmm import config, sample, predict

TARGETS = ('energy', 'forces', 'mm_potential', 'mm_potential_gradient')


def make_model():
    torch.manual_seed(731)
    c = config()
    c.update(num_layers=2, ext_num_rbf=5)
    return get_backend('tensornet_qmmm').build_model(c).eval()


def reference():
    # Execute only definitions from the unmodified pinned MIT source. Supply local
    # utility dependencies; do not execute upstream import-time global TF32 changes.
    path = Path(__file__).parent / 'fixtures/tensornet_ext_b0f7c7b.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '95c9507cd5ceedcb8700229a16286450ca2b2c8043f8b5deb7fd87ecaa3fe9ce'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    def scatter(src, index):
        return src.new_zeros((int(index.max()) + 1,) + src.shape[1:]).index_add(0, index, src)
    ns = dict(torch=torch, nn=nn, Tensor=Tensor, Optional=Optional, Tuple=Tuple, scatter=scatter)
    for name in ('CosineCutoff', 'OptimizedDistance', 'rbf_class_mapping', 'act_class_mapping'):
        ns[name] = getattr(utils, name)
    exec(compile(tree, str(path), 'exec'), ns)
    return ns


def test_full_pinned_upstream_energy_and_all_derivatives():
    model = make_model()
    c = model.config
    ref = reference()['TensorNet_Ext'](
        hidden_channels=c['embedding_dimension'], num_layers=c['num_layers'],
        num_rbf=c['num_rbf'], cutoff_upper=c['cutoff_upper'],
        ext_num_rbf=c['ext_num_rbf'], ext_cutoff_upper=c['ext_cutoff_upper'],
        dtype=torch.float64).double()
    ref.load_state_dict(model.qm_representation.state_dict(), strict=True)
    s = sample()
    out = predict(model, s, TARGETS)
    z = torch.tensor(s['z']); pos = torch.tensor(s['pos'], requires_grad=True)
    mm = torch.tensor(s['mm_pos'], requires_grad=True)
    q = torch.tensor(s['mm_charges'], requires_grad=True)
    x = ref(z, pos, mm, q, torch.zeros_like(z))[0]
    energy = model.residual_head(x).sum()
    f, esp, g = torch.autograd.grad(energy, (pos, q, mm))
    for key, value in dict(energy=energy, forces=-f, mm_potential=esp,
                           mm_potential_gradient=g/q[:, None]).items():
        torch.testing.assert_close(out[key], value, atol=2e-12, rtol=2e-11)


def test_ragged_batch_equals_individual_no_cross_edges():
    model = make_model()
    frames = [sample(), sample(), sample()]
    frames[0].update(mm_pos=np.empty((0, 3)), mm_charges=np.empty(0))
    frames[1]['z'] = frames[1]['z'][:2]; frames[1]['pos'] = frames[1]['pos'][:2]
    # Deliberately overlapping frames exposes accidental cross-sample edges.
    joined = {k: np.concatenate([s[k] for s in frames]) for k in ('z','pos','mm_pos','mm_charges')}
    joined.update(batch=np.repeat(np.arange(3), [3, 2, 3]), mm_offsets=np.array([0,0,2,4]), charge=np.zeros(3))
    out = predict(model, joined, TARGETS)
    singles = [predict(model, s, TARGETS) for s in frames]
    for key in TARGETS:
        expected = torch.stack([o[key] for o in singles]) if key == 'energy' else torch.cat([o[key] for o in singles])
        torch.testing.assert_close(out[key], expected, atol=1e-11, rtol=1e-10)
    explicit = copy.deepcopy(joined); del explicit['mm_offsets']; explicit['mm_batch'] = np.array([1,1,2,2])
    for key in TARGETS:
        torch.testing.assert_close(predict(model, explicit, TARGETS)[key], out[key])
    bad = copy.deepcopy(joined); bad['mm_offsets'] = np.array([0,2,1,4])
    with pytest.raises(ValueError, match='offsets'): predict(model, bad, TARGETS)
    del bad['mm_offsets']
    with pytest.raises(ValueError, match='require'): predict(model, bad, TARGETS)


@pytest.mark.parametrize('position', [2., 10., 12.])
def test_single_edge_trailing_qm_empty_and_cutoff(position):
    model = make_model()
    s = dict(z=np.array([1,1]), pos=np.array([[0.,0.,0.],[30.,0.,0.]]),
             mm_pos=np.array([[position,0.,0.]]), mm_charges=np.array([.3]))
    out = predict(model, s, TARGETS)
    assert all(torch.isfinite(out[k]).all() for k in TARGETS)
    if position >= 10:
        for key in ('mm_potential', 'mm_potential_gradient'):
            torch.testing.assert_close(out[key], torch.zeros_like(out[key]), atol=0, rtol=0)
    # The cosine envelope (and expnorm envelope) tends continuously to zero.
    if position == 10:
        s['mm_pos'][0,0] -= 1e-6
        near = predict(model, s, TARGETS)
        torch.testing.assert_close(near['energy'], out['energy'], atol=1e-10, rtol=0)
        assert near['mm_potential_gradient'].abs().max() < 1e-7
    s.update(mm_pos=np.empty((0,3)), mm_charges=np.empty(0))
    empty = predict(model.train(), s, TARGETS)
    assert empty['mm_potential_gradient'].shape == (0,3)
    sum(empty[k].sum() for k in TARGETS).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_charge_limit_and_gesp_parameter_finite_difference():
    model = make_model().train()
    s = sample(); s['mm_charges'][0] = 0.
    zero = predict(model, s, TARGETS)
    assert zero['mm_potential_gradient'][0].abs().max() > 1e-8
    for q in (1e-6, -1e-6):
        small = copy.deepcopy(s); small['mm_charges'][0] = q
        out = predict(model, small, TARGETS + ('mm_forces',))
        torch.testing.assert_close(-out['mm_forces']/torch.tensor(small['mm_charges'])[:,None],
                                   out['mm_potential_gradient'], atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(out['mm_potential_gradient'], zero['mm_potential_gradient'], atol=2e-7, rtol=2e-5)
    loss = (zero['mm_potential_gradient'] - .17).square().mean()
    loss.backward()
    # Both the external kernel and nonlinear response coefficient contribute.
    for parameter in (model.qm_representation.ext_layer.linears_scalar[0].weight,
                      model.qm_representation.layers[0].linears_tensor[0].weight,
                      model.residual_head[0].weight):
        index = np.unravel_index(parameter.grad.abs().argmax().item(), parameter.shape)
        analytical = parameter.grad[index].item(); original = parameter[index].item()
        assert abs(analytical) > 1e-9
        values = []
        for delta in (1e-5, -1e-5):
            with torch.no_grad(): parameter[index] = original + delta
            values.append((predict(model, s, TARGETS)['mm_potential_gradient'] - .17).square().mean().item())
        with torch.no_grad(): parameter[index] = original
        assert analytical == pytest.approx((values[0]-values[1])/2e-5, rel=2e-5, abs=1e-9)


def test_fixed_response_kernel_probe_fd_and_not_total_mixed():
    model = make_model().train(); s = sample()
    backend = get_backend('tensornet_qmmm')
    inp = backend.prepare_sample(s, 'cpu', torch.float64, model=model); inp.pop('baseline')
    inp['mm_charges'].requires_grad_(); inp['mm_pos'].requires_grad_()
    parts, message, batch, mm_batch = model._energy_with_environment(**inp)
    response = torch.autograd.grad(parts['residual'], message, create_graph=True, retain_graph=True)[0]
    out = model(**inp, properties=TARGETS)
    ext = model.qm_representation.ext_layer
    eps = 1e-5
    fd = torch.zeros_like(inp['mm_pos'])
    for a in range(len(fd)):
        for d in range(3):
            values = []
            unit = torch.zeros_like(inp['mm_charges']); unit[a] = 1
            for delta in (eps, -eps):
                probe = inp['mm_pos'].detach().clone(); probe[a,d] += delta
                values.append((response.detach() * ext(inp['pos'], probe, unit, batch, mm_batch)).sum())
            fd[a,d] = (values[0]-values[1])/(2*eps)
    torch.testing.assert_close(fd, out['mm_potential_gradient'], atol=1e-9, rtol=2e-7)
    mixed = torch.autograd.grad(out['mm_potential'].sum(), inp['mm_pos'])[0]
    assert not torch.allclose(mixed, out['mm_potential_gradient'], atol=1e-8, rtol=1e-4)


def test_legacy_rejected_even_direct_nonstrict_load():
    model = make_model(); state = model.state_dict(); state['qmmm_schema'] = torch.tensor(1)
    with pytest.raises(ValueError, match='Incompatible QMMM architecture'):
        model.load_state_dict(state, strict=False)
    with pytest.raises(ValueError, match='Incompatible QMMM architecture'):
        get_backend('tensornet_qmmm').load_model(state, config())
    old_config = config(); old_config.pop('architecture'); old_config['schema_version'] = 1
    with pytest.raises(ValueError, match='Incompatible QMMM architecture'):
        get_backend('tensornet_qmmm').build_model(old_config)


def test_config_yaml_web_defaults_and_all_target_o3_permutation():
    import yaml
    backend = get_backend('tensornet_qmmm')
    defaults = backend.default_config()
    yaml_config = yaml.safe_load((Path(__file__).parents[1]/'configs/train_tensornet_qmmm.yaml').read_text())['model_config']
    assert backend._config(yaml_config) == {**defaults, 'precision':64}
    model = make_model(); s = sample(); original = predict(model, s, TARGETS)
    rotation = np.linalg.qr(np.array([[1.,2.,3.],[4.,-.5,2.],[1.,.1,-.7]]))[0]
    rotation[:,0] *= -1
    changed = copy.deepcopy(s)
    for key in ('pos','mm_pos'): changed[key] = changed[key] @ rotation + [3.,-1.,.2]
    changed['mm_pos'] = changed['mm_pos'][::-1].copy()
    changed['mm_charges'] = changed['mm_charges'][::-1].copy()
    actual = predict(model, changed, TARGETS)
    torch.testing.assert_close(actual['energy'], original['energy'], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual['forces'], original['forces']@torch.tensor(rotation), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual['mm_potential'], original['mm_potential'].flip(0), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual['mm_potential_gradient'], original['mm_potential_gradient'].flip(0)@torch.tensor(rotation), atol=1e-10, rtol=1e-10)


def test_trusted_serialized_legacy_module_rejected(tmp_path):
    from mlpui.external_models import load_external_checkpoint
    model = make_model()
    model.config['architecture'] = 'legacy_h_plus_3'
    model.qmmm_schema.fill_(1)
    path = tmp_path/'legacy_module.pt'
    torch.save(model, path)
    with pytest.raises(ValueError, match='Incompatible QMMM architecture'):
        load_external_checkpoint(path, trusted_checkpoint=True, device='cpu')


def test_batched_offline_composition():
    from test_qmmm import offline_config
    backend = get_backend('tensornet_qmmm')
    cfg = offline_config(); cfg.update(num_layers=2)
    model = backend.build_model(cfg).eval()
    bare = backend.build_model({**cfg, 'baseline':{'mode':'residual_only'}}).eval()
    bare.load_state_dict(model.state_dict())
    one = sample()
    s = {k:np.concatenate([v,v]) for k,v in one.items()}
    s.update(batch=np.repeat([0,1],3), mm_offsets=np.array([0,2,4]), charge=np.zeros(2))
    original = predict(bare, s, TARGETS)
    for key in TARGETS: s['baseline_'+key] = np.ones(original[key].shape)*.123
    s['baseline_metadata'] = cfg['baseline']
    out = predict(model, s, TARGETS)
    for key in TARGETS: torch.testing.assert_close(out[key],original[key]+.123)
