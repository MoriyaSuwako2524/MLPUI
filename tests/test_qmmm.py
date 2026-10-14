"""Scientific and integration acceptance tests for the fixed-environment prototype."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from mlpui.backends import get_backend, find_backend
from mlpui.data import NpyDataset, NpyShards
from mlpui.external_models import load_external_checkpoint
from mlpui.training import Trainer, TrainingConfig
from mlpui.models.tensornet_qmmm.model import GasCoulombBaseline, environment


def config():
    c = get_backend('tensornet_qmmm').default_config()
    c.update(embedding_dimension=8, num_layers=1, num_rbf=4, precision=64)
    return c


def sample():
    return dict(z=np.array([8, 1, 1]), pos=np.array([[0., 0., 0.], [1., .2, .1], [-.2, .9, .3]]),
        mm_pos=np.array([[2., -.7, .5], [-1.2, 1.8, -.6]]), mm_charges=np.array([.2, -.3]))


@pytest.fixture
def model():
    torch.manual_seed(9)
    return get_backend('tensornet_qmmm').build_model(config()).eval()


def predict(model, s, props=('energy', 'forces', 'mm_forces', 'mm_potential')):
    backend = get_backend('tensornet_qmmm')
    inputs = backend.prepare_sample(s, 'cpu', torch.float64, model=model)
    return backend.forward(model, inputs, props)


def test_dual_ragged_zero_mm_and_units(tmp_path):
    arrays = dict(z=[8, 1, 1, 8, 1], pos=np.arange(15).reshape(5, 3), offsets=[0, 3, 5],
        mm_pos=np.array([[1., 2., 3.], [4., 5., 6.]]), mm_charges=[.2, -.3], mm_offsets=[0, 0, 2],
        mm_forces=np.ones((2, 3)), mm_potential=np.ones(2), baseline_forces=np.ones((5, 3)),
        energy=[1., 2.])
    for name, a in arrays.items():
        np.save(tmp_path / f'{name}.npy', a)
    (tmp_path / 'baseline_metadata.json').write_text(json.dumps(offline_config()['baseline']))
    d = NpyDataset(tmp_path, gradients=True, length_scale=2., energy_scale=3.)
    assert d[0]['mm_pos'].shape == (0, 3)
    assert d[0]['mm_charges'].shape == (0,)
    np.testing.assert_allclose(d[1]['mm_pos'], arrays['mm_pos'] * 2)
    np.testing.assert_allclose(d[-1]['pos'], arrays['pos'][3:] * 2)
    np.testing.assert_allclose(d[1]['mm_forces'], -1.5)
    np.testing.assert_allclose(d[1]['baseline_forces'], 1.5)  # baselines are always forces
    np.testing.assert_allclose(d[1]['mm_potential'], 3.)
    assert d[1]['energy'] == 6.
    del d
    np.save(tmp_path / 'mm_offsets.npy', [0, 2, 1])
    with pytest.raises(ValueError, match='MM ragged'):
        NpyDataset(tmp_path)


def test_dense_mm_independent_of_qm_ragged(tmp_path):
    arrays = dict(z=[1, 8, 1], pos=np.zeros((3, 3)), offsets=[0, 1, 3],
        mm_pos=np.zeros((2, 0, 3)), mm_charges=np.zeros((2, 0)))
    for name, a in arrays.items():
        np.save(tmp_path / f'{name}.npy', a)
    d = NpyDataset(tmp_path)
    assert len(d[0]['z']) == 1 and len(d[1]['z']) == 2
    assert d[1]['mm_pos'].shape == (0, 3)
    del d
    np.save(tmp_path / 'mm_charges.npy', np.zeros((2, 1)))
    with pytest.raises(ValueError, match='MM point layout'):
        NpyDataset(tmp_path)


def test_shard_alignment(tmp_path):
    for k in ('a', 'b'):
        np.save(tmp_path / f'{k}_z.npy', [1])
        np.save(tmp_path / f'{k}_pos.npy', np.zeros((1, 1, 3)))
        np.save(tmp_path / f'{k}_mm_pos.npy', np.ones((1, 2 if k == 'a' else 0, 3)))
        np.save(tmp_path / f'{k}_mm_charges.npy', np.ones((1, 2 if k == 'a' else 0)))
    fields = {k: '{shard}_' + k + '.npy' for k in ('z', 'pos', 'mm_pos', 'mm_charges')}
    d = NpyShards(tmp_path, ['a', 'b'], files=fields)
    assert len(d[0]['mm_pos']) == 2 and len(d[1]['mm_pos']) == 0


@pytest.mark.parametrize('reflection', [False, True])
def test_o3_translation_and_permutation(model, reflection):
    s = sample()
    original = predict(model, s)
    rotation = np.linalg.qr(np.array([[1., 2., 3.], [4., -.5, 2.], [1., .1, -.7]]))[0]
    if reflection:
        rotation[:, 0] *= -1
    moved = copy.deepcopy(s)
    for key in ('pos', 'mm_pos'):
        moved[key] = s[key] @ rotation + np.array([4., -3., .5])
    out = predict(model, moved)
    torch.testing.assert_close(out['energy'], original['energy'], atol=1e-9, rtol=1e-9)
    for prop in ('forces', 'mm_forces'):
        torch.testing.assert_close(out[prop], original[prop] @ torch.tensor(rotation), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(out['mm_potential'], original['mm_potential'], atol=1e-9, rtol=1e-9)
    permutation, mm_permutation = [2, 0, 1], [1, 0]
    s['z'], s['pos'] = s['z'][permutation], s['pos'][permutation]
    s['mm_pos'], s['mm_charges'] = s['mm_pos'][mm_permutation], s['mm_charges'][mm_permutation]
    out = predict(model, s)
    torch.testing.assert_close(out['energy'], original['energy'])
    torch.testing.assert_close(out['forces'], original['forces'][permutation])
    torch.testing.assert_close(out['mm_forces'], original['mm_forces'][mm_permutation])
    torch.testing.assert_close(out['mm_potential'], original['mm_potential'][mm_permutation])
    torch.testing.assert_close(original['forces'].sum(0) + original['mm_forces'].sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-9, rtol=0)


@pytest.mark.parametrize('field,prop,sign', [('pos', 'forces', -1), ('mm_pos', 'mm_forces', -1), ('mm_charges', 'mm_potential', 1)])
def test_finite_difference(model, field, prop, sign):
    s = sample()
    actual = predict(model, s)[prop].detach().numpy()
    for index in np.ndindex(s[field].shape):
        hi, lo = copy.deepcopy(s), copy.deepcopy(s)
        hi[field][index] += 1e-5
        lo[field][index] -= 1e-5
        derivative = (predict(model, hi, ('energy',))['energy'] - predict(model, lo, ('energy',))['energy']).item() / 2e-5
        assert actual[index] == pytest.approx(sign * derivative, rel=2e-4, abs=2e-6)


def test_zero_mm(model):
    s = sample()
    s.update(mm_pos=np.zeros((0, 3)), mm_charges=np.zeros(0))
    out = predict(model, s)
    assert out['mm_forces'].shape == (0, 3)
    assert out['mm_potential'].shape == (0,)
    assert torch.isfinite(out['energy'])


@pytest.mark.parametrize('prop', ['forces', 'mm_forces', 'mm_potential'])
def test_derivative_loss_parameter_gradient(model, prop):
    model.train()
    loss = predict(model, sample(), (prop,))[prop].square().mean()
    loss.backward()
    parameter = model.residual_head[0].weight
    analytical = parameter.grad[0, -3].item()
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None) > 0
    original = parameter[0, -3].item()
    values = []
    for delta in (1e-5, -1e-5):
        with torch.no_grad():
            parameter[0, -3] = original + delta
        values.append(predict(model, sample(), (prop,))[prop].square().mean().item())
    with torch.no_grad():
        parameter[0, -3] = original
    assert analytical == pytest.approx((values[0] - values[1]) / 2e-5, rel=3e-4, abs=2e-6)


def offline_config(kind='full_dq_dR'):
    c = config()
    c['baseline'] = dict(mode='offline', energy_definition='uma_gas_plus_gas_charge_coulomb',
        qm_force_kind=kind, uma_checkpoint_sha256='a' * 64, charge_checkpoint_sha256='b' * 64,
        source_manifest_sha256='c' * 64)
    return c


def test_offline_baseline_composition_and_rejections():
    torch.manual_seed(5)
    m = get_backend('tensornet_qmmm').build_model(offline_config()).eval()
    s = sample()
    s['baseline_metadata'] = offline_config()['baseline']
    props = ('energy', 'forces', 'mm_forces', 'mm_potential')
    with pytest.raises(ValueError, match='baseline_energy'):
        predict(m, s)
    s.update(baseline_energy=np.array(3.), baseline_forces=np.ones((3, 3)),
        baseline_mm_forces=np.ones((2, 3)) * 2, baseline_mm_potential=np.ones(2) * 4)
    out = predict(m, s)
    bare = get_backend('tensornet_qmmm').build_model(config()).eval()
    bare.load_state_dict(m.state_dict())
    residual = predict(bare, sample())
    for prop in props:
        torch.testing.assert_close(out[prop], residual[prop] + torch.as_tensor(s['baseline_' + prop]))
    with pytest.raises(ValueError, match='dq/dR'):
        Trainer('tensornet_qmmm', offline_config('fixed_charge_approximation'))
    invalid = offline_config()
    del invalid['baseline']['source_manifest_sha256']
    with pytest.raises(ValueError, match='provenance'):
        get_backend('tensornet_qmmm').build_model(invalid)


def test_differentiable_gas_charge_baseline():
    class Energy(torch.nn.Module):
        def forward(self, z, pos, charge):
            return (pos[1:] - pos[:1]).square().sum() * .05
    class Charges(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        def forward(self, z, pos, charge):
            raw = self.scale * (pos - pos.mean(0)).square().sum(-1)
            return raw - raw.mean() + charge / len(pos)
    b = GasCoulombBaseline(Energy(), Charges()).train()
    assert not b.gas_charge.training and not b.gas_charge.scale.requires_grad
    s = sample()
    backend = get_backend('tensornet_qmmm')
    model = backend.build_model(config())
    inputs = backend.prepare_sample(s, 'cpu', torch.float64, model=model)
    inputs.pop('baseline')
    inputs['pos'].requires_grad_(True)
    terms = b(**inputs)
    derivative = torch.autograd.grad(sum(terms.values()), inputs['pos'])[0]
    hi, lo = copy.deepcopy(s), copy.deepcopy(s)
    hi['pos'][0, 0] += 1e-5
    lo['pos'][0, 0] -= 1e-5
    values = []
    for state in (hi, lo):
        t = backend.prepare_sample(state, 'cpu', torch.float64, model=model)
        t.pop('baseline')
        values.append(sum(b(**t).values()).item())
    assert derivative[0, 0].item() == pytest.approx((values[0] - values[1]) / 2e-5, abs=1e-7)
    pos = inputs['pos'].detach().requires_grad_(True)
    q = b.gas_charge(inputs['z'], pos, inputs['charge']).detach()
    phi, _ = environment(pos, inputs['mm_pos'], inputs['mm_charges'])
    fixed = b.gas_energy(inputs['z'], pos, inputs['charge']) + (q * phi).sum()
    frozen_charge_derivative = torch.autograd.grad(fixed, pos)[0]
    assert not torch.allclose(derivative, frozen_charge_derivative)
    composed = b.compose(model, inputs)
    assert set(composed['components']) == {'gas', 'coulomb', 'residual'}
    torch.testing.assert_close(composed['energy'], sum(composed['components'].values()))


def write_training_data(path):
    path.mkdir()
    s = sample()
    for name in ('pos', 'mm_pos', 'mm_charges'):
        np.save(path / f'{name}.npy', np.stack([s[name], s[name]]))
    np.save(path / 'z.npy', s['z'])
    np.save(path / 'energy.npy', [1., 1.])
    np.save(path / 'forces.npy', np.zeros((2, 3, 3)))
    np.save(path / 'mm_forces.npy', np.zeros((2, 2, 3)))
    np.save(path / 'mm_potential.npy', np.zeros((2, 2)))
    return path


def test_train_checkpoint_roundtrip_and_detection(tmp_path):
    props = ('energy', 'forces', 'mm_forces', 'mm_potential')
    t = Trainer('tensornet_qmmm', config(), TrainingConfig(epochs=2, dtype=torch.float64,
        loss_weights={p: 1. for p in props}))
    data = NpyDataset(write_training_data(tmp_path / 'data'))
    initial = {k: v.clone() for k, v in t.model.state_dict().items()}
    t.fit(data, output_dir=tmp_path / 'run')
    assert any(not torch.equal(initial[k], v) for k, v in t.model.state_dict().items())
    assert find_backend(t.model).name == 'tensornet_qmmm'
    assert find_backend(t.model.state_dict()).name == 'tensornet_qmmm'
    loaded = load_external_checkpoint(tmp_path / 'run' / 'model.pt', dtype=torch.float64, device='cpu').model
    for prop, value in predict(t.model, sample()).items():
        if prop != 'components':
            torch.testing.assert_close(value, predict(loaded, sample())[prop], rtol=0, atol=0)
    saved = torch.load(tmp_path / 'run' / 'model.pt', weights_only=True)
    saved['state_dict']['unexpected'] = torch.ones(1)
    torch.save(saved, tmp_path / 'bad.pt')
    with pytest.raises(RuntimeError, match='Unexpected key'):
        load_external_checkpoint(tmp_path / 'bad.pt')
    assert t.evaluate(data)['metrics']['mm_forces']['count'] == 12


def test_cli(tmp_path):
    write_training_data(tmp_path / 'data')
    payload = dict(family='tensornet_qmmm', model_config=config(),
        training=dict(epochs=1, dtype='float64', loss_weights={'energy': 1., 'forces': 1., 'mm_forces': 1., 'mm_potential': 1.}),
        train={'directory': 'data'}, output_dir='run')
    # JSON is also valid YAML; no extra config serializer needed.
    path = tmp_path / 'train.yaml'
    path.write_text(json.dumps(payload))
    run = subprocess.run([sys.executable, '-B', '-m', 'scripts.train', str(path)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / 'run' / 'model.pt').is_file()


def test_web_export_roundtrip(tmp_path):
    from mlpui.web.datasets import export_subset
    from mlpui.web.config import dataset
    data = NpyDataset(write_training_data(tmp_path / 'data'))
    result = dataset(export_subset(data, [1, 0], tmp_path / 'export'))
    for row, index in enumerate([1, 0]):
        for key, value in data[index].items():
            np.testing.assert_array_equal(result[row][key], value)
        assert result[row]['frame_index'] == index


def test_empty_mm_only_training_and_metrics(tmp_path):
    path = tmp_path / 'empty'
    path.mkdir()
    s = sample()
    for key, value in dict(z=s['z'], pos=s['pos'][None], mm_pos=np.zeros((1, 0, 3)),
                          mm_charges=np.zeros((1, 0)), mm_forces=np.zeros((1, 0, 3))).items():
        np.save(path / f'{key}.npy', value)
    t = Trainer('tensornet_qmmm', config(), TrainingConfig(epochs=1, dtype=torch.float64,
                loss_weights={'mm_forces': 1.}))
    t.fit(path)
    assert t.history[0]['train']['mm_forces'] == 0
    assert t.evaluate(path)['metrics']['mm_forces'] == dict(mae=None, mse=None, rmse=None, count=0)


def test_checkpoint_physics_override_rejected(tmp_path):
    t = Trainer('tensornet_qmmm', config(), TrainingConfig(dtype=torch.float64))
    path = t.save(tmp_path / 'model.pt')
    altered = config()
    altered['field_scale'] = 2.
    with pytest.raises(ValueError, match='cannot be overridden'):
        load_external_checkpoint(path, model_config=altered)
    saved = torch.load(path, weights_only=True)
    saved['state_dict']['qmmm_schema'] = torch.tensor(2)
    torch.save(saved, tmp_path / 'bad_schema.pt')
    with pytest.raises(ValueError, match='weight schema'):
        load_external_checkpoint(tmp_path / 'bad_schema.pt')


def test_offline_checkpoint_and_dataset_metadata(tmp_path):
    c = offline_config()
    t = Trainer('tensornet_qmmm', c, TrainingConfig(dtype=torch.float64))
    path = t.save(tmp_path / 'model.pt')
    loaded = load_external_checkpoint(path, device='cpu').model
    s = sample()
    s.update(baseline_energy=np.array(1.), baseline_forces=np.ones((3, 3)),
             baseline_mm_forces=np.ones((2, 3)), baseline_mm_potential=np.ones(2),
             baseline_metadata=c['baseline'])
    t.model.eval()
    expected, actual = predict(t.model, s), predict(loaded, s)
    for prop in ('energy', 'forces', 'mm_forces', 'mm_potential'):
        torch.testing.assert_close(expected[prop], actual[prop], atol=0, rtol=0)
    s['baseline_metadata'] = {**c['baseline'], 'source_manifest_sha256': 'd' * 64}
    with pytest.raises(ValueError, match='exactly match'):
        predict(loaded, s)
    directory = write_training_data(tmp_path / 'data')
    np.save(directory / 'baseline_energy.npy', np.ones(2))
    with pytest.raises(ValueError, match='baseline_metadata.json'):
        NpyDataset(directory)


def test_invalid_environment_and_default_dtype():
    backend = get_backend('tensornet_qmmm')
    c = config()
    c['precision'] = 32
    m = backend.build_model(c)
    inputs = backend.prepare_sample(sample(), 'cpu', torch.float32, model=m)
    out = backend.forward(m, inputs, ('energy', 'forces'))
    assert out['energy'].dtype == torch.float32 and torch.isfinite(out['forces']).all()
    s = sample()
    s['pbc'] = [True, True, True]
    with pytest.raises(ValueError, match='nonperiodic'):
        predict(m, s)
    s = sample()
    s['mm_pos'][0] = s['pos'][0]
    with pytest.raises(ValueError, match='Coincident'):
        inputs = backend.prepare_sample(s, 'cpu', torch.float32, model=m)
        backend.forward(m, inputs, ('energy',))


def test_unrequested_offline_derivatives_are_not_returned():
    backend = get_backend('tensornet_qmmm')
    m = backend.build_model(offline_config()).eval()
    s = sample()
    s.update(baseline_metadata=offline_config()['baseline'], baseline_energy=np.array(0.),
             baseline_forces=np.zeros((3, 3)))
    out = predict(m, s, ('forces',))
    assert set(out) == {'energy', 'forces', 'components', 'derivative_contract'}


def test_web_capabilities_and_ase_boundary(model):
    from mlpui.web.config import presets
    assert 'tensornet_qmmm' in presets()
    assert get_backend('tensornet_qmmm').supports_evaluation_artifacts
    with pytest.raises(ValueError, match='ASE/MD'):
        get_backend('tensornet_qmmm').prepare_inputs(None, 'cpu', torch.float64)
    with pytest.raises(ValueError, match='dual-region'):
        get_backend('torchmdnet').prepare_sample(sample(), 'cpu', torch.float64)
