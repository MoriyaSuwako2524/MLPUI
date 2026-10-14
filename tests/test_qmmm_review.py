"""Contract tests from the post-implementation QMMM review."""
import copy
import numpy as np
import pytest
import torch

from mlpui.backends import get_backend
from mlpui.data import NpyDataset
from mlpui.external_models import load_external_checkpoint
from mlpui.models.tensornet_qmmm.model import GasCoulombBaseline, COULOMB_EV_ANGSTROM
from mlpui.training import Trainer, TrainingConfig
from test_qmmm import sample, config, offline_config, predict


class GasEnergy(torch.nn.Module):
    def forward(self, z, pos, charge):
        return .1 * (pos - pos.mean(0)).square().sum()


class NonconservingCharges(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))

    def forward(self, z, pos, charge):
        return .4 + self.scale * (pos - pos.mean(0)).square().sum(-1)


def tensors(s):
    m = get_backend('tensornet_qmmm').build_model(config())
    inputs = get_backend('tensornet_qmmm').prepare_sample(s, 'cpu', torch.float64, model=m)
    inputs.pop('baseline')
    return inputs


def test_total_charge_projection_jacobian_and_force_fd():
    baseline = GasCoulombBaseline(GasEnergy(), NonconservingCharges())
    s = sample()
    s['charge'] = 1.
    inputs = tensors(s)
    inputs['pos'].requires_grad_(True)
    q = baseline.projected_charges(inputs['z'], inputs['pos'], inputs['charge'])
    torch.testing.assert_close(q.sum(), inputs['charge'], atol=1e-14, rtol=0)
    jacobian_sum = torch.autograd.grad(q.sum(), inputs['pos'], retain_graph=True)[0]
    torch.testing.assert_close(jacobian_sum, torch.zeros_like(jacobian_sum), atol=1e-14, rtol=0)
    total = sum(baseline(**inputs).values())
    gradient = torch.autograd.grad(total, inputs['pos'])[0]
    values = []
    for delta in (1e-5, -1e-5):
        perturbed = copy.deepcopy(s)
        perturbed['pos'][1, 2] += delta
        values.append(sum(baseline(**tensors(perturbed)).values()).item())
    assert gradient[1, 2].item() == pytest.approx((values[0] - values[1]) / 2e-5, abs=1e-7)
    assert not baseline.gas_charge.scale.requires_grad


def test_mm_potential_baseline_sign_and_charge_fd():
    baseline = GasCoulombBaseline(GasEnergy(), NonconservingCharges())
    s = sample()
    inputs = tensors(s)
    inputs['mm_charges'].requires_grad_(True)
    result = sum(baseline(**inputs).values())
    potential = torch.autograd.grad(result, inputs['mm_charges'])[0]
    q = baseline.projected_charges(inputs['z'], inputs['pos'], inputs['charge'])
    distance = torch.linalg.vector_norm(inputs['pos'][:, None] - inputs['mm_pos'][None], dim=-1)
    expected = COULOMB_EV_ANGSTROM * (q[:, None] / distance).sum(0)
    torch.testing.assert_close(potential, expected)
    for j in range(len(s['mm_charges'])):
        values = []
        for delta in (1e-5, -1e-5):
            perturbed = copy.deepcopy(s)
            perturbed['mm_charges'][j] += delta
            values.append(sum(baseline(**tensors(perturbed)).values()).item())
        assert potential[j].item() == pytest.approx((values[0] - values[1]) / 2e-5, abs=1e-8)


def test_offline_force_not_claimed_to_be_total_energy_autograd():
    backend = get_backend('tensornet_qmmm')
    model = backend.build_model(offline_config()).eval()
    s = sample()
    s.update(baseline_metadata=offline_config()['baseline'], baseline_energy=np.array(2.),
             baseline_forces=np.ones((3, 3)) * .7)
    inputs = backend.prepare_sample(s, 'cpu', torch.float64, model=model)
    inputs['pos'].requires_grad_(True)
    energy_only = backend.forward(model, inputs, ('energy',))
    residual_force = -torch.autograd.grad(energy_only['energy'], inputs['pos'])[0]
    composed = predict(model, s, ('forces',))
    assert energy_only['derivative_contract'] == 'offline_component_values_not_total_energy_autograd'
    torch.testing.assert_close(composed['forces'], residual_force + .7)
    assert not torch.allclose(composed['forces'], residual_force)


def test_optional_targets_and_potential_baseline_subtraction():
    trainer = Trainer('tensornet_qmmm', offline_config(), TrainingConfig(dtype=torch.float64,
                       loss_weights={'mm_potential': 1.}))
    trainer.model.eval()
    s = sample()
    s.update(baseline_metadata=offline_config()['baseline'], baseline_energy=np.array(2.),
             baseline_mm_potential=np.array([3., -4.]))
    prediction = predict(trainer.model, s, ('mm_potential',))['mm_potential'].detach().numpy()
    s['mm_potential'] = prediction - np.array([.2, -.5])
    errors = trainer._errors(s)
    assert set(errors) == {'mm_potential'}
    np.testing.assert_allclose(errors['mm_potential'].detach(), [.2, -.5], atol=1e-12)
    # No energy, QM-force or MM-force labels are required for potential-only loss.
    del s['mm_potential']
    with pytest.raises(ValueError, match='missing enabled labels'):
        trainer._loss(s)


def test_dense_qm_ragged_mm_with_empty_middle_sample(tmp_path):
    arrays = dict(z=[8, 1], pos=np.arange(18).reshape(3, 2, 3),
        mm_pos=np.arange(9).reshape(3, 3), mm_charges=[.1, .2, .3], mm_offsets=[0, 1, 1, 3],
        mm_forces=np.arange(9).reshape(3, 3), mm_potential=[10., 20., 30.])
    for name, array in arrays.items():
        np.save(tmp_path / f'{name}.npy', array)
    data = NpyDataset(tmp_path)
    assert data[1]['mm_pos'].shape == (0, 3)
    assert data[1]['mm_potential'].shape == (0,)
    np.testing.assert_array_equal(data[2]['pos'], arrays['pos'][2])
    np.testing.assert_array_equal(data[2]['mm_pos'], arrays['mm_pos'][1:])
    np.testing.assert_array_equal(data[2]['mm_charges'], [.2, .3])
    np.testing.assert_array_equal(data[2]['mm_potential'], [20., 30.])


def test_offline_checkpoint_requires_external_sample_baseline(tmp_path):
    trainer = Trainer('tensornet_qmmm', offline_config(), TrainingConfig(dtype=torch.float64))
    path = trainer.save(tmp_path / 'model.pt')
    checkpoint = torch.load(path, weights_only=True)
    assert all(k == 'qmmm_schema' or k.startswith(('qm_representation.', 'residual_head.'))
               for k in checkpoint['state_dict'])
    loaded = load_external_checkpoint(path, device='cpu').model
    s = sample()
    s['baseline_metadata'] = offline_config()['baseline']
    with pytest.raises(ValueError, match='baseline_energy'):
        predict(loaded, s, ('energy',))
