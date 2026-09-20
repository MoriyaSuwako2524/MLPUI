"""Isolation and numerical compatibility with the former external backends."""
import os
import subprocess
import sys

import pytest
import torch

from mlpui.external_models import load_external_checkpoint
from mlpui.external_calculator import AtomicInputAdapter
from mlpui.models.torchmdnet.extensions.ops import python_neighbor_pairs
from test_external_models_integration import torchmd_args, water


def test_training_without_external_packages(tmp_path):
    code = r'''
import importlib.abc
import sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'newtonnet', 'torchmdnet', 'les'}:
            raise ImportError('External backend blocked: ' + fullname)
sys.meta_path.insert(0, BlockExternal())
from pathlib import Path
import numpy as np
from mlpui.training import Trainer, TrainingConfig
from mlpui.web.config import presets
from mlpui.calculator import CalculatorBuilder
from mlpui.data import NpyDataset
folder = Path(sys.argv[1])
np.save(folder / 'z.npy', [1, 1])
np.save(folder / 'pos.npy', [[[0., 0., 0.], [1., 0., 0.]]])
np.save(folder / 'energy.npy', [0.1])
np.save(folder / 'forces.npy', np.zeros((1, 2, 3)))
for family, config in presets().items():
    if family == 'newtonnet':
        config.update(n_features=8, n_basis=4, n_interactions=1)
    else:
        config.update(embedding_dimension=8, num_rbf=4, num_layers=1)
    trainer = Trainer(family, config, TrainingConfig(epochs=1))
    assert type(trainer.model).__module__.startswith('mlpui.models.' + family)
    trainer.fit(folder, output_dir=folder / family)
    assert trainer.evaluate(folder)['samples'] == 1
    atoms = NpyDataset.atoms(NpyDataset(folder)[0])
    atoms.calc = CalculatorBuilder.from_checkpoint(folder / family / 'model.pt', device='cpu').build()
    assert np.isfinite(atoms.get_forces()).all()
assert not any(k.startswith(('newtonnet', 'torchmdnet', 'les')) for k in sys.modules)
'''
    env = dict(os.environ, MLPUI_NEIGHBORS="python")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("loop,transpose", [(False, False), (True, False), (False, True), (True, True)])
def test_fallback_matches_reference_kernel(periodic, loop, transpose):
    reference = pytest.importorskip("torchmdnet.extensions.ops")
    torch.manual_seed(101)
    positions = torch.rand(6, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    box = torch.tensor([[8., 0., 0.], [1., 8., 0.], [0.5, 1., 8.]], dtype=torch.float64)
    args = ("brute", positions, batch, box, periodic, 0., 3., 40, loop, transpose)
    expected = reference.get_neighbor_pairs_kernel(*args)
    actual = python_neighbor_pairs(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)
    def derivatives(result):
        loss = result[1].square().sum() + result[2].pow(3).sum()
        first = torch.autograd.grad(loss, positions, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), positions)[0]
        return first, second
    for a, e in zip(derivatives(actual), derivatives(expected)):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("architecture", ["graph-network", "transformer", "equivariant-transformer", "tensornet"])
def test_external_torchmd_checkpoint_numerics(tmp_path, architecture):
    reference = pytest.importorskip("torchmdnet.models.model")
    config = torchmd_args(architecture)
    model = reference.create_model(dict(config)).eval()
    checkpoint = tmp_path / "old.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict()}, checkpoint)
    bundled = load_external_checkpoint(checkpoint, device="cpu").model
    assert type(bundled).__module__.startswith("mlpui.models.torchmdnet.")
    for periodic in (False, True):
        inputs = AtomicInputAdapter("torchmdnet").convert(water(periodic), "cpu", torch.float64)
        expected, actual = model(**inputs), bundled(**inputs)
        for key in ("y", "neg_dy", "charge"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_legacy_full_pickle_remaps_to_bundled(tmp_path, family):
    if family == "newtonnet":
        reference = pytest.importorskip("newtonnet.models.newtonnet")
        model = reference.NewtonNet(n_features=8, n_basis=4, n_interactions=1,
                                     output_properties=["energy", "gradient_force"]).double()
        model.eval()
    else:
        reference = pytest.importorskip("torchmdnet.models.model")
        model = reference.create_model(torchmd_args("tensornet")).eval()
    path = tmp_path / "old_full.pt"
    torch.save(model, path)
    bundled = load_external_checkpoint(path, family=family, trusted_checkpoint=True, device="cpu").model
    assert type(bundled).__module__.startswith("mlpui.models." + family)
    inputs = AtomicInputAdapter(family).convert(water(), "cpu", torch.float64)
    actual, expected = bundled(**inputs), model(**inputs)
    if family == "newtonnet":
        actual, expected = vars(actual), vars(expected)
    for key, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[key], rtol=1e-9, atol=1e-9)
