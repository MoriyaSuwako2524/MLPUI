"""Registry extensibility and equivalence to native model execution."""
import copy
import subprocess
import sys

import pytest
import torch
from ase import Atoms

import mlpui.backends as registry
from mlpui.backends import ModelBackend, get_backend, register_backend
from mlpui.calculator import CalculatorBuilder
from mlpui.external_models import load_external_checkpoint
from mlpui.training import Trainer, TrainingConfig, configure_charge_head
from mlpui.web.config import presets, backend_metadata, normalize
from test_training import write_data


def small_config(family):
    config = get_backend(family).default_config()
    if family == "newtonnet":
        config.update(n_features=8, n_basis=4, n_interactions=1)
    else:
        config.update(embedding_dimension=8, num_rbf=4, num_layers=1)
    return config


def test_registry_lazy_imports_and_aliases():
    result = subprocess.run([sys.executable, "-c", """
import sys
from mlpui.backends import registered_backends, get_backend
assert {b.name for b in registered_backends()} == {'newtonnet', 'torchmdnet', 'mace'}
assert get_backend('tensornet') is get_backend('torchmdnet')
assert get_backend('torchmd-net') is get_backend('torchmdnet')
assert not any(n.startswith(('mlpui.models.newtonnet', 'mlpui.models.torchmdnet', 'mlpui.models.mace')) for n in sys.modules)
"""], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    with pytest.raises(ValueError, match="Unknown model family"):
        get_backend("unregistered")
    with pytest.raises(ValueError, match="Duplicate"):
        register_backend(get_backend("newtonnet"))


def test_config_isolation_and_order():
    original = {"model": {"output_properties": ["charge", "energy", "gradient_force"]}}
    configured = configure_charge_head("newtonnet", original, ["charges"])
    assert configured == original
    configured["model"]["output_properties"].append("stress")
    assert original["model"]["output_properties"] == ["charge", "energy", "gradient_force"]
    default = presets()
    default["newtonnet"]["output_properties"].append("charge")
    assert "charge" not in presets()["newtonnet"]["output_properties"]


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_native_outputs_gradients_and_step_are_unchanged(family):
    """Compare the new dispatch with direct native calls, including force loss."""
    trainer = Trainer(family, small_config(family), TrainingConfig(
        dtype=torch.float64, loss_weights={"energy": 1., "forces": 2., "charges": 3.}))
    native = copy.deepcopy(trainer.model)
    trainer.model.train()
    native.train()
    atoms = Atoms("OH2", positions=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    z = torch.tensor(atoms.numbers)
    pos = torch.tensor(atoms.positions, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    if family == "newtonnet":
        raw = native(z=z, pos=pos, batch=batch, cell=torch.zeros(1, 3, 3, dtype=torch.float64))
        expected = dict(energy=raw.energy, forces=raw.gradient_force, charges=raw.charge)
    else:
        raw = native(z=z, pos=pos, batch=batch, box=None,
                     q=torch.zeros(1, dtype=torch.float64), s=torch.zeros(1, dtype=torch.float64))
        expected = dict(energy=raw["y"], forces=raw["neg_dy"], charges=raw["charge"])
    inputs = trainer.backend.prepare_inputs(atoms, "cpu", torch.float64)
    actual = trainer.backend.forward(trainer.model, inputs, trainer.config.loss_weights)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    sum(trainer.config.loss_weights[k] * v.square().mean() for k, v in expected.items()).backward()
    sum(trainer.config.loss_weights[k] * actual[k].square().mean() for k in expected).backward()
    for p, q in zip(trainer.model.parameters(), native.parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    native_optimizer = torch.optim.Adam(native.parameters(), lr=trainer.config.learning_rate)
    trainer.optimizer.step()
    native_optimizer.step()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_checkpoint_metadata_legacy_and_actual_capabilities(tmp_path, family):
    trainer = Trainer(family, small_config(family))
    path = trainer.save(tmp_path / "model.pt")
    saved = torch.load(path, weights_only=True)
    assert saved["backend"] == family
    assert saved["format_version"] == saved["model_config_version"] == 1
    assert "charges" in trainer.backend.training_targets
    assert "charges" not in trainer.backend.available_properties(trainer.model)
    for version in ("format_version", "model_config_version"):
        torch.save({**saved, version: 999}, tmp_path / "bad.pt")
        with pytest.raises(ValueError, match=version):
            load_external_checkpoint(tmp_path / "bad.pt", device="cpu")
    other = "torchmdnet" if family == "newtonnet" else "newtonnet"
    torch.save({**saved, "backend": other}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not requested family"):
        load_external_checkpoint(tmp_path / "bad.pt", device="cpu")
    legacy = {k: v for k, v in saved.items() if k not in {"backend", "format_version", "model_config_version"}}
    torch.save(legacy, tmp_path / "legacy.pt")
    for filename in (path, tmp_path / "legacy.pt"):
        model = load_external_checkpoint(filename, device="cpu").model
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.toy_scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, coordinates):
        return self.toy_scale * coordinates.square().sum()


class ToyBackend(ModelBackend):
    """A third, incompatible input API must work without orchestration edits."""
    name = "test_toy"
    label = "Toy"
    aliases = ("test_alias",)
    training_targets = ("energy",)

    def default_config(self):
        return {"toy": True}

    def add_charge_head(self, config):
        raise ValueError("Toy does not support charges")

    def matches(self, state):
        return isinstance(state, ToyModel) if isinstance(state, torch.nn.Module) else "toy_scale" in state

    def build_model(self, config):
        return ToyModel()

    def load_model(self, state, config):
        model = self.build_model(config)
        model.load_state_dict(state, strict=True)
        return model

    def validate_training(self, model, config, targets):
        assert set(targets) == {"energy"}

    def available_properties(self, model):
        return {"energy", "free_energy"}

    def prepare_inputs(self, atoms, device, dtype, *, length_to_angstrom=1., model=None, **kwargs):
        assert isinstance(model, ToyModel)
        return torch.tensor(atoms.positions / length_to_angstrom, device=device, dtype=dtype)

    def forward(self, model, inputs, properties):
        energy = model(inputs)
        return {"energy": energy, "free_energy": energy}


def test_third_backend_through_training_web_checkpoint_and_ase(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_backends", dict(registry._backends))
    monkeypatch.setattr(registry, "_aliases", dict(registry._aliases))
    register_backend(ToyBackend())
    assert backend_metadata()["test_toy"]["label"] == "Toy"
    config = presets()["test_toy"]
    data = write_data(tmp_path / "data")
    payload = normalize(dict(name="Third backend", family="test_toy", model_config=config,
                            training={"loss_weights": {"energy": 1.}}, train={"directory": str(data)}))
    trainer = Trainer("test_alias", payload["model_config"],
                      TrainingConfig(epochs=1, loss_weights={"energy": 1.}))
    before = trainer.model.toy_scale.detach().clone()
    trainer.fit(data, output_dir=tmp_path / "run")
    assert not torch.equal(before, trainer.model.toy_scale)
    assert trainer.evaluate(data)["metrics"]["energy"]["count"] == 2
    atoms = Atoms("H2", positions=[[0., 0., 0.], [1., 0., 0.]])
    atoms.calc = CalculatorBuilder.from_checkpoint(tmp_path / "run/model.pt", device="cpu",
                                                  properties=["energy"]).build()
    assert atoms.get_potential_energy() == pytest.approx(trainer.model.toy_scale.item())
