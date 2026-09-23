import importlib
import numpy as np
import pytest
import torch

from mlpui.data import NpyDataset, NpyShards
from mlpui.training import Trainer, TrainingConfig
from mlpui.calculator import CalculatorBuilder
from test_external_models_integration import torchmd_args


def write_data(path):
    path.mkdir(exist_ok=True)
    np.save(path / "z.npy", np.array([8, 1, 1]))
    np.save(path / "pos.npy", np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                                       [[0., 0., 0.], [1.1, 0., 0.], [0., 1.1, 0.]]]))
    np.save(path / "energy.npy", np.array([1., 1.2]))
    np.save(path / "forces.npy", np.ones((2, 3, 3)) * .1)
    return path


def test_evaluation_aggregation_and_stop(tmp_path, monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig()
    trainer.model = torch.nn.Linear(1, 1)
    errors = iter([{"energy": torch.tensor([2.]), "forces": torch.tensor([1., -1., 1.])},
                   {"energy": torch.tensor([-4.]), "forces": torch.tensor([3.] * 6)}])
    monkeypatch.setattr(trainer, "_errors", lambda sample: next(errors))
    result = trainer.evaluate(write_data(tmp_path))
    assert result["metrics"]["energy"] == {"mae": 3., "mse": 10., "rmse": 10.**.5, "count": 2}
    assert result["metrics"]["forces"]["mae"] == pytest.approx(21 / 9)
    assert result["metrics"]["forces"]["mse"] == pytest.approx(57 / 9)
    from mlpui.training import TrainingStopped
    with pytest.raises(TrainingStopped):
        trainer.evaluate(tmp_path, should_stop=lambda: True)


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_evaluation_preserves_model(tmp_path, family):
    model_config = (dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                         output_properties=["energy", "gradient_force"])
                    if family == "newtonnet" else torchmd_args("tensornet"))
    trainer = Trainer(family, model_config, TrainingConfig(dtype=torch.float64))
    original = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    events = []
    result = trainer.evaluate(write_data(tmp_path), on_progress=events.append)
    assert result["samples"] == 2
    assert result["metrics"]["forces"]["count"] == 18
    assert result["metrics"]["energy"]["rmse"] >= 0
    assert events[-1]["completed"] == 2
    assert trainer.history == []
    assert not trainer.optimizer.state
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_dense_ragged_and_validation(tmp_path):
    data = NpyDataset(write_data(tmp_path))
    assert len(data) == 2
    assert data[1]["z"].shape == (3,)
    del data  # Release Windows memory maps before replacing fixtures.
    np.save(tmp_path / "pos.npy", np.zeros((5, 3)))
    np.save(tmp_path / "z.npy", np.ones(5, dtype=int))
    np.save(tmp_path / "offsets.npy", [0, 2, 5])
    np.save(tmp_path / "forces.npy", np.zeros((5, 3)))
    data = NpyDataset(tmp_path)
    assert len(data[0]["z"]) == 2 and len(data[-1]["z"]) == 3
    with pytest.raises(IndexError):
        data[2]
    del data
    np.save(tmp_path / "offsets.npy", [0, 4, 3])
    with pytest.raises(ValueError, match="Ragged"):
        NpyDataset(tmp_path)
    np.save(tmp_path / "offsets.npy", np.array([0, 4, 3, 5], dtype=np.uint64))
    with pytest.raises(ValueError, match="Ragged"):
        NpyDataset(tmp_path)


def test_pete_shard_names_and_gradient_units(tmp_path):
    np.save(tmp_path / "full_qm_type.npy", [6, 1])
    for shard in ("w00", "w01"):
        np.save(tmp_path / f"qm_coord_{shard}.npy", np.ones((3, 2, 3)))
        np.save(tmp_path / f"qm_grad_{shard}.npy", np.full((3, 2, 3), 4.))
        np.save(tmp_path / f"energy_{shard}.npy", np.ones(3))
    files = {"z": "full_qm_type.npy", "pos": "qm_coord_{shard}.npy",
             "forces": "qm_grad_{shard}.npy", "energy": "energy_{shard}.npy"}
    data = NpyShards(tmp_path, ["w00", "w01"], files=files, gradients=True,
                     energy_scale=3, length_scale=2)
    assert len(data) == 6
    np.testing.assert_allclose(data[3]["forces"], -6)
    np.testing.assert_allclose(data[3]["pos"], 2)
    assert data[3]["energy"] == 3
    with pytest.raises(FileNotFoundError):
        NpyShards(tmp_path, ["w02"], files=files)


def test_bad_arrays_and_missing_labels(tmp_path):
    write_data(tmp_path)
    np.save(tmp_path / "z.npy", [1., 1., 1.])
    with pytest.raises(ValueError, match="integer atomic"):
        NpyDataset(tmp_path)
    np.save(tmp_path / "z.npy", np.array([{}, {}, {}], dtype=object))
    with pytest.raises(ValueError):
        NpyDataset(tmp_path)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0},
                                    {"learning_rate": float("nan")},
                                    {"loss_weights": {}}, {"loss_weights": {"forces": -1}}])
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["newtonnet", "graph-network", "transformer", "equivariant-transformer", "tensornet"])
def test_real_force_training_and_checkpoint(tmp_path, family):
    is_newton = family == "newtonnet"
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet" if is_newton else "mlpui.models.torchmdnet.models.model")
    config = (dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                   activation="silu", output_properties=["energy", "gradient_force"])
              if is_newton else torchmd_args(family))
    backend = "newtonnet" if is_newton else "torchmdnet"
    # Forces alone must update parameters through second derivatives.
    trainer = Trainer(backend, config, TrainingConfig(epochs=2, batch_size=2,
                      dtype=torch.float64, loss_weights={"forces": 1.0}))
    before = {k: v.clone() for k, v in trainer.model.named_parameters()}
    data = NpyDataset(write_data(tmp_path / "data"))
    # Check the force-loss parameter gradient against an independent finite
    # difference: merely observing a weight update does not establish correctness.
    trainer.model.train()
    loss, _ = trainer._loss(data[0])
    loss.backward()
    parameter = max((p for p in trainer.model.parameters() if p.grad is not None),
                    key=lambda p: p.grad.abs().max().item())
    flat_index = int(parameter.grad.abs().argmax())
    analytical = parameter.grad.reshape(-1)[flat_index].item()
    original = parameter.detach().reshape(-1)[flat_index].item()
    eps = 1e-5
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original + eps
    plus = trainer._loss(data[0])[0].item()
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original - eps
    minus = trainer._loss(data[0])[0].item()
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original
    assert analytical == pytest.approx((plus - minus) / (2 * eps), rel=1e-3, abs=1e-6)
    history = trainer.fit(data, data, output_dir=tmp_path / "run")
    assert len(history) == 2 and np.isfinite(history[-1]["validation"]["forces"])
    assert any(not torch.equal(before[k], v) for k, v in trainer.model.named_parameters())
    atoms = data.atoms(data[0])
    atoms.calc = CalculatorBuilder.from_checkpoint(tmp_path / "run/model.pt", device="cpu").build()
    assert np.isfinite(atoms.get_forces()).all()
    continued = Trainer(backend, config, TrainingConfig(epochs=1, dtype=torch.float64),
                        checkpoint=tmp_path / "run/model.pt")
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, continued.model.state_dict()[key])
    continued.fit(data)


@pytest.mark.integration
def test_cli(tmp_path):
    import subprocess
    import sys
    import yaml
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    write_data(tmp_path / "data")
    spec = {"family": "newtonnet", "model_config": {
        "cutoff": 3., "n_features": 8, "n_basis": 4, "n_interactions": 1,
        "output_properties": ["energy", "gradient_force"]},
        "training": {"epochs": 1}, "train": {"directory": "data"},
        "output_dir": "run"}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "scripts.train", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run/model.pt").is_file()


@pytest.mark.parametrize("kwargs", [
    {"save_interval": -1}, {"save_interval": 1.5}, {"save_interval": True},
    {"max_checkpoints": 0}, {"max_checkpoints": False}, {"test_interval": 0},
    {"test_interval": 2.5},
])
def test_invalid_save_and_test_intervals(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_periodic_saving_retention_and_test_cadence(tmp_path, family):
    importlib.import_module("mlpui.models.newtonnet.models.newtonnet" if family == "newtonnet" else "mlpui.models.torchmdnet.models.model")
    model_config = (dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                        output_properties=["energy", "gradient_force"])
                    if family == "newtonnet" else torchmd_args("tensornet"))
    data = NpyDataset(write_data(tmp_path / "train"))
    test = NpyDataset(write_data(tmp_path / "test"))
    config = TrainingConfig(epochs=5, batch_size=2, dtype=torch.float64,
                            save_interval=2, max_checkpoints=1, test_interval=2)
    trainer = Trainer(family, model_config, config)
    directory = tmp_path / "run"
    directory.joinpath("checkpoints").mkdir(parents=True)
    unrelated = directory / "checkpoints/notes.pt"
    unrelated.write_bytes(b"keep")
    snapshots = []

    def progress(event):
        if event["phase"] == "epoch_end":
            snapshots.append((event["epoch"], sorted(p.name for p in (directory / "checkpoints").glob("epoch_*.pt"))))

    history = trainer.fit(data, test_data=test, output_dir=directory, on_progress=progress)
    assert snapshots == [(1, []), (2, ["epoch_000002.pt"]), (3, ["epoch_000002.pt"]),
                         (4, ["epoch_000004.pt"]), (5, ["epoch_000004.pt"])]
    assert [r["epoch"] for r in history if "test" in r] == [2, 4]
    assert all(np.isfinite(r["test"]["forces"]) for r in history if "test" in r)
    assert unrelated.read_bytes() == b"keep"
    assert torch.load(directory / "model.pt", weights_only=True)["epoch"] == 5
    saved = torch.load(directory / "checkpoints/epoch_000004.pt", weights_only=True)
    assert saved["epoch"] == 4 and "test" in saved["history"][-1]
    atoms = test.atoms(test[0])
    atoms.calc = CalculatorBuilder.from_checkpoint(directory / "checkpoints/epoch_000004.pt", device="cpu").build()
    assert np.isfinite(atoms.get_forces()).all()
    # Test evaluation must not update weights or interfere with subsequent training.
    baseline = Trainer(family, model_config, TrainingConfig(epochs=5, batch_size=2, dtype=torch.float64))
    baseline.fit(data)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, baseline.model.state_dict()[key])


def test_failed_atomic_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    trainer = Trainer.__new__(Trainer)
    from mlpui.backends import get_backend
    trainer.backend = get_backend("newtonnet")
    trainer.family = trainer.backend.name
    trainer.model = torch.nn.Linear(1, 1)
    trainer.model_config = {}
    trainer.history = []
    trainer.config = TrainingConfig(save_interval=1, max_checkpoints=1)
    trainer._save_periodic(tmp_path, 1)
    previous = (tmp_path / "checkpoints/epoch_000001.pt").read_bytes()

    def failed_save(value, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        trainer._save_periodic(tmp_path, 2)
    assert (tmp_path / "checkpoints/epoch_000001.pt").read_bytes() == previous
    assert not (tmp_path / "checkpoints/epoch_000002.pt").exists()
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))
