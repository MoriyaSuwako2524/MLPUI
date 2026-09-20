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
    pytest.importorskip("newtonnet.models.newtonnet" if is_newton else "torchmdnet.models.model")
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
    pytest.importorskip("newtonnet.models.newtonnet")
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
