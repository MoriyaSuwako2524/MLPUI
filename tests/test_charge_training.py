import numpy as np
import pytest
import torch

from mlpui.training import Trainer, TrainingConfig
from mlpui.data import NpyDataset
from mlpui.calculator import CalculatorBuilder
from mlpui.web.config import normalize, inspect_data, presets
from test_training import write_data


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
def test_optional_charge_training_and_checkpoint(tmp_path, family):
    data = write_data(tmp_path / "data")
    np.save(data / "charges.npy", np.array([[[-.8], [.4], [.4]], [[-.7], [.3], [.4]]]))
    model = presets()[family]
    if family == "newtonnet":
        model.update(n_features=8, n_basis=4, n_interactions=1)
    else:
        model.update(embedding_dimension=8, num_rbf=4, num_layers=1)
    disabled = Trainer(family, model)
    if family == "newtonnet":
        assert "charge" not in disabled.model.output_properties
    else:
        assert "charge" not in disabled.model.output_modules
    disabled.save(tmp_path / "no-charge.pt")
    options = TrainingConfig(epochs=2, dtype=torch.float64,
                             loss_weights={"energy": 1., "forces": 1., "charges": 2.})
    trainer = Trainer(family, model, options)
    if family == "newtonnet":
        assert trainer.model.output_properties[-1] == "charge"
        assert trainer.model.aggregators[0].les is None
        charge_parameters = list(trainer.model.output_layers[-1].parameters())
    else:
        charge_parameters = list(trainer.model.output_modules["charge"].parameters())
    before = [p.detach().clone() for p in charge_parameters]
    history = trainer.fit(data, data, test_data=data, output_dir=tmp_path / "trained")
    assert all("charges" in row[split] for row in history for split in ("train", "validation", "test"))
    assert any(not torch.equal(old, p) for old, p in zip(before, charge_parameters))
    result = trainer.evaluate(data)
    assert result["metrics"]["charges"]["count"] == 6
    atoms = NpyDataset.atoms(NpyDataset(data)[0])
    atoms.calc = CalculatorBuilder.from_checkpoint(tmp_path / "trained/model.pt", device="cpu",
                                                  properties=["energy", "forces", "charges"]).build()
    assert atoms.get_charges().shape == (3,)
    assert np.isfinite(atoms.get_charges()).all()
    with pytest.raises(RuntimeError, match="Missing key"):
        Trainer(family, model, options, checkpoint=tmp_path / "no-charge.pt")


def test_charge_preview_and_ragged_shape(tmp_path):
    data = write_data(tmp_path)
    np.save(data / "qm_charge_w00.npy", [[-.8, .4, .4], [-.7, .3, .4]])
    payload = {"name": "charge test", "family": "newtonnet", "model_config": presets()["newtonnet"],
               "training": {"loss_weights": {"charges": 1.}},
               "train": {"directory": str(data), "files": {"z": "z.npy", "pos": "pos.npy",
                          "charges": "qm_charge_w00.npy"}}}
    normalized = normalize(payload)
    assert normalized["model_config"]["output_properties"][-1] == "charge"
    assert inspect_data(normalized)["train"]["samples"] == 2
    np.save(data / "qm_charge_w00.npy", np.ones(2))
    with pytest.raises(ValueError, match="charges.npy shape"):
        inspect_data(normalized)
    ragged = tmp_path / "ragged"
    ragged.mkdir()
    np.save(ragged / "z.npy", [1, 1, 1])
    np.save(ragged / "pos.npy", np.zeros((3, 3)))
    np.save(ragged / "offsets.npy", [0, 1, 3])
    np.save(ragged / "charges.npy", [[.1], [.2], [.3]])
    assert NpyDataset(ragged)[1]["charges"].shape == (2, 1)
