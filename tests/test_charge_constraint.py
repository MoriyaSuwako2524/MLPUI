import numpy as np
import pytest
import torch

from mlpui.models.charge import constrain_charges
from mlpui.training import Trainer, TrainingConfig
from mlpui.web.config import presets, normalize, inspect_data
from mlpui.calculator import CalculatorBuilder
from mlpui.data import NpyDataset
from mlpui.external_models import load_external_checkpoint
from test_training import write_data


def test_projection_ragged_batches_and_gradients():
    batch = torch.tensor([1, 0, 1, 0, 1])
    charges = torch.tensor([[.2], [-.1], [.3], [.6], [.4]], dtype=torch.float64, requires_grad=True)
    total = torch.tensor([1., -2.], dtype=torch.float64)
    result = constrain_charges(charges, batch, total)
    sums = torch.zeros(2, dtype=torch.float64).index_add(0, batch, result.flatten())
    torch.testing.assert_close(sums, total, atol=1e-14, rtol=0)
    for group in (0, 1):
        mask = batch == group
        expected = charges[mask] + (total[group] - charges[mask].sum()) / int(mask.sum())
        torch.testing.assert_close(result[mask], expected)
    assert torch.autograd.gradcheck(lambda x: constrain_charges(x, batch, total), (charges,))
    single = constrain_charges(torch.tensor([[.7]]), torch.tensor([0]), torch.tensor([-1.]))
    assert single.item() == pytest.approx(-1.)
    for q in (None, [0.], [float('nan'), 0.]):
        with pytest.raises(ValueError):
            constrain_charges(charges, batch, q)


@pytest.mark.parametrize("family", ["newtonnet", "torchmdnet"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constraint_training_save_evaluation_and_ase(tmp_path, family, dtype):
    data = write_data(tmp_path / "data")
    np.save(data / "charges.npy", [[-.8, .4, .4], [-.7, .7, 1.]])
    np.save(data / "charge.npy", [[0.], [1.]])
    config = presets()[family]
    if family == "newtonnet":
        config.update(n_features=8, n_basis=4, n_interactions=1)
    else:
        config.update(embedding_dimension=8, num_rbf=4, num_layers=1)
    config["charge_constraint"] = True
    options = TrainingConfig(epochs=1, dtype=dtype, loss_weights={"energy": 1., "forces": 1., "charges": 1.})
    trainer = Trainer(family, config, options)
    trainer.fit(data, output_dir=tmp_path / "trained")
    assert trainer.evaluate(data)["metrics"]["charges"]["count"] == 6
    checkpoint = tmp_path / "trained/model.pt"
    assert torch.load(checkpoint, weights_only=True)["model_config"]["charge_constraint"] is True
    inputs = dict(z=torch.tensor([1, 1, 1, 1, 1]),
                  pos=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [1., 0., 0.]], dtype=dtype),
                  batch=torch.tensor([0, 0, 1, 1, 1]), q=torch.tensor([1., -1.], dtype=dtype))
    if family == "newtonnet":
        inputs["cell"] = torch.zeros(2, 3, 3, dtype=dtype)
    output = trainer.model(**inputs)
    prediction = output.charge if family == "newtonnet" else output["charge"]
    sums = torch.zeros(2, dtype=dtype).index_add(0, inputs["batch"], prediction.flatten())
    torch.testing.assert_close(sums, inputs["q"], atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=0)
    atoms = NpyDataset.atoms(NpyDataset(data)[1])
    atoms.calc = CalculatorBuilder.from_checkpoint(checkpoint, device="cpu", properties=["charges"], charge=-1).build()
    assert atoms.get_charges().sum() == pytest.approx(-1., abs=2e-6)
    atoms.calc = CalculatorBuilder.from_checkpoint(checkpoint, device="cpu", properties=["charges"]).build()
    with pytest.raises(ValueError, match="explicit Q"):
        atoms.get_charges()
    no_flag = dict(trainer.model_config)
    del no_flag["charge_constraint"]
    inherited = Trainer(family, no_flag, options, checkpoint=checkpoint)
    assert inherited.model.charge_constraint is True
    assert inherited.model_config["charge_constraint"] is True
    with pytest.raises(ValueError, match="silently disabled"):
        load_external_checkpoint(checkpoint, model_config=dict(no_flag, charge_constraint=False))
    missing = dict(NpyDataset(data)[0])
    missing.pop("charge")
    with pytest.raises(ValueError, match="explicit Q"):
        trainer._errors(missing)


def test_constraint_requires_total_charge_file_in_preview(tmp_path):
    write_data(tmp_path)
    np.save(tmp_path / "charges.npy", np.zeros((2, 3)))
    payload = {"name": "hard charge", "family": "newtonnet",
               "model_config": dict(presets()["newtonnet"], charge_constraint=True),
               "training": {"loss_weights": {"charges": 1.}}, "train": {"directory": str(tmp_path)}}
    config = normalize(payload)
    with pytest.raises(ValueError, match="missing labels: charge"):
        inspect_data(config)
    np.save(tmp_path / "charge.npy", [0., 1.])
    assert inspect_data(config)["train"]["samples"] == 2
