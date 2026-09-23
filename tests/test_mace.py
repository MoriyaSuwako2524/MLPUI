"""Real bundled MACE tests: learning, physics, serialization and WebUI."""
import copy
import json
import threading
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from mlpui.backends import get_backend
from mlpui.calculator import CalculatorBuilder
from mlpui.external_models import load_external_checkpoint
from mlpui.model_patcher import ModelPatcher
from mlpui.training import Trainer, TrainingConfig
from mlpui.web.config import presets, normalize
from mlpui.web.server import make_server
from test_training import write_data
from test_web import wait_job


def config(**kwargs):
    settings = presets()["mace"]
    settings.update(atomic_numbers=[1, 8], atomic_energies={"1": -.1, "8": -.2},
                    num_channels=4, max_ell=1, max_L=1, num_interactions=2,
                    correlation=2, radial_MLP=[8], readout_channels=4,
                    charge_hidden_channels=4)
    settings.update(kwargs)
    return settings


def water():
    return Atoms("OH2", positions=[[.1, .2, .3], [1.05, .2, .3], [-.15, 1.13, .3]])


def predict(model, atoms, properties=("energy", "forces", "charges"), charge=0.):
    backend = get_backend("mace")
    return backend.forward(model, backend.prepare_inputs(atoms, "cpu", next(model.parameters()).dtype,
        model=model, charge=charge, require_charge=model.charge_constraint), properties)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_training_checkpoint_continuation_and_ase(tmp_path, dtype):
    data = write_data(tmp_path / "data")
    np.save(data / "charges.npy", [[-.8, .4, .4], [-.7, .7, 1.]])
    np.save(data / "charge.npy", [0., 1.])
    default_dtype = torch.get_default_dtype()
    options = TrainingConfig(epochs=2, dtype=dtype, loss_weights={"energy": 1., "forces": 1., "charges": 2.})
    trainer = Trainer("mace", config(charge_constraint=True), options)
    assert torch.get_default_dtype() == default_dtype
    before = [p.detach().clone() for p in trainer.model.charge_head.parameters()]
    history = trainer.fit(data, data, test_data=data, output_dir=tmp_path / "run")
    assert all(set(row[s]) == {"energy", "forces", "charges"}
               for row in history for s in ("train", "validation", "test"))
    assert any(not torch.equal(p, old) for p, old in zip(trainer.model.charge_head.parameters(), before))
    assert trainer.evaluate(data)["metrics"]["charges"]["count"] == 6
    path = tmp_path / "run/model.pt"
    resumed = Trainer("mace", trainer.model_config, options, checkpoint=path)
    assert resumed.history == [] and not resumed.optimizer.state
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    atoms = water()
    reference = predict(trainer.model, atoms, charge=1.)
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu", charge=1.,
        properties=["energy", "forces", "charges"]).build()
    np.testing.assert_allclose(atoms.get_forces(), reference["forces"].detach(), rtol=2e-5, atol=2e-6)
    assert atoms.get_potential_energy() == pytest.approx(reference["energy"].item(), abs=2e-6)
    assert atoms.get_charges().sum() == pytest.approx(1., abs=2e-6)
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu", properties=["charges"]).build()
    with pytest.raises(ValueError, match="explicit Q"):
        atoms.get_charges()
    bad = copy.deepcopy(trainer.model_config)
    bad["charge_constraint"] = False
    with pytest.raises(ValueError, match="cannot be silently disabled"):
        load_external_checkpoint(path, model_config=bad)


def test_mace_force_finite_difference_and_symmetries():
    model = get_backend("mace").build_model(config(precision=64, predict_charges=True))
    model.eval()
    atoms = water()
    result = predict(model, atoms)
    h = 1e-5
    energies = []
    for delta in (h, -h):
        displaced = atoms.copy()
        displaced.positions[1, 0] += delta
        energies.append(predict(model, displaced, ("energy",))["energy"].item())
    assert result["forces"][1, 0].item() == pytest.approx(-(energies[0]-energies[1])/(2*h), rel=1e-5, abs=1e-8)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    transformed = atoms.copy()
    transformed.positions = atoms.positions @ rotation.T + [2., -3., 4.]
    rotated = predict(model, transformed)
    torch.testing.assert_close(rotated["energy"], result["energy"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rotated["charges"], result["charges"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rotated["forces"].detach(), result["forces"].detach().numpy() @ rotation.T, atol=1e-10)
    order = [2, 0, 1]
    permuted = predict(model, atoms[order])
    torch.testing.assert_close(permuted["charges"], result["charges"][order], rtol=1e-10, atol=1e-10)
    # The independent charge head does not contribute to energy or force.
    with torch.no_grad():
        for param in model.charge_head.parameters():
            param.add_(1.)
    changed = predict(model, atoms)
    torch.testing.assert_close(changed["energy"], result["energy"], rtol=0, atol=0)
    torch.testing.assert_close(changed["forces"], result["forces"], rtol=0, atol=0)
    assert not torch.equal(changed["charges"], result["charges"])


def test_mace_periodic_stress_units_and_isolated_atom():
    model = get_backend("mace").build_model(config(precision=64))
    model.eval()
    atoms = water()
    atoms.set_cell([[7., 0., 0.], [.5, 7., 0.], [.2, .3, 7.]])
    atoms.pbc = True
    result = predict(model, atoms, ("energy", "forces", "stress"))
    h = 1e-5
    energies = []
    for delta in (h, -h):
        strained = atoms.copy()
        transform = np.diag([1+delta, 1., 1.])
        strained.set_cell(atoms.cell.array @ transform, scale_atoms=True)
        energies.append(predict(model, strained, ("energy",))["energy"].item())
    assert result["stress"][0, 0, 0].item() == pytest.approx((energies[0]-energies[1])/(2*h)/atoms.get_volume(), abs=1e-9)
    isolated = predict(model, Atoms("H"), ("energy", "forces"))
    assert torch.isfinite(isolated["energy"]).all()
    torch.testing.assert_close(isolated["forces"], torch.zeros(1, 3, dtype=torch.float64))
    wrapped = Atoms("H2", positions=[[.1, 0., 0.], [11.1, 0., 0.]], cell=[12., 12., 12.], pbc=True)
    equivalent = Atoms("H2", positions=[[.1, 0., 0.], [-.9, 0., 0.]])
    periodic_pair = predict(model, wrapped, ("energy", "forces"))
    gas_pair = predict(model, equivalent, ("energy", "forces"))
    for key in ("energy", "forces"):
        torch.testing.assert_close(periodic_pair[key], gas_pair[key], rtol=1e-10, atol=1e-10)
    mol = water()
    unscaled = predict(model, mol, ("energy", "forces"))
    mol.positions *= 2
    mol.calc = CalculatorBuilder(ModelPatcher(model), device="cpu", energy_to_ev=3., length_to_angstrom=2.).build()
    assert mol.get_potential_energy() == pytest.approx(unscaled["energy"].item()*3)
    np.testing.assert_allclose(mol.get_forces(), unscaled["forces"].detach().numpy()*1.5, atol=1e-12)


def test_mace_ragged_batch_charge_projection():
    backend = get_backend("mace")
    model = backend.build_model(config(precision=64, predict_charges=True, charge_constraint=True))
    model.train()
    graphs = [backend.prepare_inputs(atoms, "cpu", torch.float64, model=model)["data"]
              for atoms in (water(), Atoms("H2", positions=[[0., 0., 0.], [1., 0., 0.]]))]
    batch = {}
    for key in ("positions", "node_attrs", "shifts", "unit_shifts", "cell", "head"):
        batch[key] = torch.cat([g[key] for g in graphs])
    batch["batch"] = torch.tensor([0, 0, 0, 1, 1])
    batch["ptr"] = torch.tensor([0, 3, 5])
    batch["edge_index"] = torch.cat([graphs[0]["edge_index"], graphs[1]["edge_index"]+3], dim=1)
    out = model(batch, q=torch.tensor([1., -1.], dtype=torch.float64))
    torch.testing.assert_close(out["charges"][:3].sum(), torch.tensor(1., dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(out["charges"][3:].sum(), torch.tensor(-1., dtype=torch.float64), rtol=0, atol=1e-12)
    (out["charges"].square().mean()+out["forces"].square().mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.charge_head.parameters())


def test_mace_strict_heads_elements_and_configuration(tmp_path):
    trainer = Trainer("mace", config(), TrainingConfig(loss_weights={"energy": 1.}))
    path = trainer.save(tmp_path / "energy.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        Trainer("mace", config(), TrainingConfig(loss_weights={"charges": 1.}), checkpoint=path)
    with pytest.raises(PropertyNotImplementedError, match="charges"):
        CalculatorBuilder.from_checkpoint(path, properties=["charges"], device="cpu").build()
    with pytest.raises(ValueError, match="outside model atomic_numbers"):
        predict(trainer.model, Atoms("He"), ("energy",))
    for changes in (dict(atomic_numbers=[8, 1]), dict(atomic_energies={"1": 0.}),
                    dict(r_max=-1), dict(max_L=2), dict(source_version="1.0"),
                    dict(predict_charges="yes"), dict(correlation=4)):
        with pytest.raises(ValueError):
            get_backend("mace").build_model(config(**changes))
    with pytest.raises(ValueError, match="mixed PBC"):
        predict(trainer.model, Atoms("H", cell=[5, 5, 5], pbc=[True, False, False]), ("energy",))
    # Energy-only gradients remain valid, without requesting force computation.
    loss, _ = trainer._loss({"z": np.array([8, 1, 1]), "pos": water().positions, "energy": np.array(0.)})
    loss.backward()


def test_mace_charge_only_training(tmp_path):
    data = write_data(tmp_path)
    np.save(data / "charges.npy", [[-.8, .4, .4], [-.7, .3, .4]])
    trainer = Trainer("mace", config(), TrainingConfig(epochs=1, loss_weights={"charges": 1.}))
    trainer.fit(data)
    assert trainer.evaluate(data)["metrics"]["charges"]["count"] == 6


def test_mace_web_training_evaluation_and_download(tmp_path):
    data = write_data(tmp_path / "data")
    np.save(data / "charges.npy", [[-.8, .4, .4], [-.7, .3, .4]])
    np.save(data / "charge.npy", [0., 0.])
    server = make_server(tmp_path / "runs", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/presets") as response:
            catalog = json.load(response)
        assert "mace" in catalog["models"] and "charges" in catalog["backends"]["mace"]["training_targets"]
        payload = dict(name="MACE E F Q", family="mace", model_config=config(charge_constraint=True),
            training=dict(epochs=1, loss_weights={"energy": 1., "forces": 1., "charges": 1.}),
            train=dict(directory=str(data)))
        job = server.manager.start(normalize(payload))
        state = wait_job(server.manager, job["id"])
        assert state["status"] == "completed", state
        with urlopen(base + f'/api/jobs/{job["id"]}/model') as response:
            assert len(response.read()) > 1000
        evaluation = dict(name="MACE evaluation", task_type="evaluation", family="mace",
            model_config=config(charge_constraint=True), checkpoint=state["checkpoint"],
            training=dict(loss_weights={"energy": 1., "forces": 1., "charges": 1.}),
            evaluation=dict(directory=str(data)))
        job = server.manager.start(normalize(evaluation))
        result = wait_job(server.manager, job["id"])
        assert result["status"] == "completed", result
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        server.lease.close()
