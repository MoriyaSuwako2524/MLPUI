import importlib
"""Real backend round trips; install both forks before running this module."""
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from mlpui.calculator import CalculatorBuilder
from mlpui.external_calculator import AtomicInputAdapter
from mlpui.external_models import load_external_checkpoint

pytestmark = pytest.mark.integration


def torchmd_args(architecture, precision=64):
    return dict(
        model=architecture, precision=precision, embedding_dimension=16,
        num_layers=1, num_rbf=8, rbf_type="expnorm", trainable_rbf=False,
        activation="silu", cutoff_lower=0., cutoff_upper=3., max_z=100,
        max_num_neighbors=16, aggr="add", neighbor_embedding=True,
        attn_activation="silu", num_heads=4, distance_influence="both",
        equivariance_invariance_group="O(3)", derivative=True, atom_filter=-1,
        prior_model=None, output_model="Scalar", reduce_op="sum",
        pred_dict={"y": 1.0, "neg_dy": 1.0, "charge": 1.0},
    )


def water(periodic=False):
    return Atoms("OH2", positions=[[0.1, 0.2, 0.3], [1.05, 0.2, 0.3], [-0.15, 1.13, 0.3]],
                 cell=[10, 10, 10], pbc=periodic)


@pytest.mark.parametrize("architecture", ["graph-network", "transformer", "equivariant-transformer", "tensornet"])
@pytest.mark.parametrize("periodic", [False, True])
def test_torchmd_real_roundtrip(tmp_path, architecture, periodic):
    backend = importlib.import_module("mlpui.models.torchmdnet.models.model")
    torch.manual_seed(12)
    args = torchmd_args(architecture)
    model = backend.create_model(dict(args))
    model.eval()
    for name in ("y", "charge"):
        model.register_buffer(f"mean_{name}", torch.tensor(0.2, dtype=torch.float64))
        model.register_buffer(f"std_{name}", torch.tensor(1.3, dtype=torch.float64))
    path = tmp_path / "torchmd.ckpt"
    torch.save({"hyper_parameters": args,
                "state_dict": {"model." + k: v for k, v in model.state_dict().items()}}, path)
    atoms = water(periodic)
    inputs = AtomicInputAdapter("torchmdnet").convert(atoms, "cpu", torch.float64)
    expected = model(**inputs)
    atoms.calc = CalculatorBuilder.from_checkpoint(
        path, properties=["energy", "forces", "charges"], device="cpu").build()
    assert atoms.calc._dtype == torch.float64
    np.testing.assert_allclose(atoms.get_potential_energy(), expected["y"].detach().numpy().item(), rtol=1e-10)
    np.testing.assert_allclose(atoms.get_forces(), expected["neg_dy"].detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(atoms.get_charges(), expected["charge"].detach().numpy().reshape(-1), atol=1e-10)
    # Independent numerical derivative checks the ASE force sign and input path.
    force = atoms.get_forces()[1, 0]
    eps = 1e-5
    atoms.positions[1, 0] += eps
    e_plus = atoms.get_potential_energy()
    atoms.positions[1, 0] -= 2 * eps
    e_minus = atoms.get_potential_energy()
    assert force == pytest.approx(-(e_plus - e_minus) / (2 * eps), abs=1e-6)


def test_torchmd_legacy_tuple_checkpoint(tmp_path):
    backend = importlib.import_module("mlpui.models.torchmdnet.models.model")
    args = torchmd_args("graph-network", precision=32)
    created = backend.create_model(dict(args))
    model = backend.TorchMD_Net(created.representation_model, created.output_modules["y"], derivative=True)
    model.eval()
    path = tmp_path / "legacy.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": {"model." + k: v for k, v in model.state_dict().items()}}, path)
    atoms = water()
    expected = model(**AtomicInputAdapter("torchmdnet").convert(atoms, "cpu", torch.float32))
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu").build()
    np.testing.assert_allclose(atoms.get_forces(), expected[1].detach().numpy(), atol=1e-6)


def test_torchmd_rejects_missing_learned_head(tmp_path):
    backend = importlib.import_module("mlpui.models.torchmdnet.models.model")
    args = torchmd_args("tensornet")
    model = backend.create_model(dict(args))
    state = model.state_dict()
    key = next(k for k in state if k.startswith("output_modules.") and k.endswith("weight"))
    del state[key]
    path = tmp_path / "incomplete.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_external_checkpoint(path)


def test_torchmd_per_head_only_scalers(tmp_path):
    backend = importlib.import_module("mlpui.models.torchmdnet.models.model")
    args = torchmd_args("tensornet")
    model = backend.create_model(dict(args))
    state = model.state_dict()
    del state["mean"], state["std"]
    for name in ("y", "charge"):
        state[f"mean_{name}"] = torch.tensor(0.3, dtype=torch.float64)
        state[f"std_{name}"] = torch.tensor(2., dtype=torch.float64)
    path = tmp_path / "normalized.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": state}, path)
    atoms = water()
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu").build()
    assert np.isfinite(atoms.get_potential_energy())


def test_ase_pbc_overrides_training_box(tmp_path):
    backend = importlib.import_module("mlpui.models.torchmdnet.models.model")
    args = torchmd_args("tensornet")
    args["box_vecs"] = [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]
    model = backend.create_model(dict(args))
    path = tmp_path / "periodic.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": model.state_dict()}, path)
    atoms = water(False)
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu").build()
    atoms.get_potential_energy()
    distances = [m for m in atoms.calc._get_backbone().modules() if type(m).__name__ == "OptimizedDistance"]
    assert distances and all(not m.use_periodic for m in distances)
    atoms.pbc = True
    atoms.get_potential_energy()
    assert all(m.use_periodic for m in distances)


@pytest.mark.parametrize("periodic", [False, True])
def test_newton_real_roundtrip(tmp_path, periodic):
    pytest.importorskip("les")  # This case includes the charge-dependent LES energy.
    backend = importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    torch.manual_seed(3)
    config = dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1, activation="silu",
                  output_properties=["energy", "gradient_force", "charge", "dipole"] + (["stress"] if periodic else []))
    model = backend.NewtonNet(**config).double()
    model.eval()
    path = tmp_path / "newton.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)
    atoms = water(periodic)
    expected = model(**AtomicInputAdapter("newtonnet").convert(atoms, "cpu", torch.float64))
    properties = ["energy", "forces", "charges", "dipole"] + (["stress"] if periodic else [])
    atoms.calc = CalculatorBuilder.from_checkpoint(path, device="cpu", properties=properties).build()
    assert atoms.calc._dtype == torch.float64
    np.testing.assert_allclose(atoms.get_potential_energy(), expected.energy.detach().numpy().item(), atol=1e-10)
    np.testing.assert_allclose(atoms.get_forces(), expected.gradient_force.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(atoms.get_charges(), expected.charge.detach().numpy().reshape(-1), atol=1e-10)
    np.testing.assert_allclose(atoms.get_dipole_moment(), expected.dipole.detach().numpy().reshape(3), atol=1e-10)
    if periodic:
        np.testing.assert_allclose(atoms.get_stress(voigt=False), expected.stress.detach().numpy().reshape(3, 3), atol=1e-10)
    force = atoms.get_forces()[1, 0]
    atoms.positions[1, 0] += 1e-5
    plus = atoms.get_potential_energy()
    atoms.positions[1, 0] -= 2e-5
    minus = atoms.get_potential_energy()
    assert force == pytest.approx(-(plus - minus) / 2e-5, abs=1e-6)


def test_newton_trusted_serialized_model_and_yaml(tmp_path):
    backend = importlib.import_module("mlpui.models.newtonnet.models.newtonnet")
    import yaml
    config = dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                  output_properties=["energy", "gradient_force"])
    model = backend.NewtonNet(**config)
    path = tmp_path / "full_model.pt"
    torch.save(model, path)
    with pytest.raises(ValueError, match="weights-only"):
        load_external_checkpoint(path, family="newtonnet")
    atoms = water()
    atoms.calc = CalculatorBuilder.from_checkpoint(path, family="newtonnet", trusted_checkpoint=True, device="cpu").build()
    e = atoms.get_potential_energy()
    weights = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"model": config}), encoding="utf-8")
    atoms.calc = CalculatorBuilder.from_checkpoint(weights, model_config=config_path, device="cpu").build()
    assert atoms.get_potential_energy() == pytest.approx(e)
    with pytest.raises(ValueError, match="model_config"):
        load_external_checkpoint(weights)
