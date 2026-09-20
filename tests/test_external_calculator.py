import numpy as np
import pytest
import torch
from torch import nn
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from mlpui.calculator import CalculatorBuilder, SimpleOutputAdapter
from mlpui.external_calculator import AtomicInputAdapter, AtomicOutputAdapter
from mlpui.external_models import detect_family, load_external_checkpoint
from mlpui.model_patcher import ModelPatcher


class HarmonicModel(nn.Module):
    """Analytical reference for force signs, units and repeated patching."""
    mlpui_family = "torchmdnet"
    derivative = True

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, z, pos, batch, box=None, q=None, s=None):
        pos.requires_grad_(True)
        energy = self.scale * pos.square().sum()
        forces = -torch.autograd.grad(energy, pos)[0]
        return {"y": energy, "neg_dy": forces}


def test_ase_units_and_gradient_with_energy_only_request():
    atoms = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
    patcher = ModelPatcher(HarmonicModel())
    atoms.calc = CalculatorBuilder(patcher, energy_to_ev=3, length_to_angstrom=2).build()
    assert atoms.get_potential_energy() == pytest.approx(3.375)
    np.testing.assert_allclose(atoms.get_forces()[1], [-4.5, 0, 0])
    atoms.calc = CalculatorBuilder(patcher, properties=["energy"]).build()
    assert atoms.get_potential_energy() == pytest.approx(4.5)


def test_repeated_patched_calculation_restores_base_and_reapplies():
    model = HarmonicModel()
    base = ModelPatcher(model)
    atoms = Atoms("H", positions=[[1, 0, 0]])
    atoms.calc = CalculatorBuilder.from_patcher_with_patches(base, {"scale": torch.tensor(5.0)}).build()
    assert atoms.get_potential_energy() == pytest.approx(5)
    assert model.scale.item() == pytest.approx(2)
    atoms.positions[0, 0] = 2
    assert atoms.get_potential_energy() == pytest.approx(20)
    np.testing.assert_allclose(atoms.get_forces(), [[-20, 0, 0]])
    assert model.scale.item() == pytest.approx(2)


def test_tuple_force_regression():
    atoms = Atoms("H")
    results = SimpleOutputAdapter(["energy", "forces"]).convert(
        (torch.tensor([1.]), torch.tensor([[1., 2., 3.]])), atoms)
    np.testing.assert_array_equal(results["forces"], [[1, 2, 3]])


def test_output_shapes_and_missing_heads():
    atoms = Atoms("H")
    adapter = AtomicOutputAdapter(["charges", "dipole"])
    result = adapter.convert({"charge": torch.tensor([[0.5]]), "vec": torch.ones(1, 3)}, atoms)
    assert result["charges"].shape == (1,)
    assert result["dipole"].shape == (3,)
    with pytest.raises(PropertyNotImplementedError, match="magnitude"):
        adapter.convert({"charge": torch.zeros(1), "vec": torch.ones(1)}, atoms)
    with pytest.raises(PropertyNotImplementedError, match="did not produce"):
        AtomicOutputAdapter(["forces"]).convert({"energy": torch.ones(1)}, atoms)


@pytest.mark.parametrize("family", ["torchmdnet", "newtonnet"])
def test_periodic_cells(family):
    adapter = AtomicInputAdapter(family)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    inputs = adapter.convert(atoms, "cpu", torch.float64)
    key = "cell" if family == "newtonnet" else "box"
    np.testing.assert_allclose(inputs[key].reshape(3, 3), np.eye(3) * 10)
    atoms.pbc = [True, False, True]
    with pytest.raises(ValueError, match="mixed PBC"):
        adapter.convert(atoms, "cpu", torch.float64)
    atoms.pbc = True
    atoms.set_cell([0, 0, 0])
    with pytest.raises(ValueError, match="nonsingular"):
        adapter.convert(atoms, "cpu", torch.float64)


def test_stress_conversion():
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    output = AtomicOutputAdapter(["stress"], energy_to_ev=16, length_to_angstrom=2)
    stress = torch.tensor([[[1., 6., 5.], [6., 2., 4.], [5., 4., 3.]]])
    np.testing.assert_allclose(output.convert({"stress": stress}, atoms)["stress"], [2, 4, 6, 8, 10, 12])


def test_newton_rejects_unsupported_triclinic_cell():
    atoms = Atoms("H", cell=[[10, 0, 0], [1, 10, 0], [0, 0, 10]], pbc=True)
    with pytest.raises(ValueError, match="triclinic"):
        AtomicInputAdapter("newtonnet").convert(atoms, "cpu", torch.float64)


def test_detection_and_explicit_family_mismatch(tmp_path):
    state = {"model.representation_model.embedding.weight": torch.ones(2, 2)}
    assert detect_family(state) == "torchmdnet"
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": state}, path)
    with pytest.raises(ValueError, match="not requested family"):
        load_external_checkpoint(path, family="newtonnet")


def test_no_unsafe_pickle_fallback(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(HarmonicModel(), path)
    with pytest.raises(ValueError, match="weights-only"):
        load_external_checkpoint(path, family="newtonnet")


def test_validation_and_offload():
    model = HarmonicModel()
    patcher = ModelPatcher(model)
    with pytest.raises(ValueError, match="positive"):
        CalculatorBuilder(patcher, length_to_angstrom=0).build()
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        CalculatorBuilder(patcher, properties=["stress"]).build()
    atoms = Atoms("H", positions=[[1, 0, 0]])
    calc = CalculatorBuilder(patcher, keep_on_device=False).build()
    atoms.calc = calc
    atoms.get_forces()
    assert not calc._ready
    assert next(model.parameters()).device.type == "cpu"
