# MACE in MLPUI

MLPUI bundles the standard ScaleShiftMACE implementation from the official
MACE 0.3.16 release. It uses the same NPY datasets, weighted losses, task queue,
early stopping, evaluation plots and ASE interface as the other backends.
Install/update the project dependencies with `python -m pip install -e ".[test]"`.
There is no separate `mace-torch` installation, model download or native build.
The pure PyTorch/e3nn implementation uses CPU or CUDA through the usual device
setting. PyTorch >=2.3 (excluding 2.4.1) is required; tests use PyTorch 2.7.1.

## Training and evaluation

Select **MACE** in WebUI. Expand the model configuration and set:

- `atomic_numbers`: a sorted, unique list of all elements in the dataset.
  The preset is H, C, N, O, F, P, S, Cl. Unknown elements cause an explicit error.
- `atomic_energies`: a dictionary with one reference energy for every listed
  atomic number, e.g. `{"1": -13.6, "8": -2040.0}` for a model containing only H/O.
  Use your actual reference energies in the units of the converted training
  labels. These illustrative values are not recommended quantum-chemical data.
  The preset explicitly uses zeros; references are not automatically fitted.
- `r_max`: neighbor cutoff in the converted coordinate units.
- `num_channels`, `max_L`, `max_ell`, `num_interactions`, `correlation`: model
  size/angular settings. Hidden irreps are generated as `C x 0e + C x 1o + ...`
  through `max_L`; the last layer contains scalars. This adapter supports
  `0 <= max_L <= max_ell <= 3` and `1 <= correlation <= 3`.
- `avg_num_neighbors`: message normalization (preset 8); not automatically
  estimated. `atomic_inter_scale` and `atomic_inter_shift` default to 1 and 0.

The complete settings are saved with the checkpoint, including the source
version, reference energies and precision. Energy is total energy per structure;
the shared trainer's force and charge losses average over components per structure.
This is MLPUI's training recipe, not MACE's upstream CLI/EMA/SWA training recipe.

Enable **训练原子电荷预测** and supply `charges.npy` to train E/F/charges together.
The backend creates `predict_charges: true` automatically. Charge-only training
and evaluation also work. Predicted charges come from a separate invariant
scalar MLP sharing MACE features. They are supervised partial charges, not
PolarMACE's self-consistent electrostatic quantities, and do not enter the energy.

Enable **总电荷硬约束** and supply `charge.npy` for explicit total Q per structure
(including zeros for neutral molecules). It uses the existing differentiable
`q_i += (Q - sum(q))/N` projection. Total Q is not otherwise an input to the MACE
energy representation, and there is no explicit long-range electrostatic term.

Energy/force-only models have no charge head. Loading them with charge training
enabled fails strictly; this integration does not silently initialize missing
heads while continuing a checkpoint. Start a new E/F/Charge model instead.

## Python and ASE

```python
import torch
from mlpui.backends import get_backend
from mlpui.training import Trainer, TrainingConfig

model_config = get_backend("mace").default_config()
model_config["atomic_numbers"] = [1, 8]
model_config["atomic_energies"] = {"1": 0.0, "8": 0.0}  # replace as appropriate
model_config["charge_constraint"] = True
trainer = Trainer("mace", model_config, TrainingConfig(
    epochs=100, batch_size=4, dtype=torch.float64,
    loss_weights={"energy": 1.0, "forces": 1.0, "charges": 1.0}))
trainer.fit("data/train", "data/validation", output_dir="runs/mace")
```

```python
from ase.build import molecule
from mlpui.calculator import CalculatorBuilder

atoms = molecule("H2O")
atoms.calc = CalculatorBuilder.from_checkpoint(
    "runs/mace/model.pt", family="mace", device="cpu", charge=0.0,
    properties=["energy", "forces", "charges"],
).build()
energy, forces, charges = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_charges()
```

Model units must agree with ASE or be specified using the existing calculator
conversion factors. Fully periodic orthorhombic and triclinic cells support
forces and stress; gas-phase structures are nonperiodic. Mixed PBC is rejected.
The common trainer supports different atom counts through sequential structures.

## Supported checkpoints and scope

This backend loads MLPUI MACE exports or their matching bare state dictionaries
with `model_config`. `core.*` retains the vendored core's native names under a
wrapper prefix; `charge_head.*` contains the additional learned charge head.
Native upstream `.model`/foundation/PolarMACE models are not automatically
converted or downloaded. Supporting their architecture variants and heads is a
separate migration task. Checkpoints still contain weights and configuration,
not optimizer/RNG state for exact resume.

See [source provenance](../mlpui/models/VENDORED_MODELS.md) and
[backend interface](model-backends.md). Numerical tests cover float32/float64
training, checkpoint round trips, charge conservation, finite-difference forces
and stress, symmetry, periodic images, ASE units and real WebUI worker jobs.
