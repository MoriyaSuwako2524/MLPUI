# MLPUI

Python interfaces for machine-learning interatomic potentials. UMA, NewtonNet
and TorchMD-Net network implementations are included under `mlpui/models/`.
The UMA implementation remains experimental. Bundled fork revisions, local
changes and third-party licenses are recorded in
[models/VENDORED_MODELS.md](mlpui/models/VENDORED_MODELS.md).

## Install

Use a separate environment and install PyTorch appropriate for your device,
then install MLPUI. Separate NewtonNet/TorchMD-Net installations and source
checkout overlays are no longer needed:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

NewtonNet energy/force heads do not require `les`. A charge head placed before energy,
BEC and legacy full pickles containing LES objects need that optional library
(`python -m pip install -e ".[les]"`). Existing state-dict checkpoints retain their parameter
names and use the bundled implementations. Old fully serialized model files
still require explicit `trusted_checkpoint=True`; their module names are mapped
to bundled classes without importing the external model packages.

TorchMD-Net includes a PyTorch neighbor-search fallback for CPU and CUDA with
no compiler required. This uses quadratic memory/time; for large GPU workloads
compile the included acceleration sources with a C++ compiler and a CUDA toolkit
compatible with your PyTorch installation:

```bash
python scripts/build_model_extensions.py --cuda
```

For CPU-only native kernels omit `--cuda`. Builds are local to this checkout;
rebuild after PyTorch/CUDA upgrades. The loader prefers the native extension
when available. `MLPUI_NEIGHBORS=python` forces the fallback;
`MLPUI_NEIGHBORS=native` requires the extension and fails if it cannot load.
The fallback does not support CUDA graph capture or full-graph compilation.

```python
from mlpui.models.newtonnet.models.newtonnet import NewtonNet
from mlpui.models.torchmdnet.models.model import create_model
from mlpui.models.torchmdnet.extensions.ops import BACKEND
print(BACKEND)  # "python" or "native"
```

The bundled NewtonNet source retains its Regents educational/research/nonprofit
license; TorchMD-Net retains its MIT license. These licenses are shipped with
the packages and are not replaced by MLPUI's own license.

## TorchMD-Net

Loads Lightning checkpoints with `state_dict` and `hyper_parameters`, including
the fork's multiple output heads, their normalization buffers, and legacy
single-head checkpoints. Supported architectures: TensorNet, equivariant
transformer, transformer, and graph network. All weights must match strictly.

```python
from ase.build import molecule
from mlpui.calculator import CalculatorBuilder

atoms = molecule("H2O")
atoms.calc = CalculatorBuilder.from_checkpoint(
    "model.ckpt", family="torchmdnet", device="cpu",
    properties=["energy", "forces"],
).build()
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

`family` is optional when the checkpoint has recognizable weights. Use
`model_config=training_args` (or a YAML path) for a bare state dict. The model's
dtype is preserved unless `dtype=torch.float32` / `torch.float64` is supplied.

Energy scalar heads support energy, free_energy (same potential energy), and
gradient forces. The fork's `charge` head maps to ASE `charges`. A `vec` head
can map to ASE `dipole` **only if it actually outputs a 3-vector**; the current
fork's DipoleMoment head returns a magnitude and is explicitly rejected for
that ASE property. Non-energy primary output heads and ensembles are not
supported by this adapter.

## NewtonNet

Recommended portable checkpoint format:

```python
torch.save({"model_config": model_kwargs,
            "model_state_dict": model.state_dict()}, "newtonnet.pt")
```

`model_kwargs` are the arguments passed to the current fork's `NewtonNet`
constructor, including `cutoff`, `n_features`, `n_basis`, `n_interactions`,
`activation`, `layer_norm`, and the ordered `output_properties` list. Output
head ordering and activation/cutoff cannot safely be guessed from tensor shapes.

```python
atoms.calc = CalculatorBuilder.from_checkpoint(
    "newtonnet.pt", family="newtonnet", device="cpu",
    properties=["energy", "forces"],
).build()
```

Bare weights also accept `model_config="config.yml"`, either constructor settings
or the current training YAML's `model` section. Existing serialized full models
and older Lightning checkpoints containing Python objects require explicit
`trusted_checkpoint=True`; only use it for files you trust. The default reader
uses `torch.load(weights_only=True)` and never silently retries unsafe pickle.

Supported NewtonNet outputs: energy/free_energy, forces (`gradient_force` or
`direct_force`), charges, dipole, and stress in ASE Voigt order. These require
the corresponding trained/output heads in the checkpoint. Missing learned heads
are never created with random weights. Checkpoints with Hessian/BEC heads need
a specialized calculator and are rejected. Historical serialized models that
reference removed classes (e.g. `SumAggregator`) need their original backend
revision for export and an explicit architecture migration; they are not
silently treated as current NewtonNet models.

## Units, cells, and patches

### Optional supervised atomic charges

Both bundled NewtonNet and TensorNet support charge prediction. In the WebUI,
enable **训练原子电荷预测**, set its loss weight, and map the charge label file.
The standard filename is `charges.npy`; grouped QM data defaults to
`qm_charge_{shard}.npy` and can be changed to your actual filename. Dense labels
have shape `[samples, atoms]` or `[samples, atoms, 1]`; ragged labels have shape
`[total_atoms]` or `[total_atoms, 1]`. Charge labels are in elementary-charge units
and are not rescaled by the energy/length conversion settings. This is distinct
from the dataset field `charge`, which represents an input total molecular charge.

The Python/YAML equivalent is `loss_weights: {energy: 1, forces: 1, charges: 1}`.
Selecting `charges` automatically adds the matching output head. NewtonNet
appends it after existing heads, so it is an independent supervised prediction,
not an automatic LES energy correction. Existing custom configurations with
charge before energy keep the original LES behavior and require `les`.

Training, validation and periodic testing record charge MSE. Standalone
evaluation offers **评估原子电荷** and reports charge MAE/RMSE/MSE. Charge-only
training/evaluation is also allowed. ASE inference uses `properties=["charges"]`
and `atoms.get_charges()`. Total-charge conservation is optional, as described below.
Checkpoint loading remains strict: an energy-only checkpoint cannot gain a
trained charge head merely by enabling this option; train from scratch or use
a checkpoint already containing the matching head. Defaults remain energy/forces.

### Optional hard total-charge constraint

Enable **总电荷硬约束** together with charge training/evaluation. Supply a second
NumPy file for the total charge Q of every structure: `charge.npy` in standard
layout or `total_charge_{shard}.npy` in grouped layout (custom names supported).
Its shape is `[samples]` or `[samples, 1]`, in units of e. This is separate from
the atomwise `charges.npy` labels. Missing Q is an error, including for neutral
systems: write an explicit array of zeros when all structures are neutral.

The equivalent model configuration is `charge_constraint: true`. After charge
scaling, the model applies `q_i += (Q - sum(q)) / N` separately to each structure.
This differentiable projection is used during training, validation, evaluation
and inference, including batches of different atom counts. NewtonNet applies it
before any downstream LES energy term when a charge-before-energy configuration
is used. Labels are not silently changed; ensure their sums are consistent with Q.
Conservation holds to floating-point precision, not exact real arithmetic.

The setting is saved in the checkpoint's `model_config`. Loading preserves it
even if a caller's architecture configuration omits the flag; explicitly trying
to disable a saved constraint raises an error. In WebUI evaluation, enable the
option to include the Q-file mapping. ASE callers must supply Q explicitly:

```python
atoms.calc = CalculatorBuilder.from_checkpoint(
    "model.pt", device="cpu", properties=["charges"], charge=1.0,
).build()
charges = atoms.get_charges()  # sum is 1.0 within numerical precision
```

Direct model calls pass one Q per structure as `q=...` to either backend.
Leaving the constraint disabled preserves the existing unconstrained behavior.

ASE expects eV, Angstrom, eV/Angstrom, eV/Angstrom^3, elementary charges, and
e*Angstrom dipoles. MLPUI defaults to those units; it cannot infer training units.
For other units supply `energy_to_ev`, `length_to_angstrom`, and (if needed)
`dipole_to_eangstrom`. Positions and cells are converted into model units;
energy, forces, and stress are converted consistently back to ASE units.

Both backends accept nonperiodic structures and fully periodic nonsingular
cells, subject to the backend's neighbor-list/cutoff constraints. Mixed PBC is
rejected. NewtonNet's current minimum-image implementation should be used with
orthorhombic cells; this interface rejects triclinic periodic cells rather than
returning unreliable neighbor distances. Stress requires a valid periodic cell.

Full and delta fine-tuning use the existing `ModelPatcher`. Patch keys for the
new backends are the native model's `named_parameters()` / `state_dict()` keys.
Clones share a model and must be evaluated **sequentially**, not concurrently.
Weights are restored after every calculation and patches reapplied on subsequent
calls. Set `keep_on_device=False` to offload after inference.

## Basic WebUI

For manual deployment with Slurm, password-based SSH tunnels, and the existing
NewtonNet environment, see the [Chinese cluster deployment guide](docs/cluster-deployment.zh-CN.md)
and [GPU submission script](scripts/slurm/mlpui_web.sbatch).

Launch from the project directory using the environment containing your backends:

```powershell
.\.venv\Scripts\python -m mlpui.web
```

Open <http://127.0.0.1:8675>. The interface uses a task-oriented workflow inspired
by [AI Toolkit](https://github.com/ostris/ai-toolkit): create a training task,
start/stop it, and monitor its progress. It uses Python and bundled static assets;
no Node.js setup or separate frontend build is needed.

The basic UI includes:

- NewtonNet and TorchMD-Net/TensorNet presets, optional checkpoint fine-tuning,
  and an expandable model-configuration editor.
- Server-side `.npy` paths, standard/QM/custom file mappings, train/validation
  groups, gradient-to-force conversion, unit factors, and data validation.
- Epochs, batch size, learning rate, CPU/CUDA, precision, seed, and energy/force
  loss weights. Other training targets remain available through the Python API.
- One active task per workspace, persistent history, batch progress, epoch MSE
  curves, recent logs, configuration inspection, and a model download.

The **Stop** button requests cancellation at the next structure boundary and
saves the current weights. An incomplete epoch is not included in the loss
history. Saved weights can initialize a new task; exact optimizer-state resume
is not implemented. Closing the browser does not stop training. Keep the Python
WebUI process running: shutting it down requests a stop, and a crash interrupts
the worker. Failed/interrupted tasks remain visible with an error message.
Periodic saving is configurable: `save_interval=N` saves every N completed
epochs, and `max_checkpoints=K` keeps the latest K periodic weight files.
The UI defaults are N=10 and K=3; N=0 disables periodic saving. Python/YAML
configs omitted these fields retain the previous default (N=0). `model.pt`
is still saved on successful completion or a graceful stop and does not count
toward K. Writes use a temporary file and atomic replacement; older periodic
files are removed only after the new file is saved successfully.

An optional, separate **test** dataset can be evaluated every `test_interval=M`
epochs (default 1). Test evaluation runs at M, 2M, 3M, ... without updating model
weights, and reports per-property MSE in the history, logs, and chart. An ending
epoch that is not a multiple of M does not trigger an extra test. The existing
validation dataset is still evaluated every epoch. Test fields use the same
file mapping/units as training in the UI; Python/YAML can specify their own.

Each task is stored under `runs/web/<id>/` with `config.json`, `status.json`,
`train.log`, and (after completion or a graceful stop) `model.pt`. Use
`python -m mlpui.web --port 8675 --runs-dir /path/to/runs` to change the location.
Data paths refer to the machine **running the Python server**, not the browser.

For pete, start `python -m mlpui.web` in the appropriate environment on the
machine where training should run, then forward the same port from your laptop:

```text
ssh -N -L 8675:127.0.0.1:8675 pete
```

Open the local URL above. If training runs on a separate compute node, the tunnel
must target that node. The server defaults to loopback; it is a
single-user tool and does not include public hosting, accounts, or a job queue.
On a trusted cluster network, `--host 0.0.0.0` (or `MLPUI_HOST=0.0.0.0`
in the Slurm script) permits forwarding through the login node directly to the
compute node. This exposes an unauthenticated service to reachable cluster peers;
Host/Origin checks are not authentication. Keep the browser on localhost and
use the same port at both ends of the SSH tunnel.
UMA is not included in this UI.

The **数据集** page provides a persistent NPY dataset library:

- Register a server-side directory without moving its files, or upload multiple
  numeric `.npy` files from the browser (20 GiB per file; streamed to disk).
- Inspect sample counts, fields, shapes, dtypes and file sizes; search, rename,
  recheck, archive and restore records. Archiving retains files and existing tasks.
- Standard names are discovered automatically. QM prefixes and `{shard}` groups
  use the same explicit file mappings and unit conversions as training.
- Split an existing dataset using train/validation/test ratios and a random seed,
  or preserve structure order. Zero ratios omit a subset; requested nonzero subsets
  must contain at least one structure. Integer counts use largest remainders.
- Splits create independent numeric NPY copies with `offsets.npy` for variable atom
  counts. Every field follows the same structure indices. Unit conversion and
  gradient sign conversion are applied once; the new specs use scale 1 and forces.
- Choose a record when creating a training or evaluation task. Choosing a generated
  training subset also selects the validation/test siblings from that split.

The catalog and uploads live under `<runs-dir>/datasets/`. Each split stores
`split-<id>/split.json` and `<train|validation|test>_indices.npy`, indexing its
immediate source dataset (groups concatenated in their configured order).
Keep the same `--runs-dir` across cluster jobs to retain the library. Splits require
additional disk space and finish before appearing as usable records. Uploads are
sequential, not resumable across browser reloads; failed validation can be retried
with corrected mappings in the same page. Old tasks retain their data paths.
For correlated trajectories, use ordered splitting or register separate trajectory
groups; random frame splitting alone does not prevent temporal leakage.

For a standalone evaluation, choose **评估已有模型** under **新建任务**.
Supply an existing checkpoint, matching model structure configuration, and an
`.npy` dataset (standard, custom mapping, or prefixed shards). Select energy,
forces, or both, plus device and precision. Evaluation performs one pass without
updating model weights or writing a new model. The task can be stopped and has
its own progress, log, and result page. Old tasks default to the training type.

Each new evaluation generates separate reference-versus-prediction density plots
for every selected property (energy, forces, charges, etc.). The plots show the
identity line, MAE, RMSE, and scalar component count, using all evaluated values
in the configured converted dataset units. Vector properties pool their components.
Figures appear in the result page and are saved as `plots/<property>.png` and
`plots/<property>.svg`, with individual download links. Existing evaluations must
be rerun to generate figures. Update dependencies with `python -m pip install -e .`
after pulling this feature (requires Matplotlib). Python callers can use
`trainer.evaluate(data, plot_dir="evaluation/plots")`; without `plot_dir`, the
Python API retains its metrics-only behavior. Plotting retains reference/prediction
arrays in CPU memory, so memory use grows with the number of evaluated components.

Completed evaluations save `evaluation.json` in the task directory, also
accessible from the result page. It contains MAE, MSE, RMSE and scalar counts:
energy is per structure, forces are pooled over all Cartesian components across
all atoms (including variable-size structures). Units follow the configured
dataset conversion; no energy-offset fitting is applied. Unlike the training
loss, force metrics are component-weighted rather than structure-weighted.
Cancelled or failed evaluations do not publish a complete result.

Periodic files live at `checkpoints/epoch_000010.pt`, etc., and can be downloaded
from task details even while training is active or after a failed/interrupted
run. These contain weights, model config, epoch, and history, not optimizer/RNG
state: use them as initial weights for a new task, not exact training resume.

## Training from NumPy files

`mlpui.training.Trainer` provides one training interface for TorchMD-Net
(including TensorNet) and NewtonNet. UMA training is not included. All training
and validation inputs and labels are numeric `.npy` files; pickled object arrays
are rejected. Model architecture and run settings can be dictionaries or YAML.

For the existing pete dataset layout, use explicit file mappings:

```python
from mlpui.data import NpyShards
from mlpui.training import Trainer, TrainingConfig

files = {
    "z": "full_qm_type.npy",
    "pos": "qm_coord_{shard}.npy",
    "energy": "energy_{shard}.npy",
    "forces": "qm_grad_{shard}.npy",
    # "charges": "qm_esp_charge_{shard}.npy",  # requires a charge head/loss
}
root = "/scratch/moriya/codex/chorismate_mutase/calc1/npys"
train = NpyShards(root, ["w00", "w01"], files=files, gradients=True)
valid = NpyShards(root, ["w02"], files=files, gradients=True)
trainer = Trainer("newtonnet", {
    "cutoff": 5.0, "n_features": 64, "n_basis": 20,
    "n_interactions": 3, "activation": "silu",
    "output_properties": ["energy", "gradient_force"],
}, TrainingConfig(epochs=10, batch_size=4))
history = trainer.fit(train, valid, output_dir="runs/newtonnet")
```

The file mapping is arbitrary: `"pos": "{shard}_qm_coord.npy"` supports a
prefix convention as well. Fields without `{shard}` are shared between groups.
All requested files must exist; groups are never paired by independent glob order.
Pass `checkpoint="model.pt"` to `Trainer` for full-weight fine-tuning with a new
optimizer. `model_config` must describe that checkpoint's architecture. For
TorchMD-Net use `family="torchmdnet"` and its complete `create_model` settings
(the same `hyper_parameters` accepted by the checkpoint loader).

Alternatively run `python -m scripts.train configs/train_newtonnet.yaml` after
adjusting that example's paths, units, and groups. YAML-relative paths resolve
relative to the YAML file. Training and validation groups should be disjoint;
the example holds out an entire window to avoid adjacent-frame leakage.

`NpyDataset(directory)` uses default names `z.npy`, `pos.npy`, `energy.npy`,
`forces.npy`, etc. `NpyDataset(directory, files={...})` selects custom filenames.

| Field | Dense shape | Meaning |
| --- | --- | --- |
| z | (N,) or (S,N) | Integer atomic numbers, 1–118 |
| pos | (S,N,3) | Coordinates |
| energy | (S,) or (S,1) | Total energy per structure |
| forces | (S,N,3) | Forces; use `gradients=True` for energy gradients |
| charges | (S,N) | Atomic charges |
| dipole | (S,3) | Dipole vector; scalar magnitudes are rejected |
| stress | (S,3,3) | Stress tensor, periodic structures only |
| cell, pbc | (S,3,3), (S,3) | Optional cells and periodic flags |
| charge, spin | (S,) | Optional TorchMD-Net system inputs |

S is the number of structures; N is their atom count. For variable atom counts,
use concatenated `pos[A,3]`, `z[A]`, atomic labels `[A,...]`, and an integer
`offsets.npy[S+1]` starting at zero and ending at A. No padding is needed.
Coordinates and cells are multiplied by `length_scale`, energies by
`energy_scale`, forces/gradients by `energy_scale / length_scale`, and stresses
by `energy_scale / length_scale**3`. `gradients=True` additionally negates the
force label. These factors default to 1: **source units are not inferred**.
Use matching factors for train/validation. Charges and dipoles are unchanged.
For ASE inference, configure the calculator's unit conversion to match the
resulting training units. Stress uses ASE's sign convention. Mixed PBC and
NewtonNet triclinic periodic cells are rejected.

Each loss is a component-mean squared error, averaged equally over structures,
then weighted by `TrainingConfig.loss_weights`. A batch accumulates gradients
from sequential structure forwards before one optimizer step; this supports
different cells and atom counts, but is slower than a fused graph batch.
Validation retains autograd for force heads without updating parameters.
`fit` returns per-property train/validation MSE for each epoch. `model.pt`
contains model weights, architecture settings, and history, and loads through
`CalculatorBuilder.from_checkpoint`. It is an inference/fine-tuning export,
not an exact optimizer/RNG resume checkpoint. `fit` also accepts `on_progress`
(batch/validation/epoch event callback) and `should_stop` (a callable checked
between structures). Cancellation raises `TrainingStopped`; callers may then
save the current model. This interface does not train
Hessian/BEC heads or implement early stopping/distributed training.

To enable periodic saves and independent test evaluation in Python:

```python
config = TrainingConfig(save_interval=10, max_checkpoints=3, test_interval=5)
trainer = Trainer("newtonnet", model_config, config)
history = trainer.fit(train, valid, test_data=test, output_dir="runs/experiment")
```

In YAML, put these three settings under `training` and add a `test` dataset
section with the same schema as `train`/`validation`. Periodic saving requires
an `output_dir` in the Python API. Use disjoint structures for all three splits;
the WebUI rejects overlapping coordinate file paths between splits.

## Running tests

```powershell
python -m pytest -q
```

The integration tests run the actual fork implementations: checkpoint round
trips, strict weight loading, per-head normalization, precision, energy/force
agreement, finite-difference forces, charges/dipoles/stress, periodic cells,
and legacy TorchMD tuple outputs. Unit tests additionally cover conversions,
missing outputs, repeated patching, and the safe-loading default. Integration
tests skip if optional backends are not installed; check the reported skip count.
