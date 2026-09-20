# MLPUI

Python interfaces for machine-learning interatomic potentials. The existing UMA
implementation is experimental. TorchMD-Net and NewtonNet are optional backends,
using these forks rather than copying their model implementations into MLPUI:

- <https://github.com/MoriyaSuwako2524/torchmd-net>
- <https://github.com/MoriyaSuwako2524/NewtonNet>

## Install

Use a separate virtual environment. Install PyTorch appropriate for your device,
then install MLPUI and the two fork checkouts:

```powershell
python -m pip install -e ".[test]"
python -m pip install --no-build-isolation -e ../torchmd-net
python -m pip install -e ../NewtonNet
```

TorchMD-Net needs its compiled neighbor extension (and a C++/CUDA toolchain for
source builds). NewtonNet's fork requires `les`. Model dependencies load lazily:
using TorchMD-Net does not require importing NewtonNet or the UMA stack.

For a **Windows CPU test environment without a compiler**, the following setup
was tested with Python 3.11. It installs the compatible CPU wheel and overlays the
Python sources from your local forks while retaining the wheel's native extension.
Only use this source overlay in a dedicated test environment; repeat the tests
after changing either fork or the extension version.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install torchmd-net-cpu==2.4.14 ase pytest pyyaml psutil omegaconf "les @ git+https://github.com/ChengUCB/les@35b971a650cf30adb84fef7138761e81aa47b3a2"
.\.venv\Scripts\python scripts/install_local_backends.py --torchmd-root ../torchmd-net --newtonnet-root ../NewtonNet
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest -q
```

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
