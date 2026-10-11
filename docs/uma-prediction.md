# UMA prediction

Choose **New job > Job type > UMA prediction**. Select a local UMA inference
checkpoint (`.pt`) on the WebUI server, then choose an NPY dataset. The job uses
the existing CPU/GPU queue and produces energy and forces without reference labels.

## Runtime

The bundled UMA code currently contains the backbone, but not a complete predictor
with output heads and reference-energy normalization. Web prediction therefore uses
FAIR-Chem's official `load_predict_unit` and `FAIRChemCalculator` interfaces.
It loads the file you specify; it does not download model weights automatically.

FAIR-Chem uses a newer e3nn than the legacy bundled models. Keep it in a separate
Python environment rather than replacing the training environment's dependencies:

```bash
conda create -n mlpui-uma python=3.12
conda run -n mlpui-uma python -m pip install fairchem-core psutil
export MLPUI_UMA_PYTHON="$HOME/.conda/envs/mlpui-uma/bin/python"
# Start your normal WebUI/Slurm script after exporting this variable.
sbatch scripts/slurm/mlpui_web.sbatch
```

Adjust the absolute path to the environment on your cluster. Install a FAIR-Chem
release compatible with your checkpoint and cluster CUDA/PyTorch stack. The WebUI
passes the project directory to the worker through PYTHONPATH, so the separate
environment does not need an editable MLPUI install. Training jobs keep using the
normal WebUI Python. If the variable is unset, prediction uses the WebUI Python and
reports a clear error if FAIR-Chem is missing. Worker errors are retained in the log.

## Inputs and settings

- Required arrays: `z.npy` and `pos.npy`; custom mappings and `{shard}` groups also work.
- Optional arrays: `cell.npy`, `pbc.npy`, `charge.npy`, `spin.npy` and `offsets.npy`.
- For variable atom counts, use flattened positions/atomic numbers and offsets.
- Choose a task supported by the checkpoint: `omol`, `omat`, `odac`, `oc20`, `oc25`, or `omc`.
- Default settings are `{"task":"omol","charge":0,"spin":1}`. Spin means multiplicity;
  an OMOL singlet uses 1. Charge is total structure charge, not atomic charges.
- Mapped per-structure charge/spin arrays override the defaults. Leave those file
  mappings blank to use the defaults. Charge must be an integer in [-100,100];
  spin must be an integer in [0,100].
- Positions and cell must convert to angstrom using **Length scale**. Model outputs
  are always in eV and eV/angstrom; the label energy scale does not rescale predictions.
- Prediction currently uses float32 and computes both energy and forces. It does
  not train UMA, predict atomic charges, or calculate labeled evaluation metrics.

## Outputs

The completed job offers downloads for:

- `energy.npy`: one energy per structure, shape `[S]`, in eV.
- `forces.npy`: concatenated atom forces, shape `[A,3]`, in eV/angstrom.
- `offsets.npy`: integer boundaries, shape `[S+1]`.
- `prediction.json`: layout, units and counts.

For structure `i`, take `forces[offsets[i]:offsets[i+1]]`. Structures retain input
order, including group order. A cancelled or failed job does not expose results as
complete. Prediction datasets participate in managed-file reference protection.

Official references: [FAIR-Chem inference interface](https://facebookresearch.github.io/fairchem/ase-calculator/)
and [local checkpoint loading](https://facebookresearch.github.io/fairchem/fine-tuning/).
