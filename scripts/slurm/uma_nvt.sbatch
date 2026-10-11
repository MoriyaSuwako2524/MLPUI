#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=uma-nvt
#SBATCH --chdir=/home/moriya2524/mlp/sources/MLPUI
#SBATCH --output=./logs/uma-md-%j.out
#SBATCH --error=./logs/uma-md-%j.err
#SBATCH --exclude=c302,c301

# Submit from the updated MLPUI project root:
#   mkdir -p logs
#   sbatch submit_uma_md.sbatch
# Override settings via environment variables, e.g.:
#   TEMPERATURE_K=400 DURATION_PS=5 sbatch submit_uma_md.sbatch
# Or specify the number of steps instead of duration:
#   MD_STEPS=1000 sbatch submit_uma_md.sbatch
set -euo pipefail

MLPUI_ROOT="${MLPUI_ROOT:-/home/moriya2524/mlp/sources/MLPUI}"
MD_PYTHON="${MLPUI_PYTHON:-${HOME}/.conda/envs/newtonnet/bin/python}"
XYZ_FILE="${XYZ_FILE:-/scratch/moriya2524/mlp/md/test.xyz}"
UMA_CHECKPOINT="${UMA_CHECKPOINT:-/home/moriya2524/mlp/sources/UMA/UMA/checkpoints/uma-s-1p2.pt}"
ATOM_REFS="${ATOM_REFS:-/home/moriya2524/mlp/sources/UMA/UMA/references/iso_atom_elem_refs.yaml}"

TEMPERATURE_K="${TEMPERATURE_K:-300}"
DURATION_PS="${DURATION_PS:-1}"
TIMESTEP_FS="${TIMESTEP_FS:-0.1}"
FRICTION_PER_PS="${FRICTION_PER_PS:-10}"
MOLECULAR_CHARGE="${MOLECULAR_CHARGE:-0}"
SPIN_MULTIPLICITY="${SPIN_MULTIPLICITY:-1}"
MD_SEED="${MD_SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-100}"
MD_OUTPUT_DIR="${MD_OUTPUT_DIR:-/scratch/moriya2524/mlp/md/runs/uma-${SLURM_JOB_ID:?Submit this script using sbatch}}"

[[ -x "$MD_PYTHON" ]] || { echo "Python not executable: $MD_PYTHON" >&2; exit 1; }
echo "Submitted from: ${SLURM_SUBMIT_DIR:-unset}"
echo "Initial working directory: $PWD"
echo "MLPUI_ROOT: $MLPUI_ROOT"
cd "$MLPUI_ROOT"
for input_file in "$XYZ_FILE" "$UMA_CHECKPOINT" "$ATOM_REFS"; do
    [[ -r "$input_file" ]] || { echo "Input not readable: $input_file" >&2; exit 1; }
done
[[ ! -e "$MD_OUTPUT_DIR" ]] || { echo "Output already exists: $MD_OUTPUT_DIR" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${MLPUI_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "Job: ${SLURM_JOB_ID}; node: ${SLURM_NODELIST:-unknown}"
echo "Project: $MLPUI_ROOT"
echo "Python: $MD_PYTHON"
echo "XYZ: $XYZ_FILE"
echo "Checkpoint: $UMA_CHECKPOINT"
echo "Output: $MD_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$MD_PYTHON" -u - <<'PY'
import torch
import mlpui
from mlpui.calculator import CalculatorBuilder
try:
    import mlpui.scripts.nvt as nvt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Cannot import mlpui.scripts.nvt. Check MLPUI_ROOT and update the MLPUI "
        f"installation in this Python environment. Original error: {exc}"
    ) from exc
print("MLPUI:", mlpui.__file__)
print("MD module:", nvt.__file__)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is available in this allocation/environment.")
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -n "${MD_STEPS:-}" ]]; then
    TIME_ARGS=(--steps "$MD_STEPS")
else
    TIME_ARGS=(--duration-ps "$DURATION_PS")
fi

# Single-GPU, nonperiodic molecular NVT. Molecular elemental offsets are read
# from the checkpoint; do not add form_elem_refs.yaml to the total energy.
exec "$MD_PYTHON" -u -m mlpui.scripts.nvt \
    --xyz "$XYZ_FILE" \
    --checkpoint "$UMA_CHECKPOINT" \
    --atom-refs "$ATOM_REFS" \
    --task omol --device cuda \
    --temperature-k "$TEMPERATURE_K" \
    "${TIME_ARGS[@]}" \
    --timestep-fs "$TIMESTEP_FS" \
    --friction-per-ps "$FRICTION_PER_PS" \
    --charge "$MOLECULAR_CHARGE" --spin "$SPIN_MULTIPLICITY" \
    --seed "$MD_SEED" \
    --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" \
    --output "$MD_OUTPUT_DIR"
