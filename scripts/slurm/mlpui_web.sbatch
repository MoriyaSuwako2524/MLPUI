#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/%j_stdout.txt
#SBATCH --error=./logs/%j_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mlpui-web
#SBATCH --exclude=c302,c301

# Submit from the MLPUI project root, AFTER running: mkdir -p logs
# Optional overrides: MLPUI_ROOT, MLPUI_PYTHON, MLPUI_PORT, MLPUI_RUNS_DIR,
#                     MLPUI_ID_SCRIPT (your existing site-specific id.py),
#                     MLPUI_LOGIN_HOST, MLPUI_LOGIN_USER (reverse SSH tunnel).
set -euo pipefail

# Use an already prepared GPU environment, just like the working ComfyUI job.
# Set MLPUI_PYTHON if NewtonNet/MLPUI are installed at a different location.
PYTHON="${MLPUI_PYTHON:-${HOME}/.conda/envs/newtonnet/bin/python}"
[[ -x "$PYTHON" ]] || {
    echo "Python not executable: $PYTHON" >&2
    echo "Set MLPUI_PYTHON to the absolute Python path of your MLPUI/NewtonNet environment." >&2
    exit 1
}

MLPUI_ROOT="${MLPUI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
Port="${MLPUI_PORT:-2526}"
RUNS_DIR="${MLPUI_RUNS_DIR:-${MLPUI_ROOT}/runs/cluster-web}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd "$MLPUI_ROOT"
mkdir -p "$RUNS_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "This job is running on node(s): ${SLURM_NODELIST}"
echo "Hostname: $(hostname -f)"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
echo "Project: $MLPUI_ROOT"
echo "Results: $RUNS_DIR"
echo "WebUI: http://127.0.0.1:${Port} (on the compute node)"

# Fail early if the existing environment cannot use the allocated GPU.
"$PYTHON" -u - <<'PY'
import torch
from mlpui.web.server import make_server
from newtonnet.models.newtonnet import NewtonNet
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device is available. Check the allocation and PyTorch environment.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

# Opt-in only: id.py is not part of MLPUI and its behavior depends on your cluster.
# Set MLPUI_ID_SCRIPT to its absolute path if your existing connection workflow needs it.
if [[ -n "${MLPUI_ID_SCRIPT:-}" ]]; then
    [[ -f "$MLPUI_ID_SCRIPT" ]] || { echo "id.py not found: $MLPUI_ID_SCRIPT" >&2; exit 1; }
    "$PYTHON" "$MLPUI_ID_SCRIPT" "$SLURM_JOB_ID" "$Port"
fi

TUNNEL_DIR=""
WEB_PID=""
cleanup() {
    if [[ -n "$WEB_PID" ]]; then
        kill "$WEB_PID" 2>/dev/null || true
        wait "$WEB_PID" 2>/dev/null || true
    fi
    if [[ -n "$TUNNEL_DIR" ]]; then
        ssh -S "$TUNNEL_DIR/control" -O exit "$LOGIN_TARGET" 2>/dev/null || true
        rmdir "$TUNNEL_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${MLPUI_LOGIN_HOST:-}" ]]; then
    LOGIN_TARGET="${MLPUI_LOGIN_USER:-$USER}@${MLPUI_LOGIN_HOST}"
    TUNNEL_DIR=$(mktemp -d /tmp/mlpui-ssh.XXXXXXXX)
    SSH_AUTH=()
    if [[ ! -t 0 ]]; then
        # Batch jobs cannot prompt for passwords or unknown host keys.
        SSH_AUTH=(-o BatchMode=yes)
    fi
    echo "Opening reverse tunnel to $LOGIN_TARGET (may request your password)."
    if ! ssh -fNT -M -S "$TUNNEL_DIR/control" "${SSH_AUTH[@]}" \
        -o ExitOnForwardFailure=yes -o ConnectTimeout=20 \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -R "127.0.0.1:${Port}:127.0.0.1:${Port}" "$LOGIN_TARGET"; then
        echo "Reverse tunnel failed; WebUI was not started." >&2
        echo "If SSH needs a password, run this script with srun --pty bash inside a GPU allocation." >&2
        echo "Also check the login host, host key, and whether its port $Port is already in use." >&2
        exit 1
    fi
    echo "From your computer, connect to the SAME actual login node:"
    echo "ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:${Port}:127.0.0.1:${Port} $LOGIN_TARGET"
else
    echo "From your computer (replace USER, LOGIN and GPU_NODE):"
    echo "ssh -N -o ExitOnForwardFailure=yes -J USER@LOGIN -L 127.0.0.1:${Port}:127.0.0.1:${Port} USER@GPU_NODE"
fi
echo "Then open http://127.0.0.1:${Port} and select GPU / CUDA when creating a task."
echo "Stop the training in the UI and wait for model.pt before cancelling the Slurm job."

# Wait for the server and close our reverse tunnel when the script exits.
"$PYTHON" -u -m mlpui.web --port "$Port" --runs-dir "$RUNS_DIR" &
WEB_PID=$!
wait "$WEB_PID"
