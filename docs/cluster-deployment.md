# MLPUI: manual deployment on a Slurm GPU cluster

You upload code and data, authenticate interactively and submit the job yourself.
No public-key login or automatic local-to-cluster connection is required. Replace
USER, LOGIN, GPU_NODE and /path/to/... placeholders with your actual values.

## 1. Deployment layout

```text
Local browser http://127.0.0.1:2526
  -> SSH tunnel through login node
    -> allocated GPU compute node: MLPUI WebUI and workers
      -> shared NPY data and persistent run directory
```

The server must remain running in its GPU allocation. Closing the browser or local
tunnel does not stop the Slurm job. Cancelling the allocation or reaching its time
limit terminates the server and workers. Release the allocation when finished,
including when the interface is idle.

## 2. Transfer the project and data

Clone the main branch of https://github.com/MoriyaSuwako2524/MLPUI, or upload a
prepared MLPUI-deploy.zip containing code, configuration, documentation and tests.
The archive excludes local environments, datasets, large model weights and results.

```powershell
scp "C:\Users\suwak\Documents\calculations\MLPUI\dist\MLPUI-deploy.zip" USER@LOGIN:/path/to/work/
ssh USER@LOGIN
```

Extract into a new directory to avoid overwriting an existing deployment:

```bash
mkdir -p /path/to/work/mlpui-deployment
unzip /path/to/work/MLPUI-deploy.zip -d /path/to/work/mlpui-deployment
cd /path/to/work/mlpui-deployment/MLPUI
```

Code, input data and results must be on storage accessible from compute nodes.
Transfer the NPY data separately if the target cluster cannot access its original
path. Do not copy the Windows .venv to the cluster.

## 3. Prepare the Python environment

Use an explicit Python executable, as in the working ComfyUI launch script. The
example uses a NewtonNet environment; verify the actual path and installed packages.

```bash
PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
test -x "$PYTHON" || { echo "Set the actual Python path"; exit 1; }
cd /path/to/work/mlpui-deployment/MLPUI
"$PYTHON" --version
"$PYTHON" -m pip install numpy ase pyyaml psutil omegaconf torch-geometric lightning-utilities matplotlib
"$PYTHON" -m pip install --no-deps -e .
```

MLPUI declares Python 3.10 or later; the UMA inference environment has separate
requirements. --no-deps preserves the existing PyTorch installation. Install any
remaining requirements from pyproject.toml with versions compatible with the cluster.
Use a site mirror or compatible offline wheels if internet access is unavailable.
Optionally clone the existing environment with `conda create -n mlpui --clone newtonnet`
and set MLPUI_PYTHON to that environment's executable.

```bash
"$PYTHON" - <<'PY'
import inspect, torch, mlpui
from mlpui.web.server import make_server
from mlpui.models.newtonnet.models.newtonnet import NewtonNet
print('MLPUI:', mlpui.__file__)
print('NewtonNet:', inspect.getfile(NewtonNet))
print('forward:', inspect.signature(NewtonNet.forward))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
```

NewtonNet and TorchMD-Net are bundled; separate forks are unnecessary. Standard
energy/force heads do not require LES. CUDA being unavailable on a login node is
normal; the job script checks the GPU after allocation.

TensorNet can use the bundled PyTorch neighbor search without compilation. Its
cost grows quadratically with atom count. For larger GPU systems, compile the
bundled extension using a toolchain compatible with PyTorch:

```bash
"$PYTHON" scripts/build_model_extensions.py --cuda
"$PYTHON" -c 'from mlpui.models.torchmdnet.extensions.ops import BACKEND; print(BACKEND)'
```

A successful build reports native. Set MLPUI_NEIGHBORS=native to require it, or leave
the default to permit fallback. A build node without a GPU may need
TORCH_CUDA_ARCH_LIST. Do not load unavailable hard-coded modules such as GCC/9.3.0.
For UMA, follow [UMA prediction setup](uma-prediction.md) and export MLPUI_UMA_PYTHON.

## 4. Submit the WebUI job

scripts/slurm/mlpui_web.sbatch defaults to the gpu partition, one node, one task,
one GPU, four CPUs, 12 hours and excluded nodes c302,c301. Adjust these settings,
account, QOS and memory for the cluster. It uses an explicit Python path and does
not load fixed Mamba/CUDA/GCC modules or source .bashrc. The el9hw container banner
alone does not indicate which compiler modules exist.

Create logs before submission: Slurm opens log files before the script runs.

```bash
cd /path/to/work/mlpui-deployment/MLPUI
mkdir -p logs
export MLPUI_PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
export MLPUI_PORT=2526
export MLPUI_RUNS_DIR=/path/to/persistent/mlpui-runs
sbatch scripts/slurm/mlpui_web.sbatch
```

Default results go to runs/cluster-web. Use persistent storage and only one WebUI
instance per run directory. The optional, site-specific id.py is not bundled; set
MLPUI_ID_SCRIPT=/absolute/path/to/id.py to invoke it with job ID and port. Verify its
proxy behavior separately if your connection workflow depends on it.

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -f logs/JOB_ID_stdout.txt
tail -n 80 logs/JOB_ID_stderr.txt
```

The log prints the node, Python path, GPU and port. Wait for the listening message;
a PENDING allocation has not started the service yet.

## 5. Connect from a local browser

### A. SSH into the allocated compute node is allowed

```powershell
ssh -N -o ExitOnForwardFailure=yes -J USER@LOGIN -L 127.0.0.1:2526:127.0.0.1:2526 USER@GPU_NODE
```

Authenticate as prompted and keep the terminal open. Visit http://127.0.0.1:2526.
The final SSH endpoint is the compute node, so its loopback is the WebUI host.
Forwarding through only the login node cannot reach a compute-node loopback listener.
If excess key attempts prevent password authentication, configure PubkeyAuthentication
no for both hosts in your SSH client config. This does not bypass a server that
requires keys; use the authentication method supported by the administrator.

### A2. The login node can reach the compute-node service port

This does not require SSH login to the compute node:

```bash
git pull --ff-only
export MLPUI_ROOT="$PWD"
export MLPUI_HOST=0.0.0.0
export MLPUI_PORT=2526
unset MLPUI_LOGIN_HOST
mkdir -p logs
sbatch scripts/slurm/mlpui_web.sbatch
```

Use the actual node from the job log (replace c1036):

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:c1036:2526 moriya2524@sooner.oscer.ou.edu
```

The forwarding destination must be the compute node, not 127.0.0.1 on the login
node. Diagnose from the login node with:

```bash
curl --connect-timeout 5 -H 'Host: 127.0.0.1:2526' http://c1036:2526/api/jobs
```

This mode exposes an unauthenticated WebUI to reachable cluster hosts. Host/Origin
checks are not authentication; use only on a trusted network. The default listener
remains 127.0.0.1. Stop the previous server normally before reusing its run directory.

### B. Reverse tunnel from the compute node to the login node

If direct compute-node SSH is prohibited but outbound SSH is allowed, setting
MLPUI_LOGIN_HOST makes the launch script establish a reverse tunnel. It listens on
the login node loopback, exits if setup fails and cleans up its own tunnel. SSH
keepalives detect disconnection, but cannot automatically reauthenticate.

For password authentication, run an interactive GPU job from the project directory:

```bash
git pull --ff-only
export MLPUI_ROOT="$PWD"
export MLPUI_LOGIN_HOST="$(hostname -f)"
export MLPUI_LOGIN_USER=moriya2524
export MLPUI_PYTHON="$HOME/.conda/envs/newtonnet/bin/python"
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 --time=12:00:00 --exclude=c302,c301 --job-name=mlpui-web \
  --pty bash scripts/slurm/mlpui_web.sbatch
```

Keep the interactive session open. srun does not read the script's SBATCH directives,
so resource options are explicit above. hostname -f selects the actual login host;
use its site-provided external address if necessary. With noninteractive SSH
authentication already working, use sbatch instead. BatchMode=yes fails rather than
waiting for a password that cannot be entered. Never put passwords in the script.

To attach a tunnel to an existing allocation, if site policy permits:

```bash
srun --jobid=JOB_ID --overlap --nodes=1 --ntasks=1 --pty bash
hostname
# Confirm this is the node running WebUI, then keep the session open.
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

On the local machine:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

Both tunnels must use the same actual login host, not different hosts behind a
load-balanced alias. The login-node port must be free. If these mechanisms are
disabled, ask the administrator for the supported proxy workflow. Keep local and
server ports equal for the WebUI Host/Origin checks; change all ports together if needed.

## 6. Create jobs

Choose New job, a model and a training mode. Continuation loads a matching checkpoint
from the server. Enter the NPY directory and naming convention, for example
full_qm_type.npy, qm_coord_{shard}.npy, qm_grad_{shard}.npy and energy_{shard}.npy.
Use separate groups such as w00,w01 for training and w02 for validation. Check
units and whether force labels are gradients, then click Check data. Select GPU / CUDA
and set epochs, batch size and learning rate before submission.

Save interval N and retention K default to 10 and 3. N=0 disables periodic saves.
An independent test set is evaluated every M epochs; validation runs each epoch.
Optional early stopping requires separate validation data. The monitored metric must
decrease by more than min delta to reset patience. Test data never controls stopping.
best.pt tracks the actual best validation metric; model.pt retains final weights.
Continuation resets optimizer, epoch and early-stopping counters.

Evaluation computes MAE, RMSE, MSE and separate parity plots for supported targets.
UMA prediction uses unlabeled structures and writes energy/force NPY outputs; see
[UMA prediction](uma-prediction.md) for runtime and unit requirements.

## 7. Dataset management and results

Register server directories or upload NPY files (20 GiB per file). Search, rename,
tag, inspect, archive and restore entries. Archiving preserves files. Deletion
removes the catalog entry by default; original external directories are never deleted.
Managed files can be removed only when no job or other dataset references them.

Splitting creates independent NPY copies, source indices and provenance. Supply
relative train/validation/test ratios and a seed; 0 omits a subset. Too few samples
for nonempty requested subsets is an error. All fields use the same indices, with
unit/gradient conversion applied once. Selecting a generated training subset also
selects its sibling validation/test sets. Use ordered splits or separate trajectories
for correlated data. Subsets inherit tags. Files live under MLPUI_RUNS_DIR/datasets;
split.json and *_indices.npy preserve provenance. Splits require additional disk space.
Interrupted uploads can be retried on the same page; after refresh, archive incomplete
drafts and upload again.

For atomic charges, supply charges.npy. A hard total-charge constraint also requires
charge.npy (including explicit zeros for neutral structures), shape [S] or [S,1].
The constraint is saved with the checkpoint and also requires Q at evaluation.

Each job has its own MLPUI_RUNS_DIR/<job-id> directory with config.json, status.json
and train.log. Training saves model.pt, optionally best.pt and periodic
checkpoints/epoch_000010.pt. Retention excludes model.pt and best.pt. Evaluation
saves evaluation.json and plots. Prediction saves prediction.json and NPY arrays.

Stop running jobs through the interface and wait for saved outputs before cancelling
the Slurm allocation with scancel JOB_ID. Checkpoints store weights, not exact
optimizer/RNG resume state. Node failure, cancellation or time limits can lose work
since the most recent completed checkpoint. New saves succeed before old retained
checkpoints are removed.

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing Slurm log directory | Create logs before sbatch from the project root |
| ModuleNotFoundError: mlpui | Python environment and project installation |
| CUDA unavailable | GPU allocation, visibility and CUDA-enabled PyTorch |
| Address already in use | Ports on local, login and compute hosts |
| Browser cannot connect | Allocation state, listener, node and tunnel endpoint |
| Invalid host / Cross-origin | Equal ports and http://127.0.0.1:PORT |
| Weight/config mismatch | Model family, architecture and checkpoint |
| Interrupted previous jobs | The previous server exited abnormally; records do not guarantee saved weights |
| UMA dependency error | MLPUI_UMA_PYTHON and its FAIR-Chem installation |

## 9. Queue and multiple GPUs

Each visible GPU runs one job at a time, with queued jobs dispatched in submission
order when a slot is available. CPU jobs run serially in a separate slot. GPU jobs
do not fall back to CPU. Cancel pending jobs with Cancel queued job.

The default allocation has one GPU. Request more GPUs, CPUs and memory through Slurm
to run independent jobs concurrently. This is not distributed training of a single
model, and the WebUI does not submit additional Slurm allocations.

Candidates need at least 90% free GPU memory. When available, nvidia-smi UUID/process
information also excludes devices with other compute processes. This is not a
promise that every model fits. Slots are released only after workers exit. The queue
persists in the run directory: pending jobs resume after server restart; previously
running jobs are marked interrupted, not automatically retrained. Browser closure
does not affect scheduling; allocation shutdown pauses it.

References: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html) and
[OpenSSH ssh](https://man.openbsd.org/ssh.1). Site policies take precedence over
example partition, node and tunnel settings.
