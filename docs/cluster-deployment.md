# MLPUI: Manual deployment to a Slurm GPU cluster

This guide covers the current basic WebUI. Upload files, enter passwords and submit Slurm jobs manually; passwordless public-key setup and automatic cluster connections from local MLPUI are not required. Replace the placeholders `USER`, `LOGIN`, `GPU_NODE` and `/path/to/...` below.

## 1. Deployment layout

```text
Local browser http://127.0.0.1:2526
    └─ SSH tunnel through the login node
         └─ Allocated GPU compute node: 127.0.0.1:2526
              ├─ MLPUI WebUI
              ├─ Training subprocesses using the GPUs in this Slurm allocation
              ├─ .npy data on the cluster
              └─ Training results in persistent storage
```

The server manages both the interface and training jobs and must keep running inside the GPU allocation. Closing the browser or disconnecting the local SSH tunnel does not terminate the Slurm job. Cancelling the job or reaching its 12-hour limit terminates the service and training. The allocation holds the requested GPUs even when no training is running; release it when finished.

## 2. Upload the current project

Clone the latest `main` branch from `https://github.com/MoriyaSuwako2524/MLPUI`, or use the supplied `MLPUI-deploy.zip`. The deployment archive contains code, documentation, configuration and tests, excluding local virtual environments, raw data, large models in the repository root and training results. For an older deployment, first verify that the code is updated and includes `mlpui/web` and `mlpui/training.py`.

Upload from local PowerShell, or use an SFTP client that supports password authentication:

```powershell
scp "C:\Users\suwak\Documents\calculations\MLPUI\dist\MLPUI-deploy.zip" USER@LOGIN:/path/to/work/
ssh USER@LOGIN
```

On the cluster login node, extract into a new directory to avoid overwriting an existing deployment:

```bash
mkdir -p /path/to/work/mlpui-deployment
unzip /path/to/work/MLPUI-deploy.zip -d /path/to/work/mlpui-deployment
cd /path/to/work/mlpui-deployment/MLPUI
```

The project, data and results must reside on a filesystem accessible from compute nodes. The archive does not include data stored on pete. If another cluster cannot access the original paths, transfer the required `.npy` files and use their new paths in the UI. Do not upload the Windows `.venv`.

## 3. Prepare Python (first installation or code updates)

As in the working ComfyUI script, specify Python directly without relying on `module load` or `conda activate`. This example uses the NewtonNet environment. Verify its actual location; do not assume that the ComfyUI environment also contains NewtonNet:

```bash
PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
test -x "$PYTHON" || { echo "Set the actual Python path"; exit 1; }

cd /path/to/work/mlpui-deployment/MLPUI
"$PYTHON" --version
"$PYTHON" -m pip install numpy ase pyyaml psutil omegaconf torch-geometric lightning-utilities
"$PYTHON" -m pip install --no-deps -e .
```

Python 3.10 or later is required. `--no-deps` prevents installation from automatically replacing the existing PyTorch and model backends. pip does not proactively upgrade other dependencies with compatible installed versions. If the cluster has no Internet access, use its package mirror or prepare compatible offline packages.

For an isolated environment, if conda is available, first run `conda create -n mlpui --clone newtonnet`. Install with the new environment Python, then set `MLPUI_PYTHON` to its absolute path before submitting.

Check import paths and the backend interface:

```bash
"$PYTHON" - <<'PY'
import inspect
import torch
import mlpui
from mlpui.web.server import make_server
from mlpui.models.newtonnet.models.newtonnet import NewtonNet
print('MLPUI:', mlpui.__file__)
print('NewtonNet:', inspect.getfile(NewtonNet))
print('forward:', inspect.signature(NewtonNet.forward))
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
PY
```

NewtonNet and TorchMD-Net are bundled with MLPUI; separate fork installations and external package overlays are no longer needed. The NewtonNet path above should point into `MLPUI/mlpui/models/newtonnet/`. Standard energy/force outputs do not need `les`; LES heads do. CUDA being unavailable on a login node without a GPU is normal; the job script checks after GPU allocation.

TensorNet can use the bundled PyTorch neighbor search without compilation, but its memory and compute requirements grow quadratically with atom count. For large GPU jobs, compile the bundled extension using a compatible C++/CUDA toolchain:

```bash
"$PYTHON" scripts/build_model_extensions.py --cuda
"$PYTHON" -c 'from mlpui.models.torchmdnet.extensions.ops import BACKEND; print(BACKEND)'
```

Successful compilation should report `native`. Set `export MLPUI_NEIGHBORS=native` before submission to prevent an accidental slow fallback if the extension is unavailable, or keep the default when not compiling. The toolchain must match PyTorch; do not load the previously unavailable `GCC/9.3.0` module. A build node without a GPU may also require `TORCH_CUDA_ARCH_LIST` for the target GPU. The default PyTorch path has been checked on CPU; CUDA compilation and GPU execution need verification on the cluster.

## 4. Submit the WebUI job

The submission script is `scripts/slurm/mlpui_web.sbatch`. It follows the validated ComfyUI allocation: the `gpu` partition, one node, one task, one GPU, a 12-hour limit and exclusions `c302,c301`. It also requests four CPUs by default; adjust `--cpus-per-task` for the cluster. Python defaults to `$HOME/.conda/envs/newtonnet/bin/python`, overridden by `MLPUI_PYTHON`. Add account, QOS or memory `#SBATCH` directives if required by the cluster.

The previous `GCC/9.3.0` error occurred while loading modules, not during MLPUI training. The script no longer loads fixed Mamba/CUDA/GCC versions or executes `.bashrc`. The selected Python environment must already have working GPU backends. If compiling extensions later, choose a toolchain from modules actually available on the cluster. `Container param: el9hw` alone does not identify available GCC modules.

**Create logs in the project root before submitting**, because Slurm opens output files before executing the script:

```bash
cd /path/to/work/mlpui-deployment/MLPUI
mkdir -p logs
sbatch scripts/slurm/mlpui_web.sbatch
```

The defaults are port 2526 and results in `<project>/runs/cluster-web`. Override them if needed:

```bash
export MLPUI_PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
export MLPUI_PORT=2526
export MLPUI_RUNS_DIR=/path/to/persistent/mlpui-runs
sbatch scripts/slurm/mlpui_web.sbatch
```

Do not store results in node-local temporary directories that are removed when the job ends. Run only one WebUI instance per results directory at a time. Job history persists across allocations.

The `id.py` from the original script is not part of this project; its role in recording ports, proxying or tunneling has not been established. It is not called by default. If the existing cluster connection workflow depends on it, enable it explicitly:

```bash
export MLPUI_ID_SCRIPT=/absolute/path/to/id.py
sbatch scripts/slurm/mlpui_web.sbatch
```

The script executes `python "$MLPUI_ID_SCRIPT" "$SLURM_JOB_ID" "$Port"`. If this provides an existing access endpoint, follow that workflow and verify compatibility with the WebUI loopback and same-origin requirements.

Check scheduling and startup, replacing `JOB_ID` with the number returned by sbatch:

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -f logs/JOB_ID_stdout.txt
# Check errors:
tail -n 80 logs/JOB_ID_stderr.txt
```

Logs show the compute-node name, Python path, GPU name and WebUI port. `MLPUI: http://127.0.0.1:2526` indicates that the server is listening. A `PENDING` job has not started; wait before opening a compute-node tunnel.

## 5. Connect locally using manual password authentication

### Method A: SSH to the allocated compute node is allowed

Open another local PowerShell window and use the compute-node name from the logs:

```powershell
ssh -N -o ExitOnForwardFailure=yes -J USER@LOGIN -L 127.0.0.1:2526:127.0.0.1:2526 USER@GPU_NODE
```

Enter the login-node and compute-node passwords or MFA when prompted. No shell prompt after connecting is normal. Keep the terminal open and visit:

**http://127.0.0.1:2526**

`-J` routes through the login node, while the final SSH session reaches the compute node. The forwarding destination `127.0.0.1:2526` therefore refers to the WebUI host. A simple `ssh -L 2526:GPU_NODE:2526 USER@LOGIN` cannot reach a WebUI listening only on the compute-node loopback interface.

Unavailable public-key authentication does not prevent tunneling if the cluster permits password or interactive authentication. If the client fails after trying too many keys, configure `PubkeyAuthentication no` separately for **both the login and compute nodes** in local `~/.ssh/config`, and use passwords or MFA. This cannot bypass a server that allows only public keys; ask the administrator for a supported authentication method.

If local port 2526 is occupied, change `MLPUI_PORT` before submission and update both tunnel endpoints and the browser port. The WebUI checks Host/Origin; matching ports are the simplest setup.

### Method A2: Forward directly to the compute-node port through the login node, as with ComfyUI

If the login node can reach the service port on the compute node, neither compute-node SSH access nor a reverse tunnel is required. Run in the project directory:

```bash
git pull --ff-only
export MLPUI_ROOT="$PWD"
export MLPUI_HOST=0.0.0.0
export MLPUI_PORT=2526
unset MLPUI_LOGIN_HOST
mkdir -p logs
sbatch scripts/slurm/mlpui_web.sbatch
```

Read the actual compute-node name (for example c1036) from the job log, then run in local PowerShell:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:c1036:2526 moriya2524@sooner.oscer.ou.edu
```

Open `http://127.0.0.1:2526/`. Replace `c1036` with the node assigned to this job; do not use `127.0.0.1` in that position. Keep local and server ports identical to pass Host/Origin checks.

Test connectivity on the login node with `curl --connect-timeout 5 -H 'Host: 127.0.0.1:2526' http://c1036:2526/api/jobs`. JSON confirms connectivity. For timeouts or refused connections, check job logs, the node name and cluster network policies.

This mode exposes the WebUI without account authentication to the reachable cluster network. Host/Origin checks are not access authentication; use it only on a trusted network. The default remains `127.0.0.1`. Shut down any previous WebUI using the same job directory normally before starting a new instance.

### Method B: Compute-node SSH login is disallowed, but SSH from compute to login nodes is allowed

The startup script supports an automatic reverse tunnel. Set `MLPUI_LOGIN_HOST` to enable it; `MLPUI_LOGIN_USER` defaults to the current username. The tunnel listens only on the login-node `127.0.0.1`. If setup fails, the WebUI does not start. The script closes its own tunnel on exit. SSH keepalive detects broken connections and exits, but does not authenticate again automatically; recreate the tunnel.

**When a password is required**, run the following from the project directory on the login node. An interactive GPU allocation allows SSH to prompt normally; do not store passwords in files:

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

Wait for a GPU allocation and authenticate at the SSH prompt. The script then starts the WebUI and prints the local forwarding command. Keep this interactive session open. `srun` does not read `#SBATCH` directives in the script, so resource options are explicit in the command. `hostname -f` pins the actual login node; the local machine must resolve and reach that address, or use an external address supplied by the administrator.

**When compute-to-login authentication is already noninteractive**, set the same environment variables and use `mkdir -p logs` followed by `sbatch scripts/slurm/mlpui_web.sbatch`. Batch submission uses `BatchMode=yes`; authentication failure exits with a log explanation rather than waiting for an inaccessible password prompt.

To retain a running WebUI, add only the tunnel manually:

This requires the cluster to allow attached interactive steps with `srun --jobid` and SSH reverse forwarding. Do not put passwords in batch scripts. From an interactive terminal on the login node, enter the allocated compute node:

```bash
srun --jobid=JOB_ID --overlap --nodes=1 --ntasks=1 --pty bash
hostname
# Verify this is the WebUI compute node; enter the login-node password and keep the session open:
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

Then open the second tunnel from the local machine:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

Use the same browser URL. Both tunnels must reach **the same actual login node**, not different machines behind a load-balanced alias; port 2526 must also be free on that node. If the site disables these features, use its port proxy or ask the administrator for a supported workflow. Local script changes cannot override site restrictions.

## 6. Start training in the browser

1. Click **New job** and select NewtonNet. Training starts from scratch by default. For fine-tuning, enter the model path on the cluster and verify its architecture configuration.
2. Enter a `.npy` directory accessible to the cluster and choose QM windows or a custom mapping.
3. Typical files are `full_qm_type.npy`, `qm_coord_{shard}.npy`, `qm_grad_{shard}.npy` and `energy_{shard}.npy`, with training groups such as `w00, w01` and validation groups such as `w02`.
4. If labels are gradients, enable negation to obtain forces. Confirm the energy and length conversion factors; the UI does not infer source units.
5. Click **Check data**.
6. **Change Device from the default CPU to GPU · CUDA**, set epochs, batch size and learning rate, then start.
7. Set checkpoint interval N and retention K (defaults: 10 and 3; N=0 disables periodic checkpoints). For periodic testing, supply a separate test directory or groups and set interval M. Testing runs only at epochs M, 2M, 3M, etc.; validation still runs every epoch.

Physical GPU IDs are not needed; PyTorch uses devices visible within the Slurm allocation. The UI creates training subprocesses within the current allocation, without submitting additional `sbatch` jobs.

## 7. Save results and finish

For optional early stopping, select **Enable early stopping**, provide a separate validation set and choose the metric, Patience (default 10) and Min delta (default 0). The default metric is weighted total validation loss; enabled energy, force or charge MSE can also be selected. A decrease must exceed Min delta relative to the last sufficient improvement to reset the counter. Training ends after Patience consecutive epochs without such improvement. Test data does not affect this decision.

With early stopping enabled, `best.pt` stores weights at the lowest observed validation metric, even when an improvement is smaller than Min delta. `model.pt` stores the final weights without automatic rollback. The page shows the best epoch, metric and stopping reason, with a best-model download. The queue advances after early stopping. A validation set is required. Fine-tuning an older checkpoint restarts early-stopping statistics and does not restore the optimizer or patience counter.

Datasets support comma-separated tags when adding or managing them. Filter by a tag or **Untagged**; split subsets inherit tags. Expand **Delete dataset** to remove the library record by default. Source files in existing server directories are always retained. Uploaded or split data may be permanently deleted with its managed files, but deletion is rejected while any job or other dataset references those files. Deleting one subset does not delete its siblings; split indices and provenance are retained.

The sidebar **Datasets** page registers existing server directories or uploads multiple local `.npy` files (up to 20 GiB each) to the WebUI host. Standard fields are read by default; custom prefixes and `{shard}` groups are supported. Management shows sample counts, fields, array shapes, types and sizes, and supports search, rename, validation, archive and restore. Archiving does not delete files.

Expand **Create a new split from this dataset**, enter training/validation/test ratios and a random seed, then choose random or ordered splitting. Zero ratios omit a subset; too few samples for nonempty subsets cause an error. Outputs are separate NPY copies, leaving source data unchanged. Energy, forces, atomic charges, total charge and other fields use identical indices; unit and gradient conversions are applied once. Select library datasets directly when creating a job. A generated training subset automatically selects validation and test siblings from the same split.

The library and uploads are stored in `MLPUI_RUNS_DIR/datasets/`. Each `split-<ID>/split.json` records ratios, seed and source; files such as `train_indices.npy` record indices relative to the immediate source dataset. Keep the same run directory across cluster restarts. Splitting requires extra disk space; wait for completion before ending the Slurm job. Interrupted uploads may be retried on the current page. After a refresh, archive unfinished records and upload again. For time-correlated trajectories, split in original order or register separate trajectory groups.

For the optional **Hard total-charge constraint**, supply atomic-charge labels (`charges.npy`) and total charge Q per structure (`charge.npy`, shape `[structures]` or `[structures, 1]`, in e). Grouped defaults are `qm_charge_{shard}.npy` and `total_charge_{shard}.npy`. Neutral structures must explicitly supply Q=0; the file cannot be omitted. The constraint is saved with checkpoints; supply Q during evaluation as well. No additional dependency is needed.

For standalone evaluation, choose **Evaluate an existing model** under **New job**, enter its absolute path and matching architecture, and select the NPY directory or groups. Enable energy/force evaluation and choose a GPU, then start. Epochs and learning rate are not needed. Evaluation traverses the data once without updating weights. Details show MAE, RMSE and MSE; complete results are saved as `evaluation.json` with a JSON link. Force metrics aggregate all atomic Cartesian components in the configured data units. Evaluation shares the training scheduler queue.

Results are saved in `MLPUI_RUNS_DIR/<job-ID>/`:

- `config.json`: Job parameters and data mapping.
- `status.json`: Progress and losses from completed epochs.
- `train.log`: Training output and errors.
- `model.pt`: Weights saved on completion or a normal stop through the UI.
- `checkpoints/epoch_000010.pt`, etc.: Weights saved every N complete epochs, retaining the latest K files. The final `model.pt` does not count toward K. Download these files from job details; existing checkpoints remain usable after abnormal termination.

When finished, click **Stop job** if training is running. Wait for **Stopped** and the model download link. After confirming the model exists, run `scancel JOB_ID` on the login node to release the GPU. Close the local tunnel to disconnect.

**Periodic checkpoints save weights, not optimizer/RNG state for exact resumption.** Direct cancellation, node failure or an allocation time limit may lose progress since the last successful save. A periodic checkpoint or final `model.pt` can initialize another job, but its optimizer is recreated. Old checkpoints exceeding retention are removed only after a new file is written successfully. Leave time to stop long runs normally and save.

## 8. Troubleshooting

| Symptom | What to check |
| --- | --- |
| sbatch reports a missing log directory | Submit from the project root after `mkdir -p logs` |
| `ModuleNotFoundError: mlpui` | Verify the logged Python belongs to the environment where MLPUI is installed |
| `No CUDA device is available` | Check GPU allocation, CUDA visibility and whether CPU-only PyTorch was installed |
| `Address already in use` | Check ports on the compute node, local machine and reverse-tunnel login node |
| Browser cannot connect | Check that Slurm is RUNNING, the server is listening and the tunnel ends at the compute node |
| `Invalid host` or `Cross-origin` | Use identical local/server ports and `http://127.0.0.1:PORT`; existing proxies may need adaptation |
| Configuration or weights mismatch | Check NewtonNet version, architecture and checkpoint compatibility |
| Previous jobs appear interrupted | The preceding server session ended abnormally; retained job records do not guarantee saved models |

References: [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html), [OpenSSH ssh documentation](https://man.openbsd.org/ssh.1). Partition and node exclusions follow the supplied ComfyUI script; SSH and scheduling policies depend on the actual cluster configuration.


## 9. Queue and multiple GPUs

Submitted training and evaluation jobs enter a queue. Each visible GPU runs one job at a time and accepts the next job in submission order when free. Multiple GPUs run separate jobs concurrently; CPU jobs run serially in a separate slot. GPU jobs never fall back automatically to CPU. Use **Cancel queued job** to cancel a waiting job. Job details show the assigned logical GPU ID.

The script defaults to `#SBATCH --gres=gpu:1`, so GPU jobs run serially. For two concurrent jobs, change it to `#SBATCH --gres=gpu:2` within cluster quotas, increase CPU/memory resources appropriately and resubmit the WebUI allocation. The queue uses only GPUs visible in that allocation; it does not submit extra Slurm jobs or distribute one model across GPUs.

A candidate GPU must have at least 90% free memory. When nvidia-smi UUID process information is available, GPUs with other compute processes are excluded; otherwise the scheduler uses the memory check. This does not guarantee that any particular model fits. An assigned slot is released only after its subprocess exits.

The queue persists in the run directory. Restarting the WebUI with that directory resumes jobs that have not started. Previously running jobs are marked interrupted and are not retrained automatically. Closing the browser does not affect scheduling. Ending the Slurm allocation stops execution until the WebUI is restarted.
