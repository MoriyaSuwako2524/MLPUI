# MLPUI：手动部署到 Slurm GPU 集群

本指南对应当前基础 WebUI。你手动上传文件、输入密码并提交 Slurm 作业；无需配置免密公钥，也不需要让本地 MLPUI 自动连接集群。以下 `USER`、`LOGIN`、`GPU_NODE` 和 `/path/to/...` 都是需要替换的占位符。

## 1. 部署结构

```text
本地浏览器 http://127.0.0.1:2526
    └─ SSH 隧道，经登录节点中转
         └─ 已分配的 GPU 计算节点：127.0.0.1:2526
              ├─ MLPUI WebUI
              ├─ 训练子进程，使用该 Slurm 作业的 GPU
              ├─ 集群上的 .npy 数据
              └─ 持久目录中的训练结果
```

服务器同时管理界面和训练任务，必须在 GPU 作业中保持运行。关闭浏览器或断开本地 SSH 隧道不会终止 Slurm 作业；取消作业或达到 12 小时时限则会终止服务和训练。即使界面没有训练任务，Slurm 作业仍占用申请的 GPU，使用结束后应释放。

## 2. 上传当前项目

可以从 `https://github.com/MoriyaSuwako2524/MLPUI` 克隆最新的 `main` 分支，或使用提供的 `MLPUI-deploy.zip`。部署包包含代码、文档、配置和测试，不包含本机虚拟环境、原始数据、根目录大模型和训练结果。已有旧部署应先确认代码已更新，包含 `mlpui/web` 和 `mlpui/training.py`。

在本机 PowerShell 中上传，也可用支持密码登录的 SFTP 软件完成：

```powershell
scp "C:\Users\suwak\Documents\calculations\MLPUI\dist\MLPUI-deploy.zip" USER@LOGIN:/path/to/work/
ssh USER@LOGIN
```

在集群登录节点解压到新的目录，避免覆盖已有部署：

```bash
mkdir -p /path/to/work/mlpui-deployment
unzip /path/to/work/MLPUI-deploy.zip -d /path/to/work/mlpui-deployment
cd /path/to/work/mlpui-deployment/MLPUI
```

项目、数据和结果目录应位于计算节点也能访问的文件系统。部署包不包含 pete 上的数据；如果另一个集群访问不到原路径，需要自行传入对应 `.npy` 文件，并在界面使用新路径。不要上传 Windows 的 `.venv`。

## 3. 准备 Python 环境（仅首次或代码更新时）

按已经能运行的 ComfyUI 脚本方式，直接指定 Python，不依赖 `module load` 或 `conda activate`。这里默认使用 NewtonNet 环境；请先确认它的实际位置，不要默认认为 ComfyUI 环境也安装了 NewtonNet：

```bash
PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
test -x "$PYTHON" || { echo "请修改为实际 Python 路径"; exit 1; }

cd /path/to/work/mlpui-deployment/MLPUI
"$PYTHON" --version
"$PYTHON" -m pip install numpy ase pyyaml psutil omegaconf
"$PYTHON" -m pip install --no-deps -e .
```

需要 Python 3.10 或更高版本。`--no-deps` 避免安装 MLPUI 时自动替换现有 PyTorch 和模型后端。其余依赖若已有兼容版本，pip 不会主动升级它们。若集群不能联网，应使用集群的软件镜像或提前准备兼容的离线安装包。

希望隔离环境且 conda 命令可用时，可先执行 `conda create -n mlpui --clone newtonnet`，之后用新环境的 Python 安装，并在提交前将 `MLPUI_PYTHON` 设置为新环境 Python 的绝对路径。

确认导入路径和后端接口：

```bash
"$PYTHON" - <<'PY'
import inspect
import torch
import mlpui
from mlpui.web.server import make_server
from newtonnet.models.newtonnet import NewtonNet
print('MLPUI:', mlpui.__file__)
print('NewtonNet:', inspect.getfile(NewtonNet))
print('forward:', inspect.signature(NewtonNet.forward))
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
PY
```

本接口适配项目使用的 [NewtonNet fork](https://github.com/MoriyaSuwako2524/NewtonNet)，其调用为 `forward(self, z, pos, cell, batch)`。如果原环境是另一版 NewtonNet，建议在克隆的独立环境中安装匹配版本，避免破坏原工作流。该 fork 还依赖 `les`；具体安装参考项目 README。登录节点没有 GPU 时 `torch.cuda.is_available()` 为假可以正常发生，提交脚本会在分配到 GPU 后再次检查。

只使用 NewtonNet 时无需安装 TorchMD-Net。以后需要 TensorNet，再安装项目指定的 [TorchMD-Net fork](https://github.com/MoriyaSuwako2524/torchmd-net) 及与 GPU PyTorch 匹配的编译扩展。**不要在集群沿用 README 中的 Windows CPU wheel 覆盖方案。** CUDA 模块、PyTorch CUDA runtime 和驱动之间的兼容性以你现有可工作的环境为准。

## 4. 提交 WebUI 作业

提交脚本位于 `scripts/slurm/mlpui_web.sbatch`。它沿用已验证的 ComfyUI 作业设置：`gpu` 分区、单节点、单任务、一张 GPU、12 小时时限，并排除 `c302,c301`。额外默认申请 4 个 CPU，可按集群要求修改 `--cpus-per-task`。默认 Python 为 `$HOME/.conda/envs/newtonnet/bin/python`，可通过 `MLPUI_PYTHON` 覆盖。若需要账户、QOS 或内存申请，按该集群规定添加相应 `#SBATCH` 行。

之前的 `GCC/9.3.0` 报错来自模块加载阶段，不是 MLPUI 训练错误。新脚本不再加载固定版本的 Mamba/CUDA/GCC，也不执行 `.bashrc`。这依赖你指定的 Python 环境已经具备可运行的 GPU 后端；如果以后需要从源码编译扩展，再根据目标集群实际提供的模块选择工具链。`Container param: el9hw` 本身不足以判断有哪些 GCC 模块可用。

**必须先在项目根目录创建 logs，再提交**，因为 Slurm 会在脚本开始执行前打开输出文件：

```bash
cd /path/to/work/mlpui-deployment/MLPUI
mkdir -p logs
sbatch scripts/slurm/mlpui_web.sbatch
```

脚本默认端口为 2526、结果目录为 `<项目>/runs/cluster-web`。可以覆盖：

```bash
export MLPUI_PYTHON=/home/moriya2524/.conda/envs/newtonnet/bin/python
export MLPUI_PORT=2526
export MLPUI_RUNS_DIR=/path/to/persistent/mlpui-runs
sbatch scripts/slurm/mlpui_web.sbatch
```

不要把结果放到作业结束后会清理的节点本地临时目录。一个结果目录一次只运行一个 WebUI 实例，历史任务会跨作业保留。

你原脚本中的 `id.py` 不属于当前项目，尚无法判断它是记录端口、提供代理还是建立隧道。新脚本默认不调用它；如果你现有集群连接流程依赖它，可以显式启用：

```bash
export MLPUI_ID_SCRIPT=/absolute/path/to/id.py
sbatch scripts/slurm/mlpui_web.sbatch
```

脚本会执行 `python "$MLPUI_ID_SCRIPT" "$SLURM_JOB_ID" "$Port"`。如果它提供现成的访问入口，请按原流程使用，并确认兼容 WebUI 的 loopback/同源要求。

检查排队和启动情况，将 `JOB_ID` 换成 sbatch 返回的编号：

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -f logs/JOB_ID_stdout.txt
# 报错时检查：
tail -n 80 logs/JOB_ID_stderr.txt
```

日志会打印计算节点名、Python 路径、GPU 名称和 WebUI 端口。以 `MLPUI: http://127.0.0.1:2526` 表示服务已监听。`PENDING` 时尚未启动，不要急着建立计算节点隧道。

## 5. 从本机连接：手动密码认证

### 方式 A：允许 SSH 登录已分配的计算节点

在本机另开 PowerShell，使用日志里的计算节点名：

```powershell
ssh -N -o ExitOnForwardFailure=yes -J USER@LOGIN -L 127.0.0.1:2526:127.0.0.1:2526 USER@GPU_NODE
```

按提示输入登录节点和计算节点的密码或 MFA。命令连接后不出现 shell 提示符属于正常情况；保持该终端开启，然后访问：

**http://127.0.0.1:2526**

这里 `-J` 负责经过登录节点，最终 SSH 会话连接计算节点，因此转发目标 `127.0.0.1:2526` 才是运行 WebUI 的机器。普通的 `ssh -L 2526:GPU_NODE:2526 USER@LOGIN` 无法访问只监听计算节点 loopback 的 WebUI。

公钥不可用不等于 SSH 隧道不可用，前提是集群允许密码或交互式认证。如果客户端因为尝试过多密钥而失败，可以在本机 `~/.ssh/config` 中为**登录节点和计算节点分别**配置 `PubkeyAuthentication no`，继续使用密码或 MFA。若服务器只允许公钥，不允许任何密码/交互认证，则此方法也无法绕过，需要管理员提供可用的认证方式。

如果本机 2526 已被占用，请更改提交前的 `MLPUI_PORT`，并把隧道两端端口和浏览器端口一起修改。当前 WebUI 检查 Host/Origin，使用相同端口最直接。

### 方式 B：不允许 SSH 进入计算节点，但允许计算节点 SSH 回登录节点

此方式取决于集群是否允许 `srun --jobid` 附加交互步骤、SSH 反向转发。不要在批处理脚本里放密码。先在登录节点的交互终端中进入自己分配到的计算节点：

```bash
srun --jobid=JOB_ID --overlap --nodes=1 --ntasks=1 --pty bash
hostname
# 确认这里是 WebUI 所在的计算节点；输入登录节点密码，保持此会话开启：
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

再在本机开启第二段隧道：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:127.0.0.1:2526 USER@LOGIN
```

浏览器仍访问同一地址。这种情况下两段隧道必须连接**同一个实际登录节点**，不能分别落在负载均衡登录别名后面的不同机器上；登录节点的 2526 也必须空闲。若站点禁用了这些能力，应使用站点提供的端口代理或请管理员确认支持的流程，不能靠修改本地脚本解决。

## 6. 在网页上开始训练

1. 点击“新建训练”，选择 NewtonNet；默认从头训练，微调时填写集群上的模型路径，并核对结构配置。
2. 填写该集群能访问的 `.npy` 数据目录，选择 QM 窗口格式或自定义映射。
3. 类似你的现有数据可使用 `full_qm_type.npy`、`qm_coord_{shard}.npy`、`qm_grad_{shard}.npy`、`energy_{shard}.npy`，训练组例如 `w00, w01`，验证组例如 `w02`。
4. 若标签是梯度，启用取负号转换；确认能量与长度换算系数，界面不会猜测源单位。
5. 点击“检查数据”。
6. **把运行设备从默认 CPU 改为 GPU · CUDA**，设置 epochs、batch size、学习率后开始。

不需要指定集群的物理 GPU 编号，PyTorch 使用 Slurm 分配到的可见设备。网页创建的是当前 Slurm 分配中的训练子进程，不会另行 `sbatch`。

## 7. 保存与结束

结果保存在指定的 `MLPUI_RUNS_DIR/<任务ID>/`：

- `config.json`：本次任务的参数和数据映射。
- `status.json`：进度与已完成 epoch 的损失记录。
- `train.log`：训练输出和错误。
- `model.pt`：完成训练或通过界面正常停止后保存的权重。

结束使用时，先点击界面“停止训练”（如果仍在运行），等待状态变成“已停止”并出现模型下载按钮。确认模型存在后，再从登录节点执行 `scancel JOB_ID` 释放 GPU。关闭本机隧道即可断开访问。

**目前没有定期恢复 checkpoint，也没有 optimizer/RNG 的精确断点续训。** 不要把直接 `scancel`、节点故障或到达作业时限当成正常保存方式；它们可能丢失当前训练权重。保存的 `model.pt` 可以用作下一任务的初始权重，但优化器会重新创建。长训练应预留时间，主动停止并保存。

## 8. 常见问题

| 现象 | 检查位置 |
| --- | --- |
| sbatch 报日志路径不存在 | 从项目根目录提交，并提前 `mkdir -p logs` |
| `ModuleNotFoundError: mlpui` | 日志里的 Python 是否属于安装过项目的环境 |
| `No CUDA device is available` | GPU 分配、CUDA 可见性、环境是否误装 CPU PyTorch |
| `Address already in use` | 计算节点、本机或反向隧道登录节点的端口是否被占用 |
| 浏览器连接失败 | Slurm 是否 RUNNING、服务是否已监听、隧道是否最终指向计算节点 |
| `Invalid host` 或 `Cross-origin` | 使用相同本机/服务器端口和 `http://127.0.0.1:端口`；已有代理可能需要额外适配 |
| 参数或权重不匹配 | NewtonNet fork 版本、模型结构和 checkpoint 是否一致 |
| 显示之前的任务已中断 | 上个服务器会话非正常结束；任务记录保留，但不表示保存了模型 |

参考：[Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)、[OpenSSH ssh 文档](https://man.openbsd.org/ssh.1)。分区和排除节点来自你提供的 ComfyUI 脚本；目标集群的 SSH 和调度策略仍需以实际站点配置为准。
