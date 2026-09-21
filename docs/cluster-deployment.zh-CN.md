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
"$PYTHON" -m pip install numpy ase pyyaml psutil omegaconf torch-geometric lightning-utilities
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
from mlpui.models.newtonnet.models.newtonnet import NewtonNet
print('MLPUI:', mlpui.__file__)
print('NewtonNet:', inspect.getfile(NewtonNet))
print('forward:', inspect.signature(NewtonNet.forward))
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
PY
```

NewtonNet 和 TorchMD-Net 已内置在 MLPUI，不再需要安装两份 fork 或覆盖外部包。上面的 NewtonNet 路径应指向 `MLPUI/mlpui/models/newtonnet/`。标准能量/力输出无需 `les`，只有 LES 输出头需要它。登录节点没有 GPU 时 CUDA 不可用是正常现象，作业脚本会在分配到 GPU 后检查。

TensorNet 默认可以使用内置 PyTorch 邻居搜索，不需要编译，但其内存和计算量随原子数平方增长。大规模 GPU 任务建议在具备匹配 C++/CUDA 工具链的环境中编译自带扩展：

```bash
"$PYTHON" scripts/build_model_extensions.py --cuda
"$PYTHON" -c 'from mlpui.models.torchmdnet.extensions.ops import BACKEND; print(BACKEND)'
```

成功后应显示 `native`。可在提交作业前设置 `export MLPUI_NEIGHBORS=native`，避免扩展缺失时意外使用较慢的回退路径；不编译时保持默认即可。工具链版本必须与 PyTorch 匹配，不要加载之前不存在的 `GCC/9.3.0`。无 GPU 的编译节点可能还需要设置对应目标 GPU 的 `TORCH_CUDA_ARCH_LIST`。默认 PyTorch 路径已在 CPU 验证，CUDA 编译及 GPU 运行需在集群验证。

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

### 方式 A2：像 ComfyUI 一样，经登录节点直接访问计算节点端口

如果登录节点可以连接计算节点的服务端口，不需要 SSH 登录计算节点，也不需要反向隧道。在项目目录执行：

```bash
git pull --ff-only
export MLPUI_ROOT="$PWD"
export MLPUI_HOST=0.0.0.0
export MLPUI_PORT=2526
unset MLPUI_LOGIN_HOST
mkdir -p logs
sbatch scripts/slurm/mlpui_web.sbatch
```

从作业日志获取实际计算节点名（例如 c1036），在本机 PowerShell 执行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:2526:c1036:2526 moriya2524@sooner.oscer.ou.edu
```

打开 `http://127.0.0.1:2526/`。中间的 `c1036` 必须替换为本次作业节点，不能写成 `127.0.0.1`。本地和服务端端口保持一致，以通过 Host/Origin 检查。

可先在登录节点检查连通性：`curl --connect-timeout 5 -H 'Host: 127.0.0.1:2526' http://c1036:2526/api/jobs`。返回 JSON 表示通路可用；超时或拒绝连接时检查作业日志、节点名和集群网络策略。

此模式会向可达的集群网络开放没有账号认证的 WebUI；Host/Origin 检查不等同于访问认证，仅在可信网络使用。默认仍监听 `127.0.0.1`。新启动前先正常结束使用同一任务目录的旧 WebUI。

### 方式 B：不允许 SSH 进入计算节点，但允许计算节点 SSH 回登录节点

启动脚本已支持自动建立反向隧道：设置 `MLPUI_LOGIN_HOST` 即启用，`MLPUI_LOGIN_USER` 默认是当前用户名。隧道仅监听登录节点的 `127.0.0.1`，建立失败时不启动 WebUI，脚本退出时关闭自己创建的隧道。连接中断时 SSH 会通过保活检测退出，但不会自动重新认证；需要重新建立隧道。

**目前需要密码时**，在登录节点、项目目录中运行以下命令。使用交互式 GPU 作业，SSH 可以正常请求密码；不把密码写入文件：

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

等待 GPU 分配，按 SSH 提示认证，随后脚本启动 WebUI 并打印本地转发命令。保持此交互会话开启；`srun` 启动时不会读取脚本内的 `#SBATCH` 指令，所以资源参数已显式列在命令中。`hostname -f` 固定实际登录节点；本地必须能够解析并访问这个地址，否则需使用管理员提供的该节点外部地址。

**计算节点到登录节点已可免交互认证时**，设置同样的环境变量后，可以改用 `mkdir -p logs` 和 `sbatch scripts/slurm/mlpui_web.sbatch`。批处理模式使用 `BatchMode=yes`，认证失败会退出并在日志解释原因；不会等待无法输入的密码。

如果希望沿用已经启动的 WebUI，也可以只手动添加隧道：

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
7. 设置保存间隔 N 和最大保留数量 K（默认 10、3；N=0 关闭定期保存）。如需定期测试，填写独立的测试目录或分组，并设置测试间隔 M；测试仅在第 M、2M、3M…个 epoch 进行，验证集仍每个 epoch 评估。

不需要指定集群的物理 GPU 编号，PyTorch 使用 Slurm 分配到的可见设备。网页创建的是当前 Slurm 分配中的训练子进程，不会另行 `sbatch`。

## 7. 保存与结束

数据集支持标签分类：添加或管理时填写逗号分隔的标签，列表可按标签或“未分类”筛选，划分子集继承标签。在管理页展开“删除数据集”，默认只删除库中的登记记录；已有服务器目录的源文件始终保留。上传或划分生成的数据可勾选永久删除托管文件，但任何任务或其他数据集仍引用这些文件时会拒绝文件删除。删除一个划分子集不会删除其他子集，划分索引和来源记录继续保留。

侧栏的“数据集”页面可以登记服务器已有目录，或从本地浏览器上传多个 `.npy` 文件（单文件上限 20 GiB，传到运行 WebUI 的机器）。默认读取标准字段，也支持自定义前缀和 `{shard}` 分组。管理页面显示样本数、字段、数组形状、类型和大小，支持搜索、重命名、重新检查、归档和恢复；归档不删除文件。

在数据集管理中展开“基于此数据集重新划分”，填写训练／验证／测试比例及随机种子，选择随机或原顺序划分。比例为 0 的子集不生成；样本太少不能生成非空子集时会报错。输出为独立 npy 副本，原数据不变；能量、力、原子电荷、总电荷等字段按同一索引划分，单位与梯度转换只做一次。创建任务可直接选择库中的数据集；选择生成的训练集时自动带入同次划分的验证集与测试集。

数据集库和上传文件保存在 `MLPUI_RUNS_DIR/datasets/`。划分的 `split-<ID>/split.json` 记录比例、随机种子与来源，`train_indices.npy` 等记录相对于直接源数据集的样本索引。集群重启时保持同一个运行目录即可继续使用。划分需要额外磁盘空间；网页显示完成后再关闭 Slurm 作业。上传中断后可在当前页面重试，刷新页面后的未完成记录可归档并重新上传。对于时间相关轨迹，可按原顺序切分或分别登记不同轨迹分组。

可选总电荷硬约束：训练或评估电荷时勾选“总电荷硬约束”，在文件映射中分别填写原子电荷标签（`charges.npy`）和每个结构的总电荷 Q（`charge.npy`，形状 `[结构数]` 或 `[结构数, 1]`，单位 e）。分组默认文件名分别为 `qm_charge_{shard}.npy` 和 `total_charge_{shard}.npy`。中性结构也要显式提供 Q=0，不能省略文件。约束会随 checkpoint 保存；评估时同样提供 Q。无需新增依赖。

独立评估已有模型：在“新建任务”将任务类型设为“评估已有模型”，填写模型文件绝对路径和相匹配的模型结构配置，再选择 npy 数据目录或分组。勾选能量、力评估项并选择 GPU 后开始。无需设置 epochs 或学习率；评估只遍历数据一次，不更新权重。详情页显示 MAE、RMSE、MSE，完整结果保存在任务目录的 `evaluation.json`，页面提供 JSON 链接。力指标按所有原子的笛卡尔分量汇总，单位沿用数据换算设置。评估与训练共用自动调度队列。

结果保存在指定的 `MLPUI_RUNS_DIR/<任务ID>/`：

- `config.json`：本次任务的参数和数据映射。
- `status.json`：进度与已完成 epoch 的损失记录。
- `train.log`：训练输出和错误。
- `model.pt`：完成训练或通过界面正常停止后保存的权重。
- `checkpoints/epoch_000010.pt` 等：每隔 N 个完整 epoch 保存的权重，保留最新 K 个，最终 `model.pt` 不占 K 的名额。任务详情可下载这些文件，异常结束后已有文件也仍可使用。

结束使用时，先点击界面“停止训练”（如果仍在运行），等待状态变成“已停止”并出现模型下载按钮。确认模型存在后，再从登录节点执行 `scancel JOB_ID` 释放 GPU。关闭本机隧道即可断开访问。

**定期 checkpoint 保存权重，不包含 optimizer/RNG 的精确断点续训状态。** 直接 `scancel`、节点故障或到达作业时限可能丢失最近一次成功保存之后的训练进展。已有定期 checkpoint 或最终 `model.pt` 可以用作下一任务的初始权重，但优化器会重新创建。新文件写入成功后才清理超过保留数量的旧文件；长训练仍应预留时间，主动停止并保存。

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


## 9. 任务队列与多 GPU

训练和评估提交后进入队列，每张当前可见 GPU 同时执行一个任务，空闲后按提交顺序接续；多张 GPU 可同时运行不同任务。CPU 任务单独串行。GPU 任务不会自动改用 CPU，等待中的任务可点“取消排队”。详情页显示实际分配的逻辑 GPU 编号。

当前脚本默认 `#SBATCH --gres=gpu:1`，因此 GPU 任务串行。若需要两个任务并行，可按集群配额把它改成 `#SBATCH --gres=gpu:2`，同时增加适当的 CPU 和内存资源，然后重新提交 WebUI 作业。队列只使用本次 Slurm 分配给 WebUI 的可见 GPU，不会额外 sbatch，也不是把一个模型分布到多张卡上训练。

调度器要求候选 GPU 至少有 90% 空闲显存，并在可读取 nvidia-smi UUID 进程信息时排除有其他计算进程的 GPU；无法读取进程信息时使用显存检查。这个判断不保证任意模型都能装入显存。已分配的槽位直到子进程实际退出才释放。

队列保存在原运行目录：重启相同目录的 WebUI 后继续处理未开始的任务；已经运行的任务标记中断，不会自动重新训练。关闭浏览器不影响调度，结束 Slurm 作业则暂停执行，需重新启动 WebUI 才能接续。
