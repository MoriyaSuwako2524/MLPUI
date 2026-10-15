# 更新 Sooner 上的四目标训练支持

此次代码新增 `mm_potential_gradient`，训练时选择 energy、forces、
mm_potential、mm_potential_gradient 四项。现有 E/F 配置和 checkpoint 架构
仍可使用；更新代码不会自动恢复或启动训练。数学定义与适用边界见
[TensorNet-QMMM](tensornet-qmmm.md#fixed-response-potential-gradient)。

## 1. 在已登录的 Sooner 保存本地工作并更新

```bash
cd /home/moriya2524/mlp/sources/MLPUI
git status --short
git branch --show-current
git fetch origin main
git log -3 --oneline origin/main
git pull --ff-only origin main
```

应在现有 main 分支执行。若存在本地改动，先备份或提交这些改动；若 pull
报告覆盖风险或无法快进，停下来核对差异，不要使用 `reset --hard` 或强制覆盖。
本地修改的 Slurm 脚本、环境路径和数据目录需要保留。

此次不增加 Python 依赖。使用原 WebUI 的 Python 环境，并确认导入的是更新后的
仓库；不要把 Windows 虚拟环境复制到集群，也不要重新安装 PyTorch。

```bash
# PYTHON 应设为原 WebUI 作业实际使用的解释器绝对路径。
"$PYTHON" -c 'import mlpui; from mlpui.backends import get_backend; print(mlpui.__file__); print(get_backend("tensornet_qmmm").training_targets)'
"$PYTHON" -m pytest tests/test_qmmm_gesp.py tests/test_qmmm_targets.py tests/test_prepare_cm_fourtarget.py tests/test_qmmm.py tests/test_qmmm_review.py tests/test_web_qmmm.py -q
# 有 Node.js 时还可执行：node --test tests/web_ui.test.cjs
git rev-parse HEAD
```

## 2. 重启 WebUI，使新代码生效

先检查现有 WebUI 中是否有运行/排队任务及其 Slurm allocation。重启会影响同一
服务下的其他 worker；有任务运行时不要直接终止服务。空闲时按原启动方式重启
已确认的 WebUI 作业，保留原 Python、运行目录、站点资源配置和访问方式。
不要同时让两个 WebUI 写同一 runs 目录。不要删除原 E/F run。

若原先使用仓库的 `scripts/slurm/mlpui_web.sbatch`，沿用原环境变量和已修改的
站点脚本重新提交，确认新 allocation 的节点，再更新本机转发目标。旧节点名
不能保证继续有效。重新连接后查看 `/api/presets`，确认 TensorNet-QMMM 支持
`mm_potential_gradient`；该动作只检查服务版本，不创建训练。

## 3. 核验基线后才建立新数据与新 run

已有数据集摘要仅说明 residual_only、UMA-m1p1、charge-response、Q-2，
没有提供可核验的基线常数、投影约定及 E/F 文件哈希。必须从原预处理脚本和
实际文件建立 baseline contract，不能只根据标签名称推定。

```bash
"$PYTHON" -m scripts.prepare_cm_fourtarget --help
```

准备器要求原 mm10A NPY、完整 ESP NPY、已保存的气相电荷预测及 metadata、
原 E/F train/validation 目录、已核验的 contract JSON，以及一个不存在的新输出
目录。参数和 contract 字段详见 TensorNet-QMMM 文档。Pete 路径不代表 Sooner
必然可见；需使用已经授权且完成校验的本地副本。准备器不会重算 QM。

保持原11200/1600/1600/1600划分。数据准备器仅写 train/validation，原E/F逐字节
保留，ESP/GESP减去同一气相电荷库仑基线，初始权重从train-only逐帧RMS得到。
先用训练集小批次和验证小批次跑四目标 smoke并回读checkpoint，再建立独立
全量run；不附加test，不用test/guard选择权重或模型。现有E/F run保持独立。

在部署版本、基线身份和真实四目标smoke确认之前，不应把本地合成测试通过
表述为正式训练成功。checkpoint保存模型权重/配置，不包含优化器恢复状态或
外部基线数组。
