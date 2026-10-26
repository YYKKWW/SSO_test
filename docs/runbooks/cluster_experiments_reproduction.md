# 当前 cluster 实验：复现入口

本页对应 `exp` 分支的 width-256 / 1B 与 3B 实验。算法见 [说明文档](../algorithms/current_comparison_pseudocode.md)，数据与结果口径见 [结果汇总](../experiments/cluster_results_20261026.md)。

## 1. 环境与数据

已使用的 H20 环境为 Python 3.12.1、PyTorch 2.6.0+cu124、CUDA 12.4。完整 Megatron 训练环境参考已有 [H20 runbook](h20_spel_megatron_runbook.md) 和仓库环境脚本；仅安装 PyTorch 不足以保证完整 LLM 训练可运行。

启动脚本默认使用 HKU 的 `c_math_gpu` 分区、`nvidia_h20:1`、`python/3.12.1` / `cuda/12.4` 模块和 `$HOME/envs/sso_h20`。其他集群需要调整这些站点设置，不改变算法参数。

需自行准备与实验协议匹配的 OLMo 数据：训练和验证的 Megatron `.bin/.idx` 前缀，以及本地 OLMo-2 tokenizer。仓库不附带数据集。两轮实验复用同一份 3B indexed 数据源；1B 表示处理预算，不表示切换到旧论文的 1B 样本文件。

## 2. 先做针对性测试

在已安装项目依赖的环境中，从仓库根目录执行：

```bash
export PYTHONPATH="$PWD/Megatron-LM:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
P2GD_TEST_DEVICE=cpu python -B -m unittest discover -s scripts -p 'test_cluster_*.py' -v
```

本次发布目录的 38 个测试通过。它们验证方向、步长、返回点缓存、小 SVD 次数和配置匹配，不等于完整 LLM 训练测试。Slurm 启动脚本会在 H20 上重新执行对应 CUDA 预检。

## 3. 设置路径与源码清单

以下操作在 H20 登录节点的仓库根目录执行；路径占位符必须替换为实际文件。

```bash
export PROJECT_DIR="$PWD"
export ENV_DIR="$HOME/envs/sso_h20"
export TRAIN_DATA_PREFIX="/path/to/indexed/train/olmo_train_text_document"
export VALID_DATA_PREFIX="/path/to/indexed/valid/olmo_valid_text_document"
export TOKENIZER_MODEL="/path/to/OLMo-2-tokenizer"

mkdir -p logs
git ls-files -z Megatron-LM scripts slurm | xargs -0 sha256sum > source.sha256
```

`DATA_PREFIX` 不带 `.bin` 或 `.idx` 后缀。源码清单用于每次启动前检查本次 checkout；它不是原始 H20 运行的远端清单。提交后不要修改同一运行目录的代码，建议每轮使用独立 checkout。

## 4. 选择一轮提交

**3B 六组实验：** K10/K20 的 control 与混合配对，加 MuonSphere/SSO，最多同时占用 3 张 H20。

```bash
sbatch --export=ALL --array=0-5%3 slurm/cluster_selected_3b_20261025.sbatch
```

**1B 十二组网格：** 两种 LR，加 c=4/8/12 和 K40 消融，最多同时占用 3 张 H20。

```bash
sbatch --export=ALL --array=0-11%3 slurm/cluster_candidates_1b_20261024.sbatch
```

选择所需的一轮即可，不必同时提交。3B 是从头训练，非 1B checkpoint 续训；两轮均关闭 checkpoint。

## 5. 检查结果

每组输出位于 `results/<array_id>_<index>/`：

| 文件 | 用途 |
|---|---|
| `launch_config.json` | 实际参数与有效算法路径 |
| `train.log` | train/validation loss、分支与迭代统计 |
| `training_wall_seconds.txt` | 含验证、不含预检及离线审计的运行时间 |
| `TRAINING_COMPLETED` | 训练命令正常结束 |
| `COMPLETED` | 包括后处理在内全部正常结束 |
| `snapshots/spectral_call_counts.json` | 实验分支的小 SVD / full SVD 调用计数 |
| `snapshots/offline_audit.json` | 实验分支的抽样精度审计 |

基线不生成实验分支的快照/谱审计文件。先确认 Slurm 退出码、最终验证以及完成标记，再判断训练结果；不要把队列消失直接当成成功。
