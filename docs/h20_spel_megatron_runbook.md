# H20 上运行 SpEL Megatron 实验排查记录

本文档整理当前目标、已经验证的事实、失败根因和下一步执行方案。不要在本文档中写入 HPC 密码、GitHub token、SSH 私钥或其他敏感信息。

## 目标

在学校 H20 Slurm 集群上运行 `Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spel/` 下的 SpEL 训练脚本，并逐步扩展到论文风格的小规模预训练实验：

- 模型宽度：`256`、`512`
- 序列长度：`4096`
- 层数：`28`
- head dim：`128`
- FFN hidden size：`3 * hidden_size`
- token budget：先 smoke，再到 `1B` tokens
- 数据：后续使用 `allenai/olmo-mix-1124` 按组成权重抽样

## README 给出的 Megatron-LM 基线要求

`Megatron-LM/README.md` 说明当前 Megatron-LM 目录应当是完整的 NVIDIA Megatron-LM dev 分支：

```bash
git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .[mlm,dev]
```

因此正确的工程形态应当是：

```text
完整原版 NVIDIA Megatron-LM dev 分支
+ SpEL optimizer 相关 patch
+ Spectral-Sphere-Optimizer 运行脚本
```

不能只依赖当前残缺的 `SSO_test/Megatron-LM` 目录直接运行。

## 当前服务器状态

已确认 SSH key 登录可用：

```bash
ssh hpc2021 "hostname && whoami"
```

H20 节点此前已经恢复可用，之前的极小 PyTorch/SpEL smoke 曾在 H20 上完成：

```text
host: SPG-7-1
gpu: NVIDIA H20
torch.cuda.is_available(): true
optimizer: spel
exit: 0
```

当前主要问题不是 H20 GPU，而是 Megatron-LM 源码结构和数据准备。

## 已测试的 SpEL 运行文件

远端原项目目录：

```text
~/projects/SSO_test
```

SpEL 脚本目录：

```text
~/projects/SSO_test/Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spel
```

目录中有：

```text
run_true_spel_width_lr_sweep_from_h2048_ref.sh
spel.sh
```

### Dry run 结果

命令：

```bash
cd ~/projects/SSO_test
WORKSPACE=$HOME/projects/SSO_test \
MEGATRON_PATH=$HOME/projects/SSO_test/Megatron-LM \
DRY_RUN=1 \
bash Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spel/run_true_spel_width_lr_sweep_from_h2048_ref.sh
```

结果：成功生成 sweep manifest。

```text
widths: 256 512
lrs: 1e-3 3e-3 5e-3 7e-3 9e-3
train_tokens: 1000000000
train_iter: 3815
seq_length: 4096
layers: 28
head_dim: 128
global_batch: 64
```

manifest 路径：

```text
~/projects/SSO_test/results/optimizer_arena_v2/true_spel_mup_width_lr_sweep_from_h2048_ref/sweep_manifest.tsv
```

说明 wrapper 脚本本身可以展开参数。

### 直接运行 spel.sh 的第一个失败点

命令：

```bash
cd ~/projects/SSO_test
WORKSPACE=$HOME/projects/SSO_test \
MEGATRON_PATH=$HOME/projects/SSO_test/Megatron-LM \
TRAIN_TOKENS=4096 \
GLOBAL_BATCH=1 \
MICRO_BATCH=1 \
WIDTH=256 \
LR=1e-3 \
bash Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spel/spel.sh
```

失败：

```text
No .bin files found under /home/u3013198/projects/SSO_test/data/merged_data/train.
```

说明当前还没有 Megatron indexed dataset：

```text
data/merged_data/train/*.bin
data/merged_data/train/*.idx
data/merged_data/valid/*.bin
data/merged_data/valid/*.idx
```

## 当前原项目 Megatron-LM 的根因

`~/projects/SSO_test/Megatron-LM` 不是完整 Megatron-LM dev 分支。测试：

```bash
cd ~/projects/SSO_test/Megatron-LM
source ~/envs/sso_h20/bin/activate
python pretrain_gpt.py --help
```

失败：

```text
ModuleNotFoundError: No module named 'megatron.core.models'
```

缺失或不完整的关键路径包括：

```text
Megatron-LM/gpt_builders.py
Megatron-LM/megatron/core/models/gpt/gpt_model.py
Megatron-LM/megatron/core/tokenizers/text/utils/build_tokenizer.py
```

本地和远端的 `Megatron-LM/megatron/core/models` 都不完整，不能直接作为训练 Megatron 使用。

## 完整 Megatron-LM clone 验证

已在有公网的 `hpc2021-io1` 新建完整 clone：

```text
~/projects/Megatron-LM-dev-spel-v2
```

来源：

```bash
git clone -b dev --depth 1 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-dev-spel-v2
```

验证完整 clone 包含：

```text
megatron/core/models/gpt/gpt_model.py
gpt_builders.py
```

这解决了原项目中 `megatron.core.models` 缺失的问题。

## 环境依赖记录

远端 Python env：

```text
~/envs/sso_h20
```

已补装过的数据/依赖包：

```bash
pip install datasets zstandard omegaconf
```

完整 clone 测试时需要加载 module：

```bash
module purge
module load python/3.12.1
module load cuda/12.4
source ~/envs/sso_h20/bin/activate
```

如果不加载 `python/3.12.1`，Triton 可能报：

```text
fatal error: Python.h: No such file or directory
```

加载 module 后，该 `Python.h` 问题消失。

## SpEL patch 迁移结论

不能把当前 `SSO_test/Megatron-LM` 中的这些文件整文件覆盖到完整 Megatron dev clone：

```text
megatron/training/arguments.py
megatron/training/training.py
megatron/core/utils.py
megatron/core/transformer/transformer_config.py
```

原因是当前 patch 基于旧/不完整 Megatron 结构，整文件覆盖会带来版本不兼容。例如：

```text
ModuleNotFoundError: No module named 'megatron.core.models.retro'
ImportError: cannot import name 'log_single_rank' from partially initialized module 'megatron.core.utils'
```

正确做法是最小 rebase：

1. 以完整 NVIDIA Megatron-LM dev 分支为 base。
2. 迁移 SpEL 相关新增文件：

```text
emerging_optimizers/
megatron/core/optimizer/spel.py
megatron/core/optimizer/spectral_ball_optimizer.py
megatron/core/optimizer/muon_ball_optimizer.py
megatron/core/optimizer/mup_adamw.py
```

3. 对完整 dev 的原生文件做最小 patch，而不是整文件覆盖：

```text
megatron/core/optimizer/optimizer_config.py
megatron/core/optimizer/__init__.py
megatron/training/arguments.py
megatron/training/training.py
megatron/core/transformer/transformer_config.py
megatron/core/utils.py
```

4. 逐项验证：

```bash
python pretrain_gpt.py --help
python pretrain_gpt.py --help | grep spel
```

5. 再让 `spel.sh` 指向完整 patched clone：

```bash
MEGATRON_PATH=$HOME/projects/Megatron-LM-dev-spel-v2
```

## 数据准备状态

目标数据集：

```text
allenai/olmo-mix-1124
```

已确认 Hugging Face dataset config：

```text
dclm
arxiv
pes2o
starcoder
algebraic-stack
open-web-math
wiki
```

字段包含 `text`，可以用 Hugging Face `datasets` streaming 逐组件抽样。

论文表格对应大致 token 权重：

```text
dclm:             3.70T
arxiv:            20.8B
pes2o:            58.6B
starcoder:        83.0B
algebraic-stack:  11.8B
open-web-math:    12.2B
wiki:             3.66B
```

1B tokens 需要先流式抽样为 JSONL，再用 Megatron 预处理为 `.bin/.idx`。

推荐数据准备节点：

```text
hpc2021-io1
```

原因：`hpc2021-io1` 有公网，适合下载/流式读取 Hugging Face 数据。H20 计算节点只负责训练。

## 推荐下一步

### 第 1 步：整理 SpEL patch 到完整 Megatron clone

目标目录：

```text
~/projects/Megatron-LM-dev-spel-v2
```

不要继续修：

```text
~/projects/SSO_test/Megatron-LM
```

因为它不是完整 Megatron-LM。

### 第 2 步：先跑 import/argparse smoke

```bash
module purge
module load python/3.12.1
module load cuda/12.4
source ~/envs/sso_h20/bin/activate

cd ~/projects/Megatron-LM-dev-spel-v2
python pretrain_gpt.py --help
python pretrain_gpt.py --help | grep -E "spel|spectral-ball|muon-ball|spectral-mup-init"
```

### 第 3 步：准备极小 Megatron 数据

先不要直接做 1B tokens。先准备一个很小 JSONL，比如几千到几万 tokens，预处理成：

```text
~/projects/SSO_test/data/merged_data/train/*.bin
~/projects/SSO_test/data/merged_data/valid/*.bin
```

### 第 4 步：改 `spel.sh` 指向完整 Megatron clone

运行时设置：

```bash
WORKSPACE=$HOME/projects/SSO_test
MEGATRON_PATH=$HOME/projects/Megatron-LM-dev-spel-v2
```

### 第 5 步：H20 Slurm smoke

先用极小参数：

```bash
WIDTH=256
TRAIN_TOKENS=4096
GLOBAL_BATCH=1
MICRO_BATCH=1
SEQ_LENGTH=128
TRAIN_ITER=1
TRANSFORMER_IMPL=local
ATTENTION_BACKEND=unfused
```

smoke 通过后，再恢复论文风格参数：

```text
SEQ_LENGTH=4096
GLOBAL_BATCH=64
TRAIN_TOKENS=1000000000
WIDTHS="256 512"
```

## 当前结论

`spel` 脚本本身可以展开参数，H20 GPU 也不是当前阻塞点。真正阻塞点是：

1. 当前 `SSO_test/Megatron-LM` 不是完整 Megatron-LM dev 分支。
2. SpEL patch 需要 rebase 到完整 Megatron-LM，而不能整文件覆盖。
3. 还没有准备 Megatron `.bin/.idx` 数据。

完成 SpEL 最小 rebase 和小数据预处理后，才能提交真正的 H20 smoke job。
