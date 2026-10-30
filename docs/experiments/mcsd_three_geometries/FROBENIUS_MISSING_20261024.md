# Frobenius sphere 补测登记：2026-10-24

## 提交范围与状态

本轮补齐六组实验，均为 width=256、28 层、Frobenius sphere、固定
`R = sqrt(D_out)`（shape radius，`r=1`）。每个作业使用一张 H20；两条串行
队列使本批最多同时使用两张卡。账号原有的 `4319977` 1B 候选实验数组未修改。

2026-10-24 01:22 HKT 核验：`4320101` 已在 SPG-7-2 运行并完成首个训练
iteration；`4320102` 等待资源，其余四组等待前置作业结束。此时账号合计使用
4 张 H20（原数组 3 张、新批次 1 张），尚无本轮验证结果。

| 任务 | 方法 | Tokens | LR | 动量 | Seed | 前置作业 | 核验状态 |
|---|---|---:|---:|---|---:|---|---|
| 4320101 | MuonH | 3B | 0.010 | 0.95 EMA | 1234 | 无 | Running |
| 4320102 | NorMuon-Sphere | 3B | 0.010 | 0.95 EMA | 1234 | 无 | Pending: Resources |
| 4320103 | MCSD | 3B | 0.007 | 0.90 Nesterov | 1234 | 4320101 | Pending: Dependency |
| 4320104 | MCSD | 3B | 0.012 | 0.90 Nesterov | 1234 | 4320102 | Pending: Dependency |
| 4320105 | MCSD | 6B | 0.010 | 0.95 EMA | 2026 | 4320103 | Pending: Dependency |
| 4320106 | NorMuon-Sphere | 6B | 0.015 | 0.90 Nesterov | 2026 | 4320104 | Pending: Dependency |

依赖为 `afterany`，只负责限制并发，各组都从初始化独立训练，不读取前组
checkpoint。即使前组失败，后组仍会获得运行资格；不能把依赖解释为结果筛选。
按此前单卡耗时，3B 约 18.5 小时、6B 约 37 小时，两条队列各约 74 小时，
不含资源等待，实际速度受节点负载影响。

## 每组回答的问题

### 1. 公平动量对照

`4320101/4320102` 补齐四种方法在同一 `beta=0.95 EMA, LR=0.01, seed=1234`
下的比较。已有 MCSD（4197465，3.467720）和 MCSD-TP（4197469，3.467858）
可作为对应参照；这些数值是 3B 最终验证损失。

旧 `4288197/4288198` 经 Slurm 确认已取消，分别运行约 55/54 分钟，没有完整
3B 结果。新组从头重跑，并逐项核对旧配置一致性。NorMuon 的二阶动量仍为
`beta2=0.95, eps=1e-10`；本轮修改的是一阶动量 beta。

### 2. Nesterov 是否只是学习率未匹配

`4320103/4320104` 与已有 `4227544`（MCSD，0.90 Nesterov，LR=0.01，
最终 loss=3.478060）组成 `{0.007, 0.010, 0.012}` 网格。
数据、预算、seed 和算法实现相同，先检查更合适的步长能否改善 Nesterov 结果。

这里的 Nesterov 是 Muon-style 梯度组合：

```text
M_t = beta * M_(t-1) + (1-beta) * G_t
S_t = beta * M_t + (1-beta) * G_t
```

它不是在新的流形外推点重新计算梯度的 Huang--Wei/FISTA 方法。
本轮不改优化器、不增加 MCSD-unitF 归一化。MCSD-TP 的 Nesterov LR 补测仍待安排。

### 3. 6B 排序能否跨 seed 重现

`4320105/4320106` 只将原 6B 任务的 seed 从 1234 改为 2026，其余配置逐项
核对一致。对应旧任务分别为 `4288199`（MCSD，3.431077）与 `4288211`
（NorMuon-Sphere，3.426804）。

这组继续比较各方法从 3B 选定的配置，学习率和动量不同，不能单独归因于几何。
同一 seed 内两方法共享数据顺序；不同 seed 可能同时改变初始化及样本顺序。
最终应报告每个 seed 的配对差值与均值，两个 seed 仍不足以支持强统计结论。

## 固定协议与可追溯信息

- 冻结训练项目：`/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005`。
- 训练源码 commit：`fc15600a2898d1748c8b34fe8d3945b1036f03ab`，提交前确认工作树干净。
- 配置：`configs/manifold/width256_3b_shape_radius.json`；模型约 73.418M 参数。
- 序列长度 2048，global batch 128，micro batch 4，BF16 训练、FP32 master weights。
- 3B：11,445 steps，实际 3,000,238,080 tokens，warmup 229 steps。
- 6B：22,889 steps，实际 6,000,214,016 tokens，warmup 458 steps。
- cosine decay 至 peak LR 的 10%，auxiliary AdamW LR=0.001。
- 每 200 steps 和训练结束进行验证，每次 40 batches；验证集均使用既有 3B valid index。
- 6B 使用已准备的 `olmo_mix_1124_6b` 训练 index，从初始化开始，不循环 3B 代替。
- 全部关闭 checkpoint；保留日志、完整配置、源码哈希和 Slurm 登记。
- 本轮没有修改训练源码；调度脚本单独放在 H20 用户目录。

补测配置见 [JSON 计划](../../../configs/manifold/frobenius_missing_20261024.json)，
六组任务及配置校验值见 [提交登记表](reports/frobenius_missing_20261024/frobenius_missing_register_20261024.json)。
每组远端日志为 `<project>/logs/<run_dir basename>_<job_id>.out/.err`；完整参数为
登记表 `run_dir` 下的 `resolved_config.json`。

## 检查与未覆盖项

本地标准库检查通过：六组参数、两条依赖链、3B/6B 步数、禁止把 3B 默认路径
误用为 6B、取消任务重跑许可、提交状态不明确时禁止自动重试。
H20 路径检查和六组提交后配置检查通过；两个旧取消作业和两个旧 6B 作业的
实验协议已逐项比对。新增 pytest 用例已写入，本机未安装 pytest，未执行该测试套件。

尚未提交的内容：width384/512 迁移、MCSD-TP 的 Nesterov LR 网格、6B MCSD-TP/
MuonH、真实流形点外推的 LLM 实验。本轮先补以上六个缺项，不把所有因子展开成大网格。
既有完成结果参见 [动量和第二 seed 汇总](reports/followup_20261020/README.md) 与
[6B 最终结果](WIDTH256_6B_RESULTS_20261022.md)。
