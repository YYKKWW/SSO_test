# K10 混合：第二 seed 与方向机制对照

目的：验证上一轮 K10/c8 相对 MCSD-TP 的 `0.001502` 改善是否可重复，并区分收益来自多方向簇修正，还是仅来自小 gap 时更换方向/步长。

## 六组配置

| 数组后缀 | 方法 | 小 gap 时的方向 |
|---|---|---|
| `_0` | MuonSphere | 参考算法，无混合切换 |
| `_1` | SSO | 参考算法，无混合切换 |
| `_2` | MCSD-TP K10 control | 始终使用 MCSD-TP |
| `_3` | MCSD–P²GD 型 K10/c8 | 归一化动量的自适应 top≤4 簇修正 |
| `_4` | 无修正对照 | 归一化动量，不做切向/切锥修正 |
| `_5` | top-1 对照 | 归一化动量的单一顶部法向修正 |

全部使用新 seed `2345`、width256、28 层、FFN768、head dimension128、sequence4096、global batch128、microbatch4。每组单张 H20，从头训练，不读取 checkpoint。处理预算为 3B（5723 步，实际 3,000,500,224 tokens）；峰值 LR `1.5e-2`，250 步 warmup，cosine 衰减到 `1.5e-3`。沿用上一轮 indexed 数据源和每250步/结束时20 batches 的验证协议。

实验算法统一 warm block-8、FP32 QR/Ritz、K10、gap `5e-5`、簇阈值 `2e-3`；后三组统一系数 `c=8`。模型混合精度与 BF16 矩阵符号保持原配置，约束矩阵不施加额外 weight decay。无 checkpoint；仅保存日志、指标和有上限的离线审计快照。

## 只改变什么

记 `Z = -M / max(||M||_F, 1e-12)`。仅当缓存 gap≤`5e-5` 时，后三组分别使用：

```text
cone:     Q = 顶部簇上的 Frobenius 切锥型修正(Z, U_q, V_q), q≤4
ambient:  Q = Z
tangent1: Q = Z - <Z, u_1 v_1^T> u_1 v_1^T

Y = X + effective_lr * 8 * Q
对 Y 做 K10 warm block-8 / FP32 Ritz
X_next = Y * R / estimated_sigma1(Y)
缓存缩放后的奇异值和对应向量
```

三组都不再次归一化 Q，也不匹配 MCSD 的 Frobenius 长度。安全分支、谱计算和径向返回相同。路由规则相同不代表训练中触发的矩阵/时刻相同，因为轨迹会分化。

`ambient` 不是 PGD，`tangent1` 也不是多方向 P²GD；它们没有集合投影。`cone` 是近顶部簇启发式，不等同于正 gap 点的真实切锥，也没有由本实验自动获得的随机收敛保证。

## 检查与启动

- 新参数：`SPEL_PGD_CLUSTER_NEAR_PROJECTION=cone/ambient/tangent1`，默认 cone，不改变旧实验。
- [启动脚本](../../slurm/cluster_seed_mechanism_3b_20261028.sbatch)；[机制测试](../../scripts/test_cluster_near_projection.py)。
- 本地45项回归测试通过；H20两组各45项CUDA测试及4步短训练均通过：`4346696_4`（02:05）、`4346696_5`（02:03），无NaN/skips。每组1008次组件调用、1008次小SVD、0次原矩阵full SVD/svdvals。
- 短训练没有自然触发小gap分支；三种方向的公式和分支调用由CUDA单元测试覆盖。短训练仅证明集成可运行，不证明算法效果。
- 独立远端快照：`/home/u3013198/projects/cluster-seed-mechanism-3b-20261028`。
- 代码来自已完成的3B快照，只覆盖本次改动；运行前验证 `source.sha256`。不修改其他正在运行的项目。
- 提交前16张H20已有14张分配，其中4张是用户其他任务；本批并发上限1，留出另一张空卡。家目录剩余约29GB。
- 正式数组：`4346705_0`--`4346705_5`，2026-10-28 01:07 HKT提交；上表后缀与任务一一对应。每组预计约17--19小时（参考上一轮），当前串行调度全部约4--5天，不含额外排队。后续可在检查空位后调整并发。
- [提交与校验记录](../results/cluster_seed_mechanism_3b_20261028/submission.json)。本文不把短训练当成算法效果证据。

## 如何判断结果

1. 首先计算同一 seed 内 `混合 - control`、`混合 - MuonSphere`、`混合 - SSO` 的最终验证差，再与 seed1234 对照，不直接把两 seed 的绝对 loss 当成同一验证样本上的差。
2. 对比 cone / ambient / tangent1 的最终 loss、训练耗时、近分支比例、重复率和返回后 gap。只有 cone 稳定优于另两组，才支持“多方向簇修正有额外价值”。
3. 离线审计启动/中期/后期的半径、gap 和方向误差；在线不得偷偷调用原矩阵 full SVD。更新调用数须满足一次8×8小SVD/组件，full-SVD/svdvals为零。
4. 日志沿用历史 `pgd` 分支字段；本批应解释为近分支调用，不是 full-SVD PGD。q统计表示实际修正维数：ambient=0、tangent1=1；容量溢出只用于cone。

本轮不同时改变 K、gap、c 或簇容量。仍使用20 batches 验证，不声称已经完成额外100–200 batches 的固定扩展评估；两个 seed 也不足以确证 `1e-4` 量级的显著优势。
