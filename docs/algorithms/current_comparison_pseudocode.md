# 现有对比算法：运行流程与伪代码

依据 2026-10-26 的代码与 width-256 / 3B 配置整理。下文描述**实际实现**，不替代历史论文的 [MCSD / MCSD-PGD 定义](primary_mcsd_mcsd_pgd.md)。

阅读顺序：第 1–7 节看流程与参数；[第 8 节](#8-代码实现) 展开对应的 Python 源码。

## 1. 核心区别

| 方法 | 更新方向 | 谱估计 | 径向校正的位置 |
|---|---|---|---|
| **MuonSphere** | 动量的近似 matrix sign | BF16 cold PI10 | 更新前 |
| **SSO** | 用标量乘子使 matrix-sign 方向近似切向 | BF16 cold PI10 | 更新前 |
| **MCSD-TP** | 切平面投影 → matrix sign → 再次切平面投影 | FP32 warm block-8，K10 / K20 | 更新后 |
| **MCSD–P²GD 型混合** | gap 大时用 MCSD-TP；gap 小时用顶部簇方向 | 与匹配的 MCSD-TP 相同 | 两个分支均在更新后 |

**当前混合方法不做 full-SVD 球面投影。** 它使用顶部簇方向和近似径向返回，与旧版 ambient PGD 不同。

## 2. 共同流程与符号

伪代码针对一个 `m × n` 矩阵组件。`X` 为 FP32 master weight，`M` 为 Nesterov 动量，`D` 包含下降符号。

```text
每个训练步：
    BF16 前向、反向、梯度累积 → 全局梯度裁剪（阈值 1）
    embedding、output head、归一化等辅助参数 → AdamW

    对每个隐藏矩阵：
        B = 0.9 * B + 0.1 * G              # EMA 动量，初值为零
        M = 0.9 * B + 0.1 * G              # 使用本步更新后的 B
        按实际拆分规则取得各组件 X、M
        R = sqrt(m / n)                    # 目标谱半径
        h = 当前参数组学习率 * R            # 含形状缩放的步长
        按所选算法更新 X

    FP32 master weight → 同步至 BF16 模型
```

隐藏二维矩阵的 `wd_mult=0`。辅助优化器配置名虽为 `adam`，实际为 AdamW：`betas=(0.9, 0.95)`、`eps=1e-8`、weight decay 0.1；bias / norm 等按分组排除衰减。

下文共用三个算子：

| 算子 | 定义 |
|---|---|
| `norm_F(H)` | Frobenius 范数 |
| `P(H, u, v)` | 将 u、v 单位化后，计算 `H - (u.T @ H @ v) * (u @ v.T)`，使用 FP32 |
| `SIGN(H)` | 先做 Frobenius 归一化，再执行 **8 次 BF16 Newton–Schulz** |

`P` 在顶部奇异值简单、向量精确时才是真实切平面投影。`SIGN` 是 matrix sign 的近似，当前没有额外的 post-msign 谱范数归一化。

## 3. MuonSphere 与 SSO

两个基线都调用以下 PI；每次从全 1 向量开始，**不使用 warm start**。

```text
COLD_PI10(X):
    W = BF16(X)
    v = 全 1 列向量
    repeat 10 times:
        v = normalize(W.T @ (W @ v))
    u = normalize(W @ v)
    return u.T @ W @ v, u, v
```

### MuonSphere

```text
MUONSPHERE_STEP(X, M, R, h):
    Mbar = M / max(norm_F(M), 1e-8)
    sigma, u, v = COLD_PI10(X)
    X = X * R / (max(sigma, 0) + 1e-8)     # pre-radial
    return X - h * FP32(SIGN(Mbar))
```

代码名为 `MuonBall` / `muon_ball_dist`。方向不使用 u、v，也不额外投影到切平面。

### SSO

SSO 寻找标量 `lambda`，使 `F(lambda) = <Theta, SIGN(Mbar + lambda * Theta)>_F` 接近零。

```text
SSO_STEP(X, M, R, h):
    Mbar = M / max(norm_F(M), 1e-8)
    sigma, u, v = COLD_PI10(X)
    X = X * R / (max(sigma, 0) + 1e-8)     # pre-radial
    Theta = u @ v.T

    从 lambda=0 出发，若满足容差则结束；否则以 1e-3 为初始增量
    最多扩展 10 次寻找异号区间
    找到区间后最多二分 20 次，目标 abs(F(lambda)) <= 2e-4
    若搜索失败，取 lambda=0；若二分未收敛，取已见残差最小值

    return X - h * FP32(SIGN(Mbar + lambda * Theta))
```

SSO 搜索过程中会**多次调用 SIGN**。两个基线均没有 post-step 返回，因此更新后通常不严格满足谱半径约束，下一步再校正。

## 4. MCSD-TP 与混合方法

二者共用返回点的近似谱缓存 `cache=(U, s, V)`，以及第 5 节的 `RETURN`。无缓存时先执行 `RETURN(X, R, None, K)` 初始化；若本步学习率为零，仅初始化，不构造 trial。

### MCSD-TP control

```text
MCSD_DIRECTION(M, cache):
    u, v = cache.U[:, 0], cache.V[:, 0]
    H = P(M, u, v)
    H = H / max(norm_F(H), 1e-7)
    return -P(SIGN(H), u, v)

MCSD_TP_STEP(X, M, R, h, cache, K):
    D = MCSD_DIRECTION(M, cache)
    return RETURN(X + h * D, R, cache.V, K)
```

matrix sign 不保持切向性，因此其后再投影一次。control 不根据 gap 切换方向；即使接近重根，也继续使用缓存中的第一对向量。

### MCSD–P²GD 型混合

```text
MIXED_STEP(X, M, R, h, cache, K):
    s = cache.s
    gap = 1 - s[1] / s[0]

    if gap > tau:
        D = MCSD_DIRECTION(M, cache)
        alpha = h
    else:
        active = count((s[0] - s[i]) / s[0] <= cluster_tol)
        q = max(1, min(4, len(s), active))
        Z = -M / max(norm_F(M), 1e-12)
        D = CLUSTER_DIRECTION(Z, cache.U[:, :q], cache.V[:, :q])
        alpha = c * h

    return RETURN(X + alpha * D, R, cache.V, K)
```

三个参数职责不同：`tau` 决定**何时切换**，`cluster_tol` 决定**选哪些顶部方向**，`c` 决定**近区域分支步长**。

近区域顺序为：**归一化原 Nesterov 动量 → 簇方向投影 → 乘步长**。投影后不再归一化、不匹配 MCSD 方向长度，也不计算无用的 MCSD matrix sign。`c=8` 不表示实际位移是 MCSD 的 8 倍，更不是分支比例。

### 顶部簇方向

```text
CLUSTER_DIRECTION(Z, Uq, Vq):
    A = Uq.T @ Z @ Vq
    A = (A + A.T) / 2
    a, O = eigh(A)                         # 特征值升序
    b = min(a, 0)                         # 逐元素截断
    b[-1] = 0                             # 最大特征值置零
    B = O @ diag(b) @ O.T
    return Z + Uq @ (B - A) @ Vq.T
```

只分解最多 `4 × 4` 的对称矩阵：去掉所选顶部簇的一阶扩张，同时至少保留一个零变化方向。只有使用**完整真实重根子空间**时，它才是谱球面真实切锥投影；当前近似簇与 `q<=4` 截断属于 P²GD 型变体。

## 5. 共享谱估计与径向返回

### Warm block-8 Ritz

```text
WARM_RITZ(Y, previous_V, K):
    scale = max(norm_F(FP32(Y)), FP32_tiny)
    A = FP32(Y) / scale
    b = min(8, rows(Y), columns(Y))
    V = previous_V，或局部 seed=4321 生成的 n × b 随机矩阵
    V = thin_QR(V).Q

    repeat K times:
        U = thin_QR(A @ V).Q
        V = thin_QR(A.T @ U).Q

    Ubase, H = thin_QR(A @ V)              # H 为 b × b
    Uh, s, Vh = SVD(H)
    return Ubase @ Uh, scale * s, V @ Vh.T
```

矩阵乘法、QR、小 SVD 均为 FP32，关闭 TF32 / autocast；CUDA 小 SVD 使用 `gesvd`。这是固定子空间迭代，不是 block-Krylov，也不是逐方向 deflation。

### 返回与缓存

```text
RETURN(Y, R, previous_V, K):
    U, s, V = WARM_RITZ(Y, previous_V, K)
    若 s[0] <= 1e-7，报错
    factor = R / s[0]
    X_next = factor * Y
    cache_next = (U, factor * s, V)         # 返回后的近似谱
    return X_next, cache_next
```

径向缩放不改变奇异向量和相对 gap，所以可以直接复用缩放后的缓存。**gap 的变化来自 trial 更新，不是径向返回。**

缓存有效时，每组件每步仅对 trial 估计一次谱，包含一次至多 `8 × 8` 的 SVD；初始化另需一次。若外部修改权重或恢复 checkpoint，应使旧缓存失效。

## 6. 当前实验配置

| 方法 | 谱估计 | 近区域参数 |
|---|---|---|
| MCSD-TP control，两组 | warm block-8，K10 / K20 | 不切换 |
| MCSD–P²GD 型混合，两组 | warm block-8，K10 / K20 | `tau=5e-5`，`cluster_tol=2e-3`，`q<=4`，`c=8` |
| MuonSphere，一组 | BF16 cold PI10 | 无 |
| SSO，一组 | BF16 cold PI10 | 无 |

| 共同设置 | 取值 |
|---|---|
| 模型 | 28 层，width 256，FFN 中间宽度 768，2 heads，head dimension 128 |
| 结构 | SwiGLU、MHA、QK normalization、RoPE base `1e6`、pre-RMSNorm、untied embedding/head、无 dropout |
| 数据 / 起点 | 现有 3B indexed OLMo 数据源，从头训练；不是历史论文冻结的 1B 样本协议 |
| 训练 | 5723 steps，约 3B tokens；batch 128，micro-batch 4，序列长度 4096 |
| 学习率 | 峰值 `1.5e-2`，warmup 250 steps，cosine 衰减至 `1.5e-3` |
| 验证 / 资源 | 每 250 steps 验证 20 batches，末尾再验证；seed 1234，每任务 1 张 H20 |

这是**相同学习率下的匹配比较**，不是各方法独立调参后的最优比较。任务映射、启动方式和结果分别见 [3B 配置](../experiments/cluster_selected_3b_20261025.md)、[启动脚本](../../slurm/cluster_selected_3b_20261025.sbatch) 与 [前一轮 1B 结果](../experiments/cluster_candidates_1b_20261024.md)。

## 7. 实现边界

- **近似可行性：** 若第一奇异值被低估，径向返回后真实谱范数仍可能超过 R；缓存中 `s[0]=R` 不是精度证明。
- **当前不做：** 在线误差门控、自动 full-SVD fallback、函数值回溯。超过 top-4 容量仅记录，不扩维；精确审计在训练后离线完成。
- **理论定位：** 近似谱簇加径向返回不能直接继承标准投影型 P²GD 的随机收敛结论。
- **版本区分：** 旧论文 top-8 是投影秩；当前 block-8 是谱估计维数，top-4 是方向簇维数。本文 control 也不是旧论文 MCSD。

<details>
<summary>复现时需注意的配置与日志细节</summary>

- 当前 cluster 分发覆盖旧的 `exact_svd_gap_only` 等宿主参数；实际由 `cluster_spectrum_mode=warm_block8` 决定。`warm_start_uv=0` 不会关闭 block 子空间缓存。
- `pgd branches` 实际指顶部簇分支；`post projections` 指径向返回；`cluster flatten` 仅为诊断，未实际截平奇异值。分支率按**矩阵组件决策次数**计算，不按训练步数计算。
- QKV 按 head 拆分，每个组件为 `128 × 256`。FC1 拆分虽默认允许，但当前 FP32 master-copy 路径未复制 `is_fc1` 标记；近期 1B 日志为 252 组件/步，与每层 9 组件、FC1 未拆分一致。不要将这些运行描述为 gate/up 已独立优化。
- 半径按实际组件形状计算：未拆分 FC1 为 `1536 × 256`，R=`sqrt(6)`；若改为两个 `768 × 256` 组件，R 各为 `sqrt(3)`，应作为新配置记录。

</details>

## 8. 代码实现

以下是**当前源码的核心节选**，保留实际变量名，省略日志、参数检查及不启用的分支。片段依赖所在函数的上下文，不是独立脚本；点击函数链接可查看完整实现。正文的 `X/M/cache` 分别对应代码中的 `W/M/current_spectral_cache`。

| 功能 | 文件 |
|---|---|
| 动量与参数更新 | [orthogonalized_optimizer.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py) |
| MuonSphere | [muon_ball.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py) |
| SSO、PI、SIGN、半径缩放 | [spectral_ball_utils.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py) |
| 当前 control / mixed | [spectral_cluster_radial.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py) |
| 顶部簇方向 / 谱估计 | [spectral_p2gd.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_p2gd.py) / [warm_block_ritz.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/warm_block_ritz.py) |
| 宿主分发与缓存 | [spel_pgd_same_projection.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py) |

<details>
<summary>8.1 共同动量、半径与参数写回</summary>

入口：[OrthogonalizedOptimizer.step](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py#L136)。`exp_avg` 是保存的 B；下方 `grad` 经 Nesterov 处理后就是送入算法的 M。

```python
exp_avg.lerp_(grad, 1 - group["momentum_beta"])

if self.use_nesterov:
    grad = grad.lerp(exp_avg, group["momentum_beta"])
else:
    grad = exp_avg
```

矩阵算子返回已计入形状缩放的更新，父优化器执行最后的减法：

```python
group_kwargs = {k: v for k, v in group.items() if k != "params"}
grad = self.orthogonalize(p, grad, **group_kwargs)
```

```python
p.add_(grad, alpha=-group["lr"])
```

当前形状缩放见 [compute_target_radius](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L470) 与 [get_spectral_ball_scale_factor](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L480)：分别返回 `sqrt(m/n)` 的半径和更新缩放。二者用途不同，学习率缩放只应用一次。

</details>

<details>
<summary>8.2 切平面投影 P：FP32 秩一修正</summary>

函数：[project_to_tangent_plane](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py#L79)。输入矩阵 M 与顶部向量 u、v；返回去掉法向分量后的矩阵。

```python
M_fp32 = M.to(torch.float32)
u_col = _as_column_unit_vector(u, eps=eps)
v_col = _as_column_unit_vector(v, eps=eps)
```

通过形状检查后，实际修正为：

```python
coeff = torch.matmul(u_col.transpose(-2, -1), torch.matmul(M_fp32, v_col))
return M_fp32 - coeff * torch.matmul(u_col, v_col.transpose(-2, -1))
```

这里不分解矩阵；只使用传入的向量做矩阵乘法与秩一减法。

</details>

<details>
<summary>8.3 SIGN：8 次 BF16 Newton–Schulz</summary>

函数：[_small_msign](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L37)。输入 G 要求 FP32，内部归一化后转为 BF16：

```python
transpose = G.size(-2) > G.size(-1)
X = G.mT if transpose else G
X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
```

```python
X = X.to(torch.bfloat16)

coeffs = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
]

for i in range(steps):
    if i < 8:
        a, b, c = coeffs[i]
    else:
        a, b, c = coeffs[-1]
    X = _muon_newton_schulz_step(X, a, b, c)

return X.mT if transpose else X
```

每一轮实际调用：

```python
A = X @ X.mT
B = torch.addmm(A, A, A, alpha=c, beta=b)
X = torch.addmm(X, B, X, alpha=1.0, beta=a)
return X
```

`steps=8` 使用全部八组系数。源文件中的旧 FP32 警告注释与转换语句不一致；实际运行精度以这里的 BF16 转换为准。

</details>

<details>
<summary>8.4 基线 cold PI10 与 hard pre-radial</summary>

函数：[power_iteration](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L155)。`steps` 由本次配置显式传入 10。

```python
w = w.to(torch.bfloat16)
v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
for _ in range(steps):
    v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
u = torch.nn.functional.normalize(w @ v, dim=-2)
s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)

return s, u, v
```

随后 [apply_retract](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L208) 的 hard 分支原地修改 W：

```python
if max(sigma, 0.0) + 1e-8 != target_radius:
    scale_factor = target_radius / (max(sigma, 0.0) + 1e-8)
    W.mul_(scale_factor)
return 0.0
```

返回值 `0.0` 是诊断字段，不是更新后的权重；权重已由 `W.mul_` 修改。

</details>

<details>
<summary>8.5 MuonSphere：pre-radial 后计算 SIGN</summary>

函数：[compute_muon_ball_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py#L59)。当前 TP=1，`W_work=W`、`M_work=M`。

```python
M_fp32 = M_work.to(torch.float32)
M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2, -1), keepdim=True).clamp_min(1e-8))
```

```python
sigma, u, v = power_iteration(W_work, steps=power_iteration_steps)
sigma_value = sigma.item()
```

```python
retract_bias = apply_retract(
    W_work, sigma_value, target_radius,
    mode=retract_mode, alpha=retract_alpha, current_lr=current_lr
)
```

```python
Phi = msign(M_fp32, steps=msign_steps)
```

函数返回 `Phi`，组件包装层乘形状缩放，父优化器再乘学习率并减去更新。这里的 `Phi` 是正向 matrix-sign 输出；下降负号来自最终参数写回。

</details>

<details>
<summary>8.6 SSO：乘子搜索与方向构造</summary>

函数：[_compute_single_rank](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L535)。同 MuonSphere 完成动量归一化、PI 和 pre-radial 后，构造 `Theta = u @ v.transpose(-2, -1)`，调用：

```python
lambda_value, converged, residual, iterations = bisection_fn(
    G=M_fp32,
    Theta=Theta,
    initial_guess=0.0,
    initial_step=1e-3,
    tolerance_f=solver_tolerance_f,
    max_iterations=solver_max_iterations,
    max_expansions=10,
    msign_steps=msign_steps,
)
```

```python
Z = M_fp32 + lambda_value * Theta

Phi = msign(Z, steps=msign_steps)

return Phi, retract_bias, sigma_value
```

[find_bracket](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L285) 负责区间扩展；[solve_lambda_with_bisection](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py#L378) 的核心二分判断如下：

```python
λ_mid = 0.5 * (λ_L + λ_R)
f_mid = compute_f_tensor(G, Theta, λ_mid, msign_steps)
```

```python
if abs(f_mid) < abs(best_f):
    best_λ, best_f = λ_mid, f_mid
```

```python
if f_mid < 0:
    λ_L, f_L = λ_mid, f_mid
else:
    λ_R, f_R = λ_mid, f_mid
```

每次 `compute_f_tensor` 都重新执行 matrix sign；达到容差即返回，否则最多执行 20 轮并返回已见最优残差对应的乘子。

</details>

<details>
<summary>8.7 MCSD-TP 与混合路由：当前启用的两个分支</summary>

函数：[compute_cluster_radial_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py#L14)。当前缓存读取与分支选择：

```python
x = W.float()
gap = cache.rel_gap
fallback = cluster_mode != "control" and gap <= gap_threshold_rel
```

MCSD 分支执行：

```python
projected = host.project_to_tangent_plane(M.float(), cache.u1, cache.v1, eps=eps)
projected = projected / torch.linalg.vector_norm(projected).clamp_min(eps)
reference = host.msign(projected, steps=msign_steps).float()
reference = host.project_to_tangent_plane(reference, cache.u1, cache.v1, eps=eps)
direction = -reference
```

近区域分支先从缓存中选择顶部簇：

```python
relative = (cache.s_basis[0] - cache.s_basis) / cache.s_basis[0]
active = int((relative <= cluster_threshold_rel).sum())
cap = 2 if cluster_mode == "fixed2" else 4
rank = min(cap, len(cache.s_basis), active if cluster_mode == "adaptive4" else cap)
```

```python
rank = max(1, rank)
momentum = M.float()
if cluster_direction_mode == "momentum_fro":
    momentum = momentum / torch.linalg.vector_norm(momentum).clamp_min(1e-12)
direction = host.project_spectral_sphere_cone(
    -momentum, cache.u_basis[:, :rank], cache.v_basis[:, :rank]
)
if cluster_direction_mode == "momentum_fro":
    multiplier = float(pgd_lr_scale)
```

本次使用 `adaptive4 + momentum_fro`。安全分支的 `multiplier=1`；近区域分支为 8。旧的 `matched_fro` 长度匹配分支未启用。

</details>

<details>
<summary>8.8 顶部簇修正：小矩阵特征分解</summary>

函数：[project_spectral_sphere_cone](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_p2gd.py#L12)。`z` 是负的归一化动量；`u, v` 为配对的前 q 列子空间。当前运算为 FP32，并在 CUDA 上关闭 TF32。

```python
a = u.T @ z @ v
a = (a + a.T) * 0.5
if q == 1:
    b = torch.zeros_like(a)
else:
    values, vectors = torch.linalg.eigh(a)  # Ascending.
    clipped = values.clamp(max=0)
    clipped[-1] = 0  # Boundary of the sphere, including all-negative A.
    b = (vectors * clipped) @ vectors.T
return z + u @ (b - a) @ v.T
```

`vectors * clipped` 按列缩放，等价于 `vectors @ diag(clipped)`。返回的是方向，函数本身不更新权重、不做径向返回。

</details>

<details>
<summary>8.9 Warm block-8 Ritz：初始化、迭代与小 SVD</summary>

函数：[warm_block8_ritz](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/warm_block_ritz.py#L7)。实际返回 **U、s、Vh**；正文伪代码为易读使用的是 V，二者相差转置。

```python
x = matrix.detach().float()
b = min(8, *x.shape)
if initial_v is None:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    q = torch.randn(x.shape[1], b, device=x.device, dtype=torch.float32,
                    generator=generator)
else:
    if initial_v.shape != (x.shape[1], b):
        raise ValueError("warm basis has incompatible shape")
    q = initial_v.detach().to(device=x.device, dtype=torch.float32)
```

FP32 精度控制与迭代：

```python
if x.is_cuda:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
with torch.autocast(device_type=x.device.type, enabled=False):
    scale = torch.linalg.vector_norm(x).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = x / scale
    q, _ = torch.linalg.qr(q, mode="reduced")
    for _ in range(iterations):
        p, _ = torch.linalg.qr(normalized @ q, mode="reduced")
        q, _ = torch.linalg.qr(normalized.T @ p, mode="reduced")
    p, h = torch.linalg.qr(normalized @ q, mode="reduced")
    kwargs = {"driver": "gesvd"} if x.is_cuda else {}
    uh, s, vh = torch.linalg.svd(h, full_matrices=False, **kwargs)
    return p @ uh, s * scale, vh @ q.T
```

完整函数在 `finally` 中恢复先前的 TF32 / matmul 设置；节选不应脱离该保护直接替换生产函数。

</details>

<details>
<summary>8.10 Trial、径向返回与缓存写回</summary>

位置：[compute_cluster_radial_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py#L14) 的末尾。当前 `decompose(y, cache)` 调用 warm Ritz，使用 `cache.v_basis` 热启动。

```python
alpha = float(effective_lr) * multiplier
y = x + alpha * direction
u, s, vh = decompose(y, cache)
if float(s[0]) <= eps:
    raise ValueError("zero trial has no defined radial return")
factor = target_radius / s[0]
returned = y * factor
next_cache = returned_cache(u, s * factor, vh)
```

`returned_cache` 明确标记近似谱，而不是 exact cache：

```python
def returned_cache(u, s, vh):
    options = dict(is_exact=False, bounds_certified=False,
                   gap_lower=0., gap_upper=1.) if approximate else {}
    return host._spectral_point_cache_from_decomposition(u, s, vh, **options)
```

[缓存构造函数](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py#L221) 从**缩放后的**奇异值计算下一步 gap：

```python
sigma1 = float(S_returned[0].item())
sigma2 = float(S_returned[1].item()) if S_returned.numel() > 1 else 0.0
scale = max(abs(sigma1), eps)
rel_gap = max(0.0, 1.0 - sigma2 / scale)
```

核心函数没有直接把 `returned` 写入 W，而是返回一个等效更新，适配公共优化器接口：

```python
return (x - returned) / float(effective_lr), info
```

宿主 [_compute_component_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py#L2989) 定义 `effective_lr = current_lr * scale_factor`，缓存 `info.next_spectral_cache`，再执行：

```python
return update_unscaled * scale_factor
```

因此父优化器的 `W -= current_lr * update` 最终得到 `returned`，**不会重复施加 trial 步长或 c**。

</details>
