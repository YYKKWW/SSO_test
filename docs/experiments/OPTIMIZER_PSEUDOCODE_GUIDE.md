# 优化器对照：伪代码与核心实现

本文说明本项目的实际更新流程，重点为 **Frobenius sphere 的 MCSD、MCSD-TP、MuonH、NorMuon-Sphere**；其他几何见折叠区。实验结果另见 [3B/6B 汇总](FROBENIUS_3B_6B_SUMMARY.md)。

**阅读方式：** 正文看伪代码；展开“实际代码”查看 Python 核心实现。代码按对应分支整理，省略参数校验、分组遍历和审计日志；依赖原模块上下文，不是可独立运行的训练脚本。`source` 对应 S，`updated` 对应 W_new。

[核心区别](#overview) · [共同流程](#common) · [四算法实现](#frobenius) · [其他几何](#other-geometries) · [实验设置](#settings) · [源码](#sources)

<a id="overview"></a>

## 1. 核心区别

**四种方法最终都用同一个 Frobenius 球面返回公式，差别主要在方向构造和步长缩放。**

| 方法 | 符号运算前投影动量 | 符号运算后投影方向 | 逐行自适应缩放 | 更新尺度 |
|---|:---:|:---:|:---:|---|
| MCSD | 是 | 否 | 否 | 固定形状尺度 |
| MCSD-TP | 是 | 是 | 否 | 固定形状尺度 |
| MuonH | 否 | 否 | 否 | 方向归一化后乘 R |
| NorMuon-Sphere | 否 | 否 | 是 | 方向归一化后乘 R |

这里的 TP 是 **tangent projection（切向投影）**。NorMuon-Sphere 是 NorMuon 逐行统计与球面返回的项目适配版，不是原始无约束 NorMuon。

<a id="common"></a>

## 2. 共同流程

### 符号与分组

- `W`：形状为 `m × n` 的 FP32 master weight；`G`：累计并裁剪后的梯度。
- `M`：FP32 动量，初始为零；`S`：本步使用的梯度组合。
- `eta`：调度后的当前学习率；`R=r*sqrt(m)`：固定 Frobenius 半径；`k=min(m,n)`。
- `normF`、`inner`：Frobenius 范数、内积；`MSIGN`：矩阵 polar 近似，**不是逐元素 sign**。

每层分别更新 Q、K、V、O、gate、up、down 七个逻辑矩阵，融合 QKV 按 GQA 布局拆分，**不是逐头更新**。其余参数使用辅助 AdamW。受约束矩阵的 `weight_decay=wd_mult=0`。

### 动量与执行顺序

```text
每步：BF16 前向/反向 → 梯度累积与裁剪 → 对每个矩阵执行：

M = beta*M + (1-beta)*G
S = beta*M + (1-beta)*G  if Nesterov else M
W_new = 所选算法的一次更新(W, S, eta, R)

写回 FP32 master weight，并同步 BF16 模型副本。
```

这里使用刚更新的 M，不做偏差修正或 vector transport。**Muon-style Nesterov 只组合梯度，不在外推点重新计算梯度**；它不额外保存一个模型。四方法均保存 M，NorMuon-Sphere 另存一个 `m × 1` 的 FP32 行统计 v。

<details>
<summary><strong>实际代码：动量、矩阵读取与写回</strong></summary>

来源：[ManifoldMCSD.step](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)；基线使用相同数学约定。

```python
w = param.data.index_select(0, rows)
g = grad.index_select(0, rows)
values = GeometryParameters(
    group["geometry"], component["radius"], component["scale"]
)

momentum = component["momentum"]
momentum.mul_(group["momentum_beta"]).add_(
    g, alpha=1.0 - group["momentum_beta"]
)
source = (
    g.lerp(momentum, group["momentum_beta"])
    if group["use_nesterov"] else momentum
)

# 在这里执行下方对应算法，得到 updated。
param.data.index_copy_(0, rows, updated)
```

`rows` 指向一个逻辑矩阵的行。基线动量用 `m.lerp_(g, 1-beta)` 原位更新，与上式数学等价，但舍入顺序可不同。

</details>

<a id="frobenius"></a>

## 3. Frobenius 四算法

### 公共算子

在可行点 `normF(W)=R` 处：

```text
TP(Z)     = Z - W * inner(W,Z) / R^2
UNIT(Z)   = Z / max(normF(Z), tiny_FP32)
RETURN(Y) = R * Y / normF(Y)                 # Y=0 时报告错误
c         = R / sqrt(k)
```

RETURN 是到球面的最近点投影；用于切向增量时，也是一种径向 retraction。

<details>
<summary><strong>实际代码：半径、切向投影和球面返回</strong></summary>

来源：[manifold_geometry.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_geometry.py) 的 Frobenius 分支，分别摘自三个函数。

```python
# shape_geometry_parameters(shape, name, radius_factor)
m, n = shape
radius = radius_factor * math.sqrt(m)
values = GeometryParameters(
    name, radius, radius / math.sqrt(min(m, n))
)

# tangent_project(x, z, params) 的返回值
projected = z - x * (torch.sum(x * z) / params.radius**2)

# return_to_manifold(trial, params, ...) 的 Frobenius 分支
radius = params.radius
norm = torch.linalg.vector_norm(trial).item()
if norm == 0.0:
    raise GeometryFailure("zero_trial")
returned = trial * (radius / norm)
```

这里用 `projected`、`returned` 接住原函数返回表达式，便于与伪代码对照。

</details>

### MCSD：先投影动量

```text
D = MSIGN(TP(S))
W_new = RETURN(W - eta*c*D)
```

TP(S) 切向，但经过非线性 MSIGN 后，D 一般不再切向；默认不再归一化 D。

<details>
<summary><strong>实际代码：MCSD 一次更新</strong></summary>

来源：[ManifoldMCSD.step](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)，展开 Frobenius、未启用 unitF 的路径。

```python
q = tangent_project(w, source, values)
direction = -(
    partial_polar(q)
    if group["lmo_mode"] == "exact"
    else polar_express_msign(q)
)
trial = w + group["lr"] * values.scale * direction
updated, info = return_to_manifold(
    trial, values, mode="projection",
    gap_stop=group["gap_warning"],
    polar_mode=group["stiefel_return_mode"],
)
```

源码 `direction` 已含负号，因此这里用加法；默认 `lmo_mode="ns"`。Frobenius 返回不使用 gap 或 Stiefel 参数。

</details>

### MCSD-TP：再投影输出

```text
D = MSIGN(TP(S))
D = TP(D)
W_new = RETURN(W - eta*c*D)
```

第二次 TP 作用于符号输出，而非重复处理动量。默认不重新归一化，更新长度可能缩短。

<details>
<summary><strong>实际代码：MCSD-TP 一次更新</strong></summary>

来源同上，区别是符号输出后的第二次 `tangent_project`。

```python
q = tangent_project(w, source, values)
direction = -(
    partial_polar(q)
    if group["lmo_mode"] == "exact"
    else polar_express_msign(q)
)
direction = tangent_project(w, direction, values)
trial = w + group["lr"] * values.scale * direction
updated, info = return_to_manifold(
    trial, values, mode="projection",
    gap_stop=group["gap_warning"],
    polar_mode=group["stiefel_return_mode"],
)
```

</details>

### MuonH：统一方向长度

```text
D = UNIT(MSIGN(S))
W_new = RETURN(W - eta*R*D)
```

不先投影动量。非零方向下，返回前的更新长度为 `eta*R`。

<details>
<summary><strong>实际代码：MuonH 一次更新</strong></summary>

来源：[baseline_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py) 的 `muonh` 分支。

```python
sign = _polar(momentum, lmo_mode)
norm = torch.linalg.vector_norm(sign)
direction = sign / norm.clamp_min(torch.finfo(sign.dtype).tiny)
trial = w - lr * values.radius * direction
updated = return_to_manifold(trial, values, mode="projection")[0]
```

该函数参数名 `momentum` 实际接收已选择的 `source`，可以是 EMA 或 Nesterov 组合；`_polar` 在 ns 模式调用 `polar_express_msign`，exact 模式调用 `partial_polar`。

</details>

### NorMuon-Sphere：增加逐行二阶统计

```text
H = MSIGN(S)
v = beta2*v + (1-beta2)*mean(H^2, 按行)       # v 初始为零
A = H / (sqrt(v) + eps)                      # 按行广播
D = UNIT(A)
W_new = RETURN(W - eta*R*D)
```

默认 `beta2=0.95`、`eps=1e-10`，无偏差修正。统计的是 **MSIGN 输出的平方**，不是原始梯度平方；行对应原矩阵的输出维度。

<details>
<summary><strong>实际代码：NorMuon-Sphere 一次更新</strong></summary>

来源：[normuon_sphere_direction / baseline_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py)。

```python
# normuon_sphere_direction 内部：原位更新持久状态。
sign = _polar(momentum, mode)
row_second_moment.lerp_(
    sign.square().mean(dim=-1, keepdim=True), 1 - beta2
)
scaled = sign / (row_second_moment.sqrt() + eps)
direction = scaled / torch.linalg.vector_norm(scaled).clamp_min(
    torch.finfo(scaled.dtype).tiny
)

# baseline_update 内部：方向返回后完成参数更新。
trial = w - lr * values.radius * direction
updated = return_to_manifold(trial, values, mode="projection")[0]
```

`row_second_moment` 只在初始化时置零，每步复用；`mode` 对应调用者的 `lmo_mode`。

</details>

<a id="differences"></a>

### 归一化消融与公平比较

MCSD/MCSD-TP 的 `unitF` 消融只替换最后一步，默认关闭：

```text
D = D / (normF(D) + 1e-12)                   # TP 版本在第二次 TP 之后归一化
W_new = RETURN(W - eta*R*D)
```

**加 unitF 仍不等于 MuonH**：前者使用 `MSIGN(TP(S))`，后者使用 `MSIGN(S)`。只有精确满秩 polar 满足 `normF(D)=sqrt(k)`，原 MCSD 的固定尺度才等价于这一长度归一化。

比较时分开报告：**相同 LR/动量的公平对照**，以及**各自最佳 LR 的调参结果**。相同 LR 不保证相同实际位移。

<details>
<summary><strong>实际代码：unitF 开关</strong></summary>

来源：[ManifoldMCSD.step](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)，放在可选的第二次 TP 之后。

```python
if group["direction_normalization"] == "frobenius":
    unit_direction = direction / (
        torch.linalg.vector_norm(direction) + group["direction_norm_eps"]
    )
    trial = w + group["lr"] * values.radius * unit_direction
else:
    trial = w + group["lr"] * values.scale * direction
```

</details>

### 数值精度

Frobenius 四方法共用 **10 步 BF16 Polar Express，输出 FP32**，不是精确 SVD。配置 `baseline_msign_steps=8` 只控制 SSO/MuonSphere。稀疏审计仍可能调用 SVD，因此“主更新不用 SVD”不等于“训练完全没有 SVD”。

<details>
<summary><strong>实际代码：BF16 矩阵符号与 FP32 polishing</strong></summary>

来源：[polar_express_msign](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_polar.py)。`_COEFFICIENTS` 为该文件中的八组固定系数；默认 `steps=10, polish_steps=0`。

```python
magnitude = torch.linalg.vector_norm(matrix)
if magnitude.item() == 0:
    return torch.zeros_like(matrix)
transpose = matrix.shape[0] > matrix.shape[1]
x = (matrix.T if transpose else matrix) / (magnitude * 1.01)
x = x.to(torch.bfloat16)

for i in range(steps):
    a, b, c = _COEFFICIENTS[min(i, len(_COEFFICIENTS) - 1)]
    if i < len(_COEFFICIENTS) - 1:
        a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
    gram = x @ x.T
    polynomial = c * gram
    polynomial.diagonal().add_(b)
    polynomial = polynomial @ gram
    polynomial.diagonal().add_(a)
    x = polynomial @ x

x = x.float()
for _ in range(polish_steps):
    gram = x @ x.T
    x = 0.5 * (3.0 * x - gram @ x)
result = x.T if transpose else x
```

Stiefel 返回另设 `polish_steps=3`；Frobenius 方向不做这三步。

</details>

<a id="other-geometries"></a>

## 4. 其他几何与基线

<details>
<summary><strong>展开：Spectral MCSD/TP、SSO、MuonSphere、Stiefel iMuon</strong></summary>

### 半径与 MCSD/TP 的变化

当前 `shape` 协议使用原矩阵的 `m=D_out, n=D_in`：

| 几何 | 约束半径 | MCSD 尺度 c | MCSD / MCSD-TP 返回 |
|---|---|---|---|
| Frobenius sphere | `R=r*sqrt(m)` | `R/sqrt(min(m,n))` | 径向 / 径向 |
| Spectral sphere | `R=r*sqrt(m/n)` | R | top-8 近似投影 / 径向 |
| Scaled Stiefel | `R=r*sqrt(m/n)` | R | polar / polar |

Stiefel 的高矩阵满足 `W.T@W=R^2 I`，宽矩阵满足 `W@W.T=R^2 I`。内部转置不改变 R；该协议不等同于旧 `init_energy` 能量匹配。

MCSD/TP 保留第 3 节的方向流程，替换 TP 和 RETURN：

- **Spectral TP**：`N=u1@v1.T`，`TP(Z)=Z-inner(N,Z)*N`。默认用 cold BF16 PI10，末次左向量与奇异值用 FP32 修正。
- **Spectral MCSD 返回**：对 residual 逐次 PI、修正并 deflate，最多 8 次；首个奇异值移到 R，其余截断到不超过 R。不是精确 truncated SVD。
- **Spectral MCSD-TP 返回**：`R*Y/PI10(Y).sigma1`。可选 exact 模式分别使用 SVD 投影或 SVD 范数径向返回。
- **Stiefel TP**：高矩阵时 `TP(Z)=Z-W*sym(W.T@Z)/R^2`，其中 `sym(A)=(A+A.T)/2`；宽矩阵转置处理。

Spectral 近重奇异值时，当前实现仍用选定法向继续计算，**不触发 PGD**，也不保证存在唯一光滑切平面。本节不是旧 MCSD-TP-PGD 分支算法。

<details>
<summary><strong>实际代码：Spectral MCSD/TP 的 PI、方向和返回</strong></summary>

来源：[leading_triplet / spectral_return_pi](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_spectral.py) 与 [ManifoldMCSD.step](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)。先看默认 PI 的核心：

```python
work = matrix.to(matmul_dtype)  # 默认 torch.bfloat16
vector = torch.arange(
    1, matrix.shape[1] + 1, device=matrix.device, dtype=torch.float32
)
vector = torch.sin(vector) + 1.5
v = torch.nn.functional.normalize(vector.to(matmul_dtype), dim=0)
for _ in range(steps):  # 默认 10
    u = torch.nn.functional.normalize(work @ v, dim=0, eps=1e-12)
    v = torch.nn.functional.normalize(work.T @ u, dim=0, eps=1e-12)
u = torch.nn.functional.normalize(matrix @ v.float(), dim=0, eps=1e-12)
v = v.float()
sigma = u @ matrix @ v
```

方向构造和返回调用（默认 ns 路径）：

```python
_, u, v = leading_triplet(w, steps=group["power_steps"])
normal = torch.outer(u, v)

def project_spectral(vector):
    return vector - torch.sum(normal * vector) * normal

q = project_spectral(source)
direction = -polar_express_msign(q)
if group["method"] == "mcsd_tp":
    direction = project_spectral(direction)

trial = w + group["lr"] * values.scale * direction
return_mode = "radial" if group["method"] == "mcsd_tp" else "projection"
updated = spectral_return_pi(
    trial, values.radius, mode=return_mode,
    steps=group["power_steps"], rank=group["topk_rank"],
)
```

`spectral_return_pi` 的两个返回分支：

```python
if mode == "radial":
    sigma, _, _ = leading_triplet(trial, steps=steps, matmul_dtype=matmul_dtype)
    if sigma.item() <= 0:
        raise ValueError("zero or invalid spectral trial")
    return trial * (radius / sigma)

updated = trial.clone()
residual = trial.clone()
for index in range(min(rank, min(trial.shape))):
    sigma, u, v = leading_triplet(residual, steps=steps, matmul_dtype=matmul_dtype)
    if sigma.item() <= 0:
        break
    outer = torch.outer(u, v)
    desired = radius if index == 0 else min(sigma.item(), radius)
    updated.add_(outer, alpha=desired - sigma.item())
    residual.add_(outer, alpha=-sigma.item())
return updated
```

</details>

### MuonSphere 与 SSO

两者共用原 SSO 内核：**cold BF16 PI10 + 8 步 BF16 MSIGN，先缩放 W，再更新**。

```text
S0 = S / max(normF(S), 1e-8)
sigma,u,v = ORIGINAL_PI10(W)                 # 每次从全一右向量启动
W0 = W * R / (max(sigma,0) + 1e-8)
N = u@v.T

if MuonSphere:
    lambda = 0
if SSO:
    f(lambda) = inner(N, MSIGN8(S0 + lambda*N))
    lambda = 近似求根(f)

D = MSIGN8(S0 + lambda*N)
W_new = W0 - eta*R*FP32(D)                   # 不追加 post-step 投影
```

SSO 最多扩展区间 10 次、二分 20 次，容差 `2e-4`；无区间则取零，未收敛则取已评估的最小残差候选。每次 f 评估都包含一次 MSIGN8。MuonSphere 对应历史代码名 `muon_ball`。

<details>
<summary><strong>实际代码：MuonSphere</strong></summary>

来源：[compute_muon_ball_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py) 的单卡路径。

```python
W_work, M_work = W, M
M_fp32 = M_work.to(torch.float32)
M_fp32 = M_fp32 / torch.linalg.norm(
    M_fp32, dim=(-2, -1), keepdim=True
).clamp_min(1e-8)

sigma, u, v = power_iteration(W_work, steps=power_iteration_steps)
sigma_value = sigma.item()
retract_bias = apply_retract(
    W_work, sigma_value, target_radius,
    mode=retract_mode, alpha=retract_alpha, current_lr=current_lr
)
Phi = msign(M_fp32, steps=msign_steps)
```

`apply_retract` 在原位缩放 W；外层 `baseline_update` 再执行 `w - lr * values.scale * Phi.float()`。当前 shape 协议 `values.scale=R`，调用参数为 PI10、MSIGN8、hard return。

</details>

<details>
<summary><strong>实际代码：SSO</strong></summary>

来源：[\_compute_single_rank](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py)，展开 bisection 分支。

```python
M_fp32 = M.to(torch.float32)
M_fp32 = M_fp32 / torch.linalg.norm(
    M_fp32, dim=(-2, -1), keepdim=True
).clamp_min(1e-8)
sigma, u, v = power_iteration(W, steps=power_iteration_steps)
sigma_value = sigma.item()
retract_bias = apply_retract(
    W, sigma_value, target_radius,
    mode=retract_mode, alpha=retract_alpha, current_lr=current_lr
)
Theta = u @ v.transpose(-2, -1)
lambda_value, converged, residual, iterations = solve_lambda_with_bisection(
    G=M_fp32, Theta=Theta,
    initial_guess=0.0, initial_step=1e-3,
    tolerance_f=solver_tolerance_f,
    max_iterations=solver_max_iterations,
    max_expansions=10, msign_steps=msign_steps,
)
Z = M_fp32 + lambda_value * Theta
Phi = msign(Z, steps=msign_steps)
```

外层同样执行 `w - lr * values.scale * Phi.float()`，不再返回球面。完整区间搜索和二分实现见同文件 `find_bracket`、`solve_lambda_with_bisection`；返回的收敛标志不是额外一次投影。

</details>

### iMuon：分块切向方向

当前适配版使用 EMA，不启用 Nesterov。宽矩阵先转置，半径不变：

```text
Q = W/R
A = Q.T@S
K = (A-A.T)/2
H = S-Q@A                                   # 方阵时直接设 H=0

cutoff = 8*eps_FP32*normF(S)
SIGN(B) = 0 if normF(B)<=cutoff else MSIGN(B)
D = Q@SIGN(K) + SIGN(H)
W_new = R*POLAR_RETURN(W - eta*R*D)
必要时转回原方向
```

两个块分别求符号，组合后不再整体归一化。它与 Stiefel MCSD/TP 共用返回算子：默认 **10 步 BF16 Polar Express + 3 步 FP32 NS polishing**；精确参考为 full polar `U@V.T`，要求 trial 满秩。默认近似不能当成精确正交投影。

<details>
<summary><strong>实际代码：iMuon 方向与 Stiefel 返回</strong></summary>

来源：[imuon_direction / baseline_update](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py)。

```python
transpose = w.shape[0] < w.shape[1]
q = (w.T if transpose else w) / radius
m = momentum.T if transpose else momentum
qt_m = q.T @ m
k = 0.5 * (qt_m - qt_m.T)
h = m - q @ qt_m if q.shape[0] > q.shape[1] else torch.zeros_like(m)
cutoff = 8 * torch.finfo(m.dtype).eps * torch.linalg.vector_norm(m)

def block_sign(block):
    if torch.linalg.vector_norm(block) <= cutoff:
        return torch.zeros_like(block)
    return _polar(block, mode)

direction = -q @ block_sign(k) - block_sign(h)
direction = direction.T if transpose else direction

# baseline_update 接收上述带负号方向。
trial = w + lr * values.scale * direction
updated = return_to_manifold(
    trial, values, mode="projection", polar_mode=stiefel_return_mode
)[0]
```

Stiefel 的切向投影与默认返回来自 [manifold_geometry.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_geometry.py)：

```python
# tangent_project 的 Stiefel 分支
transposed = x.shape[0] < x.shape[1]
w, v = (x.T, z.T) if transposed else (x, z)
product = w.T @ v
sym = (product + product.T) * 0.5
projected = v - (w @ sym) / params.radius**2
projected = projected.T if transposed else projected

# return_to_manifold 的 Stiefel ns 分支
returned = radius * polar_express_msign(trial, polish_steps=3)
```

Stiefel MCSD/TP 使用这组 TP/返回算子；iMuon 不走前置 TP，而使用上面的两个块。

</details>

</details>

<a id="settings"></a>

## 5. 最近实验设置

| 项目 | Frobenius width-256 协议 |
|---|---|
| 模型 | 28 层，FFN=768，Q heads=4，KV groups=2，head dim=64；约 73.42M 参数 |
| 架构 | SwiGLU、GQA、QK normalization、RoPE、pre-RMSNorm、untied embeddings，dropout=0 |
| 数据与 batch | OLMo-Mix-1124；序列 2048，global batch=128，micro batch=4 |
| 预算 | 3B：11,445 步；6B：22,889 步；均从初始化训练 |
| 调度 | warmup 分别 229/458 步，cosine 衰减到 peak LR 的 10% |
| 半径与精度 | r=1；BF16 模型、FP32 master/state；梯度裁剪阈值 1 |
| 验证 | 同一 valid index，每 200 步及结束时评估 40 batches |
| 辅助 AdamW | peak LR=0.001，betas=(0.9,0.95)，eps=1e-8；仅允许衰减的辅助参数组使用 wd=0.1 |

原始网格中 MCSD/TP 为 **0.95 EMA**，MuonH/NorMuon 为 **0.9 Nesterov**；后续增加了匹配动量的对照。因此不能将原始差异全部归因于几何操作。各组 LR、beta、seed、状态及最终 loss 统一查阅 [结果汇总](FROBENIUS_3B_6B_SUMMARY.md)，不在本文重复维护。

<a id="sources"></a>

## 6. 源码与来源

| 内容 | 实现 |
|---|---|
| MCSD/TP 与基线适配 | [manifold_mcsd.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)、[manifold_baselines.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py) |
| 半径、切空间、返回与数值内核 | [manifold_geometry.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_geometry.py)、[manifold_polar.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_polar.py)、[manifold_spectral.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_spectral.py) |
| SSO / MuonSphere | [spectral_ball_utils.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py)、[muon_ball.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py) |
| 分组与训练入口 | [manifold_layout.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_layout.py)、[run.py](../../scripts/manifold/run.py)、[launch.py](../../scripts/manifold/launch.py) |

方法来源：[Hyperball / MuonH](https://arxiv.org/abs/2606.16899)、[NorMuon](https://arxiv.org/abs/2510.05491)、[iMuon](https://arxiv.org/abs/2605.09238)、[SSO 官方项目](https://github.com/Unakar/Spectral-Sphere-Optimizer)。本文描述项目适配实现，不声称原论文的理论保证直接覆盖当前 BF16 版本。

*版本范围：2026-10-26 本地源码核对；未重新核验 H20 文件。部分本地改动尚未发布，链接中的 GitHub 源码可能滞后；具体实验以运行时冻结配置和代码为准。*
