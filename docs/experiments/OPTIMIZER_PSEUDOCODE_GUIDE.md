# 当前对比优化器的实现与伪代码

本文说明本项目如何在 Megatron 的同一训练流程中运行 **MCSD、MCSD-TP、MuonH、NorMuon-Sphere、SSO、MuonSphere 和 iMuon**。主体为最近的 Frobenius sphere 3B/6B 实验，后半部分补充 spectral sphere 与 Stiefel 的实现。

**最重要的区别：MCSD 先投影动量再做矩阵符号；MCSD-TP 还会投影符号输出；MuonH 直接对动量做矩阵符号并归一化；NorMuon-Sphere 在此基础上增加逐行二阶统计。** 四种 Frobenius 方法最后都使用同一个球面返回公式，并非四种不同的返回映射。

实现核对日期：2026-10-26。以下是本项目适配版的执行说明，不把它们称为原论文完整训练方案的原样复现。近期实验见 [3B/6B 结果汇总](FROBENIUS_3B_6B_SUMMARY.md)。

[算法总览](#overview) · [共同流程](#common) · [Frobenius 四算法](#frobenius) · [实际差异](#differences) · [其他几何](#other-geometries) · [运行设置](#settings) · [源码与文献](#sources)

<a id="overview"></a>

## 1 算法总览

| 名称 | 约束 | 核心方向构造 | 参数返回方式 |
|---|---|---|---|
| MCSD | 三种几何均支持 | 切向投影动量，再求矩阵符号 | Frobenius 径向返回；spectral 近似投影；Stiefel polar 返回 |
| MCSD-TP | 三种几何均支持 | MCSD 的符号输出再投影到切空间 | Frobenius 径向返回；spectral 径向返回；Stiefel polar 返回 |
| MuonH | Frobenius sphere | 动量的矩阵符号，按 Frobenius 范数归一化 | 更新后径向返回 |
| NorMuon-Sphere | Frobenius sphere | 矩阵符号、逐行二阶缩放、整体归一化 | 更新后径向返回 |
| SSO | Spectral sphere | 求一个标量乘子，使符号方向近似满足切向条件 | 更新前 PI 估计范数并缩放；不追加更新后投影 |
| MuonSphere | Spectral sphere | SSO 中固定乘子为零，直接使用符号方向 | 同 SSO 的更新前缩放 |
| iMuon | Scaled Stiefel | 切空间分成两个块，分别求矩阵符号 | 更新后 polar 返回 |

**名称对应：** `MuonSphere` 调用历史文件 `muon_ball.py` 的内核；`NorMuon-Sphere` 是本项目把 NorMuon 的逐行统计与球面返回组合起来的适配版。当前三几何实现是 `manifold_*`，不要与旧 `spel.py`、`spel_pgd_same_projection.py` 或 cached-SVD MCSD-TP-PGD 实验混用。

<a id="common"></a>

## 2 共同符号与训练流程

### 2.1 一次更新处理什么矩阵

下面每段伪代码处理一个逻辑矩阵组件，而不是整个模型，也不是整个融合 QKV 张量。

| 符号 | 含义 |
|---|---|
| `W` | 当前 FP32 master weight，形状 `m × n`；`m=D_out`，`n=D_in` |
| `G` | 当前批次累计、完成梯度裁剪后的该组件梯度 |
| `M` | 跨步保存的 FP32 一阶动量，初始为零 |
| `S` | 本次送入方向构造的 EMA 或 Nesterov 梯度组合 |
| `eta` | 当前迭代的学习率，由 warmup/cosine 调度得到，不是始终等于 peak LR |
| `R` | 固定的矩阵约束半径；Frobenius 下为 `r * sqrt(m)` |
| `k` | `min(m,n)`；此处是矩阵短边，**不是**旧 spectral top-8 的秩参数 |
| `c` | MCSD 的固定形状尺度；Frobenius 下 `c=R/sqrt(k)` |
| `inner(A,B)` | Frobenius 内积，即所有对应元素乘积之和 |
| `normF(A)` | Frobenius 范数，即所有元素平方和的平方根 |
| `MSIGN(A)` | 实际数值矩阵符号/polar 近似；不是逐元素 `sign(A)` |

伪代码统一用 `W_trial = W - ... * D`，把 `D` 写成正号方向源；源码 MCSD 保存的是 `-D` 并使用加法，二者相同。伪代码不要求实数域等价表达式在 BF16 舍入后逐位一致。

### 2.2 逻辑矩阵与半径

每层按 `Q、K、V、O、gate、up、down` 划分为七个逻辑矩阵。融合 QKV 根据 GQA 的真实行布局收集为完整 Q、K、V，**优化器不是逐 attention head 各自更新**。初始化脚本中的 head-wise 初始化选项不改变这一优化器分组。

最近 width=256、4 个 Q 头、2 个 KV 组、head dimension=64、FFN=768、`r=1` 时：

| 组件 | 形状 `m × n` | Frobenius 半径 `R` | MCSD 固定尺度 `c` |
|---|---|---|---|
| Q、O | 256 × 256 | 16 | 1 |
| K、V | 128 × 256 | `sqrt(128)` | 1 |
| gate、up | 768 × 256 | `sqrt(768)` | `sqrt(3)` |
| down | 256 × 768 | 16 | 1 |

28 层共有 196 个这样的矩阵组件。Embedding、输出 head、归一化参数及其他未被标记的参数交给辅助 AdamW，不套用下面的球面更新。

### 2.3 整体执行顺序

```text
初始化模型和每个逻辑矩阵的固定半径
将受约束矩阵映射到指定几何，保存 FP32 master weight
为每个组件创建 M = 0；NorMuon-Sphere 另外创建行统计 v = 0

for 每个训练 iteration:
    用 BF16 模型副本执行前向和反向
    累积 micro-batch 梯度，按训练器设置执行全局梯度裁剪
    读取本步学习率 eta

    for 每个有梯度的逻辑矩阵组件:
        读取 FP32 W、G、M，以及需要的行统计 v
        M, S = MOMENTUM(G, M, beta, use_nesterov)
        按所选算法执行下面的一次矩阵更新
        写回更新后的 FP32 W 和持久状态

    用辅助 AdamW 更新其余参数
    将 FP32 master weights 同步到 BF16 模型副本
    按固定间隔验证、记录日志
```

矩阵优化器的 `weight_decay=0`、`wd_mult=0`。训练命令的全局 `weight_decay=0.1` 作用于辅助 AdamW 中允许衰减的参数组，不作用于这些受约束矩阵；归一化、bias 等仍遵循 Megatron 的分组豁免。

### 2.4 EMA 与 Muon-style Nesterov

```text
function MOMENTUM(G, M, beta, use_nesterov):
    M = beta * M + (1 - beta) * G
    if use_nesterov:
        S = beta * M + (1 - beta) * G    # 使用刚更新的 M
    else:
        S = M
    return M, S
```

四个 Frobenius 方法使用相同的一阶动量约定，无一阶动量偏差修正。`S` 是临时结果，Nesterov 不额外保存一个外推后的模型。

这里的动量保存在环境矩阵空间，未执行历史切向量的 vector transport。该 Nesterov 是梯度组合，**不是**先生成流形外推点、再在外推点重新计算梯度的 AManPG/FISTA；不能直接据此声称获得流形 Nesterov 加速定理。

<a id="frobenius"></a>

## 3 Frobenius sphere 四算法

共同约束为 `normF(W)=R`。在可行点 W 处：

```text
function TANGENT(W, Z, R):
    return Z - W * inner(W, Z) / R^2

function FROB_RETURN(Y, R):
    if normF(Y) == 0: 报告 zero_trial 错误
    return R * Y / normF(Y)

function UNIT_F(Z):
    return Z / max(normF(Z), tiny_FP32)
```

`FROB_RETURN` 在 `Y != 0` 时是到 Frobenius 球面的最近点投影；如果 trial 增量属于切空间，它也给出标准的局部径向 retraction。**两种解释可以对应同一个公式，区别在输入方向。** 浮点实现只在数值精度内保持约束。

### 3.1 MCSD

一句话：**先去掉动量的法向分量，再对剩余矩阵做 spectral-ball LMO，最后返回球面。**

```text
输入: W, S, eta, R
k = min(m, n)
c = R / sqrt(k)

Q = TANGENT(W, S, R)
D = MSIGN(Q)
Y = W - eta * c * D
W_new = FROB_RETURN(Y, R)
```

`Q` 是切向的，但非线性的 `MSIGN(Q)` 一般不再切向。MCSD 不对该输出追加 TP，也不默认每步除以 `normF(D)`。精确、满秩 polar 的 `normF(D)=sqrt(k)`，此时 `c` 将 trial 更新长度设为 `eta*R`；低秩或有限精度 NS 下不保证这一等式。

### 3.2 MCSD-TP

一句话：**在 MCSD 的基础上，再把矩阵符号输出投影一次，保证 trial 使用切向方向。**

```text
输入: W, S, eta, R
k = min(m, n)
c = R / sqrt(k)

Q = TANGENT(W, S, R)
D = MSIGN(Q)
D = TANGENT(W, D, R)          # 相比 MCSD 新增这一行
Y = W - eta * c * D
W_new = FROB_RETURN(Y, R)
```

两次 TP 不是重复计算：第一次作用于动量 `S`，第二次作用于非线性的矩阵符号输出。当前默认不在第二次 TP 后重新归一化，因此其 trial 更新长度可能比 MCSD 短。这里 `TP` 指 tangent projection，不是模型并行中的 tensor parallel。

### 3.3 MuonH

一句话：**直接把动量变成矩阵符号方向，归一化其整体长度，再按相对半径步长更新。**

```text
输入: W, S, eta, R

D = MSIGN(S)                 # 不先执行 TANGENT(W, S, R)
D = UNIT_F(D)
Y = W - eta * R * D
W_new = FROB_RETURN(Y, R)
```

非零方向下，trial 更新长度在浮点精度内为 `eta*R`；返回后的实际位移未必等于它。该实现采用 Hyperball 的固定权重范数与固定相对更新长度机制，模型、矩阵分组和数据沿用本项目协议。[Hyperball 论文](https://arxiv.org/abs/2606.16899)

### 3.4 NorMuon-Sphere

一句话：**在 MuonH 的矩阵符号之后，加上逐输出行的自适应缩放，再统一整体步长。**

额外持久状态 `v` 的形状为 `m × 1`，FP32、初始为零；`beta2=0.95`，`eps=1e-10`。

```text
输入: W, S, v, eta, R, beta2, eps

H = MSIGN(S)
row_energy[i] = mean_j(H[i,j]^2)
v = beta2 * v + (1 - beta2) * row_energy

A[i,j] = H[i,j] / (sqrt(v[i]) + eps)
D = UNIT_F(A)
Y = W - eta * R * D
W_new = FROB_RETURN(Y, R)

保存 v
```

必须注意：统计对象是 **符号输出 H 的平方**，不是原始梯度 G 的平方；分母是 `sqrt(v)+eps`，不是 `sqrt(v+eps)`；代码没有对 v 做偏差修正。行始终对应原始逻辑矩阵的输出维度，即使 MSIGN 内部为了效率转置矩阵，也会先转回再统计。

该方法保留 NorMuon 的逐行统计思想，但最终采用本项目的 `eta*R` 步长和球面返回。因此应称 **NorMuon-Sphere 适配版**，不应把其结果直接写成原始无球面约束 NorMuon 的结果。[NorMuon 论文](https://arxiv.org/abs/2510.05491)

### 3.5 MCSD-unitF 消融

它不是第五个主算法，而是改变 MCSD/MCSD-TP trial 长度的开关。在第 3.1 或 3.2 节构造 D 后，替换两行：

```text
D_unit = D / (normF(D) + 1e-12)    # 若为 TP，归一化在第二次 TP 之后
Y = W - eta * R * D_unit
W_new = FROB_RETURN(Y, R)
```

默认运行关闭该开关。它没有移除 MCSD 的前置切向投影，也没有增加 NorMuon 的逐行二阶统计。

<a id="differences"></a>

## 4 怎样理解四算法的实际差异

### 4.1 只看操作顺序

```text
MCSD:          S -> TP -> MSIGN ----------------> 固定形状尺度 -> 球面返回
MCSD-TP:       S -> TP -> MSIGN -> TP ----------> 固定形状尺度 -> 球面返回
MuonH:         S ------> MSIGN ----------------> unit-F       -> 球面返回
NorMuon-Sphere:S ------> MSIGN -> 行二阶缩放 ---> unit-F       -> 球面返回
```

**MCSD 加 unit-F 后仍不等于 MuonH。** 前者使用 `MSIGN(TANGENT(W,S,R))`，后者使用 `MSIGN(S)`；切向投影与矩阵符号一般不交换。MCSD-TP 还多一次输出 TP。只有在某些特殊输入下，这些方向才可能相同。

**相同 LR 不保证相同实际角度。** 它既受归一化影响，也受方向的法向/切向比例影响。对可行 W，记 `W_hat=W/R`，`D_T=TANGENT(W,D,R)`。对于 trial `Y=W-eta*s*D`，当 `R-eta*s*inner(W_hat,D)>0` 时，返回后的转角满足：

$$
\tan\theta=
\frac{\eta s\,\|D_T\|_F}
{R-\eta s\,\langle W/R,D\rangle_F}.
$$

因此“四方法同 LR 比较”回答的是匹配超参数下的表现；“四方法各自选最佳 LR”回答的是各自调参后的表现。二者都值得报告，但不能互相替代。

### 4.2 每个矩阵额外保存什么

| 方法 | 持久优化器状态，不含 W/G | 主路径每步矩阵符号次数 |
|---|---|---:|
| MCSD / MCSD-TP | 一个 `m × n` 的 M | 1 |
| MuonH | 一个 `m × n` 的 M | 1 |
| NorMuon-Sphere | M，加 `m × 1` 的 v | 1 |

EMA 改为此处的 Nesterov 不增加一个持久模型副本，但可能产生临时矩阵。所有方法仍由 Megatron 保存 FP32 master weight 与 BF16 模型参数；不能把表里的动量状态当作训练总显存。

### 4.3 实际 MSIGN 不是精确 SVD

Frobenius 四方法共用 `polar_express_msign`，默认 **10 步 BF16 多项式迭代**，输出转回 FP32。其简化伪代码是：

```text
function MSIGN(A):
    if normF(A) == 0: return zeros_like(A)
    如果 A 行数大于列数，先转置，记录转置标记
    X = BF16(A / (1.01 * normF(A)))
    for i = 1,...,10:
        读取该轮 Polar Express 系数 a,b,c
        B = X @ X.T
        X = (a*I + b*B + c*B@B) @ X
    X = FP32(X)
    如曾转置，转回
    return X
```

系数采用源码内的八组序列，后两轮重复最后一组；前七组另按源码进行 `1.01` 稳定化缩放。这里是代数形式，实际代码的乘法结合顺序与对角线原位操作保持不变。

精确参考 `partial_polar` 则通过 SVD 保留有效奇异模式，对数值零模式置零；它不是默认训练路径。有限步 BF16 MSIGN 不保证所有非零奇异值严格等于 1。

配置字段 `baseline_msign_steps=8` **仅传给当前 SSO/MuonSphere 内核**，没有覆盖 Frobenius 四算法的 10 步默认值。MCSD 的稀疏数值审计仍可能调用 SVD 检查方向，不能把“主更新用 NS”写成“整个训练从不调用 SVD”。审计按组件计数错开，不是每 200 步同时审计全部矩阵。

<a id="other-geometries"></a>

## 5 其他几何下的对比方法

本节说明项目已有实现，不代表这些方法参加了上面的 Frobenius 3B/6B 网格，也不把不同约束的 loss 当作单一优化器因素的比较。

### 5.1 三种半径和通用 MCSD

当前 `shape` 协议固定同一个幅度参数 r：

| 几何 | 约束 | MCSD 固定尺度 |
|---|---|---|
| Frobenius sphere | `normF(W)=R=r*sqrt(m)` | `R/sqrt(min(m,n))` |
| Spectral sphere | `norm2(W)=R=r*sqrt(m/n)` | `R` |
| Scaled Stiefel | 高矩阵 `W.T@W=R^2 I`；宽矩阵 `W@W.T=R^2 I` | `R=r*sqrt(m/n)` |

Stiefel 按用户选定规则匹配 spectral 半径，而不是额外匹配 Frobenius 总能量。宽矩阵内部转置求解时仍使用原始 `(m,n)` 决定的 R，不能转置后重算半径。旧 `init_energy` 初始化能量协议不是本表的 `shape` 协议。

通用 MCSD/TP 仍是 `Q=TP_W(S)`、`D=MSIGN(Q)`、可选 `D=TP_W(D)`、`Y=W-eta*c*D`，只替换对应几何的 TP 和返回算子。

### 5.2 Spectral MCSD 与 MCSD-TP

若顶端奇异值简单，令 `N=u1@v1.T`，使用 `TP_W(Z)=Z-inner(N,Z)*N`。默认 `pi_topk` 模式用 **10 步 cold BF16 PI** 得到向量，再用 FP32 计算末次左向量与 Rayleigh 商；初始右向量为确定性的 `sin(arange(1,n+1))+1.5`，不缓存前步向量。

```text
N = top_normal_from_PI10(W)
Q = S - inner(N,S)*N
D = MSIGN(Q)
if method == MCSD-TP:
    D = D - inner(N,D)*N
Y = W - eta*R*D

if method == MCSD-TP:
    sigma = PI10(Y).sigma1
    W_new = R * Y / sigma
else:
    W_new = copy(Y)
    residual = copy(Y)         # 两份独立副本，避免原位修改相互影响
    for j = 1,...,min(8,m,n):
        sigma,u,v = PI10(residual)
        if sigma <= 0: break
        desired = R if j == 1 else min(sigma,R)
        W_new += (desired-sigma) * u@v.T
        residual -= sigma * u@v.T
```

这是数值近似，不是经过认证的 truncated SVD 投影。参数 `gap_warning` 只用于诊断；近重奇异值时仍使用选定法向继续计算，不触发 PGD，也不能在该点宣称有唯一光滑切平面。

可选 `spectral_solver=exact` 使用 FP32 SVD：MCSD 将第一奇异值置为 R、其他奇异值截断到不大于 R；MCSD-TP 用 `R*Y/sigma1(Y)` 返回。该 exact 路径缓存返回点对应的 top normal，但这不是旧 MCSD-TP-PGD 的缓存 gap 分支算法。

### 5.3 MuonSphere

该适配器调用 `muon_ball.py` 的单卡内核，使用现有 SSO 风格的 **pre-step hard radial return**：

```text
输入: W, S, eta, R
S0 = S / max(normF(S), 1e-8)
sigma,u,v = ORIGINAL_PI(W, steps=10)
W0 = W * R / (max(sigma,0) + 1e-8)
D = ORIGINAL_MSIGN(S0, steps=8)
W_new = W0 - eta*R*FP32(D)
```

`ORIGINAL_PI` 每次从全一右向量开始，矩阵运算和输出估计使用 BF16，不是 warm start；它与第 5.2 节的 PI 初始向量及末次 FP32 修正不同。上游注释中存在“returns fp32”的文字，但当前执行路径显式转换为 BF16，应以代码为准。

**该更新没有 post-step 投影。** W0 是用估计范数缩放的基点，W_new 经方向更新后可偏离目标球面，下一步再纠偏。因此不能描述为“每步输出都经过精确球面投影”。命名对应关系和基础实现来自 [SSO 官方仓库](https://github.com/Unakar/Spectral-Sphere-Optimizer)。

### 5.4 SSO

SSO 与 MuonSphere 的主要区别是：在符号输入中加入标量乘子，尽量满足切向条件。

```text
输入: W, S, eta, R
S0 = S / max(normF(S), 1e-8)
sigma,u,v = ORIGINAL_PI(W, steps=10)
W0 = W * R / (max(sigma,0) + 1e-8)
N = u@v.T

f(lambda) = inner(N, ORIGINAL_MSIGN(S0 + lambda*N, steps=8))
从 lambda=0 开始寻找异号区间，最多扩展 10 次
若找到区间: 二分求解 f(lambda) ~= 0
             最多 20 次，停止容差 abs(f) <= 2e-4
             未达到容差时取已评估的最小残差候选
若找不到区间: lambda = 0，并记录求解失败

D = ORIGINAL_MSIGN(S0 + lambda*N, steps=8)
W_new = W0 - eta*R*FP32(D)
```

每次 f 的评估都包含一次 MSIGN，所以“二分最多 20 次”不是“一共只做 20 步 Newton--Schulz”。当前适配器继续采用近似求解输出，既不在末尾追加 TP，也不追加 post-step 投影。以上有限精度求解不保证精确切向性。[SSO 官方实现](https://github.com/Unakar/Spectral-Sphere-Optimizer)

### 5.5 iMuon 与 Stiefel 返回

为统一写法，先把宽矩阵 W、S 转置成高矩阵，半径仍取原组件的 R。定义 `Q=W/R`、`skew(A)=(A-A.T)/2`。

```text
输入: W, S, eta, R              # 当前 iMuon 适配版使用 EMA，不启用 Nesterov
必要时转置 W 和 S，使行数 >= 列数
Q = W/R
A = Q.T @ S
K = (A - A.T)/2
H = S - Q@A                   # 方阵时直接设 H=0，避免放大抵消误差

cutoff = 8 * eps_FP32 * normF(S)
SIGN_BLOCK(B) = 0 if normF(B)<=cutoff else MSIGN(B)
D = Q@SIGN_BLOCK(K) + SIGN_BLOCK(H)
Y = W - eta*R*D
W_new = R * POLAR_RETURN(Y)
必要时将 W_new 转回
```

两个块分别做符号操作，不是对 `TP_W(S)` 只做一次 MSIGN；组合方向也没有再做整体 unit-F 或 unit-spectral 归一化。这是本项目采用的 Stiefel product-ball 方向与 ambient EMA 适配，不把原 iMuon 全套理论结论直接转移到当前 BF16 实现。[iMuon 论文](https://arxiv.org/abs/2605.09238)

Stiefel 的 MCSD/TP 使用 Frobenius-正交投影：高矩阵时 `TP_W(Z)=Z-W*sym(W.T@Z)/R^2`，宽矩阵转置处理。它们与 iMuon 共用 `POLAR_RETURN`：

```text
默认数值返回: 10 步 BF16 Polar Express + 3 步 FP32 cubic NS polishing
精确参考返回: thin SVD 得到 U,Sigma,V.T，返回 U@V.T
```

polishing 在内部短边 Gram 矩阵上实施 `X <- (3X-XX.T@X)/2`。默认路径是近似 polar 返回，不保证机器精度下严格正交；精确参考要求 trial 满秩，秩亏时当前代码报错。LMO 的 partial polar 与用于满秩 Stiefel 返回的 full polar 不能混为一谈。

<a id="settings"></a>

## 6 最近实验的具体运行设置

### 6.1 Frobenius 统一协议

| 项目 | 设置 |
|---|---|
| 模型 | 28 层 Dense decoder-only；width=256，FFN=768，Q heads=4，KV groups=2，head dimension=64；73,418,496 参数 |
| 架构 | SwiGLU、GQA、QK normalization、RoPE base=1e6、pre-RMSNorm、输入输出 embedding 不共享、dropout=0 |
| 数据与 batch | OLMo-Mix-1124；sequence length=2048，global batch=128，micro batch=4 |
| 3B | 11,445 iterations，实际 3,000,238,080 tokens；warmup=229 |
| 6B | 22,889 iterations，实际 6,000,214,016 tokens；warmup=458；独立准备的 6B training index |
| 调度 | 线性 warmup，再 cosine 衰减到 peak LR 的 10%；不是从旧 3B 模型直接续训 |
| 验证 | 相同 3B valid index；每 200 steps 和训练结束评估，每次 40 batches |
| 精度与裁剪 | BF16 模型，FP32 master/state；全局梯度范数裁剪阈值 1 |
| 辅助优化器 | AdamW，peak LR=0.001，betas=(0.9,0.95)，eps=1e-8；同样的调度比例 |
| 保存与报告 | 最近补测关闭 checkpoint；保留配置与日志；报告最终计划步的 val loss，不挑最低中间点 |

### 6.2 方法与动量配置分开记录

| 实验分组 | MCSD / MCSD-TP | MuonH / NorMuon-Sphere |
|---|---|---|
| 原始 LR 网格 | 0.95 EMA | 0.9 Nesterov |
| 匹配 0.9 EMA 网格 | 0.9 EMA | 0.9 EMA |
| 完全匹配对照，LR=0.010 | 0.95 EMA | 0.95 EMA |
| 新增 LR=0.008 的 3B 补测 | 0.95 EMA | 0.95 EMA |
| 新增 LR=0.010 的 3B 动量补测 | 0.95 Nesterov | 复用已有对照，不冒充同 Nesterov 配置 |

这些是已登记的实验设置，不把“提交”“运行”和“完成”混为同一状态；最新结果以结果文档与 Slurm 核验为准。`beta=0.95` 在 NorMuon 中可能同时出现在一阶动量和二阶行统计，但二者是独立参数。

### 6.3 理解启动器

以下示例只生成解析后的配置，不提交 GPU 任务。默认项目路径以当前仓库为准；H20 正式实验使用其已冻结的训练目录。**本次只发布说明文档，示例依赖的 `width256_3b_shape_radius.json` 仍是本地实验配置，未随本文上传；仅克隆公开仓库尚不能原样运行此命令。**

```bash
python scripts/manifold/launch.py \
  --config configs/manifold/width256_3b_shape_radius.json \
  --geometry frobenius --method manifold_mcsd \
  --stage main --tokens 3000000000 --width 256 --gpus 1 \
  --lr 0.008 --momentum-beta 0.95 --no-nesterov \
  --seed 1234 --direction-normalization none --save-interval 0
```

将方法改为 `manifold_mcsd_tp`、`manifold_muonh` 或 `manifold_normuon_sphere` 即选择另三种算法；将 `--no-nesterov` 改为 `--nesterov` 才启用梯度组合。正式提交还会检查源码、数据、环境和完整配置；不要对已有实验重复添加 `--submit`。

<a id="sources"></a>

## 7 源码与文献

### 7.1 从哪段代码核对

以下相对链接可在项目中直接阅读。伪代码省略异常处理、日志、分组遍历和通信，不省略改变方向或步长的数学操作。

| 内容 | 实现位置 |
|---|---|
| MCSD/TP 动量、两次 TP、trial 和返回选择 | [manifold_mcsd.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_mcsd.py)，`ManifoldMCSD.step` |
| MuonH/NorMuon/iMuon 与 spectral 基线适配 | [manifold_baselines.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py)，`baseline_update` 等 |
| 半径、切空间、精确参考返回 | [manifold_geometry.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_geometry.py) |
| 10 步 BF16 符号与 FP32 polishing | [manifold_polar.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_polar.py) |
| QKV / FFN 逻辑分组 | [manifold_layout.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_layout.py) |
| Spectral MCSD 的 PI/top-k 返回 | [manifold_spectral.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_spectral.py) |
| SSO 原数值内核、PI、二分法 | [spectral_ball_utils.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py) |
| MuonSphere 原数值内核 | [muon_ball.py](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py)，`compute_muon_ball_update` |
| 矩阵零 weight decay 和辅助参数分组 | [optimizer/__init__.py](../../Megatron-LM/megatron/core/optimizer/__init__.py) |
| 实际训练参数与源码哈希检查 | [run.py](../../scripts/manifold/run.py)、[launch.py](../../scripts/manifold/launch.py) |

本次为本地源码阅读核对；H20 连接超时，未重新获取远端文件哈希。此前提交登记记录的冻结训练 commit 为 `fc15600a2898d1748c8b34fe8d3945b1036f03ab`。本地存在未提交修改，因此不能仅凭这个 commit 名称断言文档中的每个文件与远端逐字一致。

用于后续比对的本地 SHA256：

```text
manifold_mcsd.py     40f012b887ba8637a0a3fcb4525b643dd0e81b9425314be2a941328b698faa27
manifold_baselines.py 270c63079c5d0b9e1d26dcd0f913c0deaee3777edf13a3f2ca9338e4f1e8a0bf
manifold_geometry.py e9b703d1a1d699eac19af76f2146658f49e922bf57e891f7ce7700ff3638b2a5
manifold_polar.py    4e188c52806aaf07a568594f7c08e3fde87080b69fc8459d326941ed15f89ecd
```

### 7.2 原始方法来源

- [Hyperball / MuonH](https://arxiv.org/abs/2606.16899)：权重和更新的 Frobenius 范数控制。
- [NorMuon](https://arxiv.org/abs/2510.05491)：符号输出后的逐行二阶统计；球面包装是本项目的组合。
- [iMuon](https://arxiv.org/abs/2605.09238)：内蕴范数 LMO；本文只描述当前 Stiefel 适配实现。
- [SSO 官方项目](https://github.com/Unakar/Spectral-Sphere-Optimizer)：SSO 与 MuonSphere 基础实现。

算法来源、数值近似、超参数与本项目的适配边界应同时保留，避免把几何上理想的更新、实际 BF16 内核和整套训练方案写成同一件事。
