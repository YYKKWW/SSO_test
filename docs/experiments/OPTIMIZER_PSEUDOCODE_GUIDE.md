# 优化器对照与伪代码

本文说明本项目的实际更新流程，重点为 **Frobenius sphere 的 MCSD、MCSD-TP、MuonH、NorMuon-Sphere**；其他几何见折叠区。实验结果另见 [3B/6B 汇总](FROBENIUS_3B_6B_SUMMARY.md)。

[核心区别](#overview) · [共同流程](#common) · [四算法伪代码](#frobenius) · [其他几何](#other-geometries) · [实验设置](#settings) · [源码](#sources)

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

### MCSD：先投影动量

```text
D = MSIGN(TP(S))
W_new = RETURN(W - eta*c*D)
```

TP(S) 切向，但经过非线性 MSIGN 后，D 一般不再切向；默认不再归一化 D。

### MCSD-TP：再投影输出

```text
D = MSIGN(TP(S))
D = TP(D)
W_new = RETURN(W - eta*c*D)
```

第二次 TP 作用于符号输出，而非重复处理动量。默认不重新归一化，更新长度可能缩短。

### MuonH：统一方向长度

```text
D = UNIT(MSIGN(S))
W_new = RETURN(W - eta*R*D)
```

不先投影动量。非零方向下，返回前的更新长度为 `eta*R`。

### NorMuon-Sphere：增加逐行二阶统计

```text
H = MSIGN(S)
v = beta2*v + (1-beta2)*mean(H^2, 按行)       # v 初始为零
A = H / (sqrt(v) + eps)                      # 按行广播
D = UNIT(A)
W_new = RETURN(W - eta*R*D)
```

默认 `beta2=0.95`、`eps=1e-10`，无偏差修正。统计的是 **MSIGN 输出的平方**，不是原始梯度平方；行对应原矩阵的输出维度。

<a id="differences"></a>

### 归一化消融与公平比较

MCSD/MCSD-TP 的 `unitF` 消融只替换最后一步，默认关闭：

```text
D = D / (normF(D) + 1e-12)                   # TP 版本在第二次 TP 之后归一化
W_new = RETURN(W - eta*R*D)
```

**加 unitF 仍不等于 MuonH**：前者使用 `MSIGN(TP(S))`，后者使用 `MSIGN(S)`。只有精确满秩 polar 满足 `normF(D)=sqrt(k)`，原 MCSD 的固定尺度才等价于这一长度归一化。

比较时分开报告：**相同 LR/动量的公平对照**，以及**各自最佳 LR 的调参结果**。相同 LR 不保证相同实际位移。

### 数值精度

Frobenius 四方法共用 **10 步 BF16 Polar Express，输出 FP32**，不是精确 SVD。配置 `baseline_msign_steps=8` 只控制 SSO/MuonSphere。稀疏审计仍可能调用 SVD，因此“主更新不用 SVD”不等于“训练完全没有 SVD”。

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
