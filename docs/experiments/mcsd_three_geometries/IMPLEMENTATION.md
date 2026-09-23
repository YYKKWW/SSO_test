# MCSD 三约束语言模型实验：实现规范与验收文档

版本：2026-09-23 / v2。供另一设备上的 YYKKWW/SSO_test 项目执行。建议先读 START_HERE.md 和 RESEARCH_OBJECTIVES.md。

## 0. 本轮交付目标、状态和论文边界

论文的三组实验拟为：传统随机 PCA；第二个真实受约束任务待定；第三个为本文的固定架构语言模型实验。本轮只实现第三组，不改动论文正文或旧 PCA 结果。

**目标：** 在同一个小型 Dense Transformer 和相同训练协议上，实现 Frobenius 球面、spectral sphere、scaled Stiefel 三种参数约束。在各约束内比较 MCSD、MCSD-TP 和相应强基线；跨约束比较展示几何与框架覆盖，不能归因成算法本身优劣。

**研究任务：** 除实现和部署外，要用测试定位算法的优势、失败与开销来源，并以最小对照指导改进。具体科学假设、成功标准、诊断表、变体登记与开发/正式评价隔离见 [RESEARCH_OBJECTIVES.md](RESEARCH_OBJECTIVES.md)。该文件与本实现合同共同执行：原算法作为冻结参照，有证据的改进以独立变体检验，不静默改写原定义。

本文件是实现合同，不是已完成实验。experiment_spec.json 为待接入配置合同，不能直接当成现有 Megatron 的原生输入。文中“建议新增”的类、文件和命令需要目标设备实现。

只读依据：当前 main1.tex 的 Algorithms 1/2，SHA256 为
0FE624F3A4CF4558DA2032D1D4C286929F1EEA5029FAB075C17B799D6B516BAA。
其顺序为当前样本 → ambient EMA → 切投影 → LMO → 返回。本文件不沿用早期笔记中的延迟动量顺序。

**结论范围：**

- Frobenius sphere 与 Stiefel 支持光滑流形实验。
- spectral sphere 只在最大奇异值单重的区域具有本文使用的光滑几何。奇异值重合的情况不能自动套用纯 MCSD/MCSD-TP。
- BF16、梯度裁剪、固定 NS 迭代、近似返回与混合辅助优化器构成实际训练设置，不能笼统称为严格理论条件已验证。
- 三种几何都要实现和检查；长训由正确性、几何适用范围与实测成本决定。未通过门槛的配置应保留失败或限制说明，不能默默从结果表消失。

## 1. 仓库安排与真实接入位置

不需要新 GitHub 仓库。建议在现有 SSO_test 创建独立分支 exp/mcsd-three-geometries。另一设备检查实际 HEAD 和本地改动；本次核查基准为：

https://github.com/YYKKWW/SSO_test/tree/d48cfc1da011cce68e94b2552ed095e62eac2df3

保留旧 spectral 代码和脚本，以新文件/显式配置添加功能。不能把旧实验重新命名后当成新几何实验。

### 1.1 已核实的入口

| 现有文件（相对 SSO_test 根目录） | 用途 |
|---|---|
| slurm/spel_olmo_1b_h20.sbatch | 已有 OLMo/H20 实际启动模板，优先复制为新脚本。 |
| Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py | 旧 spectral SpEL 实现，可参考矩阵分块和接口，不能直接当三几何公共 step。 |
| Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py | **spectral sphere** 的 MuonSphere 类路线，不是 Frobenius MuonH。 |
| Megatron-LM/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py | 现有公共更新次序，包含 WD、EMA、Nesterov 和最终 add；新方法需要避免双重更新。 |
| Megatron-LM/emerging_optimizers/orthogonalized_optimizers/__init__.py | 新类导出。 |
| Megatron-LM/megatron/core/optimizer/emerging_optimizers.py | 优化器注册、config-to-kwargs 和状态初始化。 |
| Megatron-LM/megatron/core/optimizer/optimizer_config.py | 新算法配置字段。 |
| Megatron-LM/megatron/training/arguments.py | CLI 参数与 choices。 |
| Megatron-LM/megatron/core/optimizer/layer_wise_optimizer.py | 参数分组、二维权重与辅助参数的路由。 |
| Megatron-LM/megatron/training/training.py | 模型初始化、精度转换、optimizer 创建、checkpoint load 等生命周期。 |

旧启动文件默认 global batch 是 64；历史 sweep 会覆盖成 128。新方案必须显式填写 128，不依赖继承默认值。

旧启动文件保存 checkpoint 时带有 --no-save-optim；新入口移除此项，并确认没有其他上层参数禁用 RNG/optimizer 状态保存。

旧 SpEL 的 use_nesterov 参数存在 default=True 的 store_true 路径，删除命令行开关不等于关闭。新配置必须能显式关闭 Nesterov，并打印最终生效值。

### 1.2 建议新增的结构

~~~text
Megatron-LM/emerging_optimizers/orthogonalized_optimizers/
  manifold_geometry.py       # 三几何投影、回缩、检查；不依赖训练全局状态
  manifold_mcsd.py            # 清楚定义的 step，MCSD / MCSD-TP
  muon_h.py                   # 新 Frobenius MuonH 基线
  manifold_baselines.py       # iMuon 方向的明确适配（如需要）

Megatron-LM/tests/unit_tests/optimizer/
  test_manifold_geometry.py
  test_manifold_integration.py
  test_manifold_resume.py

configs/manifold_dense/
  common.json                # 从本包配置合同转成实际可读 schema
  methods/...
  machine.local.example.json # 只有字段示意，没有凭据与设备绝对路径

scripts/manifold/
  preflight.py
  launch.py
  summarize.py

slurm/manifold_dense_h20.sbatch
docs/experiments/manifold_dense_protocol.md
~~~

路径可按目标项目惯例调整，但保持“数学核心、训练适配、实验配置、统计”分离。禁止写一个同时修改全局 CLI、初始化权重和提交 Slurm 的不可单独检查函数。

## 2. 锁定模型与训练协议

### 2.1 主模型：拟定约 127M 的 Qwen3-style Dense

这是参考现有架构缩小后的研究配置，不是官方 Qwen3 已发布 checkpoint，不加载预训练权重。

| 字段 | 建议初始值 |
|---|---:|
| layers | 28 |
| hidden size | 384 |
| FFN hidden size | 1152 |
| query heads / KV groups | 6 / 3 |
| head dimension | 64 |
| sequence length | 2048 |
| activation / norm | SwiGLU / RMSNorm |
| QK norm | 开启 |
| position | RoPE，base=1,000,000 |
| norm epsilon | 1e-6 |
| embedding / output | 不共享 |
| linear bias / dropout | 关闭 / 0 |
| tokenizer | 保留已有 OLMo tokenizer，并记录 revision、vocab 与 padding |
| precision | BF16 前反向，FP32 master weights 与动量 |

现有小宽度脚本可能将 KV groups 默认设成 query heads，从而变成 MHA。本配置必须显式指定 3 个 KV groups，并检查 6 % 3 == 0。

按 vocab=100352 估算，主要矩阵参数约 126,615,552，其中非 embedding/head 矩阵约 49,545,216。精确数量以真实构建后的参数统计为准。必须同时报告总参数、非 embedding 参数、受约束参数；不能只以“约 127M”掩盖词表占比。

这是**新建议配置**；它与旧 28 层/4096 上下文实验的绝对 loss 不直接横比。若决定保留 4096 上下文，须在所有方法中一起改变，并重新生成步数预算、配置 hash 与吞吐测量。

### 2.2 固定约束范围和矩阵粒度

约束每个 Transformer block 的七个逻辑矩阵：

Q、K、V、O、gate、up、down。

不约束 token embedding、lm_head、norm scale 或 bias；这些参数的处理对所有方法保持一致。全局模型参数因而是“受约束矩阵 × 其他参数”的组合，不能称整个网络全部处于一个紧流形。

主配置按 component 拆 Q/K/V 和 gate/up，不按单 head 拆分。拟定形状（采用 out×in 记法）：

- Q/O：384×384；
- K/V：192×384；
- gate/up：1152×384；
- down：384×1152。

Megatron fused QKV 可能按 query group 交错存储，不能直接按连续三等份切片。实现逻辑矩阵读取/写回适配器，并验证与真实前向 layout 一致。FC1 的 gate/up 同样需核对 layout。

输出 parameter_manifest.json：每个逻辑块的名称、shape、所属物理参数、切片/gather 规则、是否受约束、优化器分组、半径和缩放。检查无重叠、无漏项、写回后前向值正确。

第一版 TP（tensor parallel）=PP=EP=1，distributed optimizer 关闭；支持单 GPU 或普通数据并行即可。**MCSD-TP 中 TP 表示 tangent projection，与 tensor parallel 是两回事。** 不允许将每个 tensor-parallel shard 独立投影，然后宣称约束了完整矩阵。

### 2.3 数据、batch 与调度

保留现有 OLMo 混合比例、tokenizer 和二进制数据格式。重新建立与本次序列长度匹配的样本索引时记录 seed、manifest、混合权重和实际 consumed tokens。训练、开发验证、最终 held-out 评价的数据划分固定且互不泄漏。

建议：

- global batch：128 sequences；
- sequence：2048，故每次更新 262144 tokens；
- micro batch 初始 4；按显存测量统一调整；
- accumulation = 128 / (DP world size × micro batch)，必须为整数；
- 每预算独立完成 2% warmup + cosine，最终 LR 为 peak 的 10%；
- 全局梯度裁剪 1.0，所有方法一致；准确数学检查关闭裁剪；
- 辅助 AdamW 初始 LR=1e-3、betas=(0.9,0.95)、eps=1e-8，embedding/head WD=0.1，norm/bias WD=0。

上述辅助参数是研究方案默认值，不宣称已经最优。开发阶段如需调整，只能统一改动全部方法的共同配置，冻结后再开展正式比较。

| 阶段 | 目标 tokens | steps=ceil(target/262144) | 实际 tokens |
|---|---:|---:|---:|
| pilot | 100,000,000 | 382 | 100,139,008 |
| tune | 1,000,000,000 | 3815 | 1,000,079,360 |
| main | 3,000,000,000 | 11445 | 3,000,238,080 |
| extended | 6,000,000,000 | 22889 | 6,000,214,016 |

若 global batch 改变，重新计算这张表。不能用梯度累积的 micro-step 数代替 optimizer-step 数。

若本机只有 1B 独立训练 tokens，必须明确是继续准备同混合新数据，还是重复 epoch；不允许静默重复后称使用了 3B/6B 不同数据。

## 3. 初始化与层尺度：公平比较的关键

每个种子首先建立一份标准未约束初始网络。对一个待约束矩阵，记原始 FP32 权重为 A，p=min(m,n)，R=||A||F：

- Frobenius sphere：W0=A，固定半径 R；
- spectral sphere：W0=A，固定半径 s=||A||op，初始化用准确 SVD 确认最大奇异值单重；
- scaled Stiefel：a=R/sqrt(p)，W0=a·polar_full(A)。

辅助参数直接共用原始网络的同一初值。同约束内所有方法必须加载同一个初始化文件，并核对 hash。跨约束共享随机种子和 A，但 Stiefel 权重不同，必须记录初始 loss 和奇异值统计。

**禁止把所有方法初始化成同一个 scaled-Stiefel 矩阵。** 其 p 个非零奇异值都等于 a；当 p≥2 时，spectral sphere 的顶端 gap 为零，光滑切投影从第一步就不适用。

主 MCSD 采用固定层尺度

\[
c_\ell=R_{\ell,0}/\sqrt{p_\ell},\qquad \alpha_{\ell,k}=\eta_k c_\ell .
\]

c 在训练中不更新。等价地，可将 c 放入 LMO，使用加权产品方向球
\[
\max_\ell \|S_\ell\|_{\rm op}/c_\ell\le1.
\]
两种写法只能使用一次，不得在外层再重复乘已有 Muon shape scaler。

半径和 c 是 optimizer/checkpoint 的持久状态，不在恢复时从当前权重重新估计。新训练初始化应在 BF16 转换与 FP32 master 创建前完成；若框架只能在后面 hook，必须同步修改 master/model 两份。加载 checkpoint 时不重新投影、不重新抽种子、不重新选择半径。

## 4. 算法定义：必须逐项落实

对本节公式，W 是 step 开始时可行的 FP32 master 矩阵，g 为本轮得到的梯度（实际训练可能经过共同裁剪）。

\[
M^+=\beta M+(1-\beta)g,\quad Q=P_WM^+,\quad S=-\operatorname{msign}_{partial}(Q).
\]

- beta 初值 0.95；M0=0；
- ambient EMA，不作 transport；
- 默认无 Nesterov；
- 受约束权重 WD=0；
- MCSD：\(W^+=\Pi_C(W+\eta cS)\)；
- MCSD-TP：\(D=P_WS,\quad W^+=\operatorname{Retr}_W(\eta cD)\)；
- 切投影后不再把 D 强制归一化；
- 当前样本用于本轮方向，不能先走步再更新动量；
- g=None 的块跳过更新，并保留状态；Q=0 时 LMO 返回零。

若使用 FP32 master 和 BF16 模型副本，解释清楚模型前向是在量化副本上计算；这属于实际混合精度设置。

### 4.1 Frobenius sphere

\[
C_F=\{W:\|W\|_F=R\},\quad
P_WZ=Z-W\langle W,Z\rangle_F/R^2,\quad
\Pi_F(Y)=RY/\|Y\|_F.
\]

MCSD 与 MCSD-TP 都可使用上述径向返回，区别在于 trial 前是否再次切投影。Y=0 时欧氏投影不唯一且局部回缩不适用；主运行记录 zero_trial 并停止，不用 eps 除法偷偷返回零矩阵。

**主 LMO 是谱范数 LMO，不是 Frobenius LMO。** 若 LMO 改成切向 Q 的 Frobenius 归一化，两算法在相同返回下重合，只作为一致性检查。

可检查以下单步恒等式：b=<W,S>/R²、D=P_WS，1+αb>0 时，
\[
\Pi_F(W+\alpha S)=\Pi_F\!\left(W+\frac{\alpha}{1+\alpha b}D\right).
\]
它用于角度/有效步长诊断，不表示相同 LR 下两条完整训练轨迹必然相同。

### 4.2 Scaled Stiefel

将 wide 矩阵及对应梯度转置成 tall，完成后转置回原形状。对 m≥n：

\[
C_{\rm St}=\{W:W^\top W=a^2I\},\quad
P_WZ=Z-\frac{W}{a^2}\operatorname{sym}(W^\top Z),\quad
\Pi_{\rm St}(Y)=a\,UV^\top,\quad Y=U\Sigma V^\top.
\]

两个主方法均用准确 full-polar 返回。MCSD-TP 的这一返回是 polar retraction；MCSD 的是局部最近点投影。QR 可以另设效率变体，但不能无说明替换 MCSD 的最近点返回。

监控 trial 最小奇异值；rank-deficient trial 的非唯一 full-polar 投影不应被当成通常的光滑局部返回。准确参考遇到数值 rank deficiency 应报告并停止该配置。

**partial polar 与 full polar 要分别实现：**

- LMO：对零奇异值选零，零输入返回零；
- Stiefel 返回：保留完整 p 个正交方向，不能用 partial polar 导致返回点不满足约束。

正方 scaled-Stiefel 块上，精确 partial-polar 谱 LMO 会保留切向性，MCSD/TP 可重合。不要把重合判为错误；保留 K/V、MLP 等矩形块，才能分析非平凡的 TP 差别。

### 4.3 Spectral sphere

在最大奇异值单重的区域，令 H=u1 v1ᵀ：

\[
C_S=\{W:\|W\|_{\rm op}=s\},\quad P_WZ=Z-\langle H,Z\rangle_FH.
\]

**MCSD exact return：**
若 Y=U diag(σ1,…,σp)Vᵀ，σ 降序，则
\[
\Pi_S(Y)=U\,\operatorname{diag}(s,\min(\sigma_2,s),\ldots,\min(\sigma_p,s))\,V^\top .
\]

**MCSD-TP radial return：**
\[
D=P_WS,\qquad W^+=s(W+\eta cD)/\|W+\eta cD\|_{\rm op}.
\]

这是两个不同的合法局部返回设计。径向缩放不是一般 spectral sphere 的欧氏最近点投影，不能用来替换 exact MCSD 后仍保留 exact 标签。对 diag(2,1)、s=1，最近点是 diag(1,1)，径向结果是 diag(1,0.5)。

**已知奇异点的处理必须显式：**

- exact MCSD 的 clipping 可能使多个奇异值同时等于 s，因而产生非光滑点；下一步不能任意选一个 top singular vector 后继续声称纯光滑算法。
- exact projection 已计算 SVD，可复用其奇异值得到输出 gap，避免再做一次 SVD。可以缓存该准确返回的奇异向量用于下一步，但要确保中间没有别的操作改变权重。
- 参考运行用准确 gap；建议相对 gap≤1e-4 作为数值适用范围停止阈值，1e-3 作为预警。阈值是工程诊断，不是自动满足定理步长条件的证书。
- practical 路径可以每 200 步抽样做准确 gap 审计，但这不保证审计间隔内都处于安全区。
- 一旦发现已知 tie/阈值触发，记录层、step、gap、loss、返回类型并停止 pure-smooth run。需要继续时另建定义明确的 MCSD-PGD 或其他经验变体；本月主线不默认启用。
- 不允许静默缩小步长、切 PGD、改径向返回后仍标原方法；这些必须另有 method_id 与配置。

旧 top-k / power 实现若保留，用 Practical-SPEL 名称，并报告误差与精度。它不能等同本节准确 MCSD。

## 5. 精度、训练接入与基线

### 5.1 新方法需要明确的 step

旧公共 orthogonalized optimizer 的顺序包含：

已有梯度 → decoupled WD 修改参数 → EMA → Nesterov（可选）→ orthogonalize → 外层 add。

旧 SpEL 的 orthogonalize 内还会先就地缩放 W。直接只替换 orthogonalize 很容易导致梯度对应的 W 与几何计算使用的 W 不同、重复应用步长、或者返回后再次 add。

新实现应拥有明确的 step/return 契约：

1. 读取本轮梯度，完成 unscale 与共同裁剪；
2. 在同一 W 上更新 M，计算 P_W、LMO 和最终 W_new；
3. 将 W_new 一次写入 master；
4. 由框架同步模型副本；
5. 不让外层再次添加方向或衰减受约束矩阵。

辅助参数由其固定优化器更新。必须检查一个参数只被一个 optimizer 管理。

### 5.2 准确参考和实用版本分开

准确检查用 float64 SVD；小模型集成参考可用 FP32 SVD。实用 LMO 可使用固定 NS 系数与迭代数，建议从 8 步开始，但需要保存系数来源和实际计算 dtype。

不能依赖输入为 FP32 就认定内部是 FP32；旧 spectral utilities 会在内部转 BF16。

对谱 LMO记录两项：
\[
\max(0,\|S\|_{\rm op}-1),\qquad
\langle Q,S\rangle_F+\|Q\|_*.
\]
若方向不在球内，仅有较小线性目标误差并不足以满足 main1.tex 的 inexact-LMO 条件。

有限步 NS 用作 Stiefel 返回时，必须另标 return_approximation，并报告约束误差；不能认为 inexact LMO 定理覆盖了近似返回。准确基准与快速版本使用不同 run/config id。若近似返回误差过大，可开发更准确 polar，但不能无标签偷偷改成 QR。

### 5.3 最小方法矩阵

| 约束 | 主方法与基线 |
|---|---|
| Frobenius | MCSD、MCSD-TP、**新实现的 MuonH** |
| Spectral | MCSD-Exact、MCSD-TP-Radial、已有 SSO、已有 MuonSphere（muon_ball） |
| Stiefel | MCSD、MCSD-TP、iMuon-direction + common ambient EMA |
| 无约束应用参照 | AdamW、MuonW |

共有 12 个拟定方法。所有几何内算法共享初始文件；无约束两方法只跑一次作为共同应用参照，不按三几何重复计为六个算法。

有预算时添加 Stiefel Manifold Muon 与 RSGD；RSGD 也可在第二组分类头实验中作全面对照。不能故意省略历史上表现更好的 MuonSphere，只保留较弱 SSO。

MuonH 的核心 wrapper 为：
\[
U=\text{MuonUpdate}(g,M),\quad
W^+=R\,\frac{W-\eta R\,U/\|U\|_F}{\|W-\eta R\,U/\|U\|_F\|_F}.
\]
R 是初始化保存的固定值。U=0 则保持 W，不通过除零生成 NaN。必须说明 momentum/Nesterov 采用哪个原始版本；统一 EMA 的改编版另作标注。MuonH 不自动包含 MCSD 的前置切投影。

iMuon 的方向块范数与 MCSD ambient spectral 球不同，学习率应独立调优。若复用作者方向并添加共同 EMA，名称必须包含 adaptation，不声称完整复现其训练系统。

已有 SSO/MuonSphere 的原生更新顺序、缩放和近似保留在 native baseline 描述中；框架层统一辅助参数、数据、模块粒度和评价。已有类若不能支持本次 component 粒度，需要明确适配并测试，不能暗中沿用 per-head。

## 6. 必须完成的测试和验收标准

这些检查验证几何与训练接口，避免用大规模训练发现基础错误。只对有意义的行为写测试，不为配置常量逐行写镜像测试。

### 6.1 数学与数值

覆盖 tall、wide、square、zero、rank-deficient、近重根输入：

- tangent projector 的线性、幂等、自伴性、与法空间正交；
- scaled-Stiefel 的 a² 因子和 wide 转置处理；
- spectral LMO 的范数与线性目标，partial/full polar 的区别；
- 两种返回均满足对应约束；检查 D Retr(0) 的切向一阶行为；
- Frobenius LMO 下 MCSD/TP 重合；
- 正方 Stiefel 的谱 LMO 切向保持；矩形块不能被强制期待重合；
- Frobenius 球面的有效步长恒等式；
- spectral diag(2,1) 投影、精确 top-tie 检测，以及 shared-Stiefel-init 被拒绝；
- zero_trial、秩亏 trial 和 known spectral singularity 的明确失败原因。

建议归一化残差：

\[
e_F=|\|W\|_F/R-1|,\quad
e_{\rm St}=\|W^\top W/a^2-I\|_F/\sqrt p,\quad
e_S=|\|W\|_{\rm op}/s-1|.
\]

small float64 reference 的目标容差 1e-10，FP32 几何参考为 1e-5；这些是初始验收容差，应按矩阵条件数给出解释，不能为让测试通过而任意放宽。practical BF16 存储副本与 FP32 master 分别报告，量化副本不以 float64 标准验收。

### 6.2 训练集成

- current-sample EMA 首两步与手工参考一致；
- 不发生两次 step、不对受约束权重做额外 WD、不重复乘层尺度；
- fp32 master 与模型副本同步，第一次 forward 前已经按相应约束初始化；
- QKV/FC1 分块 round trip 无损，块不重叠，并与实际 forward 对齐；
- 所有辅助参数在各方法中分组、优化器与超参数一致；
- 保存 N 步、重启继续，与连续训练 N+M 步对比：动量、半径、scheduler、RNG、data cursor、weights 均一致或在明确的确定性数值容差内；
- BF16 重启不能重建随机初值或从当前 W 推算新半径；
- 普通 DP 对完整逻辑矩阵更新，不能把单卡通过当成 Tensor Parallel 已支持。

### 6.3 真实模型短运行

从相同初始化，逐方法至多运行 100 optimizer steps。记录训练 loss、有限性、约束误差、gap（适用时）、tokens/s、显存和返回耗时。smoke 的 loss 不用于排算法优劣。

在 H20 计时，包含梯度同步、LMO、投影/返回与 optimizer 相关通信；预热/编译、验证、checkpoint 和昂贵诊断分别计时。异步 CUDA 的计时应同步或用可靠事件，不能只计 CPU launch 时间。

已观察到的硬件性能只对实际设备有效；RTX/H200/A100 的结果不能换名为 H20。

## 7. 运行阶段、搜索与预算

### A. 实现与 smoke

先几何、后 optimizer 集成、再真实模型 100 步。默认 launcher 为 dry-run。不以“能 import”作为 smoke 完成。

### B. 100M pilot

只对通过阶段 A 的配置进行稳定性和吞吐预试验；谱球面必须给出是否仍满足光滑适用范围的证据或明确经验标签。若 full SVD 返回成本高，报告真实成本后再选准确版或清楚命名的近似版。

### C. 1B 调参

每方法初始 6 个完整候选，建议 3 个对数间隔 LR × 2 个关键参数：

- MCSD/TP：各自 LR 与 beta；
- AdamW/MuonW：各自 LR 与 WD；
- MuonH、SSO、MuonSphere：各自合理 LR 与其一个关键参数。

MCSD 初始探测中心可从 eta=0.01、beta=0.95 起；这只是调试初值，不直接规定所有基线相同 LR。先校准方法尺度，再冻结候选列表。若最佳值落边界，以统一追加配额扩展。失败候选和搜索成本也计入记录。

### D. 3B 正式比较

每方法三个独立正式种子 2027/2028/2029，配对初始化与数据顺序。开发种子为 1234，不混入正式统计。

如果直接迁移 1B 选出的配置，报告为“1B 调参、3B 预算迁移”；若主张“3B 下各自充分调优的最优表现”，要给每方法相同的 3B 搜索机会，不能只为新方法调长预算。

验证按固定频率，例如每 200 steps；最终 held-out NLL 只在配置锁定后评价。最终 NLL 与最后 checkpoint 对齐；若另报验证选 checkpoint，所有方法使用同规则。

### E. 6B 延长

预先写明选择规则，保留提出的方法及其强基线。新建完整 6B 调度，不能将带不同衰减进度的 3B checkpoint 直接续训后当作同一从零预算实验；续训问题需要另设名称。

### 总预算

8 H20×24×30=5760 GPU-hours 仅为理论上限。计划内约 4000，预留约 1760。对每方法实测吞吐 v（整个作业 tokens/s）、卡数 g：

\[
GPUh\approx gD/(3600v)+\text{验证、保存、启动等对应成本}.
\]

12 方法×6 候选×1B 为 72B tokens；12 方法×3 种子×3B 为 108B tokens。两项已约 180B tokens，尚未计 3B 再调参和 6B 延长。必须逐方法按实测吞吐换算，不能只算一个最便宜算法。

若超过预算，优先保留各几何的 MCSD/TP、历史强基线和正式种子，减少扩展规模与非核心消融；任何删减规则在最终评估前锁定并说明。不保证所有谱 exact 配置都能以合理成本完成 3B。

## 8. 启动器接口与输出合同

下面是**需要实现的接口示例**，当前仓库不存在时先开发，不能把它当作现有可运行命令。

~~~bash
python scripts/manifold/preflight.py \
  --spec configs/manifold_dense/common.json \
  --machine configs/manifold_dense/machine.local.json

python scripts/manifold/launch.py \
  --spec configs/manifold_dense/common.json \
  --machine configs/manifold_dense/machine.local.json \
  --phase smoke --method f_mcsd --seed 1234 --dry-run

python scripts/manifold/launch.py \
  --spec configs/manifold_dense/common.json \
  --machine configs/manifold_dense/machine.local.json \
  --phase main --method f_mcsd --seed 2027 --dry-run
~~~

launcher 输出解析后的完整配置、训练命令、数据路径存在性、目标/实际 tokens、资源请求与 run directory。提交模式应显式区分 local/Slurm 与 dry-run；运行目录存在已完成结果时拒绝覆盖，恢复训练使用 resume 参数与兼容性检查。

每个 run：
~~~text
runs/<architecture_hash>/<geometry>/<algorithm>/<budget>/<seed>/<config_hash>/
  resolved_config.json
  source_state.json
  data_manifest.json
  parameter_manifest.json
  initialization_manifest.json
  metrics.jsonl
  timing.json
  result.json
  failure.json                 # 若失败
  checkpoints/                 # 可重定向外部存储
~~~

source_state 至少有 commit、实际 dirty diff 指纹、环境包版本、GPU 型号、CUDA、内核系数来源。保存有效配置，而非只保存用户输入的覆盖项。

result 必须区分：pass、failed_numerical、failed_geometry、failed_environment、budget_skipped、not_run。不能把未执行写成 0 loss，也不能丢掉失败作业。

## 9. 需要画什么、如何解释

1. **同约束内主表：** 最终 NLL、perplexity、三种子波动、配对差值、实际 GPU-hours、显存、约束误差。
2. **loss 对 tokens / 对实际训练时间：** 两个口径都给，标清返回与诊断开销。
3. **跨约束表：** 固定骨架下的质量与成本；同时列初始 loss、半径策略、初始化差异。标题写“不同参数几何的表现”，不写“同一优化问题的算法胜负”。
4. **机制消融：** 前置切投影、后置切投影、实际更新角度匹配。F 球面重点区分有效步长与方向；spectral 主比较同时改变返回，归因于 TP 前需加匹配返回的局部对照。原参数方法与分析变体分开，详见 RESEARCH_OBJECTIVES.md §6。
5. **精度消融：** 准确 LMO/返回与 practical NS 的可行性、方向误差和时间；谱球面给 gap 审计与停止原因。

Frobenius 球面实际角度可用稳定 atan2 计算；Stiefel 和 spectral 不共用一个 R² 分母公式，采用一般 Frobenius 夹角 <W,W_new>/(||W||F||W_new||F) 作为矩阵方向诊断，并明确其不是所有流形的测地距离。

低频诊断用固定层或固定抽样列表。若发现可疑约束误差，在额外诊断作业核查，不把新增昂贵诊断仅施加给一方后用于时间排序。

## 10. 第一轮完成与长训就绪的判据

第一轮实现完成须同时满足：

- 几何函数、step 顺序、矩阵分块和已知失败路径均有实际检查结果；
- 真实模型参数数、形状、固定 GQA、辅助参数路由都与合同一致；
- H20 短运行或清楚标注尚未取得 H20 结果；
- 可精确/可解释恢复 checkpoint；
- 每方法的实际超参数、精度、返回类型、原生/改编身份已写入配置；
- 能生成 dry-run 命令和逐方法预算估计；
- 有明确的研究诊断报告、变体决策记录及下一阶段可检验的问题，不能只交付“已跑通”；
- 不修改 main1.tex、不覆盖旧运行、不创建重复的远程仓库。

完整 3B 实验只有在运行、评估、统计均完成后才标“完成”；实现文档、短运行或成功提交 Slurm 都不等于实验结论。

## 11. 文献与代码依据

- SSO_test 固定基准：[仓库](https://github.com/YYKKWW/SSO_test/tree/d48cfc1da011cce68e94b2552ed095e62eac2df3)；[H20 启动模板](https://github.com/YYKKWW/SSO_test/blob/d48cfc1da011cce68e94b2552ed095e62eac2df3/slurm/spel_olmo_1b_h20.sbatch)。
- 约束训练结构来源：[Controlled LLM Training on Spectral Sphere](https://arxiv.org/html/2601.08393v1)。
- MuonH 定义：[Fantastic II](https://arxiv.org/html/2606.16899v1)；[后续角度分析代码](https://github.com/mangocrazz/hyperball-may-not-be-a-free-lunch)。后者不是前者原作者仓库。
- Stiefel 方向：[Intrinsic Muon](https://arxiv.org/html/2605.09238v1)；[作者 Stiefel 代码](https://github.com/1bang118/manifold-intrinsic-muon/blob/4f1d4b14c62084071e445c08f2b3b61a44b6f2fe/non-llm/src/manifolds/stiefel.py)。
- Manifold Muon 备选基线：[Modular Manifolds 作者实现](https://github.com/thinking-machines-lab/manifolds)。
- 小模型优化器实验先例：[In Search of Adam's Secret Sauce，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5bd9aa206d782e4e1f7ab5d177a10828-Abstract-Conference.html)。
- 协议依据：[Fantastic 第一篇](https://arxiv.org/abs/2509.02046)，[Scion，ICML 2025](https://proceedings.mlr.press/v267/pethick25a.html)。
- MuonM 可作为后续动量/transport 参考：[论文](https://arxiv.org/abs/2608.28442)。本次未确认其作者公开训练代码，本交接不依赖该代码。

本包模型、半径方案、配置与阶段预算是为当前研究提出的设计，不是声称完整复现上述任何一篇论文。
