# 在另一台设备上启动本项目

这份交接包用于在现有 YYKKWW/SSO_test 中实现“固定小型 Dense Transformer、三种参数约束、各约束内比较算法”的实验。它包含实现规范和配置合同，**不包含已经完成的新优化器或可直接提交的训练作业**。

建议按下面顺序阅读：

- [RESEARCH_OBJECTIVES.md](RESEARCH_OBJECTIVES.md)：实验目的、科学假设、成功标准、机制对照，以及根据测试结果诊断和优化算法的规则。
- [IMPLEMENTATION.md](IMPLEMENTATION.md)：实现公式、仓库接入、基线、检查标准、实验与验收顺序。
- [experiment_spec.json](experiment_spec.json)：锁定的建议配置；需要新写的适配器读取，不是现有 Megatron 能直接使用的原生配置。
- 本文件：启动说明和可直接交给代码助手的任务。

## 是否需要新 GitHub 仓库

现在不需要。使用现有 SSO_test 的独立分支，建议名为 exp/mcsd-three-geometries。现有 Megatron、数据索引、环境和旧结果可复用。不同设备只需要同一个分支和提交，不需要为设备分别建仓库。

本包审计的公开基准提交为 d48cfc1da011cce68e94b2552ed095e62eac2df3。目标设备可以已有更新；先记录实际提交并检查差异，不要为对齐本包执行 reset/clean 或丢弃本地改动。

如果目标设备尚未有仓库，可以正常 clone：
~~~bash
git clone https://github.com/YYKKWW/SSO_test.git
cd SSO_test
~~~

如果已有仓库，就在其根目录工作，先查看 git status 与实际分支。在确认分支名尚不存在后可创建：
~~~bash
git switch -c exp/mcsd-three-geometries
~~~

将本包四个文件复制到该项目 docs/experiments/mcsd_three_geometries/。只将文档、配置、代码、轻量统计提交到 Git；模型、数据、完整 checkpoint 留在计算设备或适当存储中。未来需要独立发布或匿名投稿时，再整理精简的专用仓库。

## 直接交给另一设备代码助手的任务

下面这段可整体粘贴；项目须选中目标设备的 SSO_test：

> 请先阅读 docs/experiments/mcsd_three_geometries/RESEARCH_OBJECTIVES.md，再阅读 IMPLEMENTATION.md 与 experiment_spec.json，在当前 SSO_test 项目实现第三组实验：固定约 127M 的 Qwen3-style Dense Transformer，分别采用 Frobenius sphere、spectral sphere 和 scaled Stiefel 参数约束，在各约束内比较 MCSD、MCSD-TP 与指定基线。先读取项目适用的 AGENTS.md，检查实际提交、工作区改动及已有环境，在独立分支工作；保留旧实验和用户改动。
>
> 你的任务同时包括研究诊断和算法优化：回答统一框架的覆盖范围、同约束内竞争力、质量与时间折中，以及切投影/步长/动量/近似误差分别有什么作用。不以必须赢过基线为目标，不把跨约束排名归因于算法。根据实际结果提出可检验解释，先排除实现错误，再做单因素对照，并明确支持或不支持原假设。
>
> main1.tex 若在本设备可见，只读不修改；若不可见，以本包中冻结的更新顺序和公式为本轮实现合同。若当前论文或代码与合同冲突，先列出具体差异，继续完成不受影响的工作。
>
> 按规范先实现准确几何参考、当前样本 ambient EMA、独立 MuonH、矩阵分块与 optimizer 接入，再完成几何测试、FP32 master 同步测试、断点恢复测试和实际模型的短运行。所有三种几何都要实现和检查；谱球面发生已知非光滑点时保留失败证据，不能偷偷替换返回或切到 PGD。
>
> 可以直接修复错误、优化数学等价的实现、登记并实现有证据的单因素候选。结构改动使用新 method_id，保留原版与强基线，写明对当前定理的影响；不能暗改数据、半径或返回来制造提升。特别检查 Frobenius 上 TP 的有效步长解释，以及 spectral 上投影与径向返回的混杂。请维护 research/algorithm_decisions.md、theory_alignment.md 和 claim_evidence.md。每轮最多提出两个结构改进候选，给出最小验证、预期反例和成本。
>
> 先交付可复现的检查结果、H20 每步成本、显存与预算估计，以及只打印命令的 dry-run 启动器。首轮运行范围是测试和每个方法或已登记变体至多 100 个训练步的 smoke；这只能用于正确性、明显失稳和成本诊断，不能判定最终赢家。1B/3B/6B 长训使用单独命令，不属于这次交接的默认批量提交范围；若本会话后续已经授权相应阶段，则直接在该预算内推进。若没有 H20 或数据，完成本地可做的部分并明确尚未验证的项目。
>
> experiment_spec.json 是待实现的配置合同，不能直接传给现有 Megatron。请输出实际新增/修改的文件、已经通过与尚未执行的检查、解析后的最终配置、基线名称对应、目标设备上可运行的下一阶段命令。不要声称在其他硬件上的 smoke 等于 H20 复现。
>
> 每轮汇报说明：本轮研究问题、共同条件与唯一变化、实测证据、已排除和仍不确定的解释、下一项算法改进及其理论身份、下一阶段成本。开发只看验证集；正式方法冻结后再评估独立种子与最终 held-out，保留失败和负结果。

## 目标设备需要填的四类信息

新建不入 Git 的 machine.local.json，由实现者定义并读取：

- 项目、Megatron 和 Python 环境路径；
- 已有 train/validation/test 数据前缀、tokenizer 路径和版本；
- Slurm partition/account（如果使用调度系统）、每次作业卡数和时间上限；
- 日志与 checkpoint 存储目录。

不要把其他设备的绝对路径、调度账号或凭据硬编码进共享配置。

## 第一轮应交付什么

1. 新优化器和几何函数可以独立检查。
2. 小 Dense 的实际总参数/非 embedding 参数、Q/K/V 形状、分块范围已输出。
3. 三种约束的测试与已知边界均有报告。
4. 受约束权重每次 step 后返回约束，辅助参数与所有方法的路由一致。
5. checkpoint 恢复保留动量、固定半径、步长调度、数据进度和随机状态。
6. dry-run 输出可复现命令；完成短运行后才据实估算整月矩阵。
7. 有首轮研究诊断报告：算法目前的瓶颈、证据、下一项最小对照，以及应保留或停止的改进方向。

本包在原设备上只完成了文档与配置校验，没有创建 GitHub 仓库、推送分支或启动 H20 训练。
