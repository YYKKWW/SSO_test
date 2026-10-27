# Documentation Index

This directory is the documentation entry point for the H20 experiment repository.

## Layout

| Path | Purpose |
|---|---|
| `algorithms/` | Notes for optimizer variants and implementation-level algorithm changes that are not standalone experiment records. |
| `experiments/` | One document per reproducible experiment family. Put results, exact Slurm job IDs, data paths, model settings, and follow-up runs here. |
| `runbooks/` | Environment setup, H20 operational notes, debugging records, and cluster-specific procedures. |
| `templates/` | Copyable templates for new experiments or new project tracks. |

## Current Documents

| Document | Purpose |
|---|---|
| [experiments/mcsd_p2gd_results_summary.md](experiments/mcsd_p2gd_results_summary.md) | 统一入口：18 组已完成 1B/3B 对比、耗时、精度及后续优先级。 |
| [algorithms/current_comparison_pseudocode.md](algorithms/current_comparison_pseudocode.md) | 当前四类算法的中文流程、伪代码与可展开 Python 源码。 |
| [experiments/cluster_results_20261028.md](experiments/cluster_results_20261028.md) | 六组 3B 最终结果、K10/K20 精度与耗时比较。 |
| [experiments/cluster_results_20261026.md](experiments/cluster_results_20261026.md) | 已完成 1B 结果与 3B 中期历史快照。 |
| [experiments/cluster_candidates_1b_20261024.md](experiments/cluster_candidates_1b_20261024.md) | 12 组匹配 1B 实验的 loss、耗时、分支与离线精度审计。 |
| [experiments/cluster_selected_3b_20261025.md](experiments/cluster_selected_3b_20261025.md) | 六组 K10/K20、MuonSphere、SSO 的 3B 配置与提交记录。 |
| [runbooks/cluster_experiments_reproduction.md](runbooks/cluster_experiments_reproduction.md) | 当前实验分支的环境要求、测试和启动流程。 |
| [algorithms/mcsd_pgd_block2_fp32.md](algorithms/mcsd_pgd_block2_fp32.md) | Algorithm note for optional MCSD-PGD FP32 gap estimators. |
| [experiments/width256_sso_mcsd_lr_sweep_1b.md](experiments/width256_sso_mcsd_lr_sweep_1b.md) | Main width-256/512 optimizer LR sweep on the 1B-token OLMo mix sample, covering SSO, MCSD-TP/SpEL-TP, MCSD-PGD, and the 250M-token MCSD-PGD projection tuning run. |
| [runbooks/h20_spel_megatron_runbook.md](runbooks/h20_spel_megatron_runbook.md) | Historical H20 setup and debugging runbook for SpEL/Megatron experiments. |
| [templates/experiment_record_template.md](templates/experiment_record_template.md) | Starting point for a new experiment record. |

## Naming Rules

Use stable, descriptive file names:

```text
docs/experiments/<scope>_<optimizer_or_baseline>_<data>_<date_or_version>.md
docs/runbooks/<cluster_or_system>_<topic>.md
```

Examples:

```text
docs/experiments/width512_sso_mcsd_lr_sweep_1b.md
docs/experiments/width256_sso_newoptimizer_lr_sweep_1b.md
docs/runbooks/h20_environment_setup.md
```

For each new experiment, keep the primary result table, commands, job IDs, and caveats in one experiment document. Avoid scattering paper-facing results across README, shell scripts, and ad hoc notes.
