# Documentation Index

This directory is the documentation entry point for the H20 experiment repository.

## Layout

| Path | Purpose |
|---|---|
| `experiments/` | One document per reproducible experiment family. Put results, exact Slurm job IDs, data paths, model settings, and follow-up runs here. |
| `runbooks/` | Environment setup, H20 operational notes, debugging records, and cluster-specific procedures. |
| `templates/` | Copyable templates for new experiments or new project tracks. |

## Current Documents

| Document | Purpose |
|---|---|
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
