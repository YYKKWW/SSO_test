# SSO Test Experiments

This repository manages one experiment track for a paper project on SSO-style optimizers in Megatron-LM. It keeps the runnable Slurm scripts, data-preparation utilities, experiment notes, and completed result summaries needed to reproduce and extend the current width-scaling study.

The active experiment is a `width=256` learning-rate sweep on a weighted 1B-token OLMo mix sample:

```text
comparison: SSO / spectral_ball_dist vs MCSD / spel_dist
LR grid:    5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
cluster:    HKU HPC2021 H20 Slurm partition
status:     completed, 10/10 jobs finished with exit code 0:0
```

## Documentation

| Document | Purpose |
|---|---|
| [docs/width256_sso_mcsd_lr_sweep_1b.md](docs/width256_sso_mcsd_lr_sweep_1b.md) | Main experiment record. Includes algorithm definitions, dataset source and acquisition, preprocessing, Slurm configuration, completed results, and instructions for adding new optimizers or learning rates. |
| [docs/h20_spel_megatron_runbook.md](docs/h20_spel_megatron_runbook.md) | Historical H20 setup and debugging runbook. Useful for understanding earlier environment, Megatron rebase, and data-preparation decisions. |
| [README.md](README.md) | Repository entry point and high-level map. Detailed experiment maintenance should happen in the main experiment document above. |

## Repository Layout

| Path | Content |
|---|---|
| `docs/` | Experiment records and runbooks. |
| `slurm/` | Slurm launchers for H20 jobs and LR sweeps. |
| `scripts/` | Data sampling, preprocessing, Megatron patching, and verification utilities. |
| `Spectral-Sphere-Optimizer/` | Reference upstream SSO scripts and paper-related materials. |
| `Megatron-LM/` | Local historical Megatron copy. The server-side 1B experiments use `~/projects/Megatron-LM-active` instead. |
| `analysis_logs_372513/` | Local copied logs from the completed width-256 sweep. Keep only if needed for local inspection; summaries live in the main document. |

Large generated directories such as `data/`, `results/`, `logs/`, `outputs/`, `checkpoints/`, and Megatron `.bin/.idx` files should not be committed.

## Active Server Layout

The H20 runs use these server paths:

```text
~/projects/SSO_test
~/projects/Megatron-LM-active -> ~/projects/Megatron-LM-dev-spel-v3
~/envs/sso_h20
```

`Megatron-LM-active` is the stable symlink used by the Slurm scripts. If the Megatron checkout is updated later, update the symlink rather than hard-coding a new checkout name in every script.

## Current Result Summary

The best completed result in the current `width=256` 1B-token sweep is:

```text
MCSD / spel_dist, LR=1.5e-2, validation loss=3.567708, PPL=35.43530
```

The best SSO result is:

```text
SSO / spectral_ball_dist, LR=1.5e-2, validation loss=3.570953, PPL=35.55044
```

See [docs/width256_sso_mcsd_lr_sweep_1b.md](docs/width256_sso_mcsd_lr_sweep_1b.md) for the full table, job IDs, commands, and caveats.

## Quick Workflow

Prepare data:

```bash
cd ~/projects/SSO_test
bash scripts/download_olmo_mix_1124_1b.sh
bash scripts/preprocess_olmo_mix_1124_1b.sh
```

Run the current sweep:

```bash
cd ~/projects/SSO_test
bash slurm/submit_width256_sso_mcsd_lr_sweep.sh
```

Monitor jobs:

```bash
squeue -u u3013198
sacct -j <job_id> --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
```

For additional learning rates or new optimizers, follow the templates in the main experiment document.

## Git Hygiene

Recommended to track:

```text
README.md
docs/
scripts/
slurm/
```

Do not commit:

```text
data/
results/
outputs/
logs/
checkpoints/
*.bin
*.idx
Hugging Face tokens
SSH private keys
HPC passwords
```

When adding paper-facing results, keep the result table and experimental assumptions in the main document synchronized with the corresponding Slurm job IDs and logs.
