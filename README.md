# SSO Test Experiments

This repository manages one experiment track for a paper project on SSO-style optimizers in Megatron-LM. It keeps the runnable Slurm scripts, data-preparation utilities, experiment notes, and completed result summaries needed to reproduce and extend the current width-scaling study.

The active experiment track is a width-scaling learning-rate sweep on a weighted 1B-token OLMo mix sample:

```text
comparison: SSO / spectral_ball_dist vs MCSD-SpEL / spel_dist vs MCSD-PGD / spel_pgd_dist
widths:     256 and 512
LR grid:    5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
cluster:    HKU HPC2021 H20 Slurm partition
status:     completed, width-256 and width-512 sweeps finished with exit code 0:0
```

## Documentation

| Document | Purpose |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index, naming rules, and where to add new experiment records. |
| [docs/experiments/width256_sso_mcsd_lr_sweep_1b.md](docs/experiments/width256_sso_mcsd_lr_sweep_1b.md) | Main experiment record. Includes algorithm definitions, dataset source and acquisition, preprocessing, Slurm configuration, completed results, and instructions for adding new optimizers or learning rates. |
| [docs/runbooks/h20_spel_megatron_runbook.md](docs/runbooks/h20_spel_megatron_runbook.md) | Historical H20 setup and debugging runbook. Useful for understanding earlier environment, Megatron rebase, and data-preparation decisions. |
| [docs/templates/experiment_record_template.md](docs/templates/experiment_record_template.md) | Template for adding a new paper-facing experiment. |
| [README.md](README.md) | Repository entry point and high-level map. Detailed experiment maintenance should happen in the main experiment document above. |

## Repository Layout

| Path | Content |
|---|---|
| `docs/` | Documentation index, experiment records, runbooks, and templates. |
| `slurm/` | Slurm launchers for H20 jobs and LR sweeps. |
| `scripts/` | Data sampling, preprocessing, Megatron patching, and verification utilities. |
| `Spectral-Sphere-Optimizer/` | Reference upstream SSO scripts and paper-related materials. |
| `Megatron-LM/` | Bundled Megatron-LM tree with the project SpEL/SSO optimizer additions. The original Megatron model and training behavior is kept unless explicitly noted in the patch summary below. |
| `analysis_logs_372513/` | Local copied logs from the completed width-256 sweep. Keep only if needed for local inspection; summaries live in the main document. |

Large generated directories such as `data/`, `results/`, `logs/`, `outputs/`, `checkpoints/`, and Megatron `.bin/.idx` files should not be committed.

## Megatron Patch Summary

This repository keeps the Megatron-LM code path close to the original project and adds the optimizer integration needed by the experiments.

What was added:

- SpEL and SSO optimizer implementations under `Megatron-LM/megatron/core/optimizer/`.
- Shared spectral/muP helper implementations under `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/`.
- CLI/config entries for `spel`, `spel_dist`, `spectral_ball`, and `spectral_ball_dist`.
- Training-time optimizer dispatch in `Megatron-LM/megatron/training/training.py` so these optimizer names call the custom builders.
- Import compatibility fixes so SpEL/SSO can be imported on H20 without eagerly compiling Triton-backed Muon utilities.
- Unit/smoke test utilities and H20 Slurm launchers for reproducibility.

What was not intentionally changed:

- GPT model architecture, transformer block layout, attention math, MLP layout, tokenizer interface, and Megatron data loader semantics.
- Width changes are controlled by launch-script arguments such as hidden size, FFN size, attention heads, and layer count, not by rewriting Megatron model code.

The H20 jobs use `TRANSFORMER_IMPL=local` because the direct `transformer_engine` + `fused` smoke test failed in the current environment. This backend change is applied consistently to all compared optimizers and is documented in the main experiment record.

As of 2026-07-04, the bundled `Megatron-LM/` tree was replaced with a clean snapshot of the H20-tested `~/projects/Megatron-LM-dev-spel-v3` checkout (`codex/spel-h20-rebase`, upstream head `3e89f3c`) plus import-time compatibility fixes for login nodes. Short H20 smoke jobs completed with this project-local Megatron: SpEL `3733419`, SSO `3733420`.

## Active Server Layout

The completed H20 runs used these server paths:

```text
~/projects/SSO_test
~/projects/Megatron-LM-active -> ~/projects/Megatron-LM-dev-spel-v3
~/envs/sso_h20
```

For a fresh clone, the Slurm scripts can also use the bundled Megatron checkout directly:

```bash
export PROJECT_DIR=$PWD
export MEGATRON_PATH=$PWD/Megatron-LM
```

On the original H20 server, `Megatron-LM-active` is the stable symlink used by the Slurm scripts. If the server-side Megatron checkout is updated later, update the symlink rather than hard-coding a new checkout name in every script.

## Current Result Summary

Status as of 2026-07-06: the `width=256` and `width=512` five-LR sweeps are complete on H20. All listed jobs finished with Slurm state `COMPLETED` and exit code `0:0`.

Best completed results:

| Width | Optimizer | LR | Val loss | PPL | Job |
|---:|---|---:|---:|---:|---:|
| `256` | MCSD / SpEL `spel_dist` | `1.5e-2` | `3.567708` | `35.43530` | `3725139` |
| `512` | SpEL `spel_dist` | `1.5e-2` | `3.321666` | `27.70647` | `3737723` |
| `512` | MCSD-PGD `spel_pgd_dist` | `1.5e-2` | `3.321784` | `27.70973` | `3737728` |
| `512` | SSO `spectral_ball_dist` | `1.5e-2` | `3.322861` | `27.73959` | `3737718` |

### Width-256 1B-token LR Sweep

This is the original completed `width=256` sweep on the 1B-token OLMo mix sample. The SpEL-PGD column is the earlier, untuned PGD implementation; the later MCSD-PGD projection tuning is listed separately below.

| LR | SSO val loss | MCSD / SpEL val loss | Earlier SpEL-PGD val loss |
|---:|---:|---:|---:|
| `5e-3` | `3.658330` | `3.657197` | `3.931570` |
| `7e-3` | `3.625447` | `3.616392` | `4.021310` |
| `9e-3` | `3.595198` | `3.596797` | `4.071300` |
| `1e-2` | `3.590277` | `3.587145` | `4.083732` |
| `1.5e-2` | `3.570953` | `3.567708` | `4.099114` |

### Width-512 1B-token LR Sweep

This sweep compares SSO, SpEL, and MCSD-PGD with the best MCSD-PGD projection setting from the 250M-token tuning run:

```text
MCSD-PGD: projection_mode=fallback_topk, projection_rank=4, gap_threshold_rel=1e-3
```

| LR | SSO val loss | SpEL val loss | MCSD-PGD val loss |
|---:|---:|---:|---:|
| `5e-3` | `3.423116` | `3.420247` | `3.418978` |
| `7e-3` | `3.371309` | `3.371672` | `3.371645` |
| `9e-3` | `3.345420` | `3.345384` | `3.346858` |
| `1e-2` | `3.338379` | `3.337422` | `3.339918` |
| `1.5e-2` | `3.322861` | `3.321666` | `3.321784` |

### MCSD-PGD Projection Tuning

The 250M-token `width=256`, `LR=1.5e-2` projection tuning run selected the MCSD-PGD configuration used for the `width=512` sweep.

| Variant | Key setting | Val loss | PPL | Job |
|---|---|---:|---:|---:|
| SpEL baseline | `branch_mode=spel` | `4.001449` | `54.67733` | `3734899` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-4` | `4.000864` | `54.64534` | `3735004` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-3` | `4.000864` | `54.64534` | `3735008` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=1e-3` | `4.001924` | `54.70327` | `3735009` |
| MCSD-PGD top-k | `fallback_topk`, `k=2`, `gap=1e-3` | `4.003179` | `54.77197` | `3735007` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-3` | `4.009463` | `55.11728` | `3734901` |
| MCSD-PGD retraction | `fallback_retraction`, `gap=5e-3` | `4.631289` | `102.6462` | `3734905` |

See [docs/experiments/width256_sso_mcsd_lr_sweep_1b.md](docs/experiments/width256_sso_mcsd_lr_sweep_1b.md) for the full table, job IDs, commands, and caveats.

## Quick Workflow

Prepare data:

```bash
cd ~/projects/SSO_test
export MEGATRON_PATH=$PWD/Megatron-LM
bash scripts/download_olmo_mix_1124_1b.sh
bash scripts/preprocess_olmo_mix_1124_1b.sh
```

Run the current sweep:

```bash
cd ~/projects/SSO_test
export MEGATRON_PATH=$PWD/Megatron-LM
bash slurm/submit_width256_sso_mcsd_lr_sweep.sh
bash slurm/submit_width256_spel_pgd_lr_sweep.sh
```

Monitor jobs:

```bash
squeue -u u3013198
sacct -j <job_id> --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
```

For additional learning rates or new optimizers, follow the current experiment document. For a new project track, copy [docs/templates/experiment_record_template.md](docs/templates/experiment_record_template.md) into `docs/experiments/`.

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
