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

Status as of 2026-07-07: the baseline `width=256` and `width=512` five-LR sweeps are complete on H20. The `width=256` and `width=512` top-k supplemental sweeps are also complete. Two `SpEL topk k=4`, `LR=1.5e-2` supplement jobs are running as `3743071` and `3743072`. The width-512 high-LR sweep for `2e-2` and `3e-2` is running as jobs `3743116`-`3743125`. `Elapsed` is Slurm wall-clock time from `sacct` on the H20 partition.

Best completed results:

| Width | Optimizer | LR | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|---:|
| `256` | MCSD / SpEL `spel_dist`, `topk k=8` | `1.5e-2` | `3.566694` | `35.39936` | `05:34:38` | `3740137` |
| `256` | MCSD / SpEL `spel_dist`, original retraction | `1.5e-2` | `3.567708` | `35.43530` | `05:26:59` | `3725139` |
| `512` | MCSD-PGD `spel_pgd_dist`, `shared_topk k=4` | `1.5e-2` | `3.320985` | `27.68762` | `10:32:10` | `3741597` |
| `512` | SpEL `spel_dist` | `1.5e-2` | `3.321666` | `27.70647` | `10:17:51` | `3737723` |
| `512` | MCSD-PGD `spel_pgd_dist` | `1.5e-2` | `3.321784` | `27.70973` | `10:22:07` | `3737728` |
| `512` | SSO `spectral_ball_dist` | `1.5e-2` | `3.322861` | `27.73959` | `11:21:05` | `3737718` |

### Width-256 1B-token LR Sweep

This is the original completed `width=256` sweep on the 1B-token OLMo mix sample. The SpEL-PGD column is the earlier, untuned PGD implementation; the later MCSD-PGD projection tuning is listed separately below.

| LR | SSO val loss | SSO elapsed | MCSD / SpEL val loss | SpEL elapsed | Earlier SpEL-PGD val loss | PGD elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.658330` | `05:59:26` | `3.657197` | `05:27:28` | `3.931570` | `05:30:42` |
| `7e-3` | `3.625447` | `05:59:23` | `3.616392` | `05:26:49` | `4.021310` | `05:30:06` |
| `9e-3` | `3.595198` | `06:00:05` | `3.596797` | `05:27:10` | `4.071300` | `05:29:09` |
| `1e-2` | `3.590277` | `06:00:42` | `3.587145` | `05:25:56` | `4.083732` | `05:28:19` |
| `1.5e-2` | `3.570953` | `06:02:39` | `3.567708` | `05:26:59` | `4.099114` | `05:29:23` |

### Width-512 1B-token LR Sweep

This sweep compares SSO, SpEL, and MCSD-PGD with the best MCSD-PGD projection setting from the 250M-token tuning run:

```text
MCSD-PGD: projection_mode=fallback_topk, projection_rank=4, gap_threshold_rel=1e-3
```

| LR | SSO val loss | SSO elapsed | SpEL val loss | SpEL elapsed | MCSD-PGD val loss | PGD elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.423116` | `11:12:16` | `3.420247` | `10:18:25` | `3.418978` | `10:22:23` |
| `7e-3` | `3.371309` | `11:14:35` | `3.371672` | `10:17:12` | `3.371645` | `10:22:21` |
| `9e-3` | `3.345420` | `11:14:06` | `3.345384` | `10:17:31` | `3.346858` | `10:24:34` |
| `1e-2` | `3.338379` | `11:17:06` | `3.337422` | `10:18:51` | `3.339918` | `10:25:18` |
| `1.5e-2` | `3.322861` | `11:21:05` | `3.321666` | `10:17:51` | `3.321784` | `10:22:07` |

### MCSD-PGD Projection Tuning

The 250M-token `width=256`, `LR=1.5e-2` projection tuning run selected the MCSD-PGD configuration used for the `width=512` sweep.

| Variant | Key setting | Val loss | PPL | Elapsed | Job |
|---|---|---:|---:|---:|---:|
| SpEL baseline | `branch_mode=spel` | `4.001449` | `54.67733` | `01:23:39` | `3734899` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-4` | `4.000864` | `54.64534` | `01:23:53` | `3735004` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-3` | `4.000864` | `54.64534` | `01:24:04` | `3735008` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=1e-3` | `4.001924` | `54.70327` | `01:24:03` | `3735009` |
| MCSD-PGD top-k | `fallback_topk`, `k=2`, `gap=1e-3` | `4.003179` | `54.77197` | `01:24:03` | `3735007` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-3` | `4.009463` | `55.11728` | `01:24:00` | `3734901` |
| MCSD-PGD retraction | `fallback_retraction`, `gap=5e-3` | `4.631289` | `102.6462` | `01:23:29` | `3734905` |

### SpEL Projection Ablation

This follow-up tests the theory-facing projection choice for SpEL/MCSD itself. It uses `width=256`, `250M` training tokens, `LR=1.5e-2`, `GLOBAL_BATCH=128`, and the same OLMo mix sample.

| Variant | Key setting | Val loss | PPL | Elapsed | Job |
|---|---|---:|---:|---:|---:|
| SpEL-PGD shared top-k | `shared_topk`, `k=2` | **`3.983814`** | `53.72153` | `01:25:28` | `3739001` |
| SpEL top-k | `topk`, `k=2` | `3.983903` | `53.72633` | `01:24:38` | `3738997` |
| SpEL top-k | `topk`, `k=8` | `3.985768` | `53.82662` | `01:25:09` | `3738999` |
| SpEL-PGD shared top-k | `shared_topk`, `k=8` | `3.985801` | `53.82839` | `01:26:30` | `3739003` |
| SpEL-PGD shared top-k | `shared_topk`, `k=4` | `3.986346` | `53.85776` | `01:25:42` | `3739002` |
| SpEL top-k | `topk`, `k=4` | `3.986554` | `53.86893` | `01:24:59` | `3738998` |
| SpEL-PGD fallback top-k | `fallback_topk`, `k=4` | `4.000864` | `54.64534` | `01:24:16` | `3739000` |
| SpEL retraction baseline | `retraction` | `4.001449` | `54.67733` | `01:23:28` | `3738995` |
| SpEL exact SVD | `exact` | `4.138569` | `62.71301` | `01:34:24` | `3738996` |

Current interpretation: the exact SVD projection variant is slower and worse in this implementation. The strongest candidates for follow-up are `SpEL topk k=2` and `SpEL-PGD shared_topk k=2`.

### Width-256 Supplemental Top-k LR Sweep

This 1B-token supplemental sweep follows the stronger projection choices from the 250M-token ablation. It compares MCSD/SpEL with `topk k=8` against MCSD-PGD with `shared_topk k=4` and `shared_topk k=8`.

| LR | SpEL top-k k=8 val loss | SpEL elapsed | PGD shared top-k k=4 val loss | PGD k=4 elapsed | PGD shared top-k k=8 val loss | PGD k=8 elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.640078` | `05:34:53` | `3.638641` | `05:36:58` | `3.640233` | `05:36:36` |
| `7e-3` | `3.599136` | `05:33:27` | `3.602398` | `05:35:31` | `3.599719` | `05:35:49` |
| `9e-3` | `3.583797` | `05:33:26` | `3.583682` | `05:36:06` | `3.584555` | `05:36:19` |
| `1e-2` | `3.580739` | `05:33:30` | `3.577118` | `05:37:24` | `3.580421` | `05:37:12` |
| `1.5e-2` | **`3.566694`** | `05:34:38` | `3.568926` | `05:37:10` | `3.566973` | `05:37:25` |

Current interpretation: `SpEL topk k=8` gives the best completed width-256 1B result so far, but the margin over the original SpEL run is small (`3.566694` versus `3.567708`). MCSD-PGD `shared_topk k=8` is very close at `1.5e-2`.

### Width-512 Supplemental Top-k LR Sweep

The matching width-512 supplemental sweep finished successfully on 2026-07-07 with the same LR grid and projection settings.

| LR | SpEL top-k k=8 val loss | SpEL elapsed | PGD shared top-k k=4 val loss | PGD k=4 elapsed | PGD shared top-k k=8 val loss | PGD k=8 elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.402194` | `10:28:34` | `3.400489` | `10:32:23` | `3.401184` | `10:30:29` |
| `7e-3` | `3.358461` | `10:29:06` | `3.358685` | `10:30:40` | `3.357924` | `10:31:30` |
| `9e-3` | `3.338081` | `10:27:11` | `3.335894` | `10:30:05` | `3.338519` | `10:32:14` |
| `1e-2` | `3.330682` | `10:28:10` | `3.331491` | `10:34:28` | `3.331737` | `10:34:34` |
| `1.5e-2` | `3.323886` | `10:28:29` | **`3.320985`** | `10:32:10` | `3.322947` | `10:31:50` |

Current interpretation: `MCSD-PGD shared_topk k=4` gives the best completed width-512 result so far at `LR=1.5e-2`.

### LR 1.5e-2 Projection Comparison

This table aligns the current best-comparison rows at the highest LR. `SpEL topk k=4` is being supplemented because the previous 1B top-k sweep only included `k=8`.

| Width | Optimizer/config | Val loss | PPL | Elapsed/status | Job |
|---:|---|---:|---:|---:|---:|
| `256` | SSO | `3.570953` | `35.55044` | `06:02:39` | `3725134` |
| `256` | SpEL original retraction | `3.567708` | `35.43530` | `05:26:59` | `3725139` |
| `256` | SpEL top-k k=4 | pending | pending | running | `3743071` |
| `256` | SpEL top-k k=8 | **`3.566694`** | `35.39936` | `05:34:38` | `3740137` |
| `256` | MCSD-PGD shared top-k k=4 | `3.568926` | `35.47848` | `05:37:10` | `3740142` |
| `256` | MCSD-PGD shared top-k k=8 | `3.566973` | `35.40925` | `05:37:25` | `3740147` |
| `512` | SSO | `3.322861` | `27.73959` | `11:21:05` | `3737718` |
| `512` | SpEL original retraction | `3.321666` | `27.70647` | `10:17:51` | `3737723` |
| `512` | SpEL top-k k=4 | pending | pending | running | `3743072` |
| `512` | SpEL top-k k=8 | `3.323886` | `27.76806` | `10:28:29` | `3741592` |
| `512` | MCSD-PGD shared top-k k=4 | **`3.320985`** | `27.68762` | `10:32:10` | `3741597` |
| `512` | MCSD-PGD shared top-k k=8 | `3.322947` | `27.74197` | `10:31:50` | `3741602` |

### Width-512 High-LR Projection Sweep

This sweep extends the width-512 LR grid beyond `1.5e-2` to find whether the validation-loss minimum is to the right of the current best point. It uses `LR={2e-2, 3e-2}` and the same 1B-token setup.

| Config | `2e-2` job | `3e-2` job |
|---|---:|---:|
| SSO | `3743116` | `3743121` |
| SpEL top-k k=4 | `3743117` | `3743122` |
| SpEL top-k k=8 | `3743118` | `3743123` |
| MCSD-PGD shared top-k k=4 | `3743119` | `3743124` |
| MCSD-PGD shared top-k k=8 | `3743120` | `3743125` |

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
bash slurm/submit_width256_spel_topk8_pgd_topk_lr_sweep.sh
bash slurm/submit_width512_spel_topk8_pgd_topk_lr_sweep.sh
bash slurm/submit_width256_512_spel_topk4_lr1p5_supplement.sh
bash slurm/submit_width512_high_lr_projection_sweep.sh
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
