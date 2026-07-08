# SSO Test Experiments

This repository manages one experiment track for a paper project on SSO-style optimizers in Megatron-LM. It keeps the runnable Slurm scripts, data-preparation utilities, experiment notes, and completed result summaries needed to reproduce and extend the current width-scaling study.

The active experiment track is a width-scaling learning-rate sweep on a weighted 1B-token OLMo mix sample:

```text
comparison: SSO vs plain SpEL vs SpEL-TP / MCSD-TP vs MCSD-TP-PGD
widths:     256 and 512
LR grid:    5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
cluster:    HKU HPC2021 H20 Slurm partition
status:     completed baseline, high-LR, and projection-supplement sweeps; plain SpEL-PGD still missing
```

## Current Testing Goal

The current paper-facing goal is to compare SSO-style spectral optimizers on a controlled 1B-token OLMo mix pretraining task at small widths. The immediate comparison set is:

- `width=256` and `width=512`
- LR grid `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2`
- SSO / `spectral_ball_dist`
- plain SpEL / `spel_dist`
- SpEL-TP / MCSD-TP / `spel_tp_dist`
- MCSD-TP-PGD / `spel_pgd_dist`
- MuonBall / `muon_ball_dist` as a new width-256 seven-LR supplement

Current completed-result conclusion: at `LR=1.5e-2`, plain SpEL with top-k projection `k=4` is the best completed row for both `width=256` and `width=512`. Width-512 high-LR tests at `2e-2` and `3e-2` are worse, so the current minimum remains near `1.5e-2`. MCSD-TP-PGD requires top-k projection; shared retraction is clearly worse in the completed supplement.

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
- CLI/config entries for `spel`, `spel_dist`, `spel_tp`, `spel_tp_dist`, `spectral_ball`, and `spectral_ball_dist`.
- Training-time optimizer dispatch in `Megatron-LM/megatron/training/training.py` so these optimizer names call the custom builders.
- Import compatibility fixes so SpEL/SSO can be imported on H20 without eagerly compiling Triton-backed Muon utilities.
- Unit/smoke test utilities under `scripts/` and H20 Slurm launchers for reproducibility.

What was not intentionally changed:

- GPT model architecture, transformer block layout, attention math, MLP layout, tokenizer interface, and Megatron data loader semantics.
- Width changes are controlled by launch-script arguments such as hidden size, FFN size, attention heads, and layer count, not by rewriting Megatron model code.

The H20 jobs use `TRANSFORMER_IMPL=local` because the direct `transformer_engine` + `fused` smoke test failed in the current environment. This backend change is applied consistently to all compared optimizers and is documented in the main experiment record.

As of 2026-07-04, the bundled `Megatron-LM/` tree was replaced with a clean snapshot of the H20-tested `~/projects/Megatron-LM-dev-spel-v3` checkout (`codex/spel-h20-rebase`, upstream head `3e89f3c`) plus import-time compatibility fixes for login nodes. Short H20 smoke jobs completed with this project-local Megatron: SpEL `3733419`, SSO `3733420`.

## Active Server Layout

The completed H20 runs used these server paths:

```text
~/projects/SSO_test
~/projects/SSO_test/Megatron-LM
~/envs/sso_h20
```

The current Slurm and preprocessing scripts default to the bundled Megatron checkout in the repository:

```bash
export PROJECT_DIR=$PWD
export MEGATRON_PATH=$PWD/Megatron-LM
```

Older server-local checkouts such as `~/projects/Megatron-LM-active` and `~/projects/Megatron-LM-dev-spel-v3` were used during development and are not the default path for the current experiment scripts.

## Current Result Summary

Status as of 2026-07-08: the baseline `width=256` and `width=512` five-LR sweeps are complete on H20. The `width=512`, `SpEL-TP top-k k=4`, `LR=1.5e-2` supplement job `3743072` completed successfully. The width-512 high-LR sweep for `2e-2` and `3e-2` completed successfully as jobs `3743116`-`3743125`. The plain SpEL vs MCSD-TP-PGD projection supplement also completed successfully: width-256 jobs `3744519`-`3744524` and width-512 jobs `3744525`-`3744530`. `Elapsed` is Slurm wall-clock time from `sacct` on the H20 partition.

Naming audit, 2026-07-08: all historical `spel_dist` rows in this repository were run while the code always executed the post-msign tangent re-projection line `Phi = project_to_tangent_plane(Phi, u, v)`. These rows are therefore labeled `SpEL-TP` or `MCSD-TP`. The current launcher now exposes that behavior explicitly as `spel_tp_dist`; new plain `spel_dist` rows mean the post-msign TP step is disabled. Historical `spel_pgd_dist` rows are labeled `MCSD-PGD`; their SpEL branch also used the same TP re-projection in that code snapshot.

Best completed results:

| Width | Optimizer | LR | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|---:|
| `256` | plain SpEL `spel_dist`, `topk k=4` | `1.5e-2` | **`3.562941`** | `35.26677` | `05:32:06` | `3744520` |
| `256` | plain SpEL `spel_dist`, `topk k=8` | `1.5e-2` | `3.566394` | `35.38876` | `05:32:28` | `3744521` |
| `256` | SpEL-TP / MCSD-TP `spel_tp_dist`, `topk k=8` | `1.5e-2` | `3.566694` | `35.39936` | `05:34:38` | `3740137` |
| `256` | MCSD-TP-PGD `spel_pgd_dist`, `shared_topk k=8` | `1.5e-2` | `3.566973` | `35.40925` | `05:37:55` | `3744524` |
| `256` | SSO `spectral_ball_dist` | `1.5e-2` | `3.570953` | `35.55044` | `06:02:39` | `3725134` |
| `512` | plain SpEL `spel_dist`, `topk k=4` | `1.5e-2` | **`3.318744`** | `27.62564` | `10:25:33` | `3744526` |
| `512` | SpEL-TP / MCSD-TP `spel_tp_dist`, `topk k=4` | `1.5e-2` | `3.319634` | `27.65023` | `10:27:54` | `3743072` |
| `512` | MCSD-TP-PGD `spel_pgd_dist`, `shared_topk k=4` | `1.5e-2` | `3.320985` | `27.68762` | `10:32:18` | `3744529` |
| `512` | plain SpEL `spel_dist`, `topk k=8` | `1.5e-2` | `3.321280` | `27.69578` | `10:28:18` | `3744527` |
| `512` | SpEL-TP / MCSD-TP `spel_tp_dist`, original retraction | `1.5e-2` | `3.321666` | `27.70647` | `10:17:51` | `3737723` |
| `512` | SSO `spectral_ball_dist` | `1.5e-2` | `3.322861` | `27.73959` | `11:21:05` | `3737718` |

### Width-256 1B-token LR Sweep

This is the original completed `width=256` sweep on the 1B-token OLMo mix sample. The MCSD-PGD column is the earlier, untuned PGD implementation; the later MCSD-PGD projection tuning is listed separately below.

| LR | SSO val loss | SSO elapsed | MCSD-TP / SpEL-TP val loss | SpEL-TP elapsed | Earlier MCSD-PGD val loss | PGD elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.658330` | `05:59:26` | `3.657197` | `05:27:28` | `3.931570` | `05:30:42` |
| `7e-3` | `3.625447` | `05:59:23` | `3.616392` | `05:26:49` | `4.021310` | `05:30:06` |
| `9e-3` | `3.595198` | `06:00:05` | `3.596797` | `05:27:10` | `4.071300` | `05:29:09` |
| `1e-2` | `3.590277` | `06:00:42` | `3.587145` | `05:25:56` | `4.083732` | `05:28:19` |
| `1.5e-2` | `3.570953` | `06:02:39` | `3.567708` | `05:26:59` | `4.099114` | `05:29:23` |

### Width-512 1B-token LR Sweep

This sweep compares SSO, SpEL-TP, and MCSD-PGD with the best MCSD-PGD projection setting from the 250M-token tuning run:

```text
MCSD-PGD: projection_mode=fallback_topk, projection_rank=4, gap_threshold_rel=1e-3
```

| LR | SSO val loss | SSO elapsed | SpEL-TP val loss | SpEL-TP elapsed | MCSD-PGD val loss | PGD elapsed |
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
| SpEL-TP baseline | `branch_mode=spel` | `4.001449` | `54.67733` | `01:23:39` | `3734899` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-4` | `4.000864` | `54.64534` | `01:23:53` | `3735004` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-3` | `4.000864` | `54.64534` | `01:24:04` | `3735008` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=1e-3` | `4.001924` | `54.70327` | `01:24:03` | `3735009` |
| MCSD-PGD top-k | `fallback_topk`, `k=2`, `gap=1e-3` | `4.003179` | `54.77197` | `01:24:03` | `3735007` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-3` | `4.009463` | `55.11728` | `01:24:00` | `3734901` |
| MCSD-PGD retraction | `fallback_retraction`, `gap=5e-3` | `4.631289` | `102.6462` | `01:23:29` | `3734905` |

### SpEL Projection Ablation

This follow-up tests the theory-facing projection choice for SpEL-TP/MCSD-TP itself. It uses `width=256`, `250M` training tokens, `LR=1.5e-2`, `GLOBAL_BATCH=128`, and the same OLMo mix sample.

| Variant | Key setting | Val loss | PPL | Elapsed | Job |
|---|---|---:|---:|---:|---:|
| MCSD-PGD shared top-k | `shared_topk`, `k=2` | **`3.983814`** | `53.72153` | `01:25:28` | `3739001` |
| SpEL-TP top-k | `topk`, `k=2` | `3.983903` | `53.72633` | `01:24:38` | `3738997` |
| SpEL-TP top-k | `topk`, `k=8` | `3.985768` | `53.82662` | `01:25:09` | `3738999` |
| MCSD-PGD shared top-k | `shared_topk`, `k=8` | `3.985801` | `53.82839` | `01:26:30` | `3739003` |
| MCSD-PGD shared top-k | `shared_topk`, `k=4` | `3.986346` | `53.85776` | `01:25:42` | `3739002` |
| SpEL-TP top-k | `topk`, `k=4` | `3.986554` | `53.86893` | `01:24:59` | `3738998` |
| MCSD-PGD fallback top-k | `fallback_topk`, `k=4` | `4.000864` | `54.64534` | `01:24:16` | `3739000` |
| SpEL-TP retraction baseline | `retraction` | `4.001449` | `54.67733` | `01:23:28` | `3738995` |
| SpEL-TP exact SVD | `exact` | `4.138569` | `62.71301` | `01:34:24` | `3738996` |

Current interpretation: the exact SVD projection variant is slower and worse in this implementation. The strongest candidates for follow-up are `SpEL-TP topk k=2` and `MCSD-PGD shared_topk k=2`.

### Width-256 Supplemental Top-k LR Sweep

This 1B-token supplemental sweep follows the stronger projection choices from the 250M-token ablation. It compares MCSD-TP/SpEL-TP with `topk k=8` against MCSD-PGD with `shared_topk k=4` and `shared_topk k=8`.

| LR | SpEL-TP top-k k=8 val loss | SpEL-TP elapsed | PGD shared top-k k=4 val loss | PGD k=4 elapsed | PGD shared top-k k=8 val loss | PGD k=8 elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.640078` | `05:34:53` | `3.638641` | `05:36:58` | `3.640233` | `05:36:36` |
| `7e-3` | `3.599136` | `05:33:27` | `3.602398` | `05:35:31` | `3.599719` | `05:35:49` |
| `9e-3` | `3.583797` | `05:33:26` | `3.583682` | `05:36:06` | `3.584555` | `05:36:19` |
| `1e-2` | `3.580739` | `05:33:30` | `3.577118` | `05:37:24` | `3.580421` | `05:37:12` |
| `1.5e-2` | **`3.566694`** | `05:34:38` | `3.568926` | `05:37:10` | `3.566973` | `05:37:25` |

Current interpretation: within this SpEL-TP / MCSD-PGD top-k sweep, `SpEL-TP topk k=8` is strongest at `3.566694`. After the later plain SpEL supplement, the best completed width-256 result overall is plain SpEL `topk k=4` at `3.562941`.

### Width-512 Supplemental Top-k LR Sweep

The matching width-512 supplemental sweep finished successfully on 2026-07-07 with the same LR grid and projection settings.

| LR | SpEL-TP top-k k=8 val loss | SpEL-TP elapsed | PGD shared top-k k=4 val loss | PGD k=4 elapsed | PGD shared top-k k=8 val loss | PGD k=8 elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| `5e-3` | `3.402194` | `10:28:34` | `3.400489` | `10:32:23` | `3.401184` | `10:30:29` |
| `7e-3` | `3.358461` | `10:29:06` | `3.358685` | `10:30:40` | `3.357924` | `10:31:30` |
| `9e-3` | `3.338081` | `10:27:11` | `3.335894` | `10:30:05` | `3.338519` | `10:32:14` |
| `1e-2` | `3.330682` | `10:28:10` | `3.331491` | `10:34:28` | `3.331737` | `10:34:34` |
| `1.5e-2` | `3.323886` | `10:28:29` | **`3.320985`** | `10:32:10` | `3.322947` | `10:31:50` |

Current interpretation: within the SpEL-TP / MCSD-PGD top-k sweep, the strongest width-512 row is MCSD-PGD `shared_topk k=4` at `1.5e-2`. After the later plain SpEL supplement, the best completed width-512 result overall is plain SpEL `topk k=4` at `3.318744`.

### LR 1.5e-2 Projection Comparison

This table aligns the current best-comparison rows at the highest LR. It is the most direct summary for the current paper-facing `width=256/512`, 1B-token comparison.

| Width | Optimizer/config | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|
| `256` | SSO | `3.570953` | `35.55044` | `06:02:39` | `3725134` |
| `256` | plain SpEL retraction | `3.571484` | `35.56934` | `05:26:24` | `3744519` |
| `256` | plain SpEL top-k k=4 | **`3.562941`** | `35.26677` | `05:32:06` | `3744520` |
| `256` | plain SpEL top-k k=8 | `3.566394` | `35.38876` | `05:32:28` | `3744521` |
| `256` | SpEL-TP original retraction | `3.567708` | `35.43530` | `05:26:59` | `3725139` |
| `256` | SpEL-TP top-k k=4 | `3.567563` | `35.43013` | `05:33:45` | `3743071` |
| `256` | SpEL-TP top-k k=8 | `3.566694` | `35.39936` | `05:34:38` | `3740137` |
| `256` | MCSD-TP-PGD shared top-k k=4 | `3.568926` | `35.47848` | `05:35:00` | `3744523` |
| `256` | MCSD-TP-PGD shared top-k k=8 | `3.566973` | `35.40925` | `05:37:55` | `3744524` |
| `512` | SSO | `3.322861` | `27.73959` | `11:21:05` | `3737718` |
| `512` | plain SpEL retraction | `3.322735` | `27.73611` | `10:16:56` | `3744525` |
| `512` | plain SpEL top-k k=4 | **`3.318744`** | `27.62564` | `10:25:33` | `3744526` |
| `512` | plain SpEL top-k k=8 | `3.321280` | `27.69578` | `10:28:18` | `3744527` |
| `512` | SpEL-TP original retraction | `3.321666` | `27.70647` | `10:17:51` | `3737723` |
| `512` | SpEL-TP top-k k=4 | `3.319634` | `27.65023` | `10:27:54` | `3743072` |
| `512` | SpEL-TP top-k k=8 | `3.323886` | `27.76806` | `10:28:29` | `3741592` |
| `512` | MCSD-TP-PGD shared top-k k=4 | `3.320985` | `27.68762` | `10:32:18` | `3744529` |
| `512` | MCSD-TP-PGD shared top-k k=8 | `3.322947` | `27.74197` | `10:32:45` | `3744530` |

### Width-512 High-LR Projection Sweep

This sweep extends the width-512 LR grid beyond `1.5e-2` to find whether the validation-loss minimum is to the right of the current best point. It uses `LR={2e-2, 3e-2}` and the same 1B-token setup. Both higher LRs are worse than the `1.5e-2` rows.

| Config | `2e-2` val loss / job | `3e-2` val loss / job |
|---|---:|---:|
| SSO | `3.327370` / `3743116` | `3.364925` / `3743121` |
| SpEL-TP top-k k=4 | `3.327565` / `3743117` | `3.373786` / `3743122` |
| SpEL-TP top-k k=8 | **`3.325627`** / `3743118` | `3.375717` / `3743123` |
| MCSD-PGD shared top-k k=4 | `3.328709` / `3743119` | **`3.372203`** / `3743124` |
| MCSD-PGD shared top-k k=8 | `3.326192` / `3743120` | `3.373343` / `3743125` |

### Plain SpEL / MCSD-TP-PGD Projection Supplement

Submitted on 2026-07-08 with `LR=1.5e-2`, `TRAIN_TOKENS=1B`, `GLOBAL_BATCH=128`, and `MICRO_BATCH=4`. Plain SpEL uses `spel_dist` with `tp_after_msign=0`; MCSD-TP-PGD uses `spel_pgd_dist` with `spel_pgd_tangent_project_after_msign=True`.

| Width | Config | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|
| `256` | plain SpEL retraction | `3.571484` | `35.56934` | `05:26:24` | `3744519` |
| `256` | plain SpEL top-k k=4 | **`3.562941`** | `35.26677` | `05:32:06` | `3744520` |
| `256` | plain SpEL top-k k=8 | `3.566394` | `35.38876` | `05:32:28` | `3744521` |
| `256` | MCSD-TP-PGD shared retraction | `4.069594` | `58.53322` | `05:28:56` | `3744522` |
| `256` | MCSD-TP-PGD shared top-k k=4 | `3.568926` | `35.47848` | `05:35:00` | `3744523` |
| `256` | MCSD-TP-PGD shared top-k k=8 | `3.566973` | `35.40925` | `05:37:55` | `3744524` |
| `512` | plain SpEL retraction | `3.322735` | `27.73611` | `10:16:56` | `3744525` |
| `512` | plain SpEL top-k k=4 | **`3.318744`** | `27.62564` | `10:25:33` | `3744526` |
| `512` | plain SpEL top-k k=8 | `3.321280` | `27.69578` | `10:28:18` | `3744527` |
| `512` | MCSD-TP-PGD shared retraction | `3.753390` | `42.66548` | `10:20:46` | `3744528` |
| `512` | MCSD-TP-PGD shared top-k k=4 | `3.320985` | `27.68762` | `10:32:18` | `3744529` |
| `512` | MCSD-TP-PGD shared top-k k=8 | `3.322947` | `27.74197` | `10:32:45` | `3744530` |

Current interpretation: plain SpEL top-k `k=4` is the best completed row in this supplement at both widths. MCSD-TP-PGD `shared_retraction` is not competitive, while `shared_topk k=4/8` remains close to SpEL-TP and SSO.

### Missing Plain SpEL-PGD Experiments

Plain SpEL-PGD means `OPTIMIZER=spel_pgd_dist` with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`, i.e. the SpEL branch inside PGD does not apply the post-msign tangent projection. To match the MCSD-TP-PGD coverage above, the minimum missing set is:

| Width | LR | Projection modes | Jobs needed |
|---:|---:|---|---:|
| `256` | `1.5e-2` | `shared_retraction`, `shared_topk k=4`, `shared_topk k=8` | `3` |
| `512` | `1.5e-2` | `shared_retraction`, `shared_topk k=4`, `shared_topk k=8` | `3` |

Keep the same settings as MCSD-TP-PGD: `branch_mode=auto`, `gap_threshold_rel=1e-3`, `sigma2_power_iteration_steps=3`, `direction_normalization=none`, `TRAIN_TOKENS=1B`, `GLOBAL_BATCH=128`, and `MICRO_BATCH=4`.

If plain SpEL-PGD is competitive at `1.5e-2`, extend it to the same five-LR grid for the best projection mode: `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` at width `256` and `512`.

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
bash slurm/submit_width256_512_spel_mcsd_tp_pgd_projection_supplement.sh
bash slurm/submit_width256_pgd_sigma2_sweep.sh
bash slurm/submit_width256_muon_ball_lr_sweep.sh
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
