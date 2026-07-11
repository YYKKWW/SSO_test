# Experiment Record: Width-256/512 Optimizer LR Sweep on OLMo Mix 1B

Last updated: 2026-07-12

This document is the primary experiment record for the `width=256` and `width=512` optimizer learning-rate sweeps. It is intended to support paper development, later reproduction, and future extensions with new optimizers or additional learning rates.

The locked paper-facing `width=256/512/1024`, seven-LR, four-optimizer matrix
is now maintained at the top of the repository `README.md`. Unless a row
exactly matches that locked optimizer and precision configuration, tables in
this document are supplementary tuning or historical exploration and must not
be used to mark a primary-matrix cell complete.

Do not put passwords, SSH private keys, Hugging Face tokens, HPC passwords, or other secrets in this file.

## Status Summary

| Field | Current status |
|---|---|
| Experiment family | Small-scale pretraining LR sweep |
| Paper role | One supporting experiment for optimizer comparison |
| Width | `256`, `512`; width-1024 MuonBall complete, one SSO row complete, and six SSO rows running |
| Data budget | `1B` training tokens |
| Dataset | Weighted sample from `allenai/olmo-mix-1124` |
| Compared optimizers | SSO / `spectral_ball_dist`, plain SpEL / `spel_dist`, MCSD-TP/SpEL-TP / `spel_tp_dist` for new runs, plain MCSD-PGD / `spel_pgd_dist`, MuonBall / `muon_ball_dist` |
| LR grid | `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` |
| Jobs completed | width-256 1B sweep: `15/15`; MCSD-PGD 250M tuning: `18/18`; SpEL projection 250M ablation: `9/9`; width-512 1B sweep: `15/15`; width-256 supplemental top-k sweep: `15/15`; width-512 supplemental top-k sweep: `15/15`; plain SpEL / MCSD-TP-PGD projection supplement: `12/12`; width-256 PGD sigma2 supplement: `6/6`; width-256 MuonBall supplement: `7/7`; width-512 MuonBall supplement: `7/7`; width-1024 memory smoke: `3/3`; MCSD-PGD phase-B sigma2/gap tuning: `15/15`; adaptive gap-probe 1B comparison: `2/2`; width-1024 SSO/MuonBall LR sweep: `8/14` complete |
| Slurm status | width-512 MuonBall jobs `3751693`-`3751699`, adaptive gap-probe jobs `3756922`-`3756923`, width-1024 MuonBall jobs `3756221`-`3756227`, and width-1024 SSO job `3756214` completed with exit code `0:0`; SSO jobs `3756215`-`3756220` remain `RUNNING` |
| Completed tuning jobs | 250M-token plain MCSD-PGD gap-threshold tuning at `width=256`, `LR=1.5e-2`, `shared_topk k=8`: phase-A `sigma2=5` jobs with no direction normalization `3749547`-`3749553`, Frobenius normalization `3749569`-`3749575`, spectral normalization `3749612`-`3749618`; phase-B spectral-normalized sigma2/gap jobs `3750042`-`3750056`; all `COMPLETED`, exit code `0:0` |
| Selected PGD default | `sigma2_power_iteration_steps=5`, `gap_threshold_rel=1e-3`, `pgd_direction_normalization=spectral`, `pgd_lr_scale=0.5` |
| Main result table | [Completed Sweep Results](#completed-sweep-results) |
| Next likely extension | collect width-1024 SSO/MuonBall results; then decide whether width-1024 SpEL/MCSD variants are worth the cost |

## Scope And Caveats

This is a controlled small-scale experiment, not the full paper sweep.

- The completed runs cover `width=256` and `width=512`; they do not yet cover widths `1024` or `2048`.
- The current run uses one 1B-token weighted sample from OLMo mix; it is not a 30B-token paper-scale run.
- The current table is a single-run comparison; paper claims should be calibrated accordingly unless repeated seeds or additional settings are added.
- Historical MCSD-TP rows map to the old `spel_dist` implementation path because the launcher did not expose an optimizer literally named `mcsd`.
- Naming audit, 2026-07-08: all historical `spel_dist` rows in this document were run while `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py` always executed the post-msign tangent re-projection line `Phi = project_to_tangent_plane(Phi, u, v)`. These rows are labeled `SpEL-TP` or `MCSD-TP`. The current launcher now exposes that behavior explicitly as `spel_tp_dist`; new plain `spel_dist` rows mean the post-msign TP step is disabled.
- The original width-256 MCSD-PGD rows use the first `spel_pgd_dist` implementation. The later width-512 MCSD-PGD rows use the selected top-k projection setting from the 250M-token tuning run: `fallback_topk`, rank `4`, gap `1e-3`. In the current historical code snapshot, the SpEL branch inside `spel_pgd_dist` also used post-msign TP re-projection.
- Forward rule from 2026-07-09: unqualified `MCSD-PGD` means the plain `spel_pgd_dist` variant with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`. Historical `MCSD-TP-PGD` rows remain in the record, but new TP-PGD jobs should not be submitted unless a future ablation explicitly needs them.
- The supplemental top-k rows use SpEL-TP `projection_mode=topk` and MCSD-PGD `projection_mode=shared_topk`. SpEL itself does not have a `shared_topk` mode; that mode only applies to the two-branch MCSD-PGD optimizer.
- The H20 runs use Megatron's `local` transformer implementation rather than the original script's `transformer_engine` + `fused` backend. A direct TE/fused smoke test failed in the current environment; see [Backend Compatibility Note](#backend-compatibility-note).

For paper use, treat the table below as an experiment record with exact job IDs and settings. If the results are later promoted into a paper figure, record the plotting script, figure version, and any post-processing assumptions in this document.

## Maintenance Rules

Use this file as the single source of truth for this experiment family.

- Add new optimizers to [Algorithms Compared](#algorithms-compared) before running the sweep.
- Add new LR results to [Completed Sweep Results](#completed-sweep-results) only after checking Slurm exit code and final validation loss.
- Keep historical runs in [Historical Baseline](#historical-baseline) if they differ in batch size, tokenization, model config, seed, or train iterations.
- Do not overwrite previous results unless the old row was factually wrong; append corrected rows with a note.
- Keep large logs, raw data, indexed data, and checkpoints out of Git. Record their server paths instead.

## Goal

Run a small-scale version of the SSO paper's width/LR sweep at `width=256`, comparing:

- SSO: `spectral_ball_dist`
- MCSD-TP: `spel_tp_dist` for new runs; historical completed rows used `spel_dist`
- MCSD-PGD / plain SpEL-PGD: `spel_pgd_dist` with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`
- MuonBall: `muon_ball_dist` as a width-256 supplement

The completed SSO/SpEL sweeps use 1B training tokens from the weighted OLMo mix sample and evaluate five learning rates:

```text
5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
```

The MuonBall supplement extends the width-256 LR grid to seven learning rates:

```text
5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2, 2e-2, 3e-2
```

The immediate goal is to determine optimizer LR sensitivity at small width and whether the current LR grid or algorithm implementation should be extended before running more expensive widths.

## Algorithms Compared

This experiment compares three optimizer implementations in the active Megatron checkout. They are run with the same model architecture, data, token budget, warmup/decay schedule, weight decay, and batch settings. The only intended difference is the optimizer algorithm.

| Display name | Megatron optimizer name | Main source files | Role in this sweep |
|---|---|---|---|
| SSO / Spectral Sphere | `spectral_ball_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`, `spectral_ball_utils.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | Main SSO baseline following the paper's `spball` scripts. |
| MCSD-TP / SpEL-TP | `spel_tp_dist` for new runs; historical rows used `spel_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | Comparison optimizer path with post-msign tangent re-projection enabled. |
| MCSD-PGD / plain SpEL-PGD | `spel_pgd_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | New algorithm under test: SpEL-style spectral retraction with an automatic PGD fallback branch. Future runs use the plain variant by default. |
| MuonBall | `muon_ball_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/muon_ball.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | Supplementary baseline from the SSO repository: Spectral Ball with lambda fixed to zero, removing the bisection solver. |

SSO follows the Spectral Sphere / Spectral Ball setup:

- It constrains selected matrix weights to a spectral ball or sphere with radius based on spectral muP scaling.
- It uses matrix-sign style updates, computed with Newton-Schulz iterations.
- It uses hard retraction in this sweep, so weights are projected back to the spectral constraint after the update.
- In the current Megatron launcher, distributed SSO is selected by `--optimizer spectral_ball_dist`.

MuonBall follows the reference `Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/muonball/muonball.sh` optimizer constants while using this repository's H20 width-256 1B training setup:

- `--optimizer muon_ball_dist`
- `momentum=0.9`
- Nesterov enabled
- `msign_steps=8`
- `radius_mode=spectral_mup`
- `scale_mode=spectral_mup`
- `power_iteration_steps=10`
- `retract_mode=hard`
- `qkv_split_mode=head`

MCSD-TP/SpEL-TP in this project is the current comparison optimizer path:

- It uses the `spel_tp_dist` Megatron optimizer entry for new runs; historical completed rows used `spel_dist`.
- It uses the same spectral-muP radius/scale choices as the SSO run for a fair small-scale comparison.
- It uses Nesterov momentum, QKV head split, `msign_steps=8`, and hard retraction in the current sweep.

Important naming note: the codebase does not expose an optimizer literally named `mcsd` in the current launcher. For completed historical rows, "MCSD-TP" means the old `spel_dist` optimizer path with post-msign tangent re-projection enabled. From the 2026-07-08 code update onward, "MCSD-TP" should use `spel_tp_dist`; plain `spel_dist` should be labeled `SpEL`.

MCSD-PGD / plain SpEL-PGD in this project is the first new algorithm extension after the SSO/MCSD baseline:

- It subclasses the SpEL path and keeps the same spectral-sphere retraction operator: `power_iteration + apply_retract`.
- It supports exact, retraction, and top-k post-projection modes; the preferred H20 runs avoid full exact SVD and use `projection_mode=shared_topk`.
- In `branch_mode=auto`, it estimates the relative top-singular-value gap and switches to a PGD-style momentum direction when the gap is below `gap_threshold_rel`.
- In `shared_*` modes both the SpEL branch and PGD branch form a trial point and apply the same post-step projection; in `fallback_*` modes the SpEL branch preserves the original SpEL direction and only the PGD branch encodes a projected trial point.
- The historical sweep uses `branch_mode=auto`, `gap_threshold_rel=5e-3`, `sigma2_power_iteration_steps=3`, and `pgd_direction_normalization=none`.
- The 2026-07-09 plain SpEL-PGD sigma2 supplement fixes `projection_mode=shared_topk`, `rank=8`, `LR=1.5e-2`, and tests `sigma2_power_iteration_steps=5,8,10`. It also contains a completed TP ablation; future MCSD-PGD runs should keep `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`.
- The selected forward default is `sigma2_power_iteration_steps=5`, `gap_threshold_rel=1e-3`, `pgd_direction_normalization=spectral`, and `pgd_lr_scale=0.5`.
- The optional `gap_estimator_mode=block2_fp32` uses one FP32 two-dimensional subspace to estimate `sigma1,u1,v1,sigma2` and therefore changes both the SpEL top-vector path and the branch rule. Cleaner theory-facing options are `gap_estimator_mode=deflated_fp32_gap_only`, which keeps the original rank-one deflation branch rule but evaluates it in FP32, and `gap_estimator_mode=block2_fp32_gap_only`, which keeps the original SpEL `u/v` path while using an FP32 block-2 top-2 estimate only for `sigma2/sigma1` and the PGD-region test. See [../algorithms/mcsd_pgd_block2_fp32.md](../algorithms/mcsd_pgd_block2_fp32.md).
- `pgd_direction_normalization` supports `none`, `fro`, and `spectral`. The `spectral` mode estimates the PGD fallback direction's leading singular value with the same `power_iteration` helper used by SpEL/SSO, then divides by that value to align update scale under spectral norm.
- Runs after the 2026-07-09 logging update report PGD branch usage at every `LOG_INTERVAL`: stdout includes per-step `spel-pgd pgd branches: used/total (rate)` and cumulative `cumulative pgd branches: used/total (rate)`, and TensorBoard/W&B receive `spel-pgd/pgd-branch-count`, `spel-pgd/total-matrix-updates`, `spel-pgd/pgd-fallback-rate`, plus the matching `spel-pgd/cumulative-*` counters.

Future comparison algorithms should be added to this table before running them. Candidate columns to add later are implementation file, optimizer CLI name, default hyperparameters, and whether it needs a separate LR grid.

## Megatron Patch Summary

The intent is to keep the original Megatron-LM model and training stack intact, then add the SpEL/SSO optimizer path required by this experiment.

Code-level additions used by the experiment:

- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py`
- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py`
- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`
- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py`
- `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py`

Integration points:

- `Megatron-LM/megatron/core/optimizer/optimizer_config.py` defines SpEL, MCSD-PGD, and SpectralBall/SSO config fields.
- `Megatron-LM/megatron/training/arguments.py` exposes optimizer choices and CLI flags such as `--optimizer spel_dist`, `--optimizer spel_tp_dist`, `--optimizer spel_pgd_dist`, `--optimizer spectral_ball_dist`, `--spel-*`, `--spel-pgd-*`, and `--spectral-ball-*`.
- `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` registers `spel`, `spel_pgd`, and `spectral_ball` for Megatron's emerging optimizer path.
- `scripts/optimizer_compare_smoke.py` includes direct smoke tests for `spectral_ball_dist`, `spel`, and `spel_pgd`.

H20 import compatibility fixes:

- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/__init__.py` avoids exporting Triton-backed Muon and spectral-clipping modules from the package root.
- `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball_utils.py` imports `newton_schulz` lazily inside the large matrix-sign path.

These two changes avoid import-time Triton driver compilation failures in environments without the expected Python development headers. They do not change the GPT architecture or the experiment-level optimizer settings.

No intentional model-architecture change was made for this sweep. GPT layer definitions, attention/MLP layout, tokenizer interface, data-loader behavior, and pretraining entry point remain Megatron-style. The width-256 scale is produced by launcher arguments, not by modifying Megatron's transformer implementation.

The main runtime deviation from the original `spball.sh` is backend selection: H20 uses `TRANSFORMER_IMPL=local` after a TE/fused smoke test failed in the current environment. See [Backend Compatibility Note](#backend-compatibility-note).

Megatron source alignment update, 2026-07-04: `SSO_test/Megatron-LM` was replaced by a clean snapshot of the H20-tested `~/projects/Megatron-LM-dev-spel-v3` checkout (`codex/spel-h20-rebase`, upstream head `3e89f3c`). The project-local copy excludes generated files such as `.git`, `__pycache__`, logs, outputs, and compiled `.so` artifacts. Two short H20 smoke jobs verified the replacement: `3733419` for `spel_dist` and `3733420` for `spectral_ball_dist`, both `COMPLETED` with exit code `0:0`.

## Server Layout

Active server paths:

```text
~/projects/SSO_test
~/projects/SSO_test/Megatron-LM
```

The active H20 training jobs use the project-local bundled checkout:

```text
MEGATRON_PATH=${PROJECT_DIR}/Megatron-LM
```

The older server-local `~/projects/Megatron-LM-dev-spel-v3` checkout and `Megatron-LM-active` symlink were development artifacts. They are not the default path for the current Slurm scripts.

For the GitHub repository, the usable Megatron checkout is bundled under:

```text
Megatron-LM/
```

After cloning the repository, users can keep the default bundled checkout or set it explicitly:

```bash
export PROJECT_DIR=$PWD
export MEGATRON_PATH=$PWD/Megatron-LM
```

Archived old Megatron copies:

```text
~/projects/_archive_megatron_cleanup_20260704
```

## Backend Compatibility Note

The original `spball.sh` defaults to:

```bash
TRANSFORMER_IMPL=transformer_engine
ATTENTION_BACKEND=fused
```

The H20 experiment script uses:

```bash
TRANSFORMER_IMPL=local
ATTENTION_BACKEND=
```

This changes the execution backend but keeps the model architecture the same. It is applied consistently to all optimizers in this experiment.

Compatibility smoke test:

| Field | Value |
|---|---|
| Job ID | `3733395` |
| Job name | `te_fused_smoke` |
| Test config | `TRANSFORMER_IMPL=transformer_engine`, `ATTENTION_BACKEND=fused`, 2 train steps |
| Slurm state | `FAILED` |
| Exit code | `1:0` |
| Node | `SPG-7-1` |
| Failure point | model construction |
| Error summary | `TypeError: 'NoneType' object is not callable` from `TESpecProvider(...)` |

Relevant stderr warning:

```text
Transformer Engine and Apex are not installed. Falling back to Torch optimizers.
```

Conclusion: in the current `sso_h20` environment, the TE/fused path is not usable for this experiment without installing or fixing Transformer Engine/Apex compatibility. The completed optimizer sweeps therefore use `local` for all compared optimizers, which preserves fairness within this experiment but should be disclosed when comparing against paper runs that used TE/fused kernels.

## Data

Dataset source:

```text
allenai/olmo-mix-1124
```

Dataset card:

```text
https://huggingface.co/datasets/allenai/olmo-mix-1124
```

Tokenizer:

```text
allenai/OLMo-2-1124-7B
```

The 1B-token training set used here is a weighted sample from the OLMo mix components. The sampling weights are based on the component token counts used by the SSO paper appendix and encoded in `scripts/sample_olmo_mix_1124.py`.

### OLMo Mix Components

The full dataset is about 3.90T tokens and 22.4TB uncompressed. Component summary:

| Component | Tokens | Uncompressed bytes | Documents | License |
|---|---:|---:|---:|---|
| DCLM-Baseline | `3.70T` | `21.3TB` | `2.95B` | `CC-BY-4.0` |
| Arxiv | `20.8B` | `77.2GB` | `3.95M` | `ODC-BY` |
| pes2o | `58.6B` | `412GB` | `38M` | `ODC-BY` |
| starcoder | `83.0B` | `458GB` | `78.7M` | `ODC-BY` |
| Algebraic-stack | `11.8B` | `44.0GB` | `2.83M` | `ODC-BY` |
| OpenWebMath | `12.2B` | `47.23GB` | `2.89M` | `ODC-BY` |
| Wiki | `3.66B` | `18.1GB` | `6.17M` | `ODC-BY` |
| Total | `3.90T` | `22.4TB` | `3.08B` | `ODC-BY` |

### Weighted 1B Sample Allocation

For `TARGET_TOKENS=1000000000`, the sampler allocates tokens approximately as follows:

| Component | Weight | Approx train tokens |
|---|---:|---:|
| DCLM-Baseline | `95.1142%` | `951,142,142` |
| Arxiv | `0.5347%` | `5,346,961` |
| pes2o | `1.5064%` | `15,064,035` |
| starcoder | `2.1336%` | `21,336,432` |
| Algebraic-stack | `0.3033%` | `3,033,372` |
| OpenWebMath | `0.3136%` | `3,136,198` |
| Wiki | `0.0941%` | `940,860` |

For validation, the default `VALID_TOKENS=10000000` uses the same component ratios.

### Prepared Data On H20

Prepared data on H20:

```text
~/projects/SSO_test/data/olmo_mix_1124_1b/jsonl
~/projects/SSO_test/data/olmo_mix_1124_1b/indexed/train/olmo_mix_1124_1b_train_text_document.{bin,idx}
~/projects/SSO_test/data/olmo_mix_1124_1b/indexed/valid/olmo_mix_1124_1b_valid_text_document.{bin,idx}
~/projects/SSO_test/data/olmo_mix_1124_1b/tokenizer/OLMo-2-1124-7B
```

Observed prepared sizes from the completed run:

```text
raw JSONL sample: about 4.1G, 33 files
Megatron indexed data: about 7.6G
```

Local repository should not track the raw data, `.bin`, `.idx`, checkpoints, or large result directories.

### How To Get The Dataset

Login to Hugging Face on the server. Use a read token from the Hugging Face settings page. Do not write the token into scripts, docs, Git commits, or shell history snippets that will be uploaded.

```bash
ssh hpc2021
cd ~/projects/SSO_test
source ~/envs/sso_h20/bin/activate
hf auth login
```

Download tokenizer and stream the weighted OLMo mix sample into JSONL shards:

```bash
cd ~/projects/SSO_test

TARGET_TOKENS=1000000000 \
VALID_TOKENS=10000000 \
TOKENS_PER_SHARD=50000000 \
SEED=42 \
bash scripts/download_olmo_mix_1124_1b.sh
```

The downloader:

- saves tokenizer files under `data/olmo_mix_1124_1b/tokenizer/OLMo-2-1124-7B`;
- streams the selected OLMo mix components from Hugging Face;
- writes `train_*.jsonl` and `valid_*.jsonl` under `data/olmo_mix_1124_1b/jsonl`;
- writes `manifest.json` when sampling is complete.

Check download progress:

```bash
cd ~/projects/SSO_test
du -sh data/olmo_mix_1124_1b
find data/olmo_mix_1124_1b/jsonl -maxdepth 1 -name '*.jsonl' | wc -l
ls -lh data/olmo_mix_1124_1b/jsonl | tail
test -f data/olmo_mix_1124_1b/jsonl/manifest.json && cat data/olmo_mix_1124_1b/jsonl/manifest.json
```

If the process is interrupted before `manifest.json` exists, rerun the command. If you intentionally want to overwrite an existing partial or completed sample:

```bash
FORCE=1 bash scripts/download_olmo_mix_1124_1b.sh
```

### Megatron Preprocessing

Convert JSONL shards into Megatron `.bin/.idx` files:

```bash
cd ~/projects/SSO_test
bash scripts/preprocess_olmo_mix_1124_1b.sh
```

The preprocessing script uses:

```text
MEGATRON=$HOME/projects/SSO_test/Megatron-LM
RAW_DIR=$HOME/projects/SSO_test/data/olmo_mix_1124_1b/jsonl
TOKENIZER_DIR=$HOME/projects/SSO_test/data/olmo_mix_1124_1b/tokenizer/OLMo-2-1124-7B
INDEXED_DIR=$HOME/projects/SSO_test/data/olmo_mix_1124_1b/indexed
```

Expected indexed prefixes after preprocessing:

```text
data/olmo_mix_1124_1b/indexed/train/olmo_mix_1124_1b_train_text_document
data/olmo_mix_1124_1b/indexed/valid/olmo_mix_1124_1b_valid_text_document
```

Check that preprocessing finished:

```bash
cd ~/projects/SSO_test
ls -lh data/olmo_mix_1124_1b/indexed/train/olmo_mix_1124_1b_train_text_document.{bin,idx}
ls -lh data/olmo_mix_1124_1b/indexed/valid/olmo_mix_1124_1b_valid_text_document.{bin,idx}
du -sh data/olmo_mix_1124_1b/indexed
```

Common dataset issue:

```text
FileNotFoundError: zstd://...hf://datasets/allenai/olmo-mix-1124...
```

This usually means the Hugging Face streaming path could not resolve one shard through the `datasets` abstraction. The current sampler avoids that path by listing Hugging Face repository files and streaming JSON/ZSTD files directly with `requests` and `huggingface_hub`.

## Main Scripts

Main Slurm entry:

```text
slurm/spel_olmo_1b_h20.sbatch
```

Sweep submitter:

```text
slurm/submit_width256_sso_mcsd_lr_sweep.sh
```

The Slurm entry now defaults to:

```bash
MEGATRON_PATH="${MEGATRON_PATH:-$PROJECT_DIR/Megatron-LM}"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"
```

For a fresh clone outside the original H20 directory layout, override `MEGATRON_PATH`:

```bash
export MEGATRON_PATH=$PWD/Megatron-LM
```

## Experimental Protocol

Use the same protocol when adding new optimizers or extra LRs, unless the change is explicitly documented.

| Item | Rule |
|---|---|
| Data | Use the same prepared OLMo mix 1B train/valid `.bin/.idx` prefixes. |
| Tokenizer | Use `allenai/OLMo-2-1124-7B`. |
| Model shape | Keep width, layer count, FFN multiplier, head dim, sequence length, normalization, RoPE, and dropout unchanged. |
| Batch/token budget | Keep `GLOBAL_BATCH=128`, `MICRO_BATCH=4`, `TRAIN_TOKENS=1B`, `TRAIN_ITER=1908` unless creating a separate experiment section. |
| LR schedule | Use cosine decay, `LR_WARMUP_ITERS=250`, and `MIN_LR=LR/10`. |
| Evaluation | Use `EVAL_INTERVAL=250`, `EVAL_ITERS=5`; report the final validation loss and PPL at iteration `1908`. |
| Checkpoints | Disable checkpoint saving for LR sweep jobs unless a run is selected for deeper analysis. |
| Result acceptance | Only add rows with Slurm state `COMPLETED`, exit code `0:0`, and a parsed final validation line. |

For paper-facing comparisons, avoid mixing results from different batch sizes, train iteration counts, data prefixes, seeds, or validation settings in the same table.

## Experiment Config

Current sweep config:

| Field | Value |
|---|---|
| Width | `256` |
| Layers | `28` |
| Head dim | `128` |
| Attention heads | `2` |
| FFN hidden size | `768` |
| Sequence length | `4096` |
| Position embeddings | `40960` |
| Precision | `bf16` |
| Transformer implementation | `local` |
| GPUs per job | `1 x NVIDIA H20` |
| Micro batch | `4` |
| Global batch | `128` |
| Train tokens | `1,000,000,000` |
| Train iterations | `1908` |
| Warmup iterations | `250` |
| LR decay | `cosine` |
| Eval interval | `250` |
| Eval iters | `5` |
| Weight decay | `0.1` |
| Checkpoint saving | disabled for sweep |

SSO parameters are aligned with `Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spball/spball.sh`:

| SSO option | Value |
|---|---|
| Optimizer | `spectral_ball_dist` |
| Momentum | `0.9` |
| Nesterov | enabled |
| QKV split mode | `head` |
| msign steps | `8` |
| Radius mode | `spectral_mup` |
| Scale mode | `spectral_mup` |
| Solver | `bisection` |
| Solver tolerance f | `2e-4` |
| Solver max iterations | `20` |
| Power iteration steps | `10` |
| Retract mode | `hard` |
| Retract alpha | `0.05` |

MCSD-TP uses:

| MCSD-TP option | Value |
|---|---|
| Optimizer | `spel_dist` |
| Momentum | `0.9` |
| Nesterov | enabled |
| QKV split mode | `head` |
| msign steps | `8` |
| Radius mode | `spectral_mup` |
| Scale mode | `spectral_mup` |
| Power iteration steps | `10` |
| Retract mode | `hard` |
| Retract alpha | `0.05` |

## Completed Sweep Results

All jobs below completed successfully with Slurm state `COMPLETED` and exit code `0:0`.

| Optimizer | Megatron optimizer | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SSO | `spectral_ball_dist` | `5e-3` | `3725130` | `1908` | `3.658330` | `38.79651` | `05:59:26` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `7e-3` | `3725131` | `1908` | `3.625447` | `37.54150` | `05:59:23` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `9e-3` | `3725132` | `1908` | `3.595198` | `36.42292` | `06:00:05` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `1e-2` | `3725133` | `1908` | `3.590277` | `36.24412` | `06:00:42` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `1.5e-2` | `3725134` | `1908` | `3.570953` | `35.55044` | `06:02:39` | `SPG-7-1` |
| MCSD-TP | `spel_dist` | `5e-3` | `3725135` | `1908` | `3.657197` | `38.75255` | `05:27:28` | `SPG-7-1` |
| MCSD-TP | `spel_dist` | `7e-3` | `3725136` | `1908` | `3.616392` | `37.20308` | `05:26:49` | `SPG-7-2` |
| MCSD-TP | `spel_dist` | `9e-3` | `3725137` | `1908` | `3.596797` | `36.48121` | `05:27:10` | `SPG-7-2` |
| MCSD-TP | `spel_dist` | `1e-2` | `3725138` | `1908` | `3.587145` | `36.13079` | `05:25:56` | `SPG-7-2` |
| MCSD-TP | `spel_dist` | `1.5e-2` | `3725139` | `1908` | `3.567708` | `35.43530` | `05:26:59` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `5e-3` | `3733609` | `1908` | `3.931570` | `50.98697` | `05:30:42` | `SPG-7-1` |
| MCSD-PGD | `spel_pgd_dist` | `7e-3` | `3733610` | `1908` | `4.021310` | `55.77414` | `05:30:06` | `SPG-7-1` |
| MCSD-PGD | `spel_pgd_dist` | `9e-3` | `3733611` | `1908` | `4.071300` | `58.63311` | `05:29:09` | `SPG-7-1` |
| MCSD-PGD | `spel_pgd_dist` | `1e-2` | `3733612` | `1908` | `4.083732` | `59.36659` | `05:28:19` | `SPG-7-1` |
| MCSD-PGD | `spel_pgd_dist` | `1.5e-2` | `3733613` | `1908` | `4.099114` | `60.28685` | `05:29:23` | `SPG-7-1` |

Best result in this sweep:

```text
MCSD-TP / spel_dist, LR=1.5e-2, val loss=3.567708, PPL=35.43530
```

Best SSO result:

```text
SSO / spectral_ball_dist, LR=1.5e-2, val loss=3.570953, PPL=35.55044
```

Best MCSD-PGD result:

```text
MCSD-PGD / spel_pgd_dist, LR=5e-3, val loss=3.931570, PPL=50.98697
```

Difference at each LR, measured as `SSO val loss - MCSD-TP val loss`:

| LR | SSO - MCSD-TP |
|---:|---:|
| `5e-3` | `+0.001133` |
| `7e-3` | `+0.009055` |
| `9e-3` | `-0.001599` |
| `1e-2` | `+0.003132` |
| `1.5e-2` | `+0.003245` |

MCSD-PGD gap to the best baseline at the same LR, measured as `MCSD-PGD val loss - min(SSO, MCSD-TP) val loss`:

| LR | Best SSO/MCSD-TP val loss | MCSD-PGD val loss | Gap |
|---:|---:|---:|---:|
| `5e-3` | `3.657197` | `3.931570` | `+0.274373` |
| `7e-3` | `3.616392` | `4.021310` | `+0.404918` |
| `9e-3` | `3.595198` | `4.071300` | `+0.476102` |
| `1e-2` | `3.587145` | `4.083732` | `+0.496587` |
| `1.5e-2` | `3.567708` | `4.099114` | `+0.531406` |

Interpretation for this one-seed sweep:

- MCSD-TP is slightly better than SSO at four of five LRs; SSO is slightly better at `9e-3`.
- Both SSO and MCSD-TP improve as LR increases up to `1.5e-2`, so it remains reasonable to add higher LRs such as `2e-2` and `3e-2` for those two baselines if training remains stable.
- The current MCSD-PGD implementation is stable (`0` skipped iterations and `0` NaN iterations for all five jobs), but it is clearly worse than both SSO and MCSD-TP on this grid.
- MCSD-PGD degrades as LR increases in this grid; its best result is at the smallest tested LR, `5e-3`.
- For paper use, treat this early MCSD-PGD sweep as a negative or diagnostic algorithm-development result unless the fallback rule or direction scaling is revised and rerun.

### Paper Draft Notes

Use these results carefully:

- The table supports an initial width-256 LR sensitivity comparison.
- For SSO and MCSD-TP, the best observed LR in the current grid is the largest tested LR, so the optimum may lie beyond `1.5e-2`.
- The small loss gaps between SSO and MCSD-TP suggest that more LRs or repeated runs may be needed before making a strong claim.
- The early MCSD-PGD sweep is not competitive in that version; include it only if the paper needs a failed variant/ablation or if a revised version is rerun.
- If this result becomes a figure, plot validation loss versus LR with one curve per optimizer and clearly state `width=256`, `1B tokens`, `global batch=128`, and `eval_iters=5`.

## MCSD-PGD Projection Tuning

This 250M-token tuning pass was run after the original width-256 MCSD-PGD sweep. It fixes `width=256`, `LR=1.5e-2`, `GLOBAL_BATCH=128`, `EVAL_INTERVAL=100`, and `EVAL_ITERS=5`. The goal was to select a usable MCSD-PGD projection rule before running width 512.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Key setting | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---|
| SpEL-TP baseline | `branch_mode=spel` | `3734899` | `477` | `4.001449` | `54.67733` | `01:23:39` | `SPG-7-1` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-4` | `3734900` | `477` | `4.009463` | `55.11728` | `01:24:08` | `SPG-7-1` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-3` | `3734901` | `477` | `4.009463` | `55.11728` | `01:24:00` | `SPG-7-1` |
| MCSD-PGD exact | `fallback_exact`, `gap=5e-3` | `3734902` | `477` | `4.011888` | `55.25106` | `01:24:06` | `SPG-7-1` |
| MCSD-PGD exact | `fallback_exact`, `gap=1e-2` | `3734903` | `477` | `4.011760` | `55.24400` | `01:24:12` | `SPG-7-1` |
| MCSD-PGD exact | `fallback_exact`, `gap=5e-3`, `direction=fro` | `3734904` | `477` | `4.011085` | `55.20674` | `01:24:10` | `SPG-7-1` |
| MCSD-PGD retraction | `fallback_retraction`, `gap=5e-3` | `3734905` | `477` | `4.631289` | `102.6462` | `01:23:29` | `SPG-7-1` |
| MCSD-PGD retraction | `fallback_retraction`, `gap=1e-2` | `3734906` | `477` | `4.762298` | `117.0145` | `01:23:28` | `SPG-7-1` |
| MCSD-PGD top-k | `fallback_topk`, `k=1`, `gap=1e-4` | `3735002` | `477` | `4.019766` | `55.68809` | `01:24:13` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=2`, `gap=1e-4` | `3735003` | `477` | `4.003179` | `54.77197` | `01:23:58` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-4` | `3735004` | `477` | `4.000864` | `54.64534` | `01:23:53` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=1e-4` | `3735005` | `477` | `4.001924` | `54.70327` | `01:24:00` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=1`, `gap=1e-3` | `3735006` | `477` | `4.019766` | `55.68809` | `01:24:02` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=2`, `gap=1e-3` | `3735007` | `477` | `4.003179` | `54.77197` | `01:24:03` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=1e-3` | `3735008` | `477` | `4.000864` | `54.64534` | `01:24:04` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=1e-3` | `3735009` | `477` | `4.001924` | `54.70327` | `01:24:03` | `SPG-7-2` |
| MCSD-PGD top-k | `fallback_topk`, `k=4`, `gap=5e-3` | `3735010` | `477` | `4.002656` | `54.74335` | `01:24:13` | `SPG-7-1` |
| MCSD-PGD top-k | `fallback_topk`, `k=8`, `gap=5e-3` | `3735011` | `477` | `4.002858` | `54.75440` | `01:23:55` | `SPG-7-1` |

Selection for width 512:

```text
projection_mode=fallback_topk
projection_rank=4
gap_threshold_rel=1e-3
```

Rationale: `gap=1e-4` and `gap=1e-3` tie at `4.000864`, and `1e-3` is the less brittle threshold. This tuning result is a narrow 250M-token selection run; it should be treated as a configuration choice for the width-512 sweep, not as a paper-scale standalone claim.

## SpEL Projection Ablation

This follow-up run tests the projection choice for SpEL-TP/MCSD-TP itself. It uses `width=256`, `250M` training tokens, `LR=1.5e-2`, `GLOBAL_BATCH=128`, `EVAL_INTERVAL=100`, and `EVAL_ITERS=5`.

The tested SpEL projection modes are:

```text
retraction: original SpEL engineering retraction
exact:      full-SVD post-step spectral-sphere projection
topk:       approximate top-k post-step spectral-sphere projection
```

The MCSD-PGD `shared_topk` runs apply top-k projection to both the safe SpEL branch and the PGD fallback branch. In these historical rows, the safe SpEL branch also used post-msign TP re-projection.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Key setting | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---|
| SpEL-TP retraction baseline | `projection_mode=retraction` | `3738995` | `477` | `4.001449` | `54.67733` | `01:23:28` | `SPG-7-1` |
| SpEL-TP exact SVD | `projection_mode=exact` | `3738996` | `477` | `4.138569` | `62.71301` | `01:34:24` | `SPG-7-1` |
| SpEL-TP top-k | `projection_mode=topk`, `k=2` | `3738997` | `477` | `3.983903` | `53.72633` | `01:24:38` | `SPG-7-1` |
| SpEL-TP top-k | `projection_mode=topk`, `k=4` | `3738998` | `477` | `3.986554` | `53.86893` | `01:24:59` | `SPG-7-1` |
| SpEL-TP top-k | `projection_mode=topk`, `k=8` | `3738999` | `477` | `3.985768` | `53.82662` | `01:25:09` | `SPG-7-1` |
| MCSD-PGD fallback top-k | `fallback_topk`, `k=4` | `3739000` | `477` | `4.000864` | `54.64534` | `01:24:16` | `SPG-7-1` |
| MCSD-PGD shared top-k | `shared_topk`, `k=2` | `3739001` | `477` | `3.983814` | `53.72153` | `01:25:28` | `SPG-7-1` |
| MCSD-PGD shared top-k | `shared_topk`, `k=4` | `3739002` | `477` | `3.986346` | `53.85776` | `01:25:42` | `SPG-7-1` |
| MCSD-PGD shared top-k | `shared_topk`, `k=8` | `3739003` | `477` | `3.985801` | `53.82839` | `01:26:30` | `SPG-7-2` |

Interpretation:

- Exact SVD projection is slower and clearly worse in this implementation: `4.138569` versus `4.001449` for the original retraction baseline.
- Top-k projection improves SpEL-TP/MCSD-TP: `topk k=2` reaches `3.983903`.
- Applying top-k to both MCSD-PGD branches is marginally best: `shared_topk k=2` reaches `3.983814`, but the gap to SpEL-TP top-k k=2 is only `0.000089`.
- The strongest follow-up candidates are `SpEL-TP topk k=2` and `MCSD-PGD shared_topk k=2`.

## Width-256 Supplemental Top-k LR Sweep

This 1B-token supplemental run follows the projection settings that were strongest or most relevant after the 250M-token ablation:

```text
SpEL-TP / MCSD-TP: projection_mode=topk, projection_rank=8
MCSD-PGD:    projection_mode=shared_topk, projection_rank=4 or 8
```

The run keeps the same width-256 model, OLMo mix 1B sample, LR grid, global batch, evaluation cadence, tokenizer, and local Megatron backend as the original width-256 sweep.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Megatron optimizer | Key setting | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `5e-3` | `3740133` | `1908` | `3.640078` | `38.09481` | `05:34:53` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `7e-3` | `3740134` | `1908` | `3.599136` | `36.56664` | `05:33:27` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `9e-3` | `3740135` | `1908` | `3.583797` | `36.01000` | `05:33:26` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `1e-2` | `3740136` | `1908` | `3.580739` | `35.90005` | `05:33:30` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `1.5e-2` | `3740137` | `1908` | `3.566694` | `35.39936` | `05:34:38` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `5e-3` | `3740138` | `1908` | `3.638641` | `38.04009` | `05:36:58` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `7e-3` | `3740139` | `1908` | `3.602398` | `36.68610` | `05:35:31` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `9e-3` | `3740140` | `1908` | `3.583682` | `36.00586` | `05:36:06` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1e-2` | `3740141` | `1908` | `3.577118` | `35.77032` | `05:37:24` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1.5e-2` | `3740142` | `1908` | `3.568926` | `35.47848` | `05:37:10` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `5e-3` | `3740143` | `1908` | `3.640233` | `38.10070` | `05:36:36` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `7e-3` | `3740144` | `1908` | `3.599719` | `36.58796` | `05:35:49` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `9e-3` | `3740145` | `1908` | `3.584555` | `36.03732` | `05:36:19` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1e-2` | `3740146` | `1908` | `3.580421` | `35.88865` | `05:37:12` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1.5e-2` | `3740147` | `1908` | `3.566973` | `35.40925` | `05:37:25` | `SPG-7-2` |

Interpretation:

- Within this SpEL-TP / MCSD-PGD top-k supplement, the best row is `SpEL-TP topk k=8`, `LR=1.5e-2`, with val loss `3.566694`.
- After the later plain SpEL projection supplement, the best completed width-256 result overall is plain SpEL `topk k=4`, `LR=1.5e-2`, with val loss `3.562941`.
- The gain from SpEL-TP top-k over the original SpEL-TP/retraction row is small: `3.566694` versus `3.567708`.
- MCSD-PGD `shared_topk k=8` is very close at `1.5e-2`, with val loss `3.566973`.
- MCSD-PGD `shared_topk k=4` is best at `1.5e-2` among its own rows, but trails the `k=8` and SpEL-TP top-k rows at the same LR.

## Width-512 Completed Sweep Results

This extension uses the same 1B-token OLMo mix sample, LR grid, tokenizer, sequence length, global batch, and local Megatron backend as the width-256 sweep. Width 512 uses hidden size `512`, head dim `128`, four attention heads, FFN hidden size `1536`, and 28 layers.

MCSD-PGD uses the selected setting from the projection tuning run:

```text
projection_mode=fallback_topk
projection_rank=4
gap_threshold_rel=1e-3
```

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Optimizer | Megatron optimizer | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SSO | `spectral_ball_dist` | `5e-3` | `3737714` | `1908` | `3.423116` | `30.66482` | `11:12:16` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `7e-3` | `3737715` | `1908` | `3.371309` | `29.11660` | `11:14:35` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `9e-3` | `3737716` | `1908` | `3.345420` | `28.37248` | `11:14:06` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `1e-2` | `3737717` | `1908` | `3.338379` | `28.17342` | `11:17:06` | `SPG-7-1` |
| SSO | `spectral_ball_dist` | `1.5e-2` | `3737718` | `1908` | `3.322861` | `27.73959` | `11:21:05` | `SPG-7-1` |
| SpEL-TP | `spel_dist` | `5e-3` | `3737719` | `1908` | `3.420247` | `30.57696` | `10:18:25` | `SPG-7-1` |
| SpEL-TP | `spel_dist` | `7e-3` | `3737720` | `1908` | `3.371672` | `29.12719` | `10:17:12` | `SPG-7-1` |
| SpEL-TP | `spel_dist` | `9e-3` | `3737721` | `1908` | `3.345384` | `28.37146` | `10:17:31` | `SPG-7-1` |
| SpEL-TP | `spel_dist` | `1e-2` | `3737722` | `1908` | `3.337422` | `28.14648` | `10:18:51` | `SPG-7-2` |
| SpEL-TP | `spel_dist` | `1.5e-2` | `3737723` | `1908` | `3.321666` | `27.70647` | `10:17:51` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `5e-3` | `3737724` | `1908` | `3.418978` | `30.53818` | `10:22:23` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `7e-3` | `3737725` | `1908` | `3.371645` | `29.12640` | `10:22:21` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `9e-3` | `3737726` | `1908` | `3.346858` | `28.41331` | `10:24:34` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `1e-2` | `3737727` | `1908` | `3.339918` | `28.21682` | `10:25:18` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `1.5e-2` | `3737728` | `1908` | `3.321784` | `27.70973` | `10:22:07` | `SPG-7-2` |

Best result at width 512:

```text
SpEL-TP / spel_dist, LR=1.5e-2, val loss=3.321666, PPL=27.70647
```

MCSD-PGD is very close at the same LR:

```text
MCSD-PGD / spel_pgd_dist, LR=1.5e-2, val loss=3.321784, PPL=27.70973
```

For this one-seed width-512 sweep:

- All three optimizers improve as LR increases up to `1.5e-2`.
- SpEL-TP is best at `9e-3`, `1e-2`, and `1.5e-2`, but the gaps to SSO and MCSD-PGD are small.
- MCSD-PGD is best at `5e-3`; SSO is best at `7e-3`.
- MCSD-PGD with top-k projection is no longer the failed behavior seen in the earlier untuned width-256 MCSD-PGD sweep, but it has not clearly beaten SpEL-TP at the best LR.

## Width-512 Supplemental Top-k LR Sweep

This run is the width-512 counterpart of the completed width-256 supplemental top-k sweep. It finished on 2026-07-07 and uses:

```text
SpEL-TP / MCSD-TP: projection_mode=topk, projection_rank=8
MCSD-PGD:    projection_mode=shared_topk, projection_rank=4 or 8
LR grid:     5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
```

The jobs are currently tracked under:

```text
RUN_ROOT=/home/u3013198/projects/SSO_test/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep
script=slurm/submit_width512_spel_topk8_pgd_topk_lr_sweep.sh
```

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Megatron optimizer | Key setting | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `5e-3` | `3741588` | `1908` | `3.402194` | `30.02990` | `10:28:34` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `7e-3` | `3741589` | `1908` | `3.358461` | `28.74491` | `10:29:06` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `9e-3` | `3741590` | `1908` | `3.338081` | `28.16504` | `10:27:11` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `1e-2` | `3741591` | `1908` | `3.330682` | `27.95739` | `10:28:10` | `SPG-7-1` |
| SpEL-TP top-k | `spel_dist` | `topk`, `k=8` | `1.5e-2` | `3741592` | `1908` | `3.323886` | `27.76806` | `10:28:29` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `5e-3` | `3741593` | `1908` | `3.400489` | `29.97874` | `10:32:23` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `7e-3` | `3741594` | `1908` | `3.358685` | `28.75137` | `10:30:40` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `9e-3` | `3741595` | `1908` | `3.335894` | `28.10350` | `10:30:05` | `SPG-7-1` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1e-2` | `3741596` | `1908` | `3.331491` | `27.98003` | `10:34:28` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1.5e-2` | `3741597` | `1908` | `3.320985` | `27.68762` | `10:32:10` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `5e-3` | `3741598` | `1908` | `3.401184` | `29.99960` | `10:30:29` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `7e-3` | `3741599` | `1908` | `3.357924` | `28.72949` | `10:31:30` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `9e-3` | `3741600` | `1908` | `3.338519` | `28.17735` | `10:32:14` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1e-2` | `3741601` | `1908` | `3.331737` | `27.98691` | `10:34:34` | `SPG-7-2` |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1.5e-2` | `3741602` | `1908` | `3.322947` | `27.74197` | `10:31:50` | `SPG-7-2` |

Interpretation:

- Within the SpEL-TP / MCSD-PGD supplemental top-k sweep, the best width-512 result is `SpEL-TP topk k=4`, `LR=1.5e-2`, with val loss `3.319634`.
- After the later plain SpEL projection supplement, the best completed width-512 result overall is plain SpEL `topk k=4`, `LR=1.5e-2`, with val loss `3.318744`.
- This is slightly better than `MCSD-PGD shared_topk k=4`, `3.320985`, and the previous width-512 best SpEL-TP/retraction row, `3.321666`.
- `SpEL-TP topk k=8` is worse than original SpEL-TP at `1.5e-2` in width 512, even though it helped in width 256.
- `MCSD-PGD shared_topk k=4` is stronger than `shared_topk k=8` at the two highest LRs in this run.

## LR 1.5e-2 Width Comparison

This table aligns the current width-256 and width-512 rows at the highest LR. It is the primary compact comparison for the current 1B-token test goal.

| Width | Optimizer/config | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---:|---|---:|---:|---:|---:|---:|---|
| `256` | SSO / `spectral_ball_dist` | `3725134` | `1908` | `3.570953` | `35.55044` | `06:02:39` | `SPG-7-1` |
| `256` | plain SpEL / `spel_dist`, retraction | `3744519` | `1908` | `3.571484` | `35.56934` | `05:26:24` | `SPG-7-1` |
| `256` | plain SpEL / `spel_dist`, top-k `k=4` | `3744520` | `1908` | **`3.562941`** | `35.26677` | `05:32:06` | `SPG-7-2` |
| `256` | MuonBall / `muon_ball_dist` | `3747998` | `1908` | `3.564250` | `35.31298` | `05:24:57` | `SPG-7-2` |
| `256` | plain SpEL-PGD shared top-k / `spel_pgd_dist`, `k=8`, `sigma2=5` | `3747964` | `1908` | `3.566324` | `35.38627` | `05:37:26` | `SPG-7-1` |
| `256` | plain SpEL / `spel_dist`, top-k `k=8` | `3744521` | `1908` | `3.566394` | `35.38876` | `05:32:28` | `SPG-7-2` |
| `256` | SpEL-TP original retraction / `spel_dist` | `3725139` | `1908` | `3.567708` | `35.43530` | `05:26:59` | `SPG-7-2` |
| `256` | SpEL-TP top-k / `spel_dist`, `k=4` | `3743071` | `1908` | `3.567563` | `35.43013` | `05:33:45` | `SPG-7-1` |
| `256` | SpEL-TP top-k / `spel_dist`, `k=8` | `3740137` | `1908` | `3.566694` | `35.39936` | `05:34:38` | `SPG-7-1` |
| `256` | SpEL-TP-PGD shared top-k / `spel_pgd_dist`, `k=8`, `sigma2=5` | `3747965` | `1908` | `3.568810` | `35.47436` | `05:38:32` | `SPG-7-1` |
| `256` | MCSD-TP-PGD shared top-k / `spel_pgd_dist`, `k=4` | `3744523` | `1908` | `3.568926` | `35.47848` | `05:35:00` | `SPG-7-2` |
| `256` | MCSD-TP-PGD shared top-k / `spel_pgd_dist`, `k=8` | `3744524` | `1908` | `3.566973` | `35.40925` | `05:37:55` | `SPG-7-1` |
| `512` | SSO / `spectral_ball_dist` | `3737718` | `1908` | `3.322861` | `27.73959` | `11:21:05` | `SPG-7-1` |
| `512` | MuonBall / `muon_ball_dist` | `3751697` | `1908` | **`3.317970`** | `27.60424` | `10:14:49` | `SPG-7-1` |
| `512` | plain SpEL / `spel_dist`, retraction | `3744525` | `1908` | `3.322735` | `27.73611` | `10:16:56` | `SPG-7-1` |
| `512` | plain SpEL / `spel_dist`, top-k `k=4` | `3744526` | `1908` | `3.318744` | `27.62564` | `10:25:33` | `SPG-7-1` |
| `512` | plain SpEL / `spel_dist`, top-k `k=8` | `3744527` | `1908` | `3.321280` | `27.69578` | `10:28:18` | `SPG-7-2` |
| `512` | SpEL-TP original retraction / `spel_dist` | `3737723` | `1908` | `3.321666` | `27.70647` | `10:17:51` | `SPG-7-2` |
| `512` | SpEL-TP top-k / `spel_dist`, `k=4` | `3743072` | `1908` | `3.319634` | `27.65023` | `10:27:54` | `SPG-7-1` |
| `512` | SpEL-TP top-k / `spel_dist`, `k=8` | `3741592` | `1908` | `3.323886` | `27.76806` | `10:28:29` | `SPG-7-1` |
| `512` | MCSD-TP-PGD shared top-k / `spel_pgd_dist`, `k=4` | `3744529` | `1908` | `3.320985` | `27.68762` | `10:32:18` | `SPG-7-1` |
| `512` | MCSD-TP-PGD shared top-k / `spel_pgd_dist`, `k=8` | `3744530` | `1908` | `3.322947` | `27.74197` | `10:32:45` | `SPG-7-2` |

## Width-512 High-LR Projection Sweep

This sweep extends the width-512 LR grid to the right of the current best `1.5e-2` point. It follows the SSO paper LR grid continuation and tests:

```text
LR grid: 2e-2, 3e-2
Configs: SSO, SpEL-TP top-k k=4, SpEL-TP top-k k=8,
         MCSD-PGD shared_topk k=4, MCSD-PGD shared_topk k=8
Run root: /home/u3013198/projects/SSO_test/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep
Script: slurm/submit_width512_high_lr_projection_sweep.sh
```

Jobs were submitted on 2026-07-07 and completed with exit code `0:0`.

| Optimizer/config | LR | Job ID | Final iter | Val loss | PPL | Elapsed | Node |
|---|---:|---:|---:|---:|---:|---:|---|
| SSO / `spectral_ball_dist` | `2e-2` | `3743116` | `1908` | `3.327370` | `27.86495` | `11:22:09` | `SPG-7-1` |
| SpEL-TP top-k / `spel_dist`, `k=4` | `2e-2` | `3743117` | `1908` | `3.327565` | `27.87039` | `10:26:30` | `SPG-7-1` |
| SpEL-TP top-k / `spel_dist`, `k=8` | `2e-2` | `3743118` | `1908` | `3.325627` | `27.81644` | `10:28:41` | `SPG-7-1` |
| MCSD-PGD shared top-k / `spel_pgd_dist`, `k=4` | `2e-2` | `3743119` | `1908` | `3.328709` | `27.90231` | `10:32:27` | `SPG-7-1` |
| MCSD-PGD shared top-k / `spel_pgd_dist`, `k=8` | `2e-2` | `3743120` | `1908` | `3.326192` | `27.83217` | `10:31:40` | `SPG-7-1` |
| SSO / `spectral_ball_dist` | `3e-2` | `3743121` | `1908` | `3.364925` | `28.93134` | `11:26:20` | `SPG-7-1` |
| SpEL-TP top-k / `spel_dist`, `k=4` | `3e-2` | `3743122` | `1908` | `3.373786` | `29.18884` | `10:29:35` | `SPG-7-2` |
| SpEL-TP top-k / `spel_dist`, `k=8` | `3e-2` | `3743123` | `1908` | `3.375717` | `29.24525` | `10:30:39` | `SPG-7-2` |
| MCSD-PGD shared top-k / `spel_pgd_dist`, `k=4` | `3e-2` | `3743124` | `1908` | `3.372203` | `29.14266` | `10:31:51` | `SPG-7-2` |
| MCSD-PGD shared top-k / `spel_pgd_dist`, `k=8` | `3e-2` | `3743125` | `1908` | `3.373343` | `29.17591` | `10:35:54` | `SPG-7-2` |

Current interpretation: both higher LRs are worse than the `1.5e-2` rows. The width-512 minimum remains near `1.5e-2`.

## Plain SpEL / MCSD-TP-PGD Projection Supplement

This supplement was submitted on 2026-07-08 after the optimizer naming audit separated plain SpEL from SpEL-TP. It directly tests the missing projection choices at `LR=1.5e-2` for both `width=256` and `width=512`.

```text
Script: slurm/submit_width256_512_spel_mcsd_tp_pgd_projection_supplement.sh
Run roots:
  /home/u3013198/projects/SSO_test/results/olmo_1b_width256_spel_mcsd_tp_pgd_projection_supplement
  /home/u3013198/projects/SSO_test/results/olmo_1b_width512_spel_mcsd_tp_pgd_projection_supplement
Train tokens: 1B
Global batch: 128
Micro batch: 4
LR: 1.5e-2
```

Plain SpEL uses `OPTIMIZER=spel_dist` and `SPEL_TANGENT_PROJECT_AFTER_MSIGN=0`. MCSD-TP-PGD uses `OPTIMIZER=spel_pgd_dist`, `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1`, `SPEL_PGD_BRANCH_MODE=auto`, `SPEL_PGD_GAP_THRESHOLD_REL=1e-3`, and `SPEL_PGD_DIRECTION_NORMALIZATION=none`.

| Width | Optimizer/config | Projection | Job ID | Final iter | Val loss | PPL | Elapsed | Node |
|---:|---|---|---:|---:|---:|---:|---:|---|
| `256` | SpEL / `spel_dist` | `retraction` | `3744519` | `1908` | `3.571484` | `35.56934` | `05:26:24` | `SPG-7-1` |
| `256` | SpEL / `spel_dist` | `topk`, `k=4` | `3744520` | `1908` | `3.562941` | `35.26677` | `05:32:06` | `SPG-7-2` |
| `256` | SpEL / `spel_dist` | `topk`, `k=8` | `3744521` | `1908` | `3.566394` | `35.38876` | `05:32:28` | `SPG-7-2` |
| `256` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_retraction` | `3744522` | `1908` | `4.069594` | `58.53322` | `05:28:56` | `SPG-7-2` |
| `256` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_topk`, `k=4` | `3744523` | `1908` | `3.568926` | `35.47848` | `05:35:00` | `SPG-7-2` |
| `256` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_topk`, `k=8` | `3744524` | `1908` | `3.566973` | `35.40925` | `05:37:55` | `SPG-7-1` |
| `512` | SpEL / `spel_dist` | `retraction` | `3744525` | `1908` | `3.322735` | `27.73611` | `10:16:56` | `SPG-7-1` |
| `512` | SpEL / `spel_dist` | `topk`, `k=4` | `3744526` | `1908` | **`3.318744`** | `27.62564` | `10:25:33` | `SPG-7-1` |
| `512` | SpEL / `spel_dist` | `topk`, `k=8` | `3744527` | `1908` | `3.321280` | `27.69578` | `10:28:18` | `SPG-7-2` |
| `512` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_retraction` | `3744528` | `1908` | `3.753390` | `42.66548` | `10:20:46` | `SPG-7-2` |
| `512` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_topk`, `k=4` | `3744529` | `1908` | `3.320985` | `27.68762` | `10:32:18` | `SPG-7-1` |
| `512` | MCSD-TP-PGD / `spel_pgd_dist` | `shared_topk`, `k=8` | `3744530` | `1908` | `3.322947` | `27.74197` | `10:32:45` | `SPG-7-2` |

Current interpretation: plain SpEL top-k `k=4` is the strongest row in this supplement at both widths. MCSD-TP-PGD `shared_retraction` is clearly not competitive, while `shared_topk k=4/8` remains close to SSO and SpEL-TP.

## MuonBall Width-256 Seven-LR Supplement

This completed supplement adds the reference MuonBall optimizer to the same width-256 1B-token setup used by the SSO and SpEL sweeps. It answers whether removing the Spectral Ball lambda solver remains competitive at the small width used in this experiment.

```text
Script: slurm/submit_width256_muon_ball_lr_sweep.sh
Run root: /home/u3013198/projects/SSO_test/results/olmo_1b_width256_muon_ball_lr_sweep
Optimizer: muon_ball_dist
Width: 256
Train tokens: 1B
Global batch: 128
Micro batch: 4
LR grid: 5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2, 2e-2, 3e-2
```

MuonBall constants follow the original SSO repository launcher `Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/muonball/muonball.sh`: `momentum=0.9`, Nesterov enabled, `msign_steps=8`, `radius_mode=spectral_mup`, `scale_mode=spectral_mup`, `power_iteration_steps=10`, `retract_mode=hard`, and `qkv_split_mode=head`.

Completed H20 results:

| LR | Job | Val loss | PPL | Elapsed |
|---:|---:|---:|---:|---:|
| `5e-3` | `3747994` | `3.639009` | `38.05410` | `05:24:57` |
| `7e-3` | `3747995` | `3.600150` | `36.60372` | `05:24:05` |
| `9e-3` | `3747996` | `3.581582` | `35.93032` | `05:26:44` |
| `1e-2` | `3747997` | `3.575525` | `35.71338` | `05:24:58` |
| `1.5e-2` | `3747998` | **`3.564250`** | `35.31298` | `05:24:57` |
| `2e-2` | `3747999` | `3.571979` | `35.58694` | `05:24:49` |
| `3e-2` | `3748000` | `3.611113` | `37.00722` | `05:25:05` |

Comparison against the completed width-256 SSO / SpEL-TP grid:

| LR | SSO width-256 val loss | SpEL-TP width-256 val loss | Plain SpEL top-k k=4 note |
|---:|---:|---:|---|
| `5e-3` | `3.658330` | `3.657197` | plain SpEL top-k k=4 not run at this LR yet |
| `7e-3` | `3.625447` | `3.616392` | plain SpEL top-k k=4 not run at this LR yet |
| `9e-3` | `3.595198` | `3.596797` | plain SpEL top-k k=4 not run at this LR yet |
| `1e-2` | `3.590277` | `3.587145` | plain SpEL top-k k=4 not run at this LR yet |
| `1.5e-2` | `3.570953` | `3.567708` | completed plain SpEL top-k k=4: `3.562941` |
| `2e-2` | not run at width 256 | not run at width 256 | high-LR extension point |
| `3e-2` | not run at width 256 | not run at width 256 | high-LR extension point |

Current interpretation: MuonBall's best width-256 row is `LR=1.5e-2`, with val loss `3.564250`. It beats width-256 SSO at the same LR (`3.570953`) and is close to plain SpEL, but remains slightly worse than plain SpEL `topk k=4` (`3.562941`).

## MuonBall Width-512 Seven-LR Supplement

This completed supplement uses the same width-512 1B-token setup and MuonBall
optimizer constants as the width-256 supplement.

| LR | Val loss | PPL | Elapsed | Job |
|---:|---:|---:|---:|---:|
| `5e-3` | `3.398522` | `29.91984` | `10:16:03` | `3751693` |
| `7e-3` | `3.351419` | `28.54322` | `10:15:21` | `3751694` |
| `9e-3` | `3.329687` | `27.92960` | `10:15:07` | `3751695` |
| `1e-2` | `3.323978` | `27.77059` | `10:15:34` | `3751696` |
| `1.5e-2` | **`3.317970`** | **`27.60424`** | `10:14:49` | `3751697` |
| `2e-2` | `3.331416` | `27.97792` | `10:13:29` | `3751698` |
| `3e-2` | `3.382643` | `29.44849` | `10:14:00` | `3751699` |

The best LR is `1.5e-2`. This is the strongest completed width-512 row,
slightly ahead of plain SpEL top-k `k=4` (`3.318744`) and SSO (`3.322861`).
The margin over plain SpEL is `0.000774` under one seed, so it is suggestive
rather than a robust ranking claim.

## Plain SpEL-PGD Sigma2 Supplement

Plain SpEL-PGD is the same `spel_pgd_dist` optimizer with the SpEL branch post-msign tangent projection disabled:

```text
SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0
```

This completed supplement tests whether increasing `sigma2_power_iteration_steps` helps the PGD branch. It uses the same width-256 1B-token setup, `LR=1.5e-2`, `projection_mode=shared_topk`, and `rank=8`. It includes both plain SpEL-PGD and a historical SpEL-TP-PGD ablation, where SpEL-TP-PGD enables the post-msign tangent projection. Future MCSD-PGD experiments use the plain variant by default.

Run settings:

```text
OPTIMIZER=spel_pgd_dist
SPEL_PGD_BRANCH_MODE=auto
SPEL_PGD_PROJECTION_MODE=shared_topk
SPEL_PGD_RANKS=8
SPEL_PGD_GAP_THRESHOLD_REL=5e-3
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=5,8,10
SPEL_PGD_DIRECTION_NORMALIZATION=none
TRAIN_TOKENS=1000000000
GLOBAL_BATCH=128
MICRO_BATCH=4
```

| Width | Variant | TP after msign | sigma2 steps | Job | Val loss | PPL | Elapsed |
|---:|---|---:|---:|---:|---:|---:|---:|
| `256` | plain SpEL-PGD | `0` | `5` | `3747964` | **`3.566324`** | `35.38627` | `05:37:26` |
| `256` | SpEL-TP-PGD | `1` | `5` | `3747965` | `3.568810` | `35.47436` | `05:38:32` |
| `256` | plain SpEL-PGD | `0` | `8` | `3747966` | `3.569803` | `35.50958` | `05:35:26` |
| `256` | SpEL-TP-PGD | `1` | `8` | `3747967` | `3.569142` | `35.48614` | `05:37:30` |
| `256` | plain SpEL-PGD | `0` | `10` | `3747968` | `3.627396` | `37.61473` | `05:35:03` |
| `256` | SpEL-TP-PGD | `1` | `10` | `3747969` | `3.610268` | `36.97597` | `05:35:16` |

Current interpretation: `sigma2_power_iteration_steps=5` is best in this sweep. Increasing sigma2 steps to `8` or `10` does not improve validation loss and is clearly worse at `10`. The best plain SpEL-PGD row (`3.566324`) is competitive with SpEL-TP and MCSD-TP-PGD, but does not beat plain SpEL `topk k=4` (`3.562941`) or MuonBall (`3.564250`) at width 256.

Branch-count caveat: jobs `3747964`-`3747969` completed before the 2026-07-09 training-log update that records PGD branch usage. Exact PGD usage counts are therefore available only for new or rerun SpEL-PGD jobs.

## Plain MCSD-PGD Gap/Normalization Tuning

Submitted on 2026-07-09 to test whether PGD fallback frequency and direction scale explain the sigma2 degradation. All jobs in this block use the plain MCSD-PGD variant only: `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`, `width=256`, `LR=1.5e-2`, `TRAIN_TOKENS=250M`, `projection_mode=shared_topk`, and `rank=8`.

Completed phase A fixes `sigma2_power_iteration_steps=5` and sweeps direction normalization plus `gap_threshold_rel`.

| Direction normalization | Gap thresholds | Jobs | Run root |
|---|---|---|---|
| `none` | `0`, `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3`, `1e-2` | `3749547`-`3749553` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_gap_threshold_tune` |
| `fro` | `0`, `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3`, `1e-2` | `3749569`-`3749575` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_gap_threshold_tune_fro` |
| `spectral` | `0`, `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3`, `1e-2` | `3749612`-`3749618` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_gap_threshold_tune_spectral` |

All phase-A jobs completed with Slurm exit code `0:0`. Best rows by normalization:

| Direction normalization | Best gap | Job | Val loss | PPL | Cumulative PGD branches |
|---|---:|---:|---:|---:|---:|
| `spectral` | `1e-4` to `2e-3` | `3749613`-`3749616` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `fro` | `5e-3` | `3749574` | `3.991130` | `54.11601` | `432/118440` (`0.004`) |
| `none` | `0` | `3749547` | `3.991379` | `54.12947` | `0/118440` (`0.000`) |

Spectral-normalized detail:

| sigma2 steps | gap | Job | Val loss | PPL | Cumulative PGD branches |
|---:|---:|---:|---:|---:|---:|
| `5` | `0` | `3749612` | `3.991379` | `54.12947` | `0/118440` (`0.000`) |
| `5` | `1e-4` | `3749613` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `5` | `5e-4` | `3749614` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `5` | `1e-3` | `3749615` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `5` | `2e-3` | `3749616` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `5` | `5e-3` | `3749617` | `3.990719` | `54.09379` | `562/118440` (`0.005`) |
| `5` | `1e-2` | `3749618` | `3.994977` | `54.32460` | `649/118440` (`0.005`) |

Current phase-A interpretation: `spectral` normalization is the best direction scaling choice. `gap_threshold_rel=1e-2` is clearly too permissive. The useful range is narrow and conservative, roughly `1e-4` to `2e-3`; `1e-3` is the safest default within that plateau.

Completed phase B keeps `spectral` normalization and tests whether changing `sigma2_power_iteration_steps` improves over the current `sigma2=5` default. It reuses the phase-A `sigma2=5` results and submits only the missing values:

| sigma2 steps | Gap thresholds | Jobs | Run root |
|---:|---|---|---|
| `3` | `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3` | `3750042`-`3750046` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_spectral_sigma2_gap_sweep/sigma2_3` |
| `8` | `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3` | `3750047`-`3750051` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_spectral_sigma2_gap_sweep/sigma2_8` |
| `10` | `1e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3` | `3750052`-`3750056` | `/home/u3013198/projects/SSO_test/results/olmo_250m_width256_pgd_spectral_sigma2_gap_sweep/sigma2_10` |

Phase-B final results:

| sigma2 steps | Best gap | Best job(s) | Val loss | PPL | Cumulative PGD branches |
|---:|---:|---|---:|---:|---:|
| `3` | `1e-4` to `5e-3` | `3750042`-`3750046` | `3.990414` | `54.07725` | `7/118440` (`0.000`) |
| `5` | `1e-4` to `2e-3` | `3749613`-`3749616` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) |
| `8` | `1e-4` to `2e-3` | `3750047`-`3750050` | `4.003870` | `54.80988` | `10790/118440` (`0.091`) |
| `10` | `5e-3` | `3750056` | `4.095759` | `60.08492` | `43638/118440` (`0.368`) |

Current parameter decision: keep `sigma2_power_iteration_steps=5`, `gap_threshold_rel=1e-3`, and `pgd_direction_normalization=spectral`. This keeps PGD useful but rare. The code now also exposes `pgd_lr_scale`, defaulting to `0.5`, so a PGD fallback takes a smaller branch-specific step instead of immediately perturbing a near-degenerate matrix as strongly as the SpEL branch. There is still no sticky PGD state or cooldown; if future logs show consecutive PGD bursts that remain harmful even with `pgd_lr_scale=0.5`, add a separate cooldown or hysteresis rule.

The earlier sigma2=10 deflated-estimator follow-up was submitted and then cancelled on 2026-07-09 after the `block2_fp32` estimator was added. Cancelled jobs: `3750496`-`3750505` and `3750507`-`3750512`. Those jobs should not be used as completed results.

Submitted block2-FP32 gap-control follow-up on 2026-07-09:

```bash
bash slurm/submit_width256_pgd_block2_fp32_gap_control_sweep.sh
```

This follow-up fixes `gap_estimator_mode=block2_fp32`, `sigma2_power_iteration_steps=10`, `pgd_direction_normalization=spectral`, `pgd_lr_scale=0.5`, `projection_mode=shared_topk`, and `projection_rank=8`. It searches gap thresholds in a range that is meaningful under FP32 estimation and should control PGD usage by the measured top-2 spectral gap rather than by BF16/deflation noise.

```text
0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
```

Decision rule for this follow-up: target a cumulative PGD branch rate of roughly `0.5%` to `2%`; within that subset, choose the lowest validation loss. If all nonzero gaps remain worse than `gap=0`, treat PGD fallback as unhelpful under this theoretically cleaner estimator and keep the best non-PGD/rare-PGD setting.

Submitted jobs:

| gap_estimator_mode | pgd_lr_scale | Gap | Job |
|---|---:|---:|---:|
| `block2_fp32` | `0.5` | `0` | `3750634` |
| `block2_fp32` | `0.5` | `1e-6` | `3750635` |
| `block2_fp32` | `0.5` | `3e-6` | `3750636` |
| `block2_fp32` | `0.5` | `1e-5` | `3750637` |
| `block2_fp32` | `0.5` | `3e-5` | `3750638` |
| `block2_fp32` | `0.5` | `1e-4` | `3750639` |
| `block2_fp32` | `0.5` | `3e-4` | `3750640` |
| `block2_fp32` | `0.5` | `1e-3` | `3750641` |

Completed FP32 main-power follow-up on 2026-07-10:

```bash
bash slurm/submit_width256_pgd_fp32_main_power_gap_lr_sweep.sh
```

This follow-up tests `spel_pgd_main_power_dtype=fp32`, while keeping
`width=256`, `LR=1.5e-2`, `250M` train tokens, `shared_topk k=8`,
`sigma2_power_iteration_steps=10`, and spectral PGD direction normalization.
It compares `block2_fp32_gap_only + cold`, `block2_fp32_gap_only + warm`, and
`block2_fp32 + warm` at gaps `0` / `1e-4`.

Summary results:

| Setting | Best job | Gap | PGD lr | Val loss | PPL | PGD branches |
|---|---:|---:|---:|---:|---:|---:|
| `block2_fp32_gap_only`, cold, FP32 main path | `3752962` | `0` | `0.5` | **`3.987389`** | `53.91395` | `0/118440` |
| `block2_fp32_gap_only`, cold, FP32 main path | `3752965` | `1e-4` | `1.0` | `3.990568` | `54.08558` | `2/118440` |
| `block2_fp32_gap_only`, warm, FP32 main path | `3752969` | `1e-4` | `1.0` | `3.999609` | `54.57683` | `2/118440` |
| `block2_fp32`, warm | `3752971` | `1e-4` | `0.2` | `3.998521` | `54.51746` | `64/118440` |

Interpretation: FP32 ordinary SpEL power iteration improves the cold no-PGD
baseline over the BF16/default path (`3.987389` versus `3.991379`). Warm-starting
the ordinary SpEL `u/v` path remains worse. Coupled `block2_fp32 + warm` is much
healthier than cold coupled block2, but it still changes the SpEL top-vector
path and does not beat gap-only cold. Gap `1e-4` remains too conservative under
the block2-FP32 region test; PGD is selected only a few times, so these rows
mainly test main-path precision rather than the PGD fallback.

1B focused follow-ups completed on 2026-07-11:

```bash
bash slurm/submit_width256_pgd_fp32_gaponly_cold_focused_1b.sh
bash slurm/submit_width256_pgd_block2_warm_gap_lr_1b.sh
```

The broader 1B sanity sweep jobs `3754433`-`3754444` were cancelled before
completion. The main follow-up keeps the clean `block2_fp32_gap_only + cold +
main_power_dtype=fp32` path and searches gaps `1e-4`, `2e-4`, `3e-4` with
`pgd_lr_scale=0.2/0.5/1`. The lower-priority coupled-block2 contrast keeps
`block2_fp32 + warm_start_uv=1`, `shared_topk k=8`, fixed default seed,
`main_power_dtype=fp32`, and only tests gaps `1e-4` / `2e-4` with
`pgd_lr_scale=0.2/0.5`.

Main gap-only cold results:

| Gap | PGD lr | Val loss | PPL | PGD branches | Elapsed | Job |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | `0.2` | `3.571691` | `35.57670` | `17/478800` (`0.004%`) | `05:53:32` | `3754476` |
| `2e-4` | `0.2` | `3.570711` | `35.54184` | `65/478800` (`0.014%`) | `05:52:05` | `3754477` |
| `3e-4` | `0.2` | `3.571269` | `35.56169` | `229/478800` (`0.048%`) | `05:51:17` | `3754478` |
| `1e-4` | `0.5` | `3.571257` | `35.56128` | `18/478800` (`0.004%`) | `05:51:43` | `3754479` |
| `2e-4` | `0.5` | `3.570093` | `35.51990` | `69/478800` (`0.014%`) | `05:51:57` | `3754480` |
| `3e-4` | `0.5` | **`3.569919`** | **`35.51371`** | `102/478800` (`0.021%`) | `05:52:10` | `3754481` |
| `1e-4` | `1.0` | `3.570949` | `35.55031` | `5/478800` (`0.001%`) | `05:51:20` | `3754482` |
| `2e-4` | `1.0` | `3.571085` | `35.55515` | `21/478800` (`0.004%`) | `05:52:30` | `3754483` |
| `3e-4` | `1.0` | `3.572173` | `35.59387` | `118/478800` (`0.025%`) | `05:51:53` | `3754484` |

Coupled block2 warm results:

| Gap | PGD lr | Val loss | PPL | PGD branches | Elapsed | Job |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | `0.2` | `3.572953` | `35.62162` | `238/478800` (`0.050%`) | `05:50:07` | `3754549` |
| `1e-4` | `0.5` | `3.579248` | `35.84659` | `265/478800` (`0.055%`) | `05:50:58` | `3754550` |
| `2e-4` | `0.2` | `3.575163` | `35.70044` | `1022/478800` (`0.213%`) | `05:50:11` | `3754558` |
| `2e-4` | `0.5` | `3.575118` | `35.69882` | `1215/478800` (`0.254%`) | `05:50:10` | `3754559` |

Interpretation:

- The best row in this follow-up is `block2_fp32_gap_only + cold +
  gap=3e-4 + pgd_lr_scale=0.5` with validation loss `3.569919`. It slightly
  improves over the width-256 SSO row in this document (`3.570953`) but remains
  worse than the earlier plain SpEL and sigma2=5 SpEL-PGD rows.
- PGD usage under gap-only estimation remains tiny: the best row only uses
  `102/478800` PGD branches (`0.021%`). This means the current improvement is
  not evidence that frequent PGD fallback is beneficial.
- `pgd_lr_scale=1.0` is not useful in this grid. It does not improve the
  low-gap rows and is worse at `gap=3e-4`.
- Coupled `block2_fp32 + warm` remains inferior. It triggers more PGD
  (`0.050%` to `0.254%`) but all four rows are worse than gap-only cold, so the
  top-vector path should still not be replaced by block2 Ritz vectors.
- Runtime is stable: all completed 1B follow-up jobs are about `05:50` to
  `05:53`. The FP32 gap estimator does not create a large wall-clock penalty in
  this setup.

Adaptive gap-probe follow-up completed on 2026-07-11. It fixes the best
gap-only configuration above and reduces FP32 gap estimation only when the
last measured `rel_gap` for that matrix is greater than `10 * gap_threshold`.
Matrices inside that warning region continue to probe every step.

| Probe interval | Val loss | PPL | Gap probes at iter 1900 | PGD branches at iter 1900 | Elapsed | Saving | Job |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Every step | `3.569919` | `35.51371` | every matrix update | `102/478800` | `05:52:10` | baseline | `3754481` |
| `5` | **`3.569197`** | **`35.48809`** | `98880/478800` (`20.65%`) | `122/478800` | `05:37:45` | `4.09%` | `3756922` |
| `10` | `3.569514` | `35.49934` | `50623/478800` (`10.57%`) | `144/478800` | **`05:37:05`** | `4.28%` | `3756923` |

K=10 is only 40 seconds faster than K=5, so halving the probe rate from about
20.7% to 10.6% has little additional wall-clock value. K=5 is preferred from
this pair. Its validation result is also best, but all three losses are within
`0.001` under one fixed seed and should be treated as effectively tied. Initial
jobs `3756915` and `3756916` were cancelled after a split-component cache-key
bug was detected and are excluded.

Mis-submitted `k=4/16` jobs `3754547`, `3754548`, `3754551`, and `3754552`
were cancelled after about two minutes and should not be compared.

Remaining coverage if plain SpEL-PGD stays in the paper comparison:

| Width | LR | Projection modes | Jobs needed |
|---:|---:|---|---:|
| `512` | `1.5e-2` | `shared_topk k=8`, `sigma2=5`, `gap=1e-3`, `spectral`, `pgd_lr_scale=0.5` | `1` |
| `256`, `512` | `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` | best plain SpEL-PGD setting if selected | optional grid |

## Width-1024 Memory Smoke

This two-iteration smoke test checks whether `width=1024`, `num_layers=28`, `seq_length=4096`, `global_batch=128`, and `micro_batch=4` fit on one H20. These are not training-quality results and should not be compared by validation loss.

| Optimizer | Job | State | Max allocated MB | Max reserved MB | Elapsed |
|---|---:|---|---:|---:|---:|
| SSO / `spectral_ball_dist` | `3748023` | `COMPLETED` | `75555.72` | `80610.00` | `00:03:07` |
| SpEL-TP / `spel_tp_dist` | `3748024` | `COMPLETED` | `74531.72` | `78978.00` | `00:02:54` |
| MuonBall / `muon_ball_dist` | `3748025` | `COMPLETED` | `74531.72` | `78978.00` | `00:02:54` |

Current interpretation: width 1024 does not OOM at `micro_batch=4`, `seq_length=4096` on H20 for these three optimizers. SSO has the highest observed allocation and is closest to the limit.

## Width-1024 SSO/MuonBall LR Sweep

Submitted on 2026-07-11:

```bash
bash slurm/submit_width1024_sso_muonball_lr_sweep.sh
```

This is the first full width-1024 1B-token comparison. It uses the same model
shape validated by the smoke test: `num_layers=28`, `hidden_size=1024`,
`seq_length=4096`, `global_batch=128`, `micro_batch=4`, and one H20 GPU per job.
The LR grid matches the width-256 MuonBall supplement:
`5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2`, `2e-2`, `3e-2`.

The submit script sets `SBATCH_TIME=2-00:00:00`. This is intentional: the
two-iteration smoke measured SSO around `50`-`57` seconds per training
iteration, so a 1B-token run with `1908` iterations can exceed the previous
1-day default.

Run root:

```text
/home/u3013198/projects/SSO_test/results/olmo_1b_width1024_sso_muonball_lr_sweep
```

MuonBall completed on 2026-07-12:

| LR | Val loss | PPL | Elapsed | Job |
|---:|---:|---:|---:|---:|
| `5e-3` | `3.224939` | `25.15204` | `21:10:55` | `3756221` |
| `7e-3` | `3.177335` | `23.98275` | `21:15:52` | `3756222` |
| `9e-3` | `3.154934` | `23.45149` | `21:14:30` | `3756223` |
| `1e-2` | **`3.148978`** | **`23.31222`** | `21:09:34` | `3756224` |
| `1.5e-2` | `3.149347` | `23.32083` | `21:10:36` | `3756225` |
| `2e-2` | `3.174094` | `23.90515` | `21:14:51` | `3756226` |
| `3e-2` | `3.234591` | `25.39598` | `21:18:39` | `3756227` |

MuonBall is best at `1e-2`; `1.5e-2` is effectively tied. The useful LR range
is approximately `9e-3` to `1.5e-2`.

SSO job `3756214` (`LR=5e-3`) completed with final val loss `3.246347`, PPL
`25.69631`, and elapsed time `23:10:10`. Jobs `3756215`-`3756220` are still
running as of 2026-07-12. Their latest common validation checkpoint is
iteration 1750:

| LR | Interim val loss | Interim PPL | Job |
|---:|---:|---:|---:|
| `5e-3` | `3.241580` | `25.57410` | `3756214` |
| `7e-3` | `3.194677` | `24.40230` | `3756215` |
| `9e-3` | `3.171328` | `23.83912` | `3756216` |
| `1e-2` | `3.165558` | `23.70197` | `3756217` |
| `1.5e-2` | **`3.157638`** | **`23.51499`** | `3756218` |
| `2e-2` | `3.175557` | `23.94014` | `3756219` |
| `3e-2` | `3.235303` | `25.41408` | `3756220` |

These SSO values are not final results. At iteration 1750, the current best LR
is `1.5e-2`; final optimizer comparisons must wait for iteration 1908.

## Historical Baseline

Earlier completed run:

| Field | Value |
|---|---|
| Job ID | `3723351` |
| Job name | `spel_olmo_1b_w256` |
| Optimizer | `spel_dist` |
| LR | `1e-3` |
| Width | `256` |
| Global batch | `64` |
| Train iterations | `3815` |
| Train tokens | about `1B` |
| Final val iter | `3815` |
| Final val loss | `3.914298` |
| Final PPL | `50.11389` |
| Slurm state | `COMPLETED 0:0` |
| Elapsed | `05:37:27` |
| Node | `SPG-7-1` |

This baseline used a different global batch and iteration count, so keep it separate from the 5-LR sweep table.

## Artifact Locations

Remote logs:

```text
~/projects/SSO_test/logs/sso_w256_lr5em3_3725130.out
~/projects/SSO_test/logs/sso_w256_lr7em3_3725131.out
~/projects/SSO_test/logs/sso_w256_lr9em3_3725132.out
~/projects/SSO_test/logs/sso_w256_lr1em2_3725133.out
~/projects/SSO_test/logs/sso_w256_lr1p5em2_3725134.out
~/projects/SSO_test/logs/mcsd_w256_lr5em3_3725135.out
~/projects/SSO_test/logs/mcsd_w256_lr7em3_3725136.out
~/projects/SSO_test/logs/mcsd_w256_lr9em3_3725137.out
~/projects/SSO_test/logs/mcsd_w256_lr1em2_3725138.out
~/projects/SSO_test/logs/mcsd_w256_lr1p5em2_3725139.out
~/projects/SSO_test/logs/spel_pgd_w256_lr5em3_3733609.out
~/projects/SSO_test/logs/spel_pgd_w256_lr7em3_3733610.out
~/projects/SSO_test/logs/spel_pgd_w256_lr9em3_3733611.out
~/projects/SSO_test/logs/spel_pgd_w256_lr1em2_3733612.out
~/projects/SSO_test/logs/spel_pgd_w256_lr1p5em2_3733613.out
```

Local copied logs:

```text
analysis_logs_372513/
```

Remote result root:

```text
~/projects/SSO_test/results/olmo_1b_width256_sso_mcsd_lr_sweep
~/projects/SSO_test/results/olmo_1b_width256_spel_pgd_lr_sweep
```

## Reproduce Current Sweep

From the server:

```bash
ssh hpc2021
cd ~/projects/SSO_test
bash slurm/submit_width256_sso_mcsd_lr_sweep.sh
bash slurm/submit_width256_spel_pgd_lr_sweep.sh
bash slurm/submit_width256_muon_ball_lr_sweep.sh
```

Monitor:

```bash
squeue -u u3013198
sacct -j <job_id> --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
tail -f logs/<job_name>_<job_id>.out
```

The MCSD-PGD sweep can also be rerun alone:

```bash
cd ~/projects/SSO_test
bash slurm/submit_width256_spel_pgd_lr_sweep.sh
```

## Add More Learning Rates

Use the same script and override `LRS`. Use a new `RUN_ROOT` to avoid mixing logs/results with the completed baseline sweep.

Example for adding `2e-2` and `3e-2`:

```bash
cd ~/projects/SSO_test
LRS="2e-2 3e-2" \
RUN_ROOT="$HOME/projects/SSO_test/results/olmo_1b_width256_sso_mcsd_lr_sweep_extra_lr" \
bash slurm/submit_width256_sso_mcsd_lr_sweep.sh
```

After completion, append rows to the completed sweep table in this document.

## Add A New Optimizer

First confirm the optimizer name exists in the active Megatron checkout:

```bash
cd ~/projects/SSO_test/Megatron-LM
grep -R "choices=.*optimizer" -n megatron/training/arguments.py
grep -R "<optimizer_name>" -n megatron/core/optimizer
```

Then submit one job by reusing `slurm/spel_olmo_1b_h20.sbatch`:

```bash
cd ~/projects/SSO_test

LR="9e-3"
LR_TAG="9em3"
OPTIMIZER="<optimizer_name>"
LABEL="<short_label>"
MIN_LR="9e-4"

sbatch \
  -J "${LABEL}_w256_lr${LR_TAG}" \
  --export=ALL,OPTIMIZER="$OPTIMIZER",WIDTH=256,NUM_LAYERS=28,HEAD_DIM=128,SEQ_LENGTH=4096,GLOBAL_BATCH=128,MICRO_BATCH=4,TRAIN_TOKENS=1000000000,LR="$LR",MIN_LR="$MIN_LR",LR_WARMUP_ITERS=250,EVAL_INTERVAL=250,EVAL_ITERS=5,LOG_INTERVAL=10,RUN_ROOT="$HOME/projects/SSO_test/results/olmo_1b_width256_new_optimizers",JOB_NAME="${LABEL}_w256_lr${LR_TAG}",SAVE_CHECKPOINT=0 \
  slurm/spel_olmo_1b_h20.sbatch
```

For a full sweep, copy `slurm/submit_width256_sso_mcsd_lr_sweep.sh` or add another `submit_one` loop with the new optimizer name and label.

## Result Row Template

Append new rows using this schema:

```markdown
| Optimizer | Megatron optimizer | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NEW | `optimizer_name` | `lr` | `job_id` | `iter` | `loss` | `ppl` | `elapsed` | `node` |
```

Also record:

```text
Slurm state:
Exit code:
Run root:
Log file:
Any instability:
```

## Parse Final Validation Loss From Logs

PowerShell parser for local copied logs:

```powershell
$lrMap = @{
  "5em3" = "5e-3"
  "7em3" = "7e-3"
  "9em3" = "9e-3"
  "1em2" = "1e-2"
  "1p5em2" = "1.5e-2"
}

Get-ChildItem -Path analysis_logs_372513 -Filter *.out | ForEach-Object {
  $txt = Get-Content -LiteralPath $_.FullName -Raw
  $name = $_.BaseName
  $opt = if ($name.StartsWith("sso_")) { "SSO" } else { "MCSD-TP" }
  $tag = [regex]::Match($name, "_lr([^_]+)_").Groups[1].Value
  $valMatches = [regex]::Matches(
    $txt,
    "validation loss at iteration\s+(\d+)[^\r\n]*lm loss value:\s*([0-9.E+-]+)[^\r\n]*lm loss PPL:\s*([0-9.E+-]+)"
  )
  if ($valMatches.Count -gt 0) {
    $m = $valMatches[$valMatches.Count - 1]
    [pscustomobject]@{
      Optimizer = $opt
      LR = $lrMap[$tag]
      Job = $name
      ValIter = $m.Groups[1].Value
      ValLoss = [double]$m.Groups[2].Value
      PPL = [double]$m.Groups[3].Value
      Finished = $txt.Contains("==== finished ====")
    }
  }
} | Sort-Object Optimizer, {[double]($_.LR -replace "e","E")} | Format-Table -AutoSize
```

## GitHub Upload Notes

Recommended files to track:

```text
docs/experiments/width256_sso_mcsd_lr_sweep_1b.md
slurm/spel_olmo_1b_h20.sbatch
slurm/submit_width256_sso_mcsd_lr_sweep.sh
scripts/preprocess_olmo_mix_1124_1b.sh
scripts/sample_olmo_mix_1124.py
scripts/patch_megatron_dev_spectral_ball.py
scripts/rebase_spel_to_megatron_dev.py
```

Do not upload:

```text
data/
results/
outputs/
logs/
checkpoints/
*.bin
*.idx
Hugging Face token
SSH private key
HPC password
```

Optional: upload small parsed result tables, but avoid uploading full training logs unless needed for reproducibility.

## References

Primary external resources:

```text
OLMo mix dataset: https://huggingface.co/datasets/allenai/olmo-mix-1124
OLMo tokenizer:   https://huggingface.co/allenai/OLMo-2-1124-7B
```

Relevant repository files:

```text
Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spball/spball.sh
Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spel/spel.sh
slurm/spel_olmo_1b_h20.sbatch
slurm/submit_width256_sso_mcsd_lr_sweep.sh
scripts/sample_olmo_mix_1124.py
scripts/download_olmo_mix_1124_1b.sh
scripts/preprocess_olmo_mix_1124_1b.sh
```
