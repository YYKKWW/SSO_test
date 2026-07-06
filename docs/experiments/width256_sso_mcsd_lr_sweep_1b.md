# Experiment Record: Width-256/512 Optimizer LR Sweep on OLMo Mix 1B

Last updated: 2026-07-07

This document is the primary experiment record for the `width=256` and `width=512` optimizer learning-rate sweeps. It is intended to support paper development, later reproduction, and future extensions with new optimizers or additional learning rates.

Do not put passwords, SSH private keys, Hugging Face tokens, HPC passwords, or other secrets in this file.

## Status Summary

| Field | Current status |
|---|---|
| Experiment family | Small-scale pretraining LR sweep |
| Paper role | One supporting experiment for optimizer comparison |
| Width | `256`, `512` |
| Data budget | `1B` training tokens |
| Dataset | Weighted sample from `allenai/olmo-mix-1124` |
| Compared optimizers | SSO / `spectral_ball_dist`, MCSD/SpEL / `spel_dist`, MCSD-PGD / `spel_pgd_dist` |
| LR grid | `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` |
| Jobs completed | width-256 1B sweep: `15/15`; MCSD-PGD 250M tuning: `18/18`; SpEL projection 250M ablation: `9/9`; width-512 1B sweep: `15/15`; width-256 supplemental top-k sweep: `15/15` |
| Slurm status | completed rows are all `COMPLETED`, all exit code `0:0`; width-512 supplemental top-k sweep submitted as jobs `3741588`-`3741602` |
| Main result table | [Completed Sweep Results](#completed-sweep-results) |
| Next likely extension | repeat selected settings or extend to width `1024` |

## Scope And Caveats

This is a controlled small-scale experiment, not the full paper sweep.

- The completed runs cover `width=256` and `width=512`; they do not yet cover widths `1024` or `2048`.
- The current run uses one 1B-token weighted sample from OLMo mix; it is not a 30B-token paper-scale run.
- The current table is a single-run comparison; paper claims should be calibrated accordingly unless repeated seeds or additional settings are added.
- The current MCSD label maps to the `spel_dist` implementation path because the active Megatron launcher does not expose an optimizer literally named `mcsd`.
- The original width-256 SpEL-PGD rows use the first `spel_pgd_dist` implementation. The later width-512 MCSD-PGD rows use the selected top-k projection setting from the 250M-token tuning run: `fallback_topk`, rank `4`, gap `1e-3`.
- The supplemental top-k rows use SpEL `projection_mode=topk` and MCSD-PGD `projection_mode=shared_topk`.
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
- MCSD: `spel_dist`
- SpEL-PGD: `spel_pgd_dist`

The current sweep uses 1B training tokens from the weighted OLMo mix sample and evaluates five learning rates:

```text
5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
```

The immediate goal is to determine optimizer LR sensitivity at small width and whether the current LR grid or algorithm implementation should be extended before running more expensive widths.

## Algorithms Compared

This experiment compares three optimizer implementations in the active Megatron checkout. They are run with the same model architecture, data, token budget, warmup/decay schedule, weight decay, and batch settings. The only intended difference is the optimizer algorithm.

| Display name | Megatron optimizer name | Main source files | Role in this sweep |
|---|---|---|---|
| SSO / Spectral Sphere | `spectral_ball_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`, `spectral_ball_utils.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | Main SSO baseline following the paper's `spball` scripts. |
| MCSD / SpEL | `spel_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | Comparison optimizer currently treated as the MCSD/SpEL path. |
| SpEL-PGD | `spel_pgd_dist` | `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spel_pgd_same_projection.py`, `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py` | New algorithm under test: SpEL-style spectral retraction with an automatic PGD fallback branch. |

SSO follows the Spectral Sphere / Spectral Ball setup:

- It constrains selected matrix weights to a spectral ball or sphere with radius based on spectral muP scaling.
- It uses matrix-sign style updates, computed with Newton-Schulz iterations.
- It uses hard retraction in this sweep, so weights are projected back to the spectral constraint after the update.
- In the current Megatron launcher, distributed SSO is selected by `--optimizer spectral_ball_dist`.

MCSD/SpEL in this project is the current comparison optimizer path:

- It uses the `spel_dist` Megatron optimizer entry.
- It uses the same spectral-muP radius/scale choices as the SSO run for a fair small-scale comparison.
- It uses Nesterov momentum, QKV head split, `msign_steps=8`, and hard retraction in the current sweep.

Important naming note: the codebase does not expose an optimizer literally named `mcsd` in the current launcher. For this experiment log, "MCSD" means the active `spel_dist` optimizer path unless a later patch adds a separate `mcsd` optimizer name.

SpEL-PGD in this project is the first new algorithm extension after the SSO/MCSD baseline:

- It subclasses the SpEL path and keeps the same spectral-sphere retraction operator: `power_iteration + apply_retract`.
- It does not use an exact SVD projection.
- In `branch_mode=auto`, it estimates the relative top-singular-value gap and switches to a PGD-style momentum direction when the gap is below `gap_threshold_rel`.
- Both the SpEL branch and PGD branch form a trial point and apply the same post-step SpEL-style retraction.
- The sweep uses `branch_mode=auto`, `gap_threshold_rel=5e-3`, `sigma2_power_iteration_steps=3`, and `pgd_direction_normalization=none`.

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

- `Megatron-LM/megatron/core/optimizer/optimizer_config.py` defines SpEL, SpEL-PGD, and SpectralBall/SSO config fields.
- `Megatron-LM/megatron/training/arguments.py` exposes optimizer choices and CLI flags such as `--optimizer spel_dist`, `--optimizer spel_pgd_dist`, `--optimizer spectral_ball_dist`, `--spel-*`, `--spel-pgd-*`, and `--spectral-ball-*`.
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
~/projects/Megatron-LM-active -> ~/projects/Megatron-LM-dev-spel-v3
```

The active H20 Megatron checkout is `Megatron-LM-dev-spel-v3`, exposed through the stable symlink `Megatron-LM-active`.

For the GitHub repository, the usable Megatron checkout is bundled under:

```text
Megatron-LM/
```

After cloning the repository, users can point scripts to the bundled checkout:

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
MEGATRON=$HOME/projects/Megatron-LM-active
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

MCSD uses:

| MCSD option | Value |
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
| MCSD | `spel_dist` | `5e-3` | `3725135` | `1908` | `3.657197` | `38.75255` | `05:27:28` | `SPG-7-1` |
| MCSD | `spel_dist` | `7e-3` | `3725136` | `1908` | `3.616392` | `37.20308` | `05:26:49` | `SPG-7-2` |
| MCSD | `spel_dist` | `9e-3` | `3725137` | `1908` | `3.596797` | `36.48121` | `05:27:10` | `SPG-7-2` |
| MCSD | `spel_dist` | `1e-2` | `3725138` | `1908` | `3.587145` | `36.13079` | `05:25:56` | `SPG-7-2` |
| MCSD | `spel_dist` | `1.5e-2` | `3725139` | `1908` | `3.567708` | `35.43530` | `05:26:59` | `SPG-7-2` |
| SpEL-PGD | `spel_pgd_dist` | `5e-3` | `3733609` | `1908` | `3.931570` | `50.98697` | `05:30:42` | `SPG-7-1` |
| SpEL-PGD | `spel_pgd_dist` | `7e-3` | `3733610` | `1908` | `4.021310` | `55.77414` | `05:30:06` | `SPG-7-1` |
| SpEL-PGD | `spel_pgd_dist` | `9e-3` | `3733611` | `1908` | `4.071300` | `58.63311` | `05:29:09` | `SPG-7-1` |
| SpEL-PGD | `spel_pgd_dist` | `1e-2` | `3733612` | `1908` | `4.083732` | `59.36659` | `05:28:19` | `SPG-7-1` |
| SpEL-PGD | `spel_pgd_dist` | `1.5e-2` | `3733613` | `1908` | `4.099114` | `60.28685` | `05:29:23` | `SPG-7-1` |

Best result in this sweep:

```text
MCSD / spel_dist, LR=1.5e-2, val loss=3.567708, PPL=35.43530
```

Best SSO result:

```text
SSO / spectral_ball_dist, LR=1.5e-2, val loss=3.570953, PPL=35.55044
```

Best SpEL-PGD result:

```text
SpEL-PGD / spel_pgd_dist, LR=5e-3, val loss=3.931570, PPL=50.98697
```

Difference at each LR, measured as `SSO val loss - MCSD val loss`:

| LR | SSO - MCSD |
|---:|---:|
| `5e-3` | `+0.001133` |
| `7e-3` | `+0.009055` |
| `9e-3` | `-0.001599` |
| `1e-2` | `+0.003132` |
| `1.5e-2` | `+0.003245` |

SpEL-PGD gap to the best baseline at the same LR, measured as `SpEL-PGD val loss - min(SSO, MCSD) val loss`:

| LR | Best SSO/MCSD val loss | SpEL-PGD val loss | Gap |
|---:|---:|---:|---:|
| `5e-3` | `3.657197` | `3.931570` | `+0.274373` |
| `7e-3` | `3.616392` | `4.021310` | `+0.404918` |
| `9e-3` | `3.595198` | `4.071300` | `+0.476102` |
| `1e-2` | `3.587145` | `4.083732` | `+0.496587` |
| `1.5e-2` | `3.567708` | `4.099114` | `+0.531406` |

Interpretation for this one-seed sweep:

- MCSD is slightly better than SSO at four of five LRs; SSO is slightly better at `9e-3`.
- Both SSO and MCSD improve as LR increases up to `1.5e-2`, so it remains reasonable to add higher LRs such as `2e-2` and `3e-2` for those two baselines if training remains stable.
- The current SpEL-PGD implementation is stable (`0` skipped iterations and `0` NaN iterations for all five jobs), but it is clearly worse than both SSO and MCSD on this grid.
- SpEL-PGD degrades as LR increases in this grid; its best result is at the smallest tested LR, `5e-3`.
- For paper use, treat SpEL-PGD as a negative or diagnostic algorithm-development result unless the fallback rule or direction scaling is revised and rerun.

### Paper Draft Notes

Use these results carefully:

- The table supports an initial width-256 LR sensitivity comparison.
- For SSO and MCSD, the best observed LR in the current grid is the largest tested LR, so the optimum may lie beyond `1.5e-2`.
- The small loss gaps between SSO and MCSD suggest that more LRs or repeated runs may be needed before making a strong claim.
- SpEL-PGD is not competitive in the current version; include it only if the paper needs a failed variant/ablation or if a revised version is rerun.
- If this result becomes a figure, plot validation loss versus LR with one curve per optimizer and clearly state `width=256`, `1B tokens`, `global batch=128`, and `eval_iters=5`.

## MCSD-PGD Projection Tuning

This 250M-token tuning pass was run after the original width-256 SpEL-PGD sweep. It fixes `width=256`, `LR=1.5e-2`, `GLOBAL_BATCH=128`, `EVAL_INTERVAL=100`, and `EVAL_ITERS=5`. The goal was to select a usable MCSD-PGD projection rule before running width 512.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Key setting | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---|
| SpEL baseline | `branch_mode=spel` | `3734899` | `477` | `4.001449` | `54.67733` | `01:23:39` | `SPG-7-1` |
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

This follow-up run tests the projection choice for SpEL/MCSD itself. It uses `width=256`, `250M` training tokens, `LR=1.5e-2`, `GLOBAL_BATCH=128`, `EVAL_INTERVAL=100`, and `EVAL_ITERS=5`.

The tested SpEL projection modes are:

```text
retraction: original SpEL engineering retraction
exact:      full-SVD post-step spectral-sphere projection
topk:       approximate top-k post-step spectral-sphere projection
```

The SpEL-PGD `shared_topk` runs apply top-k projection to both the safe SpEL branch and the PGD fallback branch.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Key setting | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---:|---:|---:|---:|---:|---|
| SpEL retraction baseline | `projection_mode=retraction` | `3738995` | `477` | `4.001449` | `54.67733` | `01:23:28` | `SPG-7-1` |
| SpEL exact SVD | `projection_mode=exact` | `3738996` | `477` | `4.138569` | `62.71301` | `01:34:24` | `SPG-7-1` |
| SpEL top-k | `projection_mode=topk`, `k=2` | `3738997` | `477` | `3.983903` | `53.72633` | `01:24:38` | `SPG-7-1` |
| SpEL top-k | `projection_mode=topk`, `k=4` | `3738998` | `477` | `3.986554` | `53.86893` | `01:24:59` | `SPG-7-1` |
| SpEL top-k | `projection_mode=topk`, `k=8` | `3738999` | `477` | `3.985768` | `53.82662` | `01:25:09` | `SPG-7-1` |
| SpEL-PGD fallback top-k | `fallback_topk`, `k=4` | `3739000` | `477` | `4.000864` | `54.64534` | `01:24:16` | `SPG-7-1` |
| SpEL-PGD shared top-k | `shared_topk`, `k=2` | `3739001` | `477` | `3.983814` | `53.72153` | `01:25:28` | `SPG-7-1` |
| SpEL-PGD shared top-k | `shared_topk`, `k=4` | `3739002` | `477` | `3.986346` | `53.85776` | `01:25:42` | `SPG-7-1` |
| SpEL-PGD shared top-k | `shared_topk`, `k=8` | `3739003` | `477` | `3.985801` | `53.82839` | `01:26:30` | `SPG-7-2` |

Interpretation:

- Exact SVD projection is slower and clearly worse in this implementation: `4.138569` versus `4.001449` for the original retraction baseline.
- Top-k projection improves SpEL/MCSD: `topk k=2` reaches `3.983903`.
- Applying top-k to both SpEL-PGD branches is marginally best: `shared_topk k=2` reaches `3.983814`, but the gap to SpEL top-k k=2 is only `0.000089`.
- The strongest follow-up candidates are `SpEL topk k=2` and `SpEL-PGD shared_topk k=2`.

## Width-256 Supplemental Top-k LR Sweep

This 1B-token supplemental run follows the projection settings that were strongest or most relevant after the 250M-token ablation:

```text
SpEL / MCSD: projection_mode=topk, projection_rank=8
MCSD-PGD:    projection_mode=shared_topk, projection_rank=4 or 8
```

The run keeps the same width-256 model, OLMo mix 1B sample, LR grid, global batch, evaluation cadence, tokenizer, and local Megatron backend as the original width-256 sweep.

All jobs below completed with Slurm state `COMPLETED` and exit code `0:0`.

| Variant | Megatron optimizer | Key setting | LR | Job ID | Final val iter | Val loss | PPL | Elapsed | Node |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `5e-3` | `3740133` | `1908` | `3.640078` | `38.09481` | `05:34:53` | `SPG-7-1` |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `7e-3` | `3740134` | `1908` | `3.599136` | `36.56664` | `05:33:27` | `SPG-7-1` |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `9e-3` | `3740135` | `1908` | `3.583797` | `36.01000` | `05:33:26` | `SPG-7-1` |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `1e-2` | `3740136` | `1908` | `3.580739` | `35.90005` | `05:33:30` | `SPG-7-1` |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `1.5e-2` | `3740137` | `1908` | `3.566694` | `35.39936` | `05:34:38` | `SPG-7-1` |
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

- The best completed width-256 1B result is now `SpEL topk k=8`, `LR=1.5e-2`, with val loss `3.566694`.
- The gain over the original SpEL/retraction row is small: `3.566694` versus `3.567708`.
- MCSD-PGD `shared_topk k=8` is very close at `1.5e-2`, with val loss `3.566973`.
- MCSD-PGD `shared_topk k=4` is best at `1.5e-2` among its own rows, but trails the `k=8` and SpEL top-k rows at the same LR.

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
| SpEL | `spel_dist` | `5e-3` | `3737719` | `1908` | `3.420247` | `30.57696` | `10:18:25` | `SPG-7-1` |
| SpEL | `spel_dist` | `7e-3` | `3737720` | `1908` | `3.371672` | `29.12719` | `10:17:12` | `SPG-7-1` |
| SpEL | `spel_dist` | `9e-3` | `3737721` | `1908` | `3.345384` | `28.37146` | `10:17:31` | `SPG-7-1` |
| SpEL | `spel_dist` | `1e-2` | `3737722` | `1908` | `3.337422` | `28.14648` | `10:18:51` | `SPG-7-2` |
| SpEL | `spel_dist` | `1.5e-2` | `3737723` | `1908` | `3.321666` | `27.70647` | `10:17:51` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `5e-3` | `3737724` | `1908` | `3.418978` | `30.53818` | `10:22:23` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `7e-3` | `3737725` | `1908` | `3.371645` | `29.12640` | `10:22:21` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `9e-3` | `3737726` | `1908` | `3.346858` | `28.41331` | `10:24:34` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `1e-2` | `3737727` | `1908` | `3.339918` | `28.21682` | `10:25:18` | `SPG-7-2` |
| MCSD-PGD | `spel_pgd_dist` | `1.5e-2` | `3737728` | `1908` | `3.321784` | `27.70973` | `10:22:07` | `SPG-7-2` |

Best result at width 512:

```text
SpEL / spel_dist, LR=1.5e-2, val loss=3.321666, PPL=27.70647
```

MCSD-PGD is very close at the same LR:

```text
MCSD-PGD / spel_pgd_dist, LR=1.5e-2, val loss=3.321784, PPL=27.70973
```

For this one-seed width-512 sweep:

- All three optimizers improve as LR increases up to `1.5e-2`.
- SpEL is best at `9e-3`, `1e-2`, and `1.5e-2`, but the gaps to SSO and MCSD-PGD are small.
- MCSD-PGD is best at `5e-3`; SSO is best at `7e-3`.
- MCSD-PGD with top-k projection is no longer the failed behavior seen in the earlier untuned width-256 SpEL-PGD sweep, but it has not clearly beaten SpEL at the best LR.

## Width-512 Supplemental Top-k LR Sweep

This run is the width-512 counterpart of the completed width-256 supplemental top-k sweep. It was submitted on 2026-07-07 and uses:

```text
SpEL / MCSD: projection_mode=topk, projection_rank=8
MCSD-PGD:    projection_mode=shared_topk, projection_rank=4 or 8
LR grid:     5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
```

The jobs are currently tracked under:

```text
RUN_ROOT=/home/u3013198/projects/SSO_test/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep
script=slurm/submit_width512_spel_topk8_pgd_topk_lr_sweep.sh
```

| Variant | Megatron optimizer | Key setting | LR | Job ID | Status |
|---|---|---|---:|---:|---|
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `5e-3` | `3741588` | submitted/running |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `7e-3` | `3741589` | submitted/running |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `9e-3` | `3741590` | submitted/running |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `1e-2` | `3741591` | submitted/running |
| SpEL top-k | `spel_dist` | `topk`, `k=8` | `1.5e-2` | `3741592` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `5e-3` | `3741593` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `7e-3` | `3741594` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `9e-3` | `3741595` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1e-2` | `3741596` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=4` | `1.5e-2` | `3741597` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `5e-3` | `3741598` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `7e-3` | `3741599` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `9e-3` | `3741600` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1e-2` | `3741601` | submitted/running |
| MCSD-PGD shared top-k | `spel_pgd_dist` | `shared_topk`, `k=8` | `1.5e-2` | `3741602` | submitted/running |

Update this section with final val loss, PPL, elapsed time, and node after `sacct` reports `COMPLETED`.

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
```

Monitor:

```bash
squeue -u u3013198
sacct -j <job_id> --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
tail -f logs/<job_name>_<job_id>.out
```

The SpEL-PGD sweep can also be rerun alone:

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
cd ~/projects/Megatron-LM-active
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
  $opt = if ($name.StartsWith("sso_")) { "SSO" } else { "MCSD" }
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
