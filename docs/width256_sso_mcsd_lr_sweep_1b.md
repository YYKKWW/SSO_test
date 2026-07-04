# Experiment Record: Width-256 SSO vs MCSD LR Sweep on OLMo Mix 1B

Last updated: 2026-07-04

This document is the primary experiment record for the `width=256` SSO-vs-MCSD learning-rate sweep. It is intended to support paper development, later reproduction, and future extensions with new optimizers or additional learning rates.

Do not put passwords, SSH private keys, Hugging Face tokens, HPC passwords, or other secrets in this file.

## Status Summary

| Field | Current status |
|---|---|
| Experiment family | Small-scale pretraining LR sweep |
| Paper role | One supporting experiment for optimizer comparison |
| Width | `256` |
| Data budget | `1B` training tokens |
| Dataset | Weighted sample from `allenai/olmo-mix-1124` |
| Compared optimizers | SSO / `spectral_ball_dist`, MCSD / `spel_dist` |
| LR grid | `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` |
| Jobs completed | `10/10` |
| Slurm status | all `COMPLETED`, all exit code `0:0` |
| Main result table | [Completed Sweep Results](#completed-sweep-results) |
| Next likely extension | add higher LRs such as `2e-2`, `3e-2`; add new optimizer rows |

## Scope And Caveats

This is a controlled small-scale experiment, not the full paper sweep.

- The current run fixes `width=256`; it does not yet cover widths `512`, `1024`, or `2048`.
- The current run uses one 1B-token weighted sample from OLMo mix; it is not a 30B-token paper-scale run.
- The current table is a single-run comparison; paper claims should be calibrated accordingly unless repeated seeds or additional settings are added.
- The current MCSD label maps to the `spel_dist` implementation path because the active Megatron launcher does not expose an optimizer literally named `mcsd`.
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

The current sweep uses 1B training tokens from the weighted OLMo mix sample and evaluates five learning rates:

```text
5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
```

The immediate goal is to determine whether the two optimizers have similar LR sensitivity at small width and whether the current LR grid should be extended upward before running more expensive widths or new algorithms.

## Algorithms Compared

This experiment compares two optimizer implementations in the active Megatron checkout. Both are run with the same model architecture, data, token budget, warmup/decay schedule, weight decay, and batch settings. The only intended difference is the optimizer algorithm.

| Display name | Megatron optimizer name | Main source files | Role in this sweep |
|---|---|---|---|
| SSO / Spectral Sphere | `spectral_ball_dist` | `Megatron-LM/megatron/core/optimizer/spectral_ball_optimizer.py`, `emerging_optimizers/orthogonalized_optimizers/spectral_ball.py`, `spectral_ball_utils.py` | Main SSO baseline following the paper's `spball` scripts. |
| MCSD / SpEL | `spel_dist` | `Megatron-LM/megatron/core/optimizer/spel.py`, `emerging_optimizers/orthogonalized_optimizers/spel.py` | Comparison optimizer currently treated as the MCSD/SpEL path. |

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

Future comparison algorithms should be added to this table before running them. Candidate columns to add later are implementation file, optimizer CLI name, default hyperparameters, and whether it needs a separate LR grid.

## Server Layout

Active server paths:

```text
~/projects/SSO_test
~/projects/Megatron-LM-active -> ~/projects/Megatron-LM-dev-spel-v3
```

The active Megatron checkout is `Megatron-LM-dev-spel-v3`, exposed through the stable symlink `Megatron-LM-active`. The old `~/projects/SSO_test/Megatron-LM` package is kept for history/smoke scripts, but it is not the main runnable Megatron dev package for the 1B experiments.

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

Conclusion: in the current `sso_h20` environment, the TE/fused path is not usable for this experiment without installing or fixing Transformer Engine/Apex compatibility. The completed SSO-vs-MCSD sweep therefore uses `local` for both optimizers, which preserves fairness within this experiment but should be disclosed when comparing against paper runs that used TE/fused kernels.

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
MEGATRON_PATH="${MEGATRON_PATH:-$HOME/projects/Megatron-LM-active}"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"
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

Best result in this sweep:

```text
MCSD / spel_dist, LR=1.5e-2, val loss=3.567708, PPL=35.43530
```

Best SSO result:

```text
SSO / spectral_ball_dist, LR=1.5e-2, val loss=3.570953, PPL=35.55044
```

Difference at each LR, measured as `SSO val loss - MCSD val loss`:

| LR | SSO - MCSD |
|---:|---:|
| `5e-3` | `+0.001133` |
| `7e-3` | `+0.009055` |
| `9e-3` | `-0.001599` |
| `1e-2` | `+0.003132` |
| `1.5e-2` | `+0.003245` |

Interpretation for this one-seed sweep: MCSD is slightly better at four of five LRs; SSO is slightly better at `9e-3`. Both optimizers improve as LR increases up to `1.5e-2`, so it is reasonable to add higher LRs such as `2e-2` and `3e-2` next if training remains stable.

### Paper Draft Notes

Use these results carefully:

- The table supports an initial width-256 LR sensitivity comparison.
- The best observed LR in the current grid is the largest tested LR, so the optimum may lie beyond `1.5e-2`.
- The small loss gaps between SSO and MCSD suggest that more LRs or repeated runs may be needed before making a strong claim.
- If this result becomes a figure, plot validation loss versus LR with one curve per optimizer and clearly state `width=256`, `1B tokens`, `global batch=128`, and `eval_iters=5`.

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
```

Local copied logs:

```text
analysis_logs_372513/
```

Remote result root:

```text
~/projects/SSO_test/results/olmo_1b_width256_sso_mcsd_lr_sweep
```

## Reproduce Current Sweep

From the server:

```bash
ssh hpc2021
cd ~/projects/SSO_test
bash slurm/submit_width256_sso_mcsd_lr_sweep.sh
```

Monitor:

```bash
squeue -u u3013198
sacct -j <job_id> --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
tail -f logs/<job_name>_<job_id>.out
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
docs/width256_sso_mcsd_lr_sweep_1b.md
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
