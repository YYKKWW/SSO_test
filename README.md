# SSO Test Experiments

This repository manages one experiment track for a paper project on SSO-style optimizers in Megatron-LM. It keeps the runnable Slurm scripts, data-preparation utilities, experiment notes, and completed result summaries needed to reproduce and extend the current width-scaling study.

The active experiment track is a width-scaling learning-rate sweep on a weighted 1B-token OLMo mix sample:

```text
comparison: SSO vs plain SpEL vs SpEL-TP / MCSD-TP vs plain MCSD-PGD
widths:     256 and 512
LR grid:    5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2
cluster:    HKU HPC2021 H20 Slurm partition
status:     completed baseline, high-LR, projection, MuonBall, width-256 PGD sigma2, and block2-FP32 gap sweeps
```

## Current Testing Goal

The current paper-facing goal is to compare SSO-style spectral optimizers on a controlled 1B-token OLMo mix pretraining task at small widths. The immediate comparison set is:

- `width=256` and `width=512`
- LR grid `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2`
- SSO / `spectral_ball_dist`
- plain SpEL / `spel_dist`
- SpEL-TP / MCSD-TP / `spel_tp_dist`
- plain SpEL-PGD / `spel_pgd_dist` with post-msign TP disabled
- MCSD-TP-PGD historical rows only; future MCSD-PGD runs default to the plain variant
- MuonBall / `muon_ball_dist` as a width-256 seven-LR supplement

Current completed-result conclusion: at `LR=1.5e-2`, plain SpEL with top-k projection `k=4` is still the best completed row for both `width=256` and `width=512`. The new width-256 MuonBall sweep is competitive and beats SSO at its best LR, but remains slightly behind plain SpEL `topk k=4`. The width-256 plain SpEL-PGD sigma2 sweep is also competitive when `sigma2_power_iteration_steps=5`, but still trails plain SpEL `topk k=4`. The block2-FP32 gap-control sweep did not improve MCSD-PGD: tight gaps almost never trigger PGD, while larger gaps trigger too much PGD and hurt validation loss. Width-512 high-LR tests at `2e-2` and `3e-2` are worse, so the current width-512 minimum remains near `1.5e-2`.

Forward rule for PGD experiments: `MCSD-PGD` now means the plain `spel_pgd_dist` variant with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`. Do not submit new `MCSD-TP-PGD` jobs unless a later experiment explicitly reopens the TP ablation.

## Documentation

| Document | Purpose |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index, naming rules, and where to add new experiment records. |
| [docs/algorithms/mcsd_pgd_block2_fp32.md](docs/algorithms/mcsd_pgd_block2_fp32.md) | Algorithm note for optional MCSD-PGD FP32 gap estimators. |
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
- Training-log support for SpEL-PGD branch diagnostics. New `spel_pgd_dist` runs print per-step and cumulative `spel-pgd pgd branches: used/total (rate)` counts at each `LOG_INTERVAL` and write the same counters under `spel-pgd/*` TensorBoard/W&B keys.
- SpEL-PGD direction normalization supports `none`, `fro`, and `spectral`; `spectral` estimates the PGD direction's leading singular value with the same `power_iteration` helper used by SpEL/SSO and divides by that value.

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

Status as of 2026-07-09: the baseline `width=256` and `width=512` five-LR sweeps are complete on H20. The `width=512`, `SpEL-TP top-k k=4`, `LR=1.5e-2` supplement job `3743072` completed successfully. The width-512 high-LR sweep for `2e-2` and `3e-2` completed successfully as jobs `3743116`-`3743125`. The plain SpEL vs MCSD-TP-PGD projection supplement also completed successfully: width-256 jobs `3744519`-`3744524` and width-512 jobs `3744525`-`3744530`. The width-256 plain SpEL-PGD / SpEL-TP-PGD sigma2 supplement completed as jobs `3747964`-`3747969`. The width-256 MuonBall seven-LR supplement completed as jobs `3747994`-`3748000`. The width-1024 two-iteration memory smoke completed as jobs `3748023`-`3748025`. The width-256 block2-FP32 MCSD-PGD gap-control sweep completed as jobs `3750634`-`3750641`. `Elapsed` is Slurm wall-clock time from `sacct` on the H20 partition.

The 250M-token MCSD-PGD gap-threshold tuning on 2026-07-09 completed successfully. It used the plain variant only, `width=256`, `LR=1.5e-2`, `shared_topk k=8`, and `sigma2_power_iteration_steps=5`. Best row: spectral direction normalization with `gap_threshold_rel` in `1e-4` to `2e-3`, final val loss `3.990190`, PPL `54.06516`, and cumulative PGD branch rate about `0.5%`.

| Direction normalization | Best gap | Best val loss | PPL | Cumulative PGD rate | Jobs |
|---|---:|---:|---:|---:|---|
| `spectral` | `1e-4` to `2e-3` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) | `3749613`-`3749616` |
| `fro` | `5e-3` | `3.991130` | `54.11601` | `432/118440` (`0.004`) | `3749574` |
| `none` | `0` | `3.991379` | `54.12947` | `0/118440` (`0.000`) | `3749547` |

Interpretation: spectral normalization helps slightly over both unnormalized PGD and Frobenius normalization. Larger `gap_threshold_rel=1e-2` is worse across all normalizations, so the PGD fallback should remain rare.

The phase-B sigma2/gap follow-up also completed on 2026-07-09. It kept spectral normalization and tested `sigma2_power_iteration_steps=3,8,10` against conservative gap thresholds. `sigma2=3` is close but almost never triggers PGD; `sigma2=8` and `10` trigger PGD much more often and are worse.

| sigma2 steps | Best val loss | PPL | Cumulative PGD rate | Jobs |
|---:|---:|---:|---:|---|
| `3` | `3.990414` | `54.07725` | `7/118440` (`0.000`) | `3750042`-`3750046` |
| `5` | **`3.990190`** | `54.06516` | `555/118440` (`0.005`) | `3749613`-`3749616` |
| `8` | `4.003870` | `54.80988` | `10790/118440` (`0.091`) | `3750047`-`3750050` |
| `10` | `4.095759` | `60.08492` | `43638/118440` (`0.368`) | `3750056` |

The block2-FP32 gap-control sweep also completed on 2026-07-09. It fixed `sigma2_power_iteration_steps=10`, `gap_estimator_mode=block2_fp32`, spectral PGD direction normalization, `pgd_lr_scale=0.5`, `shared_topk k=8`, `width=256`, `LR=1.5e-2`, and `250M` train tokens. Runtime was about `01:30` to `01:31` per job, versus about `01:26` for the comparable deflated-estimator sigma2 sweeps, so the observed overhead is roughly 5%. The result is not preferred: the best final loss is only the nearly no-PGD row, and larger gaps become worse as PGD rate rises. The `gap=0` row is also worse than the earlier deflated-estimator MCSD-PGD baseline because this coupled `block2_fp32` mode replaces the SpEL branch's top singular vectors with block-2 Ritz vectors even when PGD is never selected.

| block2-FP32 gap | Val loss | PPL | Cumulative PGD rate | Elapsed | Job |
|---:|---:|---:|---:|---:|---:|
| `0` | `4.030535` | `56.29100` | `0/118440` (`0.000`) | `01:31:26` | `3750634` |
| `1e-6` | `4.030535` | `56.29100` | `0/118440` (`0.000`) | `01:31:18` | `3750635` |
| `3e-6` | `4.030535` | `56.29100` | `0/118440` (`0.000`) | `01:30:56` | `3750636` |
| `1e-5` | `4.030484` | `56.28816` | `3/118440` (`0.000`) | `01:31:27` | `3750637` |
| `3e-5` | **`4.030419`** | `56.28451` | `51/118440` (`0.000`) | `01:30:55` | `3750638` |
| `1e-4` | `4.031507` | `56.34573` | `665/118440` (`0.006`) | `01:31:05` | `3750639` |
| `3e-4` | `4.046161` | `57.17754` | `7081/118440` (`0.060`) | `01:31:10` | `3750640` |
| `1e-3` | `4.141737` | `62.91203` | `37562/118440` (`0.317`) | `01:30:14` | `3750641` |

Current preferred MCSD-PGD setting remains the earlier deflated-estimator setting: `sigma2_power_iteration_steps=5`, `gap_threshold_rel=1e-3`, `pgd_direction_normalization=spectral`, and `pgd_lr_scale=0.5` for future runs. The smaller PGD-specific step is the current mitigation for repeated PGD triggers caused by near-degenerate spectra; if future logs still show harmful consecutive PGD bursts, add an explicit cooldown or hysteresis rule. The earlier sigma2=10 deflated-estimator follow-ups `3750496`-`3750505` and `3750507`-`3750512` were cancelled.

Active follow-up submitted on 2026-07-10: test whether warm-started SpEL `u/v`
fixes the block2 gauge issue while comparing gap-only high-precision branch
rules. It uses `width=256`, `LR=1.5e-2`, `250M` train tokens, `shared_topk k=8`,
`sigma2_power_iteration_steps=10`, `pgd_lr_scale=0.5`, and gaps `0`, `3e-5`,
`1e-4`.

| Gap estimator | Warm start `u/v` | Gap jobs |
|---|---:|---|
| `deflated_fp32_gap_only` | `0` | `3751735` (`0`), `3751736` (`3e-5`), `3751737` (`1e-4`) |
| `deflated_fp32_gap_only` | `1` | `3751738` (`0`), `3751739` (`3e-5`), `3751740` (`1e-4`) |
| `block2_fp32_gap_only` | `0` | `3751741` (`0`), `3751742` (`3e-5`), `3751743` (`1e-4`) |
| `block2_fp32_gap_only` | `1` | `3751744` (`0`), `3751745` (`3e-5`), `3751746` (`1e-4`) |
| `block2_fp32` | `0` | `3751747` (`0`), `3751748` (`3e-5`), `3751749` (`1e-4`) |
| `block2_fp32` | `1` | `3751750` (`0`), `3751751` (`3e-5`), `3751752` (`1e-4`) |

Additional warm-start PGD step-size follow-up submitted on 2026-07-10 before
the Slurm QOS submit limit was reached. These use `warm_start_uv=1`,
`shared_topk k=8`, `sigma2_power_iteration_steps=10`, and add `gap=3e-4` to
test whether a smaller PGD step can tolerate more frequent PGD triggers.

| Gap estimator | PGD lr scale | Gap jobs |
|---|---:|---|
| `deflated_fp32_gap_only` | `0.05` | `3751767` (`3e-5`), `3751768` (`1e-4`), `3751769` (`3e-4`) |
| `deflated_fp32_gap_only` | `0.1` | `3751770` (`3e-5`), `3751771` (`1e-4`), `3751772` (`3e-4`) |
| `deflated_fp32_gap_only` | `0.2` | `3751773` (`3e-5`), `3751774` (`1e-4`), `3751775` (`3e-4`) |
| `block2_fp32_gap_only` | `0.05` | `3751776` (`3e-5`) |

Because the user submit limit was reached, a lightweight deferred submitter was
started on the H20 login node. It waits until the active Slurm job count drops
below `26`, then submits the remaining finite task list and exits after either
completion or an eight-hour deadline. Log:
`logs/deferred_submit_width256_pgd_extra_20260710.log`.

Deferred task list:

| Group | Pending combinations |
|---|---|
| `block2_fp32_gap_only` lr-scale remainder | `pgd_lr_scale=0.05` with gaps `1e-4`, `3e-4`; `pgd_lr_scale=0.1,0.2` with gaps `3e-5`, `1e-4`, `3e-4` |
| `deflated_power` warm/cold baseline | `warm_start_uv=0,1`, `pgd_lr_scale=0.5`, gaps `0`, `3e-5`, `1e-4`, `3e-4` |

Latest result update, 2026-07-10 09:25 HKT: the deferred submitter completed
its finite submission list. Most jobs completed successfully; only the
`deflated_power warm_start_uv=1` baseline jobs `3752230`, `3752390`-`3752392`
were still running at this checkpoint. Main conclusion: the old coupled
`block2_fp32` failure was indeed mostly a top-vector/gauge issue, but warm
starting the SpEL `u/v` path is not preferred. The cleanest result is
`block2_fp32_gap_only` with `warm_start_uv=0`, which preserves the original SpEL
path and recovers the no-PGD baseline.

| Gap estimator | Warm `u/v` | PGD lr | Gap | Val loss | PPL | PGD rate | Job |
|---|---:|---:|---:|---:|---:|---:|---:|
| `block2_fp32_gap_only` | `0` | `0.5` | `0` | **`3.991379`** | `54.12947` | `0/118440` (`0.000`) | `3751741` |
| `block2_fp32_gap_only` | `0` | `0.5` | `3e-5` | **`3.991379`** | `54.12947` | `0/118440` (`0.000`) | `3751742` |
| `block2_fp32_gap_only` | `0` | `0.5` | `1e-4` | `3.991830` | `54.15391` | `4/118440` (`0.000`) | `3751743` |
| `block2_fp32_gap_only` | `1` | `0.5` | `0` | `4.004952` | `54.86921` | `0/118440` (`0.000`) | `3751744` |
| `block2_fp32` | `0` | `0.5` | `0` | `4.030535` | `56.29100` | `0/118440` (`0.000`) | `3751747` |
| `block2_fp32` | `1` | `0.5` | `0` | `4.001238` | `54.66580` | `0/118440` (`0.000`) | `3751750` |
| `block2_fp32` | `1` | `0.5` | `1e-4` | `3.999915` | `54.59349` | `59/118440` (`0.000`) | `3751752` |
| `deflated_fp32_gap_only` | `0` | `0.5` | `0` | `3.991379` | `54.12947` | `0/118440` (`0.000`) | `3751735` |
| `deflated_fp32_gap_only` | `0` | `0.5` | `3e-5` | `4.089620` | `59.71717` | `41034/118440` (`0.346`) | `3751736` |
| `deflated_fp32_gap_only` | `1` | `0.05` | `1e-4` | `4.076410` | `58.93351` | `48823/118440` (`0.412`) | `3751768` |
| `deflated_power` | `0` | `0.5` | `0` | `3.991379` | `54.12947` | `0/118440` (`0.000`) | `3752226` |
| `deflated_power` | `0` | `0.5` | `3e-5` | `4.109033` | `60.88779` | `47267/118440` (`0.399`) | `3752227` |

Interpretation:

- `block2_fp32_gap_only + cold` is the cleanest high-precision region-test
  implementation so far. It keeps the SpEL `u/v` path intact and recovers the
  no-PGD baseline. At gaps up to `1e-4`, however, it almost never triggers PGD
  and therefore does not improve over the baseline.
- Coupled `block2_fp32` improves dramatically with warm start (`4.030535` to
  about `4.00` at `gap=0`), confirming the previous failure was largely a
  top-vector gauge issue. It still does not beat `block2_fp32_gap_only + cold`.
- `deflated_fp32_gap_only` and `deflated_power` with `sigma2=10` are too
  aggressive: even tiny nonzero gaps trigger PGD about 35-43% of matrix updates
  and validation loss degrades. Reducing `pgd_lr_scale` to `0.05` does not fix
  this because PGD has already become too frequent.
- Warm-starting the original SpEL `u/v` path is not currently helpful; it raises
  the `gap=0` baseline from `3.991379` to roughly `4.00`.

Latest result update, 2026-07-10 12:45 HKT: the FP32 main-power follow-up
completed as jobs `3752962`-`3752973`. These jobs set
`spel_pgd_main_power_dtype=fp32`, `width=256`, `LR=1.5e-2`, `250M` train tokens,
`shared_topk k=8`, `sigma2_power_iteration_steps=10`, spectral PGD direction
normalization, and gaps `0` / `1e-4`. The best result is
`block2_fp32_gap_only + cold + gap=0`, final val loss `3.987389`, PPL
`53.91395`. This is better than the matching BF16 main-path no-PGD baseline
`3.991379`, suggesting that FP32 ordinary SpEL power iteration is a useful
main-path ablation. However, `gap=1e-4` still triggers PGD only `2`-`5` times
out of `118440` matrix updates in the gap-only runs, so the PGD fallback itself
has not shown a benefit in this setting.

Precision comparison note: these are not true FP16 training runs. The model
uses `--bf16`; the comparison below is between the original BF16/default SpEL
power-iteration path and the new FP32 SpEL main-power path inside
`spel_pgd_dist`. No completed plain `spel_dist` FP32-vs-FP16 pair exists yet.

| Setting | Gap estimator | Warm `u/v` | Gap | PGD lr | Val loss | PPL | PGD rate | Job |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16/default SpEL main path | `block2_fp32_gap_only` | `0` | `0` | `0.5` | `3.991379` | `54.12947` | `0/118440` (`0.000`) | `3751741` |
| FP32 SpEL main path | `block2_fp32_gap_only` | `0` | `0` | `0.5` | **`3.987389`** | **`53.91395`** | `0/118440` (`0.000`) | `3752962` |
| BF16/default SpEL main path | `block2_fp32_gap_only` | `0` | `1e-4` | `0.5` | `3.991830` | `54.15391` | `4/118440` (`0.000`) | `3751743` |
| FP32 SpEL main path | `block2_fp32_gap_only` | `0` | `1e-4` | `0.5` | **`3.990730`** | **`54.09435`** | `3/118440` (`0.000`) | `3752964` |
| BF16/default SpEL main path | `block2_fp32_gap_only` | `1` | `0` | `0.5` | `4.004952` | `54.86921` | `0/118440` (`0.000`) | `3751744` |
| FP32 SpEL main path | `block2_fp32_gap_only` | `1` | `0` | `0.5` | **`4.002185`** | **`54.71757`** | `0/118440` (`0.000`) | `3752966` |

Interpretation: FP32 improves the ordinary SpEL top-vector path by about
`0.001`-`0.004` validation loss in the matched rows above, but the absolute
effect is modest. This supports using FP32 for the SpEL main power iteration
when testing MCSD-PGD, while keeping the PGD region estimator separated from
the main `u/v` path.

| Gap estimator | Main power dtype | Warm `u/v` | Gap | PGD lr | Val loss | PPL | PGD rate | Job |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `block2_fp32_gap_only` | `fp32` | `0` | `0` | `0.5` | **`3.987389`** | `53.91395` | `0/118440` (`0.000`) | `3752962` |
| `block2_fp32_gap_only` | `fp32` | `0` | `1e-4` | `0.2` | `3.992790` | `54.20593` | `5/118440` (`0.000`) | `3752963` |
| `block2_fp32_gap_only` | `fp32` | `0` | `1e-4` | `0.5` | `3.990730` | `54.09435` | `3/118440` (`0.000`) | `3752964` |
| `block2_fp32_gap_only` | `fp32` | `0` | `1e-4` | `1.0` | `3.990568` | `54.08558` | `2/118440` (`0.000`) | `3752965` |
| `block2_fp32_gap_only` | `fp32` | `1` | `0` | `0.5` | `4.002185` | `54.71757` | `0/118440` (`0.000`) | `3752966` |
| `block2_fp32_gap_only` | `fp32` | `1` | `1e-4` | `0.2` | `3.999950` | `54.59540` | `2/118440` (`0.000`) | `3752967` |
| `block2_fp32_gap_only` | `fp32` | `1` | `1e-4` | `0.5` | `4.002501` | `54.73490` | `2/118440` (`0.000`) | `3752968` |
| `block2_fp32_gap_only` | `fp32` | `1` | `1e-4` | `1.0` | `3.999609` | `54.57683` | `2/118440` (`0.000`) | `3752969` |
| `block2_fp32` | `fp32` | `1` | `0` | `0.5` | `3.999673` | `54.58032` | `0/118440` (`0.000`) | `3752970` |
| `block2_fp32` | `fp32` | `1` | `1e-4` | `0.2` | `3.998521` | `54.51746` | `64/118440` (`0.001`) | `3752971` |
| `block2_fp32` | `fp32` | `1` | `1e-4` | `0.5` | `4.001473` | `54.67866` | `104/118440` (`0.001`) | `3752972` |
| `block2_fp32` | `fp32` | `1` | `1e-4` | `1.0` | `4.002787` | `54.75053` | `97/118440` (`0.001`) | `3752973` |

Interpretation:

- `main_power_dtype=fp32` improves the cold no-PGD baseline relative to the
  BF16/default main path (`3.987389` vs `3.991379`).
- Warm-starting the ordinary SpEL `u/v` path remains harmful: the gap-only
  `gap=0` baseline worsens from `3.987389` to `4.002185`.
- Coupled `block2_fp32 + warm` is now reasonable but still worse than
  gap-only cold. It changes the SpEL top-vector path, so it is not the preferred
  theory-facing implementation.
- `gap=1e-4` is too conservative under block2-FP32 gap estimation. The next
  useful experiment is `block2_fp32_gap_only + cold + main_power_dtype=fp32`
  with gaps around `1e-4` to `3e-4`, sweeping `pgd_lr_scale=0.2/0.5/1`.

1B focused follow-up completed on 2026-07-11. The broader 1B sanity sweep jobs
`3754433`-`3754444` were cancelled before completion. The main line is
`block2_fp32_gap_only + main_power_dtype=fp32 + warm_start_uv=0`; the
lower-priority contrast is coupled `block2_fp32 + warm_start_uv=1`.

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

Interpretation: `gap_only + cold` is still the cleaner implementation. Its best
row, `gap=3e-4, pgd_lr_scale=0.5`, slightly beats the SSO width-256 row in this
README (`3.569919` vs `3.570953`) but remains worse than the older plain SpEL
and sigma2=5 SpEL-PGD rows. PGD usage is still very small under gap-only
estimation. Coupled `block2_fp32 + warm` triggers more PGD but gives worse loss,
so replacing the SpEL top-vector path with block2 Ritz vectors remains a bad
tradeoff. Mis-submitted `k=4/16` jobs `3754547`, `3754548`, `3754551`, and
`3754552` were cancelled after about 2 minutes and should not be used.

Adaptive gap-probe timing follow-up submitted on 2026-07-11. The optimizer now
tracks the last measured relative gap separately for every matrix/component.
When `rel_gap > gap_probe_safe_multiplier * gap_threshold_rel`, it runs the
FP32 gap estimator every `gap_probe_interval` optimizer steps; otherwise it
returns to probing every step. The first step is always probed, and skipped
steps use the ordinary SpEL branch without reusing a previous PGD decision.
The default interval is `1`, which preserves the previous behavior.

The first 1B comparison fixes the best previous configuration:
`block2_fp32_gap_only`, `main_power_dtype=fp32`, `warm_start_uv=0`,
`gap=3e-4`, `pgd_lr_scale=0.5`, `shared_topk k=8`, sigma2 steps `10`, and
seed `1234`. Job `3756922` uses interval `5`; job `3756923` uses interval
`10`. Both use safe multiplier `10`. Initial jobs `3756915` and `3756916`
were cancelled after 1 minute 43 seconds because their anonymous split-component
cache keys were not parameter-specific; no result from those jobs should be
used. Compare the replacement jobs' elapsed time, validation
loss, PGD count, and logged cumulative gap-probe count against the every-step
baseline job `3754481` (`05:52:10`, val loss `3.569919`).

Naming audit, 2026-07-08: all historical `spel_dist` rows in this repository were run while the code always executed the post-msign tangent re-projection line `Phi = project_to_tangent_plane(Phi, u, v)`. These rows are therefore labeled `SpEL-TP` or `MCSD-TP`. The current launcher now exposes that behavior explicitly as `spel_tp_dist`; new plain `spel_dist` rows mean the post-msign TP step is disabled. Historical `spel_pgd_dist` rows may be labeled `MCSD-TP-PGD` when they used the TP branch. From 2026-07-09 onward, unqualified `MCSD-PGD` means plain `spel_pgd_dist` with post-msign TP disabled.

Best completed results:

| Width | Optimizer | LR | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|---:|
| `256` | plain SpEL `spel_dist`, `topk k=4` | `1.5e-2` | **`3.562941`** | `35.26677` | `05:32:06` | `3744520` |
| `256` | MuonBall `muon_ball_dist` | `1.5e-2` | `3.564250` | `35.31298` | `05:24:57` | `3747998` |
| `256` | plain SpEL-PGD `spel_pgd_dist`, `shared_topk k=8`, `sigma2=5` | `1.5e-2` | `3.566324` | `35.38627` | `05:37:26` | `3747964` |
| `256` | plain SpEL `spel_dist`, `topk k=8` | `1.5e-2` | `3.566394` | `35.38876` | `05:32:28` | `3744521` |
| `256` | SpEL-TP / MCSD-TP `spel_tp_dist`, `topk k=8` | `1.5e-2` | `3.566694` | `35.39936` | `05:34:38` | `3740137` |
| `256` | MCSD-TP-PGD `spel_pgd_dist`, `shared_topk k=8` | `1.5e-2` | `3.566973` | `35.40925` | `05:37:55` | `3744524` |
| `256` | MCSD-PGD `block2_fp32_gap_only`, FP32 main, `gap=3e-4`, `pgd_lr=0.5` | `1.5e-2` | `3.569919` | `35.51371` | `05:52:10` | `3754481` |
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
| `256` | MuonBall | `3.564250` | `35.31298` | `05:24:57` | `3747998` |
| `256` | plain SpEL-PGD shared top-k k=8, sigma2=5 | `3.566324` | `35.38627` | `05:37:26` | `3747964` |
| `256` | plain SpEL top-k k=8 | `3.566394` | `35.38876` | `05:32:28` | `3744521` |
| `256` | SpEL-TP original retraction | `3.567708` | `35.43530` | `05:26:59` | `3725139` |
| `256` | SpEL-TP top-k k=4 | `3.567563` | `35.43013` | `05:33:45` | `3743071` |
| `256` | SpEL-TP top-k k=8 | `3.566694` | `35.39936` | `05:34:38` | `3740137` |
| `256` | SpEL-TP-PGD shared top-k k=8, sigma2=5 | `3.568810` | `35.47436` | `05:38:32` | `3747965` |
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

### Plain SpEL-PGD Sigma2 Supplement

Submitted on 2026-07-09 with `width=256`, `LR=1.5e-2`, `TRAIN_TOKENS=1B`, `GLOBAL_BATCH=128`, `MICRO_BATCH=4`, `SPEL_PGD_PROJECTION_MODE=shared_topk`, and `SPEL_PGD_RANKS=8`. Plain SpEL-PGD disables the post-msign tangent projection with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`; SpEL-TP-PGD enables it with `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1`. This is now treated as a historical TP ablation; future MCSD-PGD experiments default to the plain variant.

| Width | Variant | sigma2 steps | Val loss | PPL | Elapsed | Job |
|---:|---|---:|---:|---:|---:|---:|
| `256` | plain SpEL-PGD | `5` | **`3.566324`** | `35.38627` | `05:37:26` | `3747964` |
| `256` | SpEL-TP-PGD | `5` | `3.568810` | `35.47436` | `05:38:32` | `3747965` |
| `256` | SpEL-TP-PGD | `8` | `3.569142` | `35.48614` | `05:37:30` | `3747967` |
| `256` | plain SpEL-PGD | `8` | `3.569803` | `35.50958` | `05:35:26` | `3747966` |
| `256` | SpEL-TP-PGD | `10` | `3.610268` | `36.97597` | `05:35:16` | `3747969` |
| `256` | plain SpEL-PGD | `10` | `3.627396` | `37.61473` | `05:35:03` | `3747968` |

Current interpretation: `sigma2_power_iteration_steps=5` is best in this sweep. Increasing sigma2 steps to `8` or `10` does not improve validation loss and is worse at `10`. The best plain SpEL-PGD row (`3.566324`) is competitive with SpEL-TP and MCSD-TP-PGD, but it does not beat plain SpEL `topk k=4` (`3.562941`) or MuonBall (`3.564250`) at width 256.

### MuonBall Width-256 Seven-LR Supplement

Submitted on 2026-07-09 with the same width-256 1B-token setup and the MuonBall constants from `Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/muonball/muonball.sh`: `momentum=0.9`, Nesterov enabled, `msign_steps=8`, `radius_mode=spectral_mup`, `scale_mode=spectral_mup`, `power_iteration_steps=10`, `retract_mode=hard`, and `qkv_split_mode=head`.

| LR | MuonBall val loss | PPL | Elapsed | Job |
|---:|---:|---:|---:|---:|
| `5e-3` | `3.639009` | `38.05410` | `05:24:57` | `3747994` |
| `7e-3` | `3.600150` | `36.60372` | `05:24:05` | `3747995` |
| `9e-3` | `3.581582` | `35.93032` | `05:26:44` | `3747996` |
| `1e-2` | `3.575525` | `35.71338` | `05:24:58` | `3747997` |
| `1.5e-2` | **`3.564250`** | `35.31298` | `05:24:57` | `3747998` |
| `2e-2` | `3.571979` | `35.58694` | `05:24:49` | `3747999` |
| `3e-2` | `3.611113` | `37.00722` | `05:25:05` | `3748000` |

Current interpretation: MuonBall's best width-256 row is `LR=1.5e-2`, with val loss `3.564250`. It beats width-256 SSO at the same LR (`3.570953`) and is close to plain SpEL, but it is still slightly worse than plain SpEL `topk k=4` (`3.562941`).

### Width-1024 Memory Smoke

This two-iteration smoke test checks whether `width=1024`, `num_layers=28`, `seq_length=4096`, `global_batch=128`, and `micro_batch=4` fit on one H20. These are not training-quality results.

| Optimizer | State | Max allocated MB | Max reserved MB | Elapsed | Job |
|---|---|---:|---:|---:|---:|
| SSO | `COMPLETED` | `75555.72` | `80610.00` | `00:03:07` | `3748023` |
| SpEL-TP | `COMPLETED` | `74531.72` | `78978.00` | `00:02:54` | `3748024` |
| MuonBall | `COMPLETED` | `74531.72` | `78978.00` | `00:02:54` | `3748025` |

Current interpretation: width 1024 does not OOM at `micro_batch=4`, `seq_length=4096` on H20 for these three optimizers. SSO has the highest observed allocation and is closest to the limit.

### Width-1024 SSO/MuonBall LR Sweep

Submitted on 2026-07-11 after the width-1024 memory smoke. This run uses the
same 1B-token setup, `global_batch=128`, `micro_batch=4`, and the width-256
MuonBall seven-LR grid. Slurm wall time is explicitly set to `2-00:00:00`
because the smoke test estimates SSO at more than 24 hours for 1B tokens.

| Optimizer | LR | Job | State at submit |
|---|---:|---:|---|
| SSO | `5e-3` | `3756214` | `RUNNING` |
| SSO | `7e-3` | `3756215` | `RUNNING` |
| SSO | `9e-3` | `3756216` | `RUNNING` |
| SSO | `1e-2` | `3756217` | `RUNNING` |
| SSO | `1.5e-2` | `3756218` | `RUNNING` |
| SSO | `2e-2` | `3756219` | `RUNNING` |
| SSO | `3e-2` | `3756220` | `RUNNING` |
| MuonBall | `5e-3` | `3756221` | `RUNNING` |
| MuonBall | `7e-3` | `3756222` | `RUNNING` |
| MuonBall | `9e-3` | `3756223` | `RUNNING` |
| MuonBall | `1e-2` | `3756224` | `RUNNING` |
| MuonBall | `1.5e-2` | `3756225` | `RUNNING` |
| MuonBall | `2e-2` | `3756226` | `RUNNING` |
| MuonBall | `3e-2` | `3756227` | `RUNNING` |

### Remaining Plain SpEL-PGD Coverage

The completed plain SpEL-PGD sigma2 supplement covers only `width=256`, `shared_topk k=8`, and `LR=1.5e-2`. The remaining minimum coverage, if this optimizer stays in the paper comparison, is:

| Width | LR | Projection modes | Jobs needed |
|---:|---:|---|---:|
| `512` | `1.5e-2` | `shared_topk k=8`, `sigma2=5` | `1` |
| `256`, `512` | `5e-3`, `7e-3`, `9e-3`, `1e-2`, `1.5e-2` | best plain SpEL-PGD setting if selected | optional grid |

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
