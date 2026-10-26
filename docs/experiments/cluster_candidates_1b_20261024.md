# Width-256 normalized-momentum candidates: 1B comparison

Array **4319977** was submitted on 2026-10-24 at 01:11 HKT. It compares five
mixed MCSD-P2GD-type candidates, three matched MCSD-TP controls, and four
MuonSphere/SSO baselines. Each task uses one H20; at most three tasks run
concurrently. This is supplementary research, not an update to the frozen
paper experiment or the independent reproduction repository.

## Completed results on October 25

All 12 tasks completed with exit code 0; the last finished at 01:40:43 HKT
on 2026-10-25. Checked again at 10:00 HKT: this array occupies no GPUs.
Every training log reports zero skipped and NaN iterations. Other unrelated
jobs remain in the account; none was changed during this result check.

Losses below are FINAL validation at iteration 1908, not the minimum over
evaluation times. Time is training-launch wall time, including validation
but excluding preflight tests and post-training numerical audits.

| Suffix | Peak LR | Method | K | c | Final val | Training time |
|---|---:|---|---:|---:|---:|---|
| _0 | 2e-2 | MCSD-TP | 20 | - | 3.589549 | 6:00:02 |
| _1 | 2e-2 | Mixed | 20 | 8 | **3.587868** | 5:59:09 |
| _2 | 2e-2 | MuonSphere | - | - | 3.593271 | 5:25:51 |
| _3 | 2e-2 | SSO | - | - | 3.591197 | 6:04:47 |
| _4 | 2e-2 | Mixed | 20 | 4 | 3.588366 | 6:00:03 |
| _5 | 2e-2 | Mixed | 20 | 12 | 3.590739 | 6:00:25 |
| _6 | 2e-2 | MCSD-TP | 40 | - | 3.591258 | 6:25:55 |
| _7 | 2e-2 | Mixed | 40 | 8 | 3.590239 | 6:23:41 |
| _8 | 1.5e-2 | MuonSphere | - | - | **3.582948** | 5:24:52 |
| _9 | 1.5e-2 | SSO | - | - | 3.591530 | 6:03:41 |
| _10 | 1.5e-2 | MCSD-TP | 20 | - | 3.588559 | 5:59:40 |
| _11 | 1.5e-2 | Mixed | 20 | 8 | **3.587659** | 6:02:46 |

### What improved and what did not

At LR 2e-2, K20/c=8 improves final loss by 0.001681 over its matched
MCSD-TP control, 0.005403 over MuonSphere and 0.003329 over SSO. It beats
its control at seven of the eight evaluation times. At LR 1.5e-2, c=8
improves its matched control by 0.000900 and SSO by 0.003871, but remains
0.004711 worse than MuonSphere. It improves the control at all six
evaluations from step 750 onward, after losing at 250/500.

Thus the mixed branch now has a small observed 1B benefit over MCSD-TP,
unlike the 100M pilots, but **does not beat the best tested MuonSphere**.
The best hybrid across the two LRs is 3.587659, versus 3.582948 for
MuonSphere. Its difference from the LR 2e-2 hybrid is only 0.000209.
All these are single-seed comparisons, not statistically established gains.

The 100M preference for LR 2e-2 did not establish the best LR for 1B:
MuonSphere in particular improves by 0.010323 when using 1.5e-2. This is
why both matched LRs must remain in the comparison.

Increasing c to 12 reduces near-region persistence further, but makes loss
0.002871 worse than c=8 and 0.001190 worse than the K20 control. A lower
branch rate is not an optimization objective. K40/c=8 remains 0.001019
better than its K40 control, but is 0.002371 worse than K20/c=8. K40 also
makes the control worse in this seed, so increased spectral accuracy alone
does not explain or solve the optimization behavior.

K20/c=8 costs 10.22% more training wall time than same-LR MuonSphere, is
within 0.25% of its MCSD-TP control, and is 1.54% faster than SSO by the
observed wall clock. K40 adds 6.83% over K20/c=8. Small timing differences
are not precise kernel-speed comparisons on a shared cluster.

### Branch persistence

Statistics are through the last training log at step 1900, not the full
1908 steps: 478800 component decisions, including 252 zero-LR startup
calls. The full completed run has 480816 component calls. Repeat percentage
is reconstructed from integer call and burst-start counts; calls per burst
also include episodes still active at the logging boundary.

| LR | K | c | Cluster calls | Cumulative % | Repeat % | Calls/burst | Step-1900 % | Capacity overflows |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2e-2 | 20 | 4 | 7598 | 1.587 | 75.88 | 4.15 | 11.11 | 137 |
| 2e-2 | 20 | 8 | 4569 | 0.954 | 59.82 | 2.49 | 9.13 | 145 |
| 2e-2 | 20 | 12 | 3766 | 0.787 | 46.63 | 1.87 | 3.97 | 74 |
| 2e-2 | 40 | 8 | 4998 | 1.044 | 61.76 | 2.62 | 7.54 | 155 |
| 1.5e-2 | 20 | 8 | 9966 | 2.081 | 69.13 | 3.24 | 11.90 | 1015 |

More than 97.9% of cumulative component decisions remain outside the
cluster branch in each mixed run. Late-step rates are higher, and repeated
near-region visits are reduced, not eliminated. These are P2GD-type cluster
calls, despite the legacy log label `pgd branches`; they are not full-SVD
PGD projections. The overflow counts also preclude claiming complete
top-eigenspace/tangent-cone treatment.

### Numerical and call-count audit

All eight experimental runs completed with exactly 480816 small Ritz SVDs
for 480816 component calls and zero full SVD/svdvals calls inside optimizer
updates. Original-matrix decompositions occurred only in tests/offline
audits. The four baselines were not instrumented with this call guard.

| Suffix | Samples | False safe | False cluster | Mean gap absolute error | Max returned radius-relative error |
|---|---:|---:|---:|---:|---:|
| _0 control K20 | 28 | 0 | 0 | 1.17e-5 | 1.37e-4 |
| _1 c8 K20 | 44 | 0 | 1 | 5.88e-6 | 3.52e-5 |
| _4 c4 K20 | 44 | 0 | 0 | 4.00e-6 | 1.15e-4 |
| _5 c12 K20 | 44 | 0 | 0 | 4.97e-6 | 6.90e-5 |
| _6 control K40 | 28 | 0 | 0 | 1.38e-6 | 3.93e-6 |
| _7 c8 K40 | 44 | 0 | 0 | 8.88e-7 | 5.13e-6 |
| _10 control LR1.5e-2 | 28 | 0 | 0 | 5.64e-6 | 7.39e-6 |
| _11 c8 LR1.5e-2 | 44 | 0 | 0 | 3.91e-6 | 4.89e-6 |

K40 has smaller observed errors, but these audits sample different training
trajectories, not a fixed-matrix replay. Samples cover scheduled shapes and
early/late cluster calls; they are NOT population false-routing rates or
uniform feasibility certificates. Zero observed false-safe calls does not
exclude missed branches elsewhere. The auditor's stored note mentions
early selection; the 1B observer also includes late samples as configured.

### Next priority

Retain K20/c=8 as the principal mixed candidate and c=4 as a nearby step
ablation. Keep both LRs visible, with emphasis on verifying LR1.5e-2 against
its stronger MuonSphere baseline. Prioritize a second-seed matched comparison
before broadening width or token budget; the observed control gains are
small. A c=4 (or c=6) test at LR1.5e-2 would address the remaining step-size
gap in this grid. Do not promote c=12 or K40 to the main configuration based
on lower branch rate or higher numerical accuracy alone.

All methods have a higher final loss than their step-1750 loss. The evaluator
advances its validation iterator between evaluations and uses only five
batches each time; this shared increase alone does not demonstrate
overfitting. Prefer same-step comparisons, and use a fixed, larger validation
sample for confirmation. No checkpoint was retained for retrospective
reevaluation in this grid.

No new jobs were submitted or canceled during this result check. Raw server
extraction and parsed tables are saved in
`output/cluster_candidates_1b_20261024/final_raw_20261025.json` and
`output/cluster_candidates_1b_20261024/final_summary_20261025.json`.

## Why these candidates

The [completed 100M tuning](cluster_momentum_lr_tune_100m_20261021.md)
supports increasing the normalized-momentum branch step, but has NOT
established an advantage over matched MCSD-TP or MuonSphere:

| Pilot at width 256 | Final validation loss | Interpretation |
|---|---:|---|
| LR 1.5e-2, K20, c=1 | 4.969451 | Original normalized-momentum reference |
| LR 1.5e-2, K20, c=4 | 4.957274 | Larger branch step improves the pilot |
| LR 1.5e-2, K20, c=8 | 4.956690 | Best tested coefficient at this LR |
| LR 1.5e-2, K20, MCSD-TP | 4.947171 | Still better than the hybrid |
| LR 1.5e-2, MuonSphere | 4.947061 | Still better than the hybrid |
| LR 1.5e-2, SSO | 4.956870 | Difference from c=8 is inconclusive |
| LR 2e-2, K20, c=4 | 4.834641 | Most promising tested LR, not proof of transfer |
| LR 2e-2, K20, MCSD-TP | 4.828390 | Higher LR also benefits the control |

At LR 1.5e-2, increasing c from 1 to 8 reduced repeated cluster calls from
90.81% to 76.27%. Repetition remains substantial. Larger-step audits also
found occasional routing errors: LR 2e-2/c=4 had three false-cluster calls
among 24 selected snapshots and maximum returned radius-relative error
3.52e-4. These samples are not population error rates.

The new grid therefore tests c=4/8/12, retains LR 1.5e-2 as a long-training
anchor, adds baselines at BOTH LRs, and pairs a K40 hybrid with its own K40
control. K40 is a precision candidate, not an already demonstrated remedy.
No new K20/K40 numerical replay result is claimed here. Weak c<=1 and
LR 1e-2 configurations are not promoted to 1B.

## Submitted grid

K is the number of fixed bilateral subspace iterations, not the retained
rank. All experimental arms retain a warm block of eight vectors. A dash
means that the cluster settings do not apply to that baseline.

| Job | Method | Peak LR | K | Branch coefficient c |
|---|---|---:|---:|---:|
| 4319977_0 | MCSD-TP control | 2e-2 | 20 | N/A |
| 4319977_1 | Mixed MCSD-P2GD type | 2e-2 | 20 | 8 |
| 4319977_2 | MuonSphere | 2e-2 | - | - |
| 4319977_3 | SSO | 2e-2 | - | - |
| 4319977_4 | Mixed MCSD-P2GD type | 2e-2 | 20 | 4 |
| 4319977_5 | Mixed MCSD-P2GD type | 2e-2 | 20 | 12 |
| 4319977_6 | MCSD-TP control | 2e-2 | 40 | N/A |
| 4319977_7 | Mixed MCSD-P2GD type | 2e-2 | 40 | 8 |
| 4319977_8 | MuonSphere | 1.5e-2 | - | - |
| 4319977_9 | SSO | 1.5e-2 | - | - |
| 4319977_10 | MCSD-TP control | 1.5e-2 | 20 | N/A |
| 4319977_11 | Mixed MCSD-P2GD type | 1.5e-2 | 20 | 8 |

Each hybrid has a control with the same LR, K, seed, data and schedule.
The two baselines retain their reference BF16 PI10/NS8 hard-retraction
implementations. SSO uses at most 20 solver iterations and tolerance 2e-4.
The MuonSphere implementation remains named `muon_ball_dist` in code.

## Algorithm and step meaning

Common mixed settings: routing gap 5e-5, cluster-membership tolerance 2e-3,
adaptive top<=4, FP32 warm block-8 QR/Ritz, and approximate radial return.
The routing threshold and cluster tolerance have different roles.

```text
cached spectrum of the returned X -> estimated gap and tangent normal
safe branch: unchanged MCSD-TP direction
near branch:
    Z = -M / max(||M||_F, 1e-12)
    Q = selected top-cluster cone projection of Z
    Y = X + effective_lr * c * Q
both branches:
    fixed K-step warm block-8 iteration on trial Y
    small 8x8 Ritz SVD
    X_next = R * Y / estimated_sigma1(Y)
    cache vectors and scaled singular-value estimates of X_next
```

There is no post-cone normalization, MCSD Frobenius-length matching,
function-value backtracking, online numerical accuracy gate, or automatic
full-SVD fallback. Full decompositions occur only in tests and post-training
offline audits, not in experimental optimizer updates. The runtime guard
rejects unintended original-matrix SVD/svdvals calls; it is not an accuracy
gate. Steady-state spectral work is one small Ritz SVD per component call.

The coefficient c multiplies the near-branch trial step after Frobenius
momentum normalization. It is NOT a target branch rate or the ratio between
the realized near-branch and MCSD update norms. Old PGD scales 0.10/0.25
used different normalization and are not comparable to c=4/8/12.

Legacy launch flags such as `exact_svd_gap_only`, `tp_radial_pgd_exact` and
`warm_start_uv=0` are bypassed by the cluster dispatch. The actual solver is
`fixed_warm_block8_fp32_ritz`, the cache is approximate, and the actual
return is `approximate_radial_both_branches`. The launch metadata records
these effective choices explicitly. Baselines ignore the cluster settings.

This is a near-top-cluster P2GD-type direction, not exact tangent-cone
projection at every iterate. Capping the cluster at four can omit active
directions. Approximate radial returns can violate the exact radius. This
experiment does not establish stochastic Clarke convergence.

## Matched training protocol

- Width 256, 28 layers, FFN 768, head dimension 128, sequence length 4096.
- Global batch 128, micro-batch 4, seed 1234, TP=PP=DP=1, one H20 per run.
- Budget 1B tokens: 1908 iterations, actually 1,000,341,504 processed tokens.
- Warmup 250 iterations; cosine decay to 10% of the peak LR by 1908.
- Validation every 250 iterations and at 1908; five validation batches.
- BF16 training and NS8 matrix sign; FP32 master weights and Ritz arithmetic.
- Experimental constrained-matrix weight decay is zero; auxiliary settings
  retain the existing weight-decay 0.1 protocol. Baselines are unmodified.
- No checkpoints; fixed source snapshot; logs every ten iterations.

Data come from the same existing **3B-indexed source** as the recent 100M
pilots, but each new run processes only **1B tokens**. All 12 new arms share
the same source, seed and budget. This is not the locked historical
1B-sample corpus and must not be merged into its paper result table.
Changing peak LR also changes the auxiliary optimizer's LR, so comparisons
must retain same-LR controls.

The source and data receipts are in
`output/cluster_candidates_1b_20261024/`. The data manifest records file
sizes and hashes of the index files, not full hashes of the large bin files.
Environment: Python 3.12.1, PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA H20,
TF32 override disabled.

## Measurements and decision criteria

Report validation at the same iteration, final validation/train loss,
training-only wall time and step time, cumulative and late-stage branch
rates, repeat rate, burst duration, and cluster-cap overflow. Branch rates
count matrix-component decisions, not training iterations. Do not select
a winner only by reducing branch frequency.

Experimental arms save at most 48 snapshots each. Scheduled samples cover
steps 2/50/250/750/1250/1750/1900; up to eight early cluster samples and
eight additional cluster samples after step 1000 are retained. Exact
reference computations run AFTER training, recording gap/radius and
direction discrepancies. Selected samples are not an unbiased estimate of
the whole run's numerical errors.

Compare each hybrid first to its matching MCSD-TP control, then to
MuonSphere and SSO at the same LR. Compare K20/K40 at LR 2e-2/c=8 together
with the two K-specific controls. Evaluate both the best observed LR per
method and the same-LR differences. A small single-seed advantage requires
another seed before claiming reproducible superiority.

## Historical startup verification and resources

At 01:16 HKT on 2026-10-24, tasks _0 and _1 had reached step 20/1908,
and _2 had logged step 10/1908. All three passed 35 CUDA preflight tests
and reported zero NaN/skipped iterations. Tasks _3 through _11 were pending
only because of the three-task array concurrency limit. No validation was
available at that startup check; completed results now appear above.

Local checks passed 46 tests, shell syntax validation, and patch whitespace
checks. The remote source checksum manifest passed before training.
Observed initial K20 time is about 11.2 seconds per step, suggesting around
six hours per K20 run plus evaluation. K40 cost is not yet measured; allow
roughly six to eight hours per run and about 24--32 hours for the grid at
three-way concurrency, subject to scheduling and actual throughput.

Both H20 nodes were healthy and had four free GPUs in total before launch.
The three-job cap leaves one of those slots available to other users.
The home filesystem had about 30 GB free. Checkpoints are disabled and
snapshot storage is bounded; the 3B corpus is reused, not copied.

Remote isolated snapshot:
`/home/u3013198/projects/cluster-candidates-1b-20261024`.

Launcher: `slurm/cluster_candidates_1b_20261024.sbatch`.
Patch archive SHA256:
`1cf3816bd669c25f5f47fdcfefe1f853aed2f48ed8e498498bc90172c57fd61e`.
Submission receipt and startup checks:
`output/cluster_candidates_1b_20261024/submission.json` and
`output/cluster_candidates_1b_20261024/startup_check.json`.
