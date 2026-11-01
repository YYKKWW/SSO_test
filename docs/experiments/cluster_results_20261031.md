# Second-Seed 3B Results and H20 Cache Cleanup

Checked on **2026-10-31, HKT (UTC+8)**. Six second-seed jobs completed;
four third-seed jobs failed before training. No jobs were running or queued
under this account, and no replacement jobs were submitted in this check.

## Completed Comparison

Matched settings: width 256, 28 layers, 3B processed tokens (5723 steps),
seed 2345, peak LR `1.5e-2`, one H20 per run, 20 validation batches.
Experimental methods use warm block-8, FP32 QR/Ritz, K10, and approximate
radial return. Near-direction arms use gap `5e-5`, Frobenius-normalized
momentum and multiplier c8; the cluster arm uses tolerance `2e-3`, top<=4.
The [launch protocol](cluster_seed_mechanism_3b_20261028.md) is unchanged.

| Method | Final validation loss | Minus MCSD-TP | Train + validation time | Job |
|---|---:|---:|---|---|
| MuonSphere | 3.436023 | +0.007410 | 16:35:03 | `4346705_0` |
| SSO | 3.435041 | +0.006428 | 18:28:11 | `4346705_1` |
| MCSD-TP K10 | 3.428613 | 0 | 17:39:34 | `4346705_2` |
| Cluster hybrid K10/c8 | 3.432620 | +0.004007 | 17:41:00 | `4346705_3` |
| **Ambient near direction, c8** | **3.427854** | **-0.000759** | 17:40:58 | `4346705_4` |
| Top-1 near correction, c8 | 3.428247 | -0.000366 | 17:44:21 | `4346705_5` |

All six have Slurm `COMPLETED`, exit `0:0`, both completion markers, and
final validation at step 5723. No NaN or skipped iterations were logged.
Times exclude preflight tests and post-training offline audits. They are
not optimizer-only timings. The cluster hybrid is 6.63% slower than
MuonSphere and 4.26% faster than SSO; it takes 86 seconds longer than control.

The four experimental runs each recorded **1,442,196 small Ritz SVDs and
zero original-matrix full-SVD/svdvals calls inside updates**. Full SVD was
used only in the separate offline audits. Baselines lack these counters.

## What Replicated

| Seed | MCSD-TP K10 | Cluster hybrid K10/c8 | Hybrid minus control |
|---|---:|---:|---:|
| 1234 | 3.443943 | **3.442441** | -0.001502 |
| 2345 | **3.428613** | 3.432620 | +0.004007 |

**The control-relative hybrid gain did not replicate.** The mean paired
difference across these two seeds is +0.001253, unfavorable to the hybrid.
At seed 2345 MCSD-TP beats MuonSphere by 0.007410 and SSO by 0.006428, but
at seed 1234 MCSD-TP was worse than both. Neither method is established as
a robust baseline winner by two seeds. Initialization and data indexing/order
can change with seed; compare paired differences rather than isolated losses.

The ambient and top-1 variants both beat the multi-direction cluster arm in
the mechanism seed. Their smaller improvements over MCSD-TP, 0.000759 and
0.000366, still need replication. The data do **not** isolate a positive
contribution from multi-direction cone correction. Equal c does not imply
equal displacement, and the trigger locations diverge across trajectories.

## Branch Behavior and Precision

Statistics cover step 5720, not the final 5723. The common denominator is
1,441,440 matrix-component calls, including zero-LR initialization calls.
Legacy `pgd` log fields mean near-branch use, not full-SVD PGD.

| Near direction | Calls | Rate | Repeat rate | Mean burst length | Mean estimated post-return gap |
|---|---:|---:|---:|---:|---:|
| Cluster top<=4 | 27,027 | 1.8750% | 60.19% | 2.51 | `9.378e-5` |
| Ambient | 12,397 | 0.8600% | 8.40% | 1.09 | `4.102e-4` |
| Top-1 | 13,444 | 0.9327% | 13.80% | 1.16 | `3.576e-4` |

The cluster arm used q=2/3/4 on 4,368 / 11,510 / 11,149 calls. It recorded
3,592 capacity overflows (13.29% of near calls). Its longer repetition and
smaller post-return gaps accompany worse loss, but this does not establish
causation or show that increasing c would fix it. Ambient plus radial return
is not PGD with a metric projection; top-1 is not a complete tangent-cone
direction at a true multiple singular value.

| Offline audit | Samples | Mean absolute gap error | Maximum radius error | Mean relative direction error | False-safe / false-near |
|---|---:|---:|---:|---:|---:|
| MCSD-TP control | 36 | `1.035e-4` | `2.284e-4` | 1.351% | 0 / 0 |
| Cluster hybrid | 52 | `6.912e-5` | `1.503e-4` | 0.878% | 0 / 0 |
| Ambient | 52 | `6.642e-5` | `1.503e-4` | 2.162% | 0 / 1 |
| Top-1 | 52 | `6.624e-5` | `1.503e-4` | 0.927% | 0 / 0 |

These are bounded, selected samples against an FP32 full-SVD oracle, not
population error rates. Control route discrepancies are hypothetical. The
sample mean gap errors exceed the routing threshold; zero sampled misses do
not certify every training decision. Keep startup and later samples distinct
in further analysis. Neither approximate feasibility nor stochastic
convergence follows from successful execution.

## Third-Seed Startup Failure

Jobs `4360583_0--3` started at **2026-10-30 10:43:35 HKT** and failed four
seconds later with exit 127. Slurm spooled the wrapper into its own directory;
the wrapper tried to find the shared launcher next to that copy. This was a
launcher-path error, not numerical divergence, OOM, or an algorithm result.
There are no third-seed training losses.

The local wrapper now resolves its shared launcher using `PROJECT_DIR`.
All five regression tests pass, including execution of a relocated wrapper
and explicit rejection of a missing snapshot. The original CUDA audit had
passed, but did not exercise Slurm spooling. The failed remote snapshot and
logs remain unchanged; a corrected training submission has **not** been made.

## Storage Cleanup

The queue and login processes were checked before deletion. Only regenerable
index-cache arrays (`document_index`, `sample_index`, `shuffle_index`) and
Python `.pyc` files were deleted from the experiment projects. Index-description
text files were retained. Dataset directories, environments, source files,
checkpoints, training logs, JSON metrics, configs and audit snapshots were
excluded.

The cleanup removed **5,950 files**, comprising 378 index arrays and 5,572
compiled Python files: **1,448,594,800 logical bytes (1.35 GiB)**. This is a
file-size total, not a claim that `df` immediately reflected the same quota
change. NFS reported **34,450,964,480 available bytes (32.09 GiB)** immediately
before and after deletion; delayed quota updates and other changes prevent
attributing all free-space changes since the initial 28-GiB check to this
cleanup. Future runs can rebuild these caches from the retained datasets and
launch configurations.

At **12:49:35 HKT**, a follow-up check found all 5,950 selected paths absent,
all six training-log SHA256 hashes unchanged, all launch configurations and
completion markers present, and all 192 audit matrix snapshots retained.
The 3B train/validation `.bin` and `.idx` files remained present. The latest
NFS reading was **50,149,195,776 available bytes (46.71 GiB)**. This later
quota change is larger than the deleted cache files and is not attributed
solely to this cleanup. The user queue remained empty.

The full deletion manifest is preserved locally and on H20 at
`projects/cluster-seed-mechanism-3b-20261028/storage_cleanup/cache_cleanup_20261031.json`.

## Next Decisions

1. Keep MCSD-TP K10 as the engineering reference. Do not expand the current
   top<=4 hybrid as a presumed winner or increase width/budget yet.
2. Complete the originally planned third-seed comparison after correcting its
   submission. The failed run provides no evidence for selecting a new seed.
3. Replicate ambient/top-1 at seed 1234 before interpreting their small gains.
   Preserve c8, K10, routing and normalization for a matched comparison.
4. Use the retained matrices for local mechanism tests, especially actual
   displacement and return-induced direction changes. The newly derived
   cone-extension MCSD-TP with a single LMO is a **separate, untrained**
   candidate; the results above do not test its theory or LLM performance.

[Consolidated results](mcsd_p2gd_results_summary.md) |
[Full metrics, curves, audits and failure messages](../results/cluster_seed_mechanism_3b_20261028/final_20261031.json) |
[Deletion manifest](../results/cluster_seed_mechanism_3b_20261028/cache_cleanup_20261031.json)
