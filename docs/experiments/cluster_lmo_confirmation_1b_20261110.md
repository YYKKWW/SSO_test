# Paired LMO Confirmation: 1B Tokens

[Results summary](mcsd_p2gd_results_summary.md) |
[Previous candidate results](cluster_lmo_shape_1b_20261104.md) |
[Launcher](../../slurm/cluster_lmo_confirmation_1b_20261110.sbatch)

## Why This Batch

On 2026-11-10 the account had no active jobs. The latest completed spectral
experiments remain the two shape/rank runs completed on November 5; no newer
spectral endpoint was found. The selected seed-2345 candidates are:

| Method | Final validation loss | Difference vs. MCSD-TP | Training + validation |
|---|---:|---:|---:|
| MuonSphere | 3.573290 | +0.000621 | 5:33:30 |
| SSO | 3.579043 | +0.006374 | 6:11:37 |
| MCSD-TP K10 | 3.572669 | 0 | 5:54:15 |
| Single-LMO adaptive top<=4, c0.5 | 3.571923 | -0.000746 | 5:55:10 |
| Single-LMO 25% shape blend, c0.5 | **3.571611** | **-0.001058** | 5:56:14 |

The blend's additional gain is only **0.000312** and is not present at every
late validation point. These candidates were selected using this seed, so a
confirmation is more informative than another gap/scale sweep. Neither lower
repeat rates nor more accurate spectra alone have predicted better loss.

## Five Matched Runs

| Job | Method | Original launcher arm | Purpose |
|---|---|---:|---|
| `4419145_0` | MCSD-TP K10 | 2 | Fresh matched control |
| `4419145_1` | Single-LMO adaptive top<=4, c0.5 | 8 | Confirm the unblended candidate |
| `4419145_2` | Single-LMO 25% shape blend, c0.5 | 11 | Isolate the selected shape change |
| `4419145_3` | MuonSphere | 0 | Matched practical baseline |
| `4419145_4` | SSO | 1 | Matched spectral-sphere baseline |

## Completed Results, Checked 2026-11-12

All five completed with Slurm exit 0:0, final evaluation at step 1908, and
no logged NaN or skipped iterations. Times below are training plus validation,
excluding preflight, smoke and offline audit. The pending status recorded at
submission below is historical, not the current status.

| Method | Final val loss | Difference vs. matched MCSD-TP | Time |
|---|---:|---:|---:|
| MCSD-TP K10 | 3.575916 | 0 | 5:55:15 |
| Single-LMO c0.5 | 3.575186 | -0.000730 | 5:56:11 |
| 25% shape blend c0.5 | 3.575362 | -0.000554 | 5:55:37 |
| MuonSphere | **3.571786** | -0.004130 | **5:33:30** |
| SSO | 3.580017 | +0.004101 | 6:11:00 |

The unblended improvement over MCSD-TP repeats: -0.000746 at seed 2345 and
-0.000730 here. The blend also improves on control, but its advantage over
unblended reverses: -0.000312 previously versus **+0.000176** here. The
two-seed mean blend advantage over unblended is only 0.000068. Neither
candidate consistently beats MuonSphere; at seed 1234 both lose by more
than 0.003. Do not present the selected seed-2345 endpoint as a general win.

At the last regular statistics line (1900, not 1908), unblended uses the
near branch 5.995% of the time, with 92.19% consecutive repeats and 66
top-four overflow events. The blend uses it 4.234% of the time, with 87.57%
repeats and one overflow. Fewer repeats did not improve the endpoint here.
Each experimental arm recorded 480,816 component updates and the same number
of small Ritz SVDs, with zero full original-matrix SVD or svdvals calls online.

[Collected endpoints, curves, configurations, audits and Slurm accounting](../results/cluster_lmo_confirmation_1b_20261110/confirmation_results_20261112.json).
The new step-aware A2/K-LMO pilot is a different algorithm family and protocol;
see its [100M experiment plan](kernel_lmo_h20_pilot_20261112.md).

All five use **seed 1234**, width 256, 28 layers, FFN 768, sequence 4096,
global batch 128, microbatch 4, LR 0.015, 250 warmup iterations, and cosine
decay to 0.0015. Each run starts fresh and processes **1,000,341,504 tokens**
in 1908 iterations from the existing 3B indexed source. This is a **1B token
budget**, not a 3B run or the historical paper's data sample. Evaluation uses
20 batches every 250 steps and at the final step.

Experimental arms keep cached warm block-8/K10 FP32 QR/Ritz, BF16 NS8,
Nesterov 0.9, matrix weight decay zero and approximate radial return. Near
arms keep routing threshold `5e-5`, cluster tolerance `2e-4`, adaptive `q<=4`
and multiplier `c=0.5`. The blend retains its existing matched unblended
Frobenius length. Baselines retain their reference BF16 cold PI10 and hard
pre-step radial normalization. Their return maps are not identical to the
experimental return; this is documented, not changed in this batch.

The source is frozen from the successful November 4 snapshot, with only the
new wrapper and its configuration tests added. The old seed-1234 control
`4373351_0` used a different source manifest, including optimizer files.
A new control avoids assuming that these versions are equivalent. The old
result remains visible as historical evidence, not as a substitute for this
batch's control. Seed 1234 is a second-seed confirmation, not a blind holdout:
other algorithms have already been evaluated on it.

## Checks and Resources

- One H20 per run; array concurrency capped at **two GPUs**.
- Checkpoints disabled; retain logs, configurations and bounded audit samples.
- Each allocation runs CUDA checks and a separate-process four-step smoke
  before fresh main training. A failed check stops that arm.
- Each experimental main run must record zero original-matrix full-SVD and
  `svdvals` calls online, and one small Ritz SVD per matrix-component update.
- Full SVD remains an **offline sampled audit**, not an online fallback.
- Expected main runtime is about 5.5-6.2 hours per arm; roughly 18 hours over
  three waves at two concurrent GPUs, excluding queue delays and startup.

This is the approximate engineering path. The recent local exact-projection
study and its convergence safeguard are **not** being deployed here. The
cluster surrogate, finite subspace iterations and radial return do not
automatically inherit standard P2GD convergence guarantees.

## Analysis Fixed Before Completion

1. Primary endpoint: final validation loss at step 1908, not the best
   intermediate value. Report each candidate minus the new matched control
   and blend minus unblended, then the two-seed paired differences.
2. Compare both candidates with MuonSphere and SSO within this seed. Do not
   substitute older eval5 or Frobenius-geometry results.
3. Report wall time with the same training-plus-validation definition,
   node placement, all validation points, NaN/skipped counts and source hash.
4. Report cumulative and late near rates, repetition, burst length, top-4
   overflow, sampled radius/gap/direction errors, and spectral-call counts.
   These are diagnostics, not optimization objectives or certificates.
5. If the gain reverses, do not promote the blend or hide the unfavorable
   result. If both candidates improve on control in both seeds, a further
   paired seed is preferable to claiming significance from two seeds. Only
   then consider a matched longer-budget comparison, given the previous
   wide-hybrid ranking reversal at 3B.

## Submission

Submitted on **2026-11-10 at 23:21 HKT**, array **`4419145_0-4%2`**.
At **23:42 HKT**, all five jobs were **PENDING (Resources)**: no CUDA test,
smoke or main training had started. These jobs are not completed results.

Local and remote CPU checks both passed **79 tests** (78 cluster tests plus
one result-collector test). All **3137** files from the successful source
snapshot were verified and copied without modification; only the new
wrapper, its configuration tests and the initial protocol document were
added. The frozen remote protocol document records the pre-submission plan;
this local living report records subsequent job status. Logs and result
metadata are outside the source manifest.

- Source directory: `/home/u3013198/projects/cluster-lmo-confirmation-1b-20261110`.
- Source manifest SHA256: `494ed0399edc488f598fec88f39fb9000658d17525000f8ec8410deb5bcdb9d5`.
- [Submission receipt and complete arm configurations](../results/cluster_lmo_confirmation_1b_20261110/submission.json).
- [Source provenance](../results/cluster_lmo_confirmation_1b_20261110/provenance.json).
- [Last checked job status](../results/cluster_lmo_confirmation_1b_20261110/status_20261110.json).

About **31 GB** was free before submission. No checkpoint or old result was
deleted. Scheduler allocation, not a free-GPU snapshot, determines the actual
start time; the runtime estimate excludes queue delays.
