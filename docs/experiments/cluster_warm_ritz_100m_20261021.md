# Warm block-8 Ritz with unified approximate radial returns

## Research priority: the mixed MCSD-P2GD method

The user's objective is the theoretical and practical feasibility of the
MIXED method. Pure MCSD-TP is a shared diagnostic control, not a prerequisite
that must first be optimized or the main direction of the next study.
The present short-run loss regressions do not disprove mixed-method
convergence; conversely, successful GPU execution does not prove it.

The running implementation combines three departures from a reference
projected-projected update: near-root clustering instead of the actual
complete top multiplicity, radial instead of metric return, and approximate
spectral information. Its Frobenius-length matching is another separate
step-rule modification. These must be isolated before attributing a failure
or gain to P2GD itself.

### Proposed mechanism ladder, not submitted

Keep the MCSD branch, routing threshold, momentum samples and scalar step
rule matched. Start on synthetic spectral-sphere problems with known top
multiplicity and high-precision reference geometry:

| Hybrid | Small-gap direction | Small-gap return | Question |
|---|---|---|---|
| H1 | Complete true-cone projected momentum | Metric sphere projection | Reference mixed mechanism |
| H2 | Same true-cone direction | Exact radial return | What changes when projection is removed? |
| H3 | Near-top cluster direction | Exact radial return | What changes when nearby roots are merged or capped? |
| H4 | Same near-cluster rule, warm block-8 Ritz | Approximate radial return | What changes from the spectral approximation? |

For the mathematical reference use raw cone-projected momentum, not silently
the current norm-matching factor. Evaluate length matching as a separate
engineering ablation. Historical raw-scale0.25 and matched-c1 runs do not
form an isolated comparison. Retain predetermined stochastic step schedules;
do not introduce a loss-based line search or online accuracy fallback.

At a feasible X, let U,V span its actual complete top singular subspace.
For a direction D define kappa=lambda_max(sym(U.T D V)). Direct differentiation
of the exact radial map gives

    d/dh [ R*(X+h*D)/||X+h*D||_2 ] at h=0+
        = D - (kappa/R)*X.

A true tangent-cone direction has kappa=0 and is preserved to first order
at that fixed X. A near-cluster direction need not have kappa=0 at X. For a
simple top root, kappa=u1.T D v1, even when the algorithm uses two or four
nearby pairs. This is a testable source of first-order direction distortion,
not a claim that it alone caused the observed loss regression. Pointwise
first-order agreement does not provide uniform control across strata.

Measure kappa, actual objective directional decrease on known objectives,
returned displacement versus the intended direction, actual/used top rank,
constraint error, and appropriate stationarity residuals. Include exact
multiple roots, small positive gaps, and gaps that shrink with the step.
Distinguish a genuinely useful repeated nonsmooth branch from stagnation;
a low branch percentage is not itself the success criterion.

The existing stochastic/stratified notes remain conditional proof drafts.
Neither their hypotheses nor theorems automatically cover fixed cluster
tolerances, capped ranks, fixed-precision returns, the norm-matching factor,
or constant-momentum finite-horizon LLM training. These require separate
analysis; local tests can falsify claims but cannot establish a theorem.
Relevant primary references: [Olikier et al., stratified methods](https://arxiv.org/abs/2303.16040),
[Olikier, closed-set retractions](https://arxiv.org/abs/2402.08536),
and [Davis et al., stochastic tame optimization](https://arxiv.org/abs/1804.07795).
They motivate the analysis rather than directly certify this implementation.

After mechanism screening, use width256 short LLM runs to check the mixed
variants, retaining one common MCSD-TP control and the MuonSphere/SSO
references. Do not require a control-only tuning campaign before this work.
No new H20 job was submitted as part of this research-priority clarification.

## Completed results, verified 2026-10-21 16:05 HKT

All four tasks completed on SPG-7-1 with exit code 0, completed training and
offline audits, and zero logged NaN/skipped iterations. The last task finished
at 15:20:49 HKT. This array no longer occupies a GPU. No new experiment or
cancellation was issued during this status check.

These are **width-256, 100M-token, single-seed pilots**, not 1B results.
All four validations are at steps 50/100/150/191. Launch configurations match
the preceding exact controls and MuonSphere/SSO in architecture, data prefixes,
tokenizer, seed, batch, learning-rate schedule and evaluation settings.
Final loss is at step191; cluster statistics are the last logged cumulative
counts at step190 (47880 decisions, including 252 initial zero-LR decisions).

| Method | Job | Final val | Cluster % | Job elapsed | Last s/step |
|---|---|---:|---:|---|---:|
| MCSD-TP, warm block8 K10 | 4291611_0 | **4.946553** | 0 | 36:55 | 10.824 |
| Cluster hybrid, warm block8 K10 | 4291611_1 | 4.958415 | 1.353 | 36:41 | 10.823 |
| MCSD-TP, warm block8 K20 | 4291611_2 | 4.947171 | 0 | 38:04 | 11.250 |
| Cluster hybrid, warm block8 K20 | 4291611_3 | 4.960459 | 1.447 | 37:56 | 11.205 |
| MuonSphere reference | 4290694_0 | 4.947061 | n/a | 34:23 | 10.133 |
| SSO reference | 4290694_1 | 4.956870 | n/a | 38:04 | 11.117 |
| Exact MCSD-TP reference | 4290663_0 | 4.947731 | 0 | 53:20 | 16.116 |
| Exact broad-cluster reference | 4290694_8 | 4.947553 | 1.429 | 53:15 | 16.085 |

Job elapsed is Slurm time excluding queue wait but including setup, checks,
validation and, for the four new runs, the final offline audit. The separately
timed training processes took 36:19, 36:16, 37:39 and 37:30 respectively.
Comparisons with old elapsed times therefore include slightly different
diagnostic overhead; the per-step speed difference supports the same trend.

### Spectral and branch audit

Every run counted exactly **48132 small Ritz SVDs for 48132 component calls**,
with **zero original-matrix full SVDs or svdvals calls inside the update**.
This count includes bootstrap. Full SVD was used only in the separate tests
and post-training diagnostics, never as an update fallback.

| Method | Audit samples | Mean absolute gap error | Max relative radius error after return | Mean relative direction discrepancy | False-cluster samples |
|---|---:|---:|---:|---:|---:|
| Control K10 | 16 | 4.18e-4 | 6.14e-4 | 4.07% | 0 |
| Hybrid K10 | 24 | 2.62e-4 | 2.93e-4 | 13.00% | 2 |
| Control K20 | 16 | 1.85e-5 | 3.55e-5 | 2.07% | 0 |
| Hybrid K20 | 24 | 1.60e-5 | 9.48e-5 | 1.56% | 0 |

These samples deliberately cover selected shapes/steps and the first eight
cluster calls, not a random population. No false-safe decision was observed
in them, which does NOT certify that none occurred during training. Control
misclassification counts are hypothetical because its router is disabled.
Direction discrepancy includes any difference in the selected branch.

Early errors dominate: among the four scheduled step190 samples, mean gap
errors are 5.18e-7, 1.06e-6, 6.20e-7 and 1.51e-6 respectively. These late
samples do not cover every component or every late cluster event.

Hybrid K10 made 648 cluster calls with 254 burst starts (394 repeats, 60.8%)
and 100 detected top-four capacity overflows. Hybrid K20 made 693 calls with
261 starts (432 repeats, 62.3%) and 105 overflows. The overall cluster share
is small, but the step190 shares reached 30/252=11.9% and 34/252=13.5%.
Radial return has not eliminated repeated near-region visits.

### Interpretation and next decision

1. Approximation removes the former full-SVD bottleneck in this pilot:
   K10 reduces elapsed time by about31%, and K20 by about29%, versus matched
   exact-spectrum variants. Runtime is now much closer to MuonSphere/SSO.
2. Control K10 is0.000508 below MuonSphere and0.001178 below the exact control.
   This tiny single-seed difference is not evidence of robust superiority;
   it is evidence that the faster control did not show a clear loss penalty.
3. The broad-cluster hybrid is worse than its own matched control by0.011862
   (K10) and0.013288 (K20). More iterations improve sampled accuracy but do
   not recover the former exact hybrid's loss. Spectral trajectories, early
   misrouting, clustered directions and capacity truncation are not isolated
   by these runs, so none can yet be assigned sole responsibility.
4. K10 control demonstrates that this approximate backend can be useful,
   but mixed MCSD-P2GD is the research priority. Follow the mechanism ladder
   above instead of making control-only optimization a prerequisite. Do not
   promote either broad-cluster hybrid directly to1B/3B. A tighter cluster or
   top2 is an engineering ablation, not automatically exact tangent-cone
   geometry. K20 is a useful precision reference, not an established better
   optimizer.

Raw logs, configs, audits and accounting are stored locally under
`output/cluster_warm_ritz_h20_20261021/`. Structured extraction is
`final_summary_20261021.json`. No matrices/checkpoints were deleted.

## Purpose and scope

This is an engineering ablation of the completed cluster-radial pilots, not
a new convergence claim. No online residual certificate, accuracy gate,
adaptive iteration budget, or automatic full-SVD fallback is used.
The frozen paper reproduction repository is unchanged.

The preceding exact-spectrum results on the same 100M-token protocol were:

| Method | Final validation loss | Job elapsed (mm:ss) |
|---|---:|---:|
| MuonSphere, 4290694_0 | 4.947061 | 34:23 |
| SSO, 4290694_1 | 4.956870 | 38:04 |
| Exact MCSD-TP control, 4290663_0 | 4.947731 | 53:20 |
| Exact adaptive top<=4, broad cluster, 4290694_8 | 4.947553 | 53:15 |

The broad-cluster hybrid used the cluster branch in 1.429% of decisions
through step 190, but did not beat MuonSphere. These small, single-seed loss
differences are exploratory. This new test asks whether approximate spectra
retain useful behavior and reduce the measured cost.

## Four matched configurations

| Array index | Direction | Block dimension | Fixed iterations |
|---|---|---:|---:|
| 0 | MCSD-TP control | 8 | 10 |
| 1 | MCSD-TP / adaptive top<=4 cluster | 8 | 10 |
| 2 | MCSD-TP control | 8 | 20 |
| 3 | MCSD-TP / adaptive top<=4 cluster | 8 | 20 |

- Routing threshold: relative gap <= 5e-5.
- Cluster membership tolerance: (s1-si)/s1 <= 2e-3; at most four directions.
- Cluster displacement coefficient: c=1 relative to the MCSD reference's
  Frobenius length. This is NOT the older raw PGD learning-rate multiplier.
- Every matrix component has its own cached right subspace. Only bootstrap
  uses a fresh local random seed; it does not consume the model/data RNG.
- FP32 master weights, FP32 block matrix products, QR and Ritz solve; TF32
  disabled. Matrix sign remains the existing BF16 NS8 implementation.
- Both branches return by Y * R / estimated_sigma1(Y); no singular-value
  clipping or metric projection is performed.

## Spectral implementation

`warm_block_ritz.py` first divides Y by its Frobenius norm for scaling.
Starting with the cached n-by-8 right basis Q, each of K rounds performs

    P = qr(Y_scaled @ Q)
    Q = qr(Y_scaled.T @ P)

It then computes `P, H = qr(Y_scaled @ Q)` and an SVD of H (at most 8-by-8).
This provides paired approximate left/right vectors and Ritz singular values.
The original matrix is never fully decomposed in this update path.

After radial scaling with a=R/estimated_sigma1(Y), the next cache stores
`U, a*S, V`. Its gap is therefore the gap of the *scaled approximate spectrum*,
not an unscaled pre-return value. Scalar scaling preserves that relative gap.
The cache is explicitly marked approximate and uncertified. A missed leading
direction can make the real returned norm larger than R, even when cached
sigma1 equals R. Warm start and more iterations do not eliminate this risk.

There is one small Ritz SVD per nonzero update after bootstrap. With this
training schedule the first zero-LR step initializes the cache using one
small Ritz SVD, without an additional trial solve. Each K-step solve uses
2K+2 reduced QR calls, so GPU speedup is a hypothesis, not a guarantee.

## Training protocol

Width 256; 28 layers; FFN width 768; head dimension 128; sequence length 4096;
global batch 128; micro-batch 4; seed 1234; one H20 per job, TP=PP=DP=1.
Each run processes 191 iterations (100,139,008 tokens), with 25 warmup
iterations, peak LR 0.015, cosine decay to 0.0015, and five validation batches
every 50 steps plus the final evaluation. Hidden constrained matrices have
zero weight decay; auxiliary parameters retain the reference optimizer setup.
No checkpoints are written.

The input is the existing **3B-indexed OLMo-Mix training source**, consumed
for 100M tokens, with the same validation prefix and tokenizer as the preceding
18-arm pilots. This is not a repeat of the historical 1B-sample paper table.

Launcher: `slurm/cluster_warm_ritz_100m.sbatch`. It defaults to four array
tasks with concurrency one, each requesting 12 CPUs, 80 GB RAM and one H20.
The source is copied into a separate immutable-for-the-run directory and
checked against `source.sha256` before each task.

## Validation and offline audit

Local checks include eight block-Ritz tests (including a width-256 sampled
update and offline audit) and eleven exact-path regression tests. CUDA checks
run in the allocated H20 job before training.

`pretrain_cluster_ritz_observed.py` counts spectral calls inside the update.
Any original-matrix SVD or svdvals call there is a programming error; this
guard is not a numerical accuracy gate and never selects another algorithm.
At most 24 bounded matrix samples per run are saved at selected steps/shapes
and early cluster calls. They contain the intended FP32 returned matrix;
the final in-place parameter write can differ by a few rounding units.

Only AFTER training, `audit_cluster_ritz_snapshots.py` computes FP32 full-SVD
reference spectra. It reports radius deviation, gap error, false-safe and
false-cluster decisions, and direction discrepancy. This audit has no input
to training. Its time is excluded from `training_wall_seconds.txt`.
The selected samples are not an unbiased estimate of overall error rates.
Legacy online residual/interval summary fields are not measured in this
mode; their default zeros must not be interpreted as zero numerical error.

Report final validation loss, training seconds/step, total training time,
cluster/repeat rate, cache hits, cluster dimension/cap overflow, and offline
radius/direction errors before deciding whether to run 1B or 3B experiments.
Control-versus-hybrid comparisons must use the same iteration budget.

## Submission status

Submitted array **4291611**, indices 0--3, on 2026-10-21 at about 12:50 HKT.
All four local configurations and the uploaded code passed eight new CPU
tests and eleven exact-path regression tests using both the local runtime
and the H20 environment. CUDA checks are performed within each allocated job.

| Job | Configuration | Latest verified status |
|---|---|---|
| 4291611_0 | MCSD-TP control, warm block-8 K=10 | Completed, exit0 |
| 4291611_1 | Adaptive top<=4 hybrid, warm block-8 K=10 | Completed, exit0 |
| 4291611_2 | MCSD-TP control, warm block-8 K=20 | Completed, exit0 |
| 4291611_3 | Adaptive top<=4 hybrid, warm block-8 K=20 | Completed, exit0 |

Remote snapshot: `/home/u3013198/projects/cluster-warm-ritz-20261021`.
Results: `results/4291611_<index>/`; receipts: `submission.txt`; source
hashes: `source.sha256`. The resource check at 12:30 HKT found SPG-7-1
mixed with one available GPU; SPG-7-2 was draining and was not treated as
available. Existing jobs 4288211 and 4288199 are not modified.

Historical first live check at 12:53 HKT: task 0 passed all 19 CUDA tests, built the
model and data, and entered real training. Step 1 performed 252 zero-LR
cache bootstraps, with no full-SVD fallback, skipped or NaN iterations.
At that time tasks 1--3 waited under the one-task array concurrency limit,
and validation had not yet run. All four have since completed, as above.

At 12:55 HKT, the latest collected log was step 10/191: train loss 10.14080,
11.016 s/iteration for the reported early window, zero NaN/skipped iterations.
All 252 components reused their cached spectrum on that step; cumulative
hits/bootstrap were 2268/252 and exact-solver fallback remained zero.
Four matrix-shape snapshots were saved. The early timing is encouraging
relative to the old exact control's roughly 16.1 s/iteration, but is not a
steady-state or final wall-time comparison. No loss-quality conclusion is
available before validation. The first task's Slurm numeric job ID is 4291612;
its stable array identifier is 4291611_0.
