# Finite-Step Repair versus MCSD-TP-PGD: Paired 100M Study

[Previous results](direct_top_repair_100m_20261112.md) |
[Algorithm](../../scripts/direct_top_pgd_pk.py) |
[Tests](../../scripts/test_direct_top_pgd_pk.py) |
[Same-state replay](../../scripts/audit_direct_pk.py)

## Opt-In Radial Return (Local Implementation, Not an H20 Result)

The completed results below use **full sphere projection** in the PGD branch.
They do not evaluate the newly requested whole-matrix normalization:

```text
safe: D = D0 (MCSD-TP)
near: D = -M / ||M||_F * (rho * ||D0||_F)
Y = X + alpha * D
X_next = R * Y / ||Y||_2
```

The existing zero-D0 reference remains sqrt(min(matrix shape)). Here `||Y||_2`
is the largest singular value, not the Frobenius norm or elementwise absolute
value. The radius R remains the existing layer/component target. This mode is
**ambient momentum + radial return**, not Euclidean PGD or a tangent-cone
direction. The usual PGD projection convergence argument does not carry over
merely by changing the return.

For a controlled comparison, only the return changes: direction, rho, routing,
data and seed remain fixed. All singular values scale by R/sigma1(Y); the trial
relative gap is preserved, without clipping several singular values to R.
An already zero-gap trial stays zero-gap, so exit from the near region is not
guaranteed. This exact-cache implementation still makes one full trial SVD per
component per update (plus bootstrap); radial normalization alone does not
require full SVD, but replacing the spectral estimator is a separate experiment.

Select `DirectPKAdapter("pgd_half", ambient_return="radial")` or `pgd_full`.
The legacy mode selects rho=0.5 or 1.0; statistics label it
`ambient_radial_half/full`, count `ambient_radial_calls`, and leave `pgd_calls`
at zero. The host's two-way counter routes these calls under `spel`; use the
adapter's dedicated counters to distinguish near ambient from MCSD updates.
The launcher accepts `--study direct_pk --arm 2 --ambient-return radial`
(arm 3 for rho=1); the choice propagates to training and snapshot replay.
Default launch behavior and all completed projected results are unchanged.
No additional 100M jobs were submitted. At the user's subsequent request,
the [3B radial follow-up](direct_ambient_radial_3b_20261113.md) submits only
the missing exact control and rho=0.5/1.0 radial arms, reusing historical
MuonSphere/SSO comparisons instead of rerunning them.

## Results: All Six Completed

Verified **2026-11-13 17:12 UTC+08:00**. All six jobs completed with exit 0:0;
each passed 26 CUDA tests and its separate smoke. All 20 regular training
reports per arm show zero skipped/NaN iterations. All four experimental arms
also completed the offline replay. No jobs were added or modified in this check.

**Main finding:** finite-step repair improves on MCSD-TP in both paired seeds,
but does not consistently beat MuonSphere. Neither tested PGD scale is a
candidate for longer training at this window and direction normalization.

### Seed 2345: Final Validation and Time

All rows use width 256, 100M tokens, LR .015 and eval20; final step 191.
Times are main training plus validation, excluding checks, smoke, offline
replay and queue time. These are fixed-LR comparisons, not per-method LR optima.

| Method | Final loss | Difference vs. control | Time (mm:ss) | Job |
|---|---:|---:|---:|---|
| MCSD-TP control | 4.922614 | 0 | 56:57 | 4428745_0 |
| Finite-step repair | 4.915827 | -0.006787 | 57:37 | 4428745_1 |
| MCSD-TP-PGD, rho=0.5 | 5.210238 | +0.287624 | 57:38 | 4428745_2 |
| MCSD-TP-PGD, rho=1.0 | 5.341342 | +0.418728 | 57:42 | 4428745_3 |
| **MuonSphere** | **4.903361** | **-0.019253** | **38:16** | 4428745_4 |
| SSO | 4.936409 | +0.013795 | 41:52 | 4428745_5 |

Finite-step repair adds 40 seconds (1.17%) to exact MCSD-TP, but is 50.6%
slower than MuonSphere and has 0.012466 higher loss in this seed. The common
full-SVD path remains the dominant engineering cost; PGD adds only 41-45
seconds to control here, so its main failure in this comparison is loss.

| Method | Step 50 | Step 100 | Step 150 | Step 191 |
|---|---:|---:|---:|---:|
| MCSD-TP | 6.516366 | 5.628499 | 5.104277 | 4.922614 |
| Finite-step | 6.472855 | 5.596349 | 5.093357 | 4.915827 |
| PGD half | 6.566698 | 5.836784 | 5.383227 | 5.210238 |
| PGD full | 6.691874 | 6.028179 | 5.552951 | 5.341342 |
| MuonSphere | 6.473329 | 5.605504 | 5.083833 | 4.903361 |
| SSO | 6.494344 | 5.628458 | 5.114617 | 4.936409 |

### Paired Seeds

Seed 1234 uses the preceding same-budget pilot references; seed 2345 uses
the fresh six-arm batch. No 1B/3B endpoints are mixed into this table.

| Method | Seed 1234 | Seed 2345 | Two-seed mean |
|---|---:|---:|---:|
| MCSD-TP | 4.918329 | 4.922614 | 4.920472 |
| Finite-step repair | 4.912478 | 4.915827 | 4.914153 |
| MuonSphere | 4.919347 | 4.903361 | 4.911354 |
| SSO | 4.927444 | 4.936409 | 4.931927 |

Finite minus control is **-0.005851 / -0.006787**, mean **-0.006319**.
Finite minus MuonSphere is **-0.006869 / +0.012466**, mean **+0.002799**.
The improvement over control repeats, but two seeds do not establish
statistical significance or reliable superiority over MuonSphere.

### Branch Behavior and Actual Step

The denominator is 47,880 positive-LR matrix-component updates, not 191
training iterations. There are 252 separate bootstrap solves. Every tested
experimental update makes exactly one trial SVD: 48,132 full SVD calls total.

| Method | Actual repair/PGD calls | Repeat fraction within near calls | Max q | q>4 within near calls |
|---|---:|---:|---:|---:|
| MCSD-TP | 0; window eligibility 7.75% | 36.61% eligibility repeats | 4 | 0% |
| Finite-step | 6,411 repairs (13.39%) | 56.34% | 7 | 12.40% |
| PGD half | 12,986 projections (27.12%) | 61.31% | 8 | 8.52% |
| PGD full | 14,984 projections (31.29%) | 55.39% | 12 | 30.93% |

The PGD half/full **returned spectral caches** have zero top gap in
10,942/12,986 (84.26%) and 14,067/14,984 (93.88%) projection calls. These
are algebraic plateaus of the clipped cached singular values, not assertions
of exact multiplicity after FP32 reconstruction/outer weight updates.
Both modes use full projection; no rank-four/eight cap caused these results.
PGD full has a lower repeat fraction but worse loss, so repetition alone
does not rank algorithm quality.

| Method | Mean returned displacement / R, all calls | Near returned/trial length, ratio of sums |
|---|---:|---:|
| MCSD-TP | 0.1006542 | 99.47% |
| Finite-step | 0.1008310 | 99.64% |
| PGD half | 0.0903682 | 99.24% |
| PGD full | 0.0974928 | 94.90% |

Finite-step repair changes the overall mean returned step by only +0.176%.
PGD directions match the prescribed half/full D0 trial norm; the zero-D0
exception never activates. Projection does not generally erase the whole
step. These aggregate metrics cannot isolate whether direction shape,
clipping, amplitude, or altered branch occupancy causes the loss deficit.
The window 2e-4 and these normalized-momentum steps must not be conflated
with historical PGD settings using different normalization or smaller gaps.

### Evidence and Next Decision

[Collected JSON](../results/direct_top_pgd_pk_20261113/final_results_20261113.json)
contains parsed validation curves, health checks, launch checks, accounting,
all counters and replay summaries. Raw evidence is under
`docs/results/direct_top_pgd_pk_20261113/h20/` (logs, configurations and audits;
no checkpoints or dataset cache downloaded).

The 32 bounded snapshots generate 128 same-state candidate returns; all have
nonpositive momentum inner products, including the poor PGD candidates.
This again shows why a favorable momentum score is not a validation-loss
oracle or proof of full stochastic objective descent.

- Retain **finite-step repair + matched MCSD-TP control** as the leading
  mechanism candidate. Compare to a fresh MuonSphere baseline when moving
  to 1B; only then decide whether longer budgets or approximation are justified.
- Do not promote the tested PGD rho=0.5/1.0, window=2e-4 configurations, nor
  the preceding Kernel-LMO variants, directly to long training.
- Preserve q capacity in any speed study: finite repair needed q up to 7
  in both seeds. A fixed top-four approximation is not an equivalent update.
- No new run is submitted by this results update. The completed batch now
  occupies zero GPUs; the separately recorded four-GPU account budget remains.

## Why These Candidates

The completed seed-1234 / width-256 / 100M study gives 4.918329 for MCSD-TP,
4.918715 for multiple-root repair, and **4.912478** for finite-step repair.
The same-budget Kernel-LMO exact / warm variants give 5.078964 / 5.082469.
Only the finite-step candidate is promoted to this paired replication. Neither
Kernel-LMO nor the multiple-root correction receives another training run.

This study changes to **seed 2345** and adds fresh MuonSphere/SSO references.
Two normalized-momentum PGD scales explicitly test actual trial length;
they are not a reuse of the old, differently normalized `pgd_lr_scale`.

| Arm | Method | Near-region direction | Return |
|---:|---|---|---|
| 0 | MCSD-TP control | D0 | Radial |
| 1 | Finite-step repair | Top-block repair of D0 retaining spectral slack | Radial |
| 2 | MCSD-TP-PGD, rho=0.5 | Normalized negative momentum, half D0 length | Full sphere projection |
| 3 | MCSD-TP-PGD, rho=1.0 | Normalized negative momentum, full D0 length | Full sphere projection |
| 4 | MuonSphere | Original reference implementation | Reference hard pre-return |
| 5 | SSO | Original reference implementation | Reference hard pre-return |

No P2GD convergence claim is attached to the finite-step repair. The PGD
arms are normalized-momentum, variable-step experimental variants, not the
paper's deterministic raw-gradient theorem implemented verbatim.

## Fixed Protocol

Width 256, 28 layers, FFN 768, head dimension 128, sequence length 4096;
global batch 128, microbatch 4, TP=PP=1. Each arm uses one H20 and processes
100,139,008 tokens in 191 steps. Peak LR .015, minimum .0015, 25 warmup steps,
cosine decay through step 191; eval20 at 50/100/150/191. All six share seed
2345 and the existing 3B indexed source, not a 3B training budget and not the
original frozen paper's 1B dataset. Nesterov momentum .9, BF16 training,
FP32 master weights. Auxiliary AdamW and reference baseline settings are
unchanged; constrained matrix WD=0.

The first four arms share full FP32 SVD caches, the BF16 NS8 MCSD-TP direction
generator, and the relative spectral window **2e-4**. q counts all values
within this window; q>1 is the near-region condition. There is no top-four
or top-eight cap. In safe calls all four use the identical MCSD-TP formula
and radial return. Initializations and oracle streams match, but independent
training paths do not have identical numerical D0 after they separate.

## Explicit PGD Step

Let alpha be the effective spectral-muP step for the component, M its
Nesterov momentum, and D0 its current MCSD-TP descent direction. In the near
region, the PGD variant uses

```
reference = ||D0||_F
if reference <= 1e-12:
    reference = sqrt(min(matrix shape))
D = -rho * reference * M / ||M||_F     # D = 0 when M = 0
Y = X + alpha * D
U, s, Vh = full_FP32_SVD(Y)
s_next = min(s, R); s_next[0] = R
X_next = (U * s_next) @ Vh
cache = (U, s_next, Vh)
```

Thus the ordinary near-region **trial** Frobenius norm is exactly 0.5 or 1.0
times alpha*||D0||_F. Projection may change the actual returned displacement;
both quantities are logged. The explicit degenerate-D0 exception prevents a
vanishing tangent direction from disabling an ambient fallback; its count is
recorded. There is no post-projection step matching, spectral normalization
of momentum, or loss backtracking.

This is a projection onto the **sphere**, not merely the ball: the largest
singular value is raised to R even when the trial lies strictly inside. All
singular values exceeding R are clipped, including directions beyond rank
eight. The transformed values supply the next gap; no trial-spectrum cache
is incorrectly reused as a returned-point cache.

After bootstrap each component update uses one full SVD of its trial, for
both radial and PGD returns. D0 is computed even on PGD calls to make the
reference amplitude explicit; this study is not a minimum-cost PGD kernel.

## Verification and Measurements

26 local CPU tests (9 existing pilot, 9 direct-repair, 8 new PGD) passed.
They check unchanged control/finite trajectories, common safe path, negative
momentum alignment, exact trial-length ratios, degenerate D0, projection of
both interior and exterior trials, correction beyond eight singular values,
one trial SVD, host dispatch, and returned-spectrum caches.

The [initial eight-snapshot local replay](../results/direct_top_pgd_pk_20261113/local_replay.json)
has three near states. Mean returned lengths / R on those states are
0.08437 (control), 0.08451 (finite), 0.04228 (PGD half) and 0.08457 (PGD full).
All returned momentum scores are nonpositive. PGD full produces a zero-gap
returned plateau on one of the three states; PGD half on none. This small,
selected sample is not an LLM-loss test and does not establish burst rates.
The [additional replay](../results/direct_top_pgd_pk_20261113/matched_replay.json)
of 16 just-completed control/finite snapshots ran on the server CPU to avoid
transferring large tensors. It reuses the saved D0. Six states are near;
their mean returned step / R is 0.10294 (control), 0.10318 (finite), 0.05162
(PGD half), and 0.10325 (PGD full). All 64 returned momentum scores are
nonpositive. PGD half/full create zero-gap plateaus in 2/6 and 3/6 near
states, respectively. These results justify retaining both amplitude arms,
but also show that a larger step does not automatically eliminate plateaus.

Main-run statistics distinguish window eligibility, actual top repairs and
actual PGD projections. They report repeated near/PGD calls, projected top
plateaus, clipped ranks, q histograms, D0/trial/returned lengths and return
correction lengths. The inherited `radial_length` field means general return
correction in this study; for PGD it is not a radial operation.

Each GPU allocation verifies frozen source, reruns all 26 tests on CUDA,
performs a separate four-step smoke, then starts fresh main training.
Afterwards eight bounded snapshots are replayed against all four candidate
updates with the same stored D0. These FP64 audits measure momentum scores,
not actual minibatch-loss descent, and are excluded from training timing.

## Resource Limits and Decision

Six short tasks, one H20 each, **at most two concurrent**. Checkpoints are
disabled; at most 32 component snapshots per experimental arm. No existing
job is canceled or modified. No 1B/3B follow-on is automatically submitted.

Historical scheduling update, 2026-11-13: the user requested a total budget of four H20
GPUs across current work. The two existing single-GPU training/serial chains
are unchanged. Array arm `4428745_0` remains eligible; arms 1-5 now depend on
`afterany:4428808`, the single-GPU GS cost benchmark. The array throttle stays
at two. While the benchmark runs, only arm 0 can join it; afterwards the
array can use two GPUs. This limits the known queued workload to four GPUs
without changing experiment settings or canceling training. Both H20 nodes
were fully allocated at the check, so only the two earlier training jobs
were actually running. Future submissions must preserve the same budget.

Judge the paired final loss, validation trajectory and actual wall time.
Compare finite repair against control across both seeds. Compare the two PGD
scales against the same new-seed control and fresh MuonSphere/SSO. A larger
gap or lower repetition count is not itself an optimization success. Use
the measured returned step and projection rank to explain, not replace, loss.

## Submission and Completion

Submitted **2026-11-13 00:14:58 UTC+08:00**, array **4428745**. All six were
pending at the first check and have since completed. CPU tests and all
per-allocation CUDA tests passed (26/26 each). Times below are UTC+08:00 on
November 13. Slurm elapsed includes checks, smoke and replay.

| Job | Method | Start | End | Slurm elapsed | State |
|---|---|---|---|---|---|
| 4428745_0 | MCSD-TP | 02:49:14 | 03:49:04 | 59:50 | Completed, 0:0 |
| 4428745_1 | Finite-step | 03:32:02 | 04:32:07 | 1:00:05 | Completed, 0:0 |
| 4428745_2 | PGD half | 03:49:10 | 04:49:01 | 59:51 | Completed, 0:0 |
| 4428745_3 | PGD full | 04:32:11 | 05:32:05 | 59:54 | Completed, 0:0 |
| 4428745_4 | MuonSphere | 04:49:02 | 05:29:00 | 39:58 | Completed, 0:0 |
| 4428745_5 | SSO | 05:29:02 | 06:12:50 | 43:48 | Completed, 0:0 |

[Submission receipt](../results/direct_top_pgd_pk_20261113/submission.json).
Frozen project: `/home/u3013198/projects/direct-top-pgd-pk-100m-20261113`.
The 3,158-file source manifest SHA256 is
`20e3eab05f0eee638f038bb0d6731310c718603b8c04f4f2603d71ef121760d5`.
Home storage was about 29 GB free at the pre-submission check. The source
document inside the frozen project is not changed after submission; this
local report records later queue observations.
