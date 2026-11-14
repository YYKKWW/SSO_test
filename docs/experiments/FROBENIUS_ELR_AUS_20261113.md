# Frobenius Sphere: ELR, AUS, and Tangent Directions

**H20 snapshot: November 14, 2026, 11:11 HKT.** [Experiment overview](FROBENIUS_3B_6B_SUMMARY.md)

[400-step pilot results](#does-muonh-already-use-an-almost-tangent-input) |
[3B replay progress](#controlled-h20-tests) | [Width-384 transfer](#width-384-transfer)

**Question.** Does projecting the Nesterov input before the LMO change the
update direction enough to affect validation loss, once the actual angular
step size is held fixed? A related question is whether the raw loss is nearly
invariant to rescaling an individual logical matrix, which would make its raw
gradient nearly tangent even without an explicit projection.

## Completed 3B Results

Width 256, 28 layers, approximately 73.4M parameters, seed 1234, beta=0.95
Nesterov, Frobenius radius `R=sqrt(D_out)`. Entries are final validation loss.

| Method | LR 0.008 | LR 0.010 | LR 0.012 |
|---|---:|---:|---:|
| MCSD + unit-F | 3.464257 (4419137) | **3.460665** (4390468) | 3.463637 (4419139) |
| MCSD + unit-F + row adaptation | 3.462687 (4419138) | **3.460299** (4390469) | 3.463414 (4419140) |

At the same peak LR of 0.010, MuonH finished at **3.458490** and NorMuon-Sphere
at **3.455796**. The best normalized MCSD variant trails MuonH by 0.001809.
Both normalization variants selected 0.010 on this grid. The second-seed
replication of the row-adaptive variant (`4428698`) finished in 18:36:36,
with final loss **3.464919** at step 11,445. Matched 3B endpoints at peak LR
0.010 and beta=0.95 Nesterov are:

| Method | Seed 1234 | Seed 2026 |
|---|---:|---:|
| NorMuon-Sphere | **3.455796** | **3.462039** |
| MuonH | 3.458490 | 3.463144 |
| MCSD-TP | 3.461153 | 3.463603 |
| MCSD | 3.461687 | 3.464190 |
| MCSD + unit-F + row adaptation | 3.460299 | 3.464919 |

The combined variant improves on plain MCSD by 0.001388 at seed 1234 but
trails it by 0.000729 at seed 2026. This comparison does not isolate the
row adaptation: seed 2026 lacks a unit-F-only run. No evidence yet supports
a consistent benefit across seeds. These results do not include AUS replay.

The 3B fixed-point extrapolation run (`4424382`, gamma=0.95, peak LR 0.00125,
no EMA, seed 1234) finished in 18:35:26 with **3.511619** at step 11,445.
This is 0.049932 above the ordinary MCSD Nesterov endpoint (3.461687), under
the respective tested recipes, not a coefficient-only or LR-matched ablation.

## What the Existing Logs Say About ELR

We use the relative **trial** step `ELR = ||W_trial - W||_F / ||W||_F` and
the normalized **returned** chord `AUS = ||W_next/||W_next||_F - W/||W||_F||_F`,
following the distinction in [From LR to ELR](https://hy.tencent.ai/research/elr?langVersion=en).
Only constrained FP32 master matrices with zero weight decay enter the comparison.

| Method, peak LR 0.010 | Median ELR / LR | Median AUS / LR | Final val. loss |
|---|---:|---:|---:|
| MCSD | 0.998551 | 0.998435 | 3.461687 |
| MCSD-TP | 0.998470 | 0.998453 | 3.461153 |
| MCSD + unit-F | 1.000000 | 0.999964 | 3.460665 |
| MCSD + unit-F + row adaptation | 1.000000 | 0.999960 | 3.460299 |
| MuonH | 1 by update rule | Not recorded | 3.458490 |

MCSD rows use 209 sparse attention samples per run, steps 1000–11400, not
full-model traces. Some earlier samples have AUS/LR near 0.80. For nonzero
directions, MuonH and MCSD-TP + unit-F both take a trial step of length
`LR * R`, so ELR/LR is one on the sphere up to rounding. That does not imply
the same AUS: the radial component of the direction affects the movement
left after radial normalization. The old MuonH run cannot settle this.

## Controlled H20 Tests

Same seed, data order, 3B schedule (11,445 steps), peak LR 0.010, 229 warmup
steps, and auxiliary AdamW for all three full-budget runs. Only constrained
matrices are replayed. The full runs follow the three six-step checks
(`4427410`, `4427411`, `4427412`) in a single-GPU lane.

| Full 3B arm | Job | Status and test |
|---|---:|---|
| MuonH | 4427413 | Completed: **3.458490** at step 11,445, 18:30:02; complete trace |
| MCSD-TP + unit-F | 4427414 | Running, 1,640/11,445; val. loss 3.914616 at step 1,600 |
| MCSD-TP + unit-F, AUS replay | 4427415 | Pending; match MuonH's recorded AUS and compare loss at the same token count |

All three six-step CUDA checks completed. The replay matched `6 x 196`
matrix steps with maximum absolute chord error `6.19e-8` and RMS error
`6.48e-9`, below the `2e-6` absolute tolerance. This verifies replay on
the short trajectory only; there is no 3B AUS-matched loss yet.

The completed MuonH trace has 11,445 steps and 196 logical matrices per
step. Its endpoint reproduces the uninstrumented result at the logged
precision. The non-replayed MCSD-TP control began behind MuonH but had
slightly lower loss at the latest matched validation step:

| Step | MuonH | MCSD-TP + unit-F | MCSD-TP minus MuonH |
|---:|---:|---:|---:|
| 400 | 4.918553 | 4.988851 | +0.070298 |
| 1000 | 4.079984 | 4.086126 | +0.006142 |
| 1200 | 4.011436 | 4.012223 | +0.000787 |
| 1400 | 3.967501 | 3.963995 | -0.003506 |
| 1600 | 3.916692 | 3.914616 | -0.002076 |

Across matched steps 401--1400 and all 196 logical matrices, the median
ratio `AUS_MCSD-TP / AUS_MuonH` is 1.00948 (10th/90th percentiles:
1.00397/1.01806). The nominal-LR control thus moves about 0.95% farther
at the median. The two early validation leads are not evidence of a
sustained crossover or a benefit from the direction alone.

The replay checks the returned
FP32 chord against its target (absolute tolerance `2e-6`); it does not match
direction or impose the reference gradient. The control and replay also
sample paired directions at the same `W`, gradient `G`, and Nesterov input
`S`: compare `msign(P_W S)` and `msign(S)`, with the same post-projection
and unit-F normalization on both. Compare cosine between directions and
alignment with `G`; greater alignment predicts a larger first-order decrease
at the same angular step. This comparison isolates the first projection;
the full MuonH vs. MCSD-TP runs differ in other parts of the update as well.

### Does MuonH Already Use an Almost Tangent Input?

MuonH pilot `4428742` completed in 40:27 with the full 3B LR schedule and
an early exit after 400 updates (104,857,600 tokens). Validation loss was
4.918553 at step 400, matching the full reference at that point. The
pilot sampled all 196 logical matrices at each of steps 1, 25, 100, 229,
and 400 (980 observations). No NaNs or skipped updates were reported.

The first three columns report the median absolute radial cosine
`|<W,Z>|/(||W||_F ||Z||_F)`. The last reports the share of matrices for
which `msign(P_W S)` aligns better with the current gradient than
`msign(S)`, after both directions receive the same post-projection and
unit-F normalization. Differences within `1e-6` are ties.

| Step | Raw gradient `G` | Nesterov input `S` | `msign(S)` | Pre-projection wins |
|---:|---:|---:|---:|---:|
| 25 | 0.080% | 0.230% | 1.972% | 71.43% |
| 100 | 0.194% | 2.222% | 12.230% | 96.94% |
| 229 | 0.252% | 4.266% | 11.123% | 64.80% |
| 400 | 0.313% | 7.471% | 12.609% | 34.69% |

The raw gradient was nearly tangent, but the actual input to MuonH's
matrix sign was not as close. At step 400, its median ELR/LR was 1.000000
while median AUS/LR was 0.990733 (10th percentile 0.980489). The equal-AUS
alignment comparison favors neither direction overall at steps 229 and
400 pooled (49.74% pre-projection wins); the result varies by matrix type:

| Matrix | Pre-projection wins, steps 229 and 400 |
|---|---:|
| Query | 91.07% |
| Key | 94.64% |
| Value | 35.71% |
| Attention output | 46.43% |
| MLP gate | 33.93% |
| MLP up | 32.14% |
| MLP down | 14.29% |

At steps 25, 229, and 400, a separate no-backward check scales one logical
matrix at a time by 0.9 or 1.1 in an early, middle, or late layer. It compares
masked token loss on the same microbatch, records a repeated unperturbed
forward, and restores the BF16 weight before the training forward. The
finite differences use one microbatch; the optimizer's radial cosines use
the accumulated gradient seen by the optimizer after gradient clipping.
Do not compare them as identical samples. Across 126 perturbations, the
median absolute loss change was `1.43e-4` (maximum `4.68e-3`), and repeated
unperturbed forwards matched. Q and K median changes were `3.39e-5` and
`2.96e-5`; the MLP gate reached `4.68e-3` on one perturbation.

Exact scale invariance of the loss for a matrix implies `<W,G>_F = 0`,
but small `<W,G>_F` at one iterate does not prove scale invariance. Even if
`G` is nearly tangent, older gradients in `S` or the nonlinear `msign`
operation can create a radial component. Interpret scale sensitivity,
direction alignment, and AUS-matched validation loss as separate evidence.

These are first-order comparisons on a single early MuonH trajectory,
not paired validation-loss outcomes. The result motivates testing an
attention-Q/K-only pre-projection ablation if the pattern persists later,
not replacing the full MCSD method before the replay results are in.

## Which Projection Matters?

Write `P_W Z = Z - W <W,Z>_F / R^2` on the Frobenius sphere. The true
Riemannian gradient of the current batch is `P_W G`. In these experiments,
the LMO instead takes either the ambient Nesterov source `S` or its current
tangent projection `P_W S`. All three directions below use unit-F scaling
and the same radial return. The minus sign is applied in the trial step.

| Method | Before matrix sign | After matrix sign, before unit-F |
|---|---|---|
| MuonH | `S` | No explicit projection |
| MCSD + unit-F | `P_W S` | No explicit projection |
| MCSD-TP + unit-F | `P_W S` | Project the sign onto the tangent space |

The MuonH step still moves in the tangent direction `-P_W msign(S)` to
first order: the radial return removes the radial part to first order, but
does not undo the choice of input to `msign`. If `P_W S = S`, MuonH and
MCSD-TP have the same tangent ray at that state; with a matched, small AUS,
they return the same point in exact arithmetic. This does not follow merely
from a small radial component in the current `G` because `S` contains history.

**Exact local check.** Take `W = diag(1,0)` and `S = G = [[-1,-1],[-1,2]]`.
Using exact polar factors, the unnormalized alignment with `P_W G` is
2.828427 after pre-projection vs. 2.773501 with the ambient input. After
applying the same post-projection and unit-F scaling to both directions,
the alignments are 2.309401 and 2.425356, respectively. Thus the exact LMO
advantage before normalization does not guarantee a larger predicted
decrease at the same AUS. This is a mathematical check, not an explanation
of the observed LLM result on its own.

At width 256 and peak LR 0.010, the existing MCSD-TP runs without the
unit-F ablation trailed MuonH by 0.002663 (seed 1234) and 0.000459
(seed 2026). The size of the gap is seed-sensitive. The existing 3B
comparison cannot separate the projection, actual angular step and
direction normalization effects.

## Width-384 Transfer

There is an encouraging but confounded historical result: at width 384,
3B tokens, and initialization-energy-matched radii, MCSD and MCSD-TP
finished at 3.299155 and 3.299154, vs. MuonH at 3.309457. The radius and
optimizer protocol differ from the present study. This is a reason to
repeat the comparison, not evidence that model width caused the reversal.
See the [historical ledger](FROBENIUS_3B_6B_HISTORY.md).

The new runs transfer the width-256 settings to width 384 (approximately
127M parameters, 28 layers, 6 query heads, 3 KV groups) without retuning.
All arms use shape radii, 3B tokens, seed 1234, peak LR 0.010,
beta=0.95 Nesterov, the same auxiliary AdamW and a single H20. All three
record per-matrix AUS. Both MCSD arms record same-state, post-projected
direction comparisons on sampled steps, covering every logical matrix at
those steps. The audit is observational and does not alter the updates.

| Arm | Six-step trace check | Full 3B job | Status |
|---|---:|---:|---|
| MuonH | 4428778 | 4428781 | Running, 7,650/11,445; val. loss 3.400546 at step 7,600 |
| MCSD + unit-F | 4428779 | 4428783 | Pending, after MuonH |
| MCSD-TP + unit-F | 4428780 | 4428784 | Pending, after MCSD + unit-F |

All three six-step H20 checks completed with 196 matrices per step and
no NaN or skipped iterations reported. The full jobs run serially. No
checkpoints are saved. The current MuonH loss is an intermediate point,
not a comparison with width 384 MCSD or with the older radius protocol.

Compare the within-width validation-loss differences relative to MuonH.
MCSD + unit-F is the pre-projection-only comparison; MCSD-TP + unit-F
adds explicit post-projection. The baseline and MCSD are separate code
paths with small arithmetic differences, so the paired direction audit
is the stricter test of the pre-projection operation. At fixed tokens,
width 384 also has fewer tokens per parameter. A change in the gap is a
transfer result at this budget, not a scaling law or a retuned comparison.

Interpret the mechanism jointly: if the width-256 loss gap shrinks under
AUS replay, step length contributes. If it persists and the same-state
ambient direction aligns better with the current gradient, that supports
a direction-quality explanation. If the two rays are nearly identical,
look first at the actual AUS and at what changes once trajectories diverge.

## Separate Spectral-Sphere Runs

At 11:05 HKT the spectral `direct_pk` radial-return jobs were still running.
Their protocol is width 256, seed 2345, sequence length 4096, peak LR 0.015,
and 5,723 steps for 3B tokens, not the Frobenius protocol above.

| Job | Configured arm | Val. loss at step 3,500 |
|---|---|---:|
| 4434107_0 | `none`, exact solver, radial-return configuration | 3.653660 |
| 4434107_2 | `pgd_half`, exact solver, radial-return configuration | 3.649616 |

The latter leads by 0.004044 at this point; neither is a final result. Both
logs reported zero cumulative PGD branches at the snapshot. That counter
does not by itself identify which direct-kernel trial rule was used, so the
loss difference is not evidence of more PGD branching. The separate spectral
ELR prefix `4434317` and branch array `4434318` are queued with dependencies.
