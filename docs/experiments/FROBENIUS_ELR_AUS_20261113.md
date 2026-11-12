# Frobenius Sphere: ELR, AUS, and Tangent Directions

**H20 snapshot: November 13, 2026, 00:14 HKT.** [Experiment overview](FROBENIUS_3B_6B_SUMMARY.md)

Later update on November 13: [the width-384 matched transfer](#width-384-transfer)
is submitted; it has no training results yet.

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
Both normalization variants selected 0.010 on this grid. Job **4428698** is
the second-seed (2026) replication of the row-adaptive winner; no final
endpoint yet. These results do not include any AUS replay.

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

| Full 3B arm | Job | Test |
|---|---:|---|
| MuonH | 4427413 | Record the actual AUS at each step for each logical matrix |
| MCSD-TP + unit-F | 4427414 | Keep the original cosine schedule and record AUS |
| MCSD-TP + unit-F, AUS replay | 4427415 | Match MuonH's recorded AUS, then compare loss at the same token count |

All three were pending at this snapshot. The single-GPU lane starts with
pilot `4428742` after the running job `4424382` exits. The first smoke
job starts after the pilot exits, even if the pilot fails; the remaining
smoke and full-budget runs require a successful predecessor.

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

The separate MuonH pilot (`4428742`, pending) uses the same 3B schedule but
stops after 400 updates (104,857,600 tokens). It records the signed radial cosine of `G`,
`S`, and `msign(S)` for every logical constrained matrix at steps
1, 25, 100, 229, and 400, plus ELR, AUS, and paired direction metrics.
Report the absolute cosine distributions by matrix type and layer, not a
mean of signed values that can cancel.

At steps 25, 229, and 400, a separate no-backward check scales one logical
matrix at a time by 0.9 or 1.1 in an early, middle, or late layer. It compares
masked token loss on the same microbatch, records a repeated unperturbed
forward, and restores the BF16 weight before the training forward. The
finite differences use one microbatch; the optimizer's radial cosines use
the accumulated gradient seen by the optimizer after gradient clipping.
Do not compare them as identical samples.

Exact scale invariance of the loss for a matrix implies `<W,G>_F = 0`,
but small `<W,G>_F` at one iterate does not prove scale invariance. Even if
`G` is nearly tangent, older gradients in `S` or the nonlinear `msign`
operation can create a radial component. Interpret scale sensitivity,
direction alignment, and AUS-matched validation loss as separate evidence.

Local validation: 97 tests passed, including bitwise equality of a MuonH
update with the probe on and off, and restoration of the perturbed weights
after a forward failure. The H20 runs have not yet validated the new diagnostic.

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

| Arm | Six-step trace check | Full 3B job |
|---|---:|---:|
| MuonH | 4428778 | 4428781 |
| MCSD + unit-F | 4428779 | 4428783 |
| MCSD-TP + unit-F | 4428780 | 4428784 |

All six jobs were pending when submitted. The first check waits for the
seed-2026 run `4428698` and the separate pilot array `4428745` to finish;
the width-384 jobs then run serially, and the full runs require all three
checks to pass. No checkpoints are saved. The wider trace and launcher
changes passed 241 local tests; H20 execution is not yet verified.

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
