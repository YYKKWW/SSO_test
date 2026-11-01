# Frobenius Sphere Experiments

**Results and next steps | Updated November 2, 2026**

MCSD and MCSD-TP are evaluated against MuonH and NorMuon-Sphere using
Megatron Dense LMs. The completed matched 3B comparison favors
**NorMuon-Sphere**. True point extrapolation now runs on H20, but its
1000-step pilot has **not beaten Muon-style Nesterov**.

[3B comparison](#latest) | [Momentum and LR](#comparisons) |
[6B results](#six-b) | [Point pilot](#point-pilot) |
[Next tests](#next) | [Full history](FROBENIUS_3B_6B_HISTORY.md) |
[Related papers](FROBENIUS_RELATED_WORK.md)

| Evidence | Coverage | Interpretation |
|---|---|---|
| Completed 3B | 60 runs in the verified October 29 ledger | LR, momentum, normalization and seed comparisons |
| Completed 6B | 4 runs, two methods and two seeds | Earlier frozen configurations, not the new Nesterov setting |
| Completed point pilots | 6 runs, 1000 steps each; 3 integration checks | 262M tokens per pilot; separate from full-budget results |
| New point follow-ups | 4 submitted, maximum 2 GPUs concurrently | Three continuations to step 3000; one fresh cap/LR test |

This is a dated evidence summary, not a live dashboard. All losses are
validation cross-entropy; lower is better. No historical results were removed:
[complete LR tables and 64-run ledger](FROBENIUS_3B_6B_HISTORY.md#ledger),
[all-project inventory](ALL_EXPERIMENTS.md).

<a id="latest"></a>

## Matched 3B Comparison

**Fixed:** width 256, 3B tokens, Frobenius radius `R=sqrt(D_out)`, peak
LR=0.010, beta=0.95 Nesterov, seed=1234. Same model, data and auxiliary AdamW.
Nesterov here combines gradients; it does not extrapolate the model point.

| Method | Final val. loss | Difference from NorMuon | H20 wall time |
|---|---:|---:|---|
| **NorMuon-Sphere** | **3.455796** | 0 | 18:31:38 |
| MuonH | 3.458490 | +0.002694 | 18:28:51 |
| MCSD-TP | 3.461153 | +0.005357 | 18:33:09 |
| MCSD | 3.461687 | +0.005891 | 18:28:58 |

![Matched 3B validation curves](frobenius_3b_6b/nesterov095_matched_20261029.png)

[Vector figure](frobenius_3b_6b/nesterov095_matched_20261029.svg) |
[Curves, configurations and jobs](frobenius_3b_6b/latest_batch_20261029.json)

This is a single-seed, fixed-LR comparison, not a statistically established
method ranking or a comparison of globally optimal configurations.

<a id="comparisons"></a>

## Momentum and Learning Rate

At beta=0.95, LR=0.010 and seed=1234, enabling Muon-style Nesterov helps
**all four methods**, not just MCSD.

| Method | EMA | Nesterov | Change |
|---|---:|---:|---:|
| MCSD | 3.467720 | 3.461687 | -0.006033 |
| MCSD-TP | 3.467858 | 3.461153 | -0.006705 |
| MuonH | 3.464988 | 3.458490 | -0.006498 |
| NorMuon-Sphere | 3.463500 | 3.455796 | -0.007704 |

The MCSD/MCSD-TP improvement repeats at seed 2026 (3.464190 / 3.463603).
The two matched Nesterov baselines still need that seed.

**LR=0.015 and 0.020 were tested.** Those results use earlier momentum
settings and remain in the [full LR grids](FROBENIUS_3B_6B_HISTORY.md#comparisons).
They cannot substitute for the requested beta=0.95 Nesterov grid.
The six-run LR=0.012/0.015 grid and MCSD-NorUnit plan are recorded as
[prepared, not submitted](FROBENIUS_NESTEROV_NORUNIT_20261029.md);
they are not the four new point-extrapolation jobs.

<a id="six-b"></a>

## Completed 6B Comparison

Earlier frozen configurations, trained from initialization on the new 6B
index. Learning rate and momentum differ between methods.

| Method / configuration | Seed 1234 | Seed 2026 | Two-seed mean |
|---|---:|---:|---:|
| MCSD / LR=0.010 / 0.95 EMA | 3.431077 | 3.428305 | 3.429691 |
| NorMuon-Sphere / LR=0.015 / 0.9 Nesterov | **3.426804** | **3.414900** | **3.420852** |

The mean difference is +0.008839 for MCSD. More tokens did not reverse this
earlier comparison. The new 0.95 Nesterov configuration and point variants
have no completed 6B results.

<details>
<summary>6B validation curves and exact run records</summary>

![Seed-1234 6B validation curves](frobenius_3b_6b/validation_loss_6b.png)

[CSV measurements](frobenius_3b_6b/validation_6b.csv) |
[All four jobs, times and qualifications](FROBENIUS_3B_6B_HISTORY.md#six-b)

</details>

<a id="point-pilot"></a>

## New Point-Extrapolation Pilot

The model holds live extrapolated Y and one accepted-point X buffer.
Gradient evaluation is at Y; no gradient EMA is used. All six pilots
completed 1000 steps with the unchanged 3B cosine horizon.

| Selected configuration | Val. loss at 1000 |
|---|---:|
| **Nesterov beta=0.95, LR=0.010** | **4.088546** |
| Point cap=0.90, LR=0.005 | 4.194326 |
| Point cap=0.90, LR=0.0025 | 4.200407 |
| No momentum, LR=0.010 | 4.662017 |

Point extrapolation leads early but trails Nesterov from step 600 in the
best tested point configuration. **These losses cannot be compared directly
with the final 3B/6B tables.**

![Point-pilot validation and difference from Nesterov](frobenius_point_20261101/validation.png)

[All six results, train curves, algorithm and resume checks](MCSD_POINT_PILOT_20261031.md) |
[Machine-readable data](frobenius_point_20261101/results.json)

<a id="next"></a>

## Next Tests

| Priority | Test | Status / decision |
|---|---|---|
| Now | Continue cap=0.90 with LR=0.005 and 0.0025; continue Nesterov control to step 3000 | Submitted: 4373194, 4373196, 4373195 |
| Now | Cap=0.95 with LR=0.0025, fresh 1000-step pilot | Submitted: 4373197 |
| Mechanism | Sample accepted/live angular displacement and compare X/Y on identical diagnostic batches | Proposed; no extra production evaluations enabled |
| Fairness | Baseline second seed; balanced LR=0.012/0.015 Nesterov grid | Still needed before stronger method-level claims |
| Transfer | Frozen best candidates, paired seeds, then 6B and larger widths | Deferred until full 3B evidence |

The four new jobs use two serial single-GPU lanes and do not automatically
promote to 3B or 6B. Original checkpoints, LR horizon and validation protocol
are preserved. [Submission details](MCSD_POINT_PILOT_20261031.md#follow-up-tests).
At the first step-1600 readback, point cap=0.90/LR=0.005 remains above the
Nesterov control: **4.014438 versus 3.911879**. Both are partial runs.

**Current research priority:** distinguish direction quality from actual
angular step size and stochastic-gradient effects. Relevant evidence and
limits are summarized in the [recent-paper reading list](FROBENIUS_RELATED_WORK.md).
The experiments do not yet establish universal superiority or a scaling law.

<a id="ledger"></a>

## Reproduction and Records

| Need | Resource |
|---|---|
| Full historical tables, including LR=0.015/0.020 | [Preserved 3B/6B report](FROBENIUS_3B_6B_HISTORY.md) |
| Every older protocol, cancellation and small test | [Project inventory](ALL_EXPERIMENTS.md) |
| Algorithm pseudocode and actual code paths | [Implementation guide](OPTIMIZER_PSEUDOCODE_GUIDE.md) |
| Redraw point-pilot figures on CPU | [Command and released JSON](MCSD_POINT_PILOT_20261031.md#reproduce-the-figures) |
| Environment, launch and resume | [Experiment 3 guide](mcsd_three_geometries/README.md) |
| Current scientific context | [Recent papers and proposed diagnostics](FROBENIUS_RELATED_WORK.md) |

**Protocol:** 28 layers, width 256, FFN 768, 4 query heads / 2 KV groups,
73,418,496 parameters; sequence length 2048, global batch 128, micro-batch 4,
one H20 per run. BF16 model, FP32 master weights; hidden-matrix weight decay
zero, auxiliary AdamW LR=0.001. Shape-based radii use r=1.

3B means 11445 steps; 6B means 22889. Both use 2% warmup and cosine decay to
10% of peak. Validation uses 40 batches every 200 steps and at the scheduled
endpoint. Its iterator advances, so align iteration and seed. Different
budget horizons have different LR schedules. Older width-384
initialization-energy-matched radii belong to a separate protocol.
