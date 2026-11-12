# H20 Frobenius Experiment Readout

[Experiment overview](FROBENIUS_3B_6B_SUMMARY.md) | [Validation data](frobenius_h20_20261112/validation.json)

**Verified November 12, 2026, 11:23 HKT.** Both LR=0.012 runs are training.
At the same step, both trail their LR=0.010 counterparts. The lowest
**completed** normalized-MCSD result remains **3.460299**. Neither the
fixed-point full-budget run nor the seed-2026 replicate has started.

The common protocol is width 256, 3B tokens, 11,445 updates, and 229 warmup
updates. Unless noted otherwise, the seed is 1234. The normalized runs use
beta=0.95 Muon-style Nesterov. Lower validation cross-entropy is better.
Validation batches advance, so intermediate comparisons match by step.

## Full 3B Results

| Method | LR=0.008 final | LR=0.010 final | LR=0.012 final |
|---|---:|---:|---|
| MCSD + unit-F | 3.464257 (4419137) | **3.460665** (4390468) | Running |
| MCSD + unit-F + row adaptation | 3.462687 (4419138) | **3.460299** (4390469) | Running |

Row adaptation lowers final loss by 0.001570 at LR=0.008 and by 0.000366
at LR=0.010 in seed 1234. Both variants improved on plain MCSD (3.461687)
at LR=0.010, but the best normalized variant still trails matched MuonH
(3.458490) and NorMuon-Sphere (3.455796). The LR=0.008 jobs took 18:36:00
and 18:43:57; the respective LR=0.010 jobs took 18:36:31 and 18:35:23.

| LR=0.010, beta=0.95 Nesterov | Seed 1234 | Seed 2026 |
|---|---:|---:|
| NorMuon-Sphere | **3.455796** | **3.462039** (4390471) |
| MuonH | 3.458490 | 3.463144 (4390470) |
| MCSD-TP | 3.461153 | 3.463603 |
| MCSD | 3.461687 | 3.464190 |
| MCSD + unit-F + row adaptation | 3.460299 | Not yet tested |

## LR=0.012 Progress

| Job | Method | Last training step | Last validation | Same-step LR=0.010 |
|---|---|---:|---:|---:|
| 4419139 | MCSD + unit-F | 7630 | 3.561833 at 7600 | 3.547785 |
| 4419140 | MCSD + unit-F + row adaptation | 6770 | 3.620917 at 6600 | 3.604839 |

Neither run has logged NaN or skipped iterations. At their latest shared
validation step (6600), the three-LR comparison is:

| Method | 0.008 | 0.010 | 0.012 | 0.012 minus 0.010 |
|---|---:|---:|---:|---:|
| MCSD + unit-F | 3.596117 | 3.606068 | 3.622383 | +0.016315 |
| MCSD + unit-F + row adaptation | 3.594388 | 3.604839 | 3.620917 | +0.016078 |

The LR=0.008 runs lead their LR=0.010 counterparts at step 6600 but lose
at the final endpoint. This observed reversal is why the replication
gate waits for all six final results instead of choosing from partial curves.

## Fixed-Point Extrapolation

| Peak LR | Fixed 0.95 at step 3000 | Capped FISTA at step 3000 | Fixed minus capped |
|---|---:|---:|---:|
| 0.00125 | 3.789622 (4419135) | 3.788594 | +0.001028 |
| 0.0025 | 3.812622 (4419136) | 3.811040 | +0.001582 |

Both pilots completed without logged NaN/skipped updates. Neither beat
the matched capped-FISTA schedule or the ordinary Nesterov reference
(3.778030 at step 3000, LR=0.010). The pilots processed 786M tokens under
the 3B schedule, not the full budget. See the
[earlier point comparisons](FROBENIUS_RESULTS_20261104.md#point-extrapolation-at-step-3000).

| Job | Status at the snapshot | Next action |
|---|---|---|
| 4424382 | Waiting for 4419139 | Full 3B fixed gamma=0.95, LR=0.00125, from initialization on one H20 |
| 4424383 | Waiting for both LR=0.012 jobs to succeed | CPU-only gate checks six final results and submits one seed-2026 replicate. No training job ID yet. |

No jobs were submitted, stopped, or changed as part of this readout.
Frozen training source: `7220c1eaf53c179f0f7dcc8eabb2b38e3a94450f`.
