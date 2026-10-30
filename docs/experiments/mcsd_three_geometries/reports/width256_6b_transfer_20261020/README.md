# Width-256 momentum control and 6B transfer (2026-10-20)

These are submitted experiments, not final results. The training source is the
clean H20 snapshot at commit `fc15600a2898d1748c8b34fe8d3945b1036f03ab`
in `/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005`.
Each job writes its complete resolved configuration, source-file hashes, and
logs under `results/manifold/<run_id>` and `logs/` in that snapshot.

## Common protocol

- Width 256, 28 layers, Frobenius sphere, shape radius factor `r=1`, seed
  `1234`, one H20 per job, BF16 training with FP32 optimizer master weights.
- Sequence length 2048, global batch 128, micro-batch 4, auxiliary AdamW LR
  0.001, cosine decay to 10% of peak LR, and 2% warmup.
- The same 3B validation index is used for every run: 40 validation batches
  every 200 steps and at the final step. Checkpoint saving is disabled.
- The 3B momentum controls use the original 3B training index and 11,445
  steps (3,000,238,080 processed tokens).
- The 6B comparison starts each method from initialization on the same
  prepared 6B training index: 22,889 steps (6,000,214,016 processed tokens).
  The index contains 6,000,011,141 tokens; Megatron may repeat a small
  fraction at the end to fill the final global batch. This is not a resumed
  continuation of the 3B runs.

The 6B data marker is
`/home/u3013198/projects/SSO_test/data/olmo_mix_1124_6b/PREPARED`.
It points to
`indexed/train/olmo_mix_1124_6b_train_text_document` and to the existing
3B validation index. Data preparation job `4287721` completed successfully.

## Jobs

| Budget | Method | Peak LR | Momentum | Job ID | Initial status |
|---|---|---:|---|---:|---|
| 3B | MuonH | 0.010 | beta=0.95 EMA | 4288197 | Running |
| 3B | NorMuon-Sphere | 0.010 | beta=0.95 EMA | 4288198 | Running |
| 6B | MCSD | 0.010 | beta=0.95 EMA | 4288199 | Running |
| 6B | MCSD-TP | 0.010 | beta=0.95 EMA | 4288200 | Running |
| 6B | MuonH | 0.015 | beta=0.90 Nesterov | 4288207 | Pending: GPU capacity |
| 6B | NorMuon-Sphere | 0.015 | beta=0.90 Nesterov | 4288211 | Pending: GPU capacity |

The two 3B jobs isolate the momentum difference at the common LR 0.010:
the existing MCSD and MCSD-TP jobs at beta=0.95 EMA, LR 0.010, seed 1234
form their matched counterparts. In contrast, the 6B jobs transfer each
method's previously selected 3B configuration without retuning on 6B. They
are budget-matched, but **not** matched in learning rate or momentum.
The initial Slurm `afterany` dependencies were resource scheduling only,
not checkpoint or data dependencies. At 13:10 CST on October 20, both were
removed at the user's request. All six jobs are now eligible to run
concurrently, subject to available GPUs. SPG-7-1 has eight GPUs allocated
(four to this batch); SPG-7-2 is drained. Thus only four of the six jobs
are currently running, and the other two are pending for capacity.

At startup, all four running jobs reached training iterations with zero
logged NaN/skipped iterations. Final validation loss and runtime remain
pending. Do not infer ranking from the early training losses, which have
different warmup horizons between 3B and 6B.

## Reduced scope (2026-10-20)

At the user's request, the 3B momentum controls `4288197` and `4288198`,
the 6B MCSD-TP run `4288200`, and the pending 6B MuonH run `4288207` were
cancelled. Continue only 6B MCSD `4288199` and 6B NorMuon-Sphere `4288211`.
The cancelled jobs saved no checkpoints; any later repeat must start from
initialization. Their logs and resolved configurations remain available.
The initial-status table above is retained as a submission record, not as
the current queue state.

## Interim validation (2026-10-21, about 21:00 HKT)

Both retained 6B jobs were still running with no logged NaN or skipped
iterations. MCSD had reached approximately 19,900/22,889 steps and
NorMuon-Sphere approximately 19,400/22,889 steps. Compare only matching
validation steps:

| Step | MCSD val loss | NorMuon-Sphere val loss | MCSD minus NorMuon |
|---:|---:|---:|---:|
| 18,000 | 3.466087 | 3.458941 | +0.007146 |
| 19,000 | 3.452495 | 3.446115 | +0.006380 |
| 19,200 | 3.451551 | 3.444718 | +0.006833 |
| 19,400 | 3.440374 | 3.433389 | +0.006985 |

NorMuon-Sphere currently has the lower validation loss; this is not a final
6B result. These are selected 3B configurations transferred to 6B, so LR
and momentum differ between methods. Do not interpret this pair as a
matched-hyperparameter ablation or as a statistically established ranking.

An updated interim [validation-loss figure](../width256_6b_live_20261021/validation_loss.svg)
shows the full trajectory and a late-stage detail panel. Its source
[CSV snapshot](../width256_6b_live_20261021/validation.csv)
contains MCSD through step 20,200 and NorMuon-Sphere through step 19,800;
at their latest common step 19,800 the losses are 3.442554 and 3.437066,
respectively. The figure is not a final-loss comparison.
