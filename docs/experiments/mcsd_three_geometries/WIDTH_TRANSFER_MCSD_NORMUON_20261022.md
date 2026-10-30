# MCSD vs NorMuon-Sphere: fixed-budget width transfer

Status (2026-10-22 HKT): locally prepared and dry-run checked. Both 6B parent
jobs have now completed successfully; the frozen H20 project is clean, about
30 GB of home storage is available, and the helper was copied to the remote
home directory. **Not submitted:** SSH became intermittent before the remote
dry-run/submission; no new Slurm IDs or training results exist for this bundle.

## Question and comparison

At fixed 3B training tokens, does the width-256 ordering of MCSD and
NorMuon-Sphere persist at widths 384 and 512? This is a **learning-rate
transfer / fixed-token width test**, not a fully tuned scaling-law comparison:
the methods use different width-256-selected LRs and momentum rules. A matched
momentum/LR ablation and multiple seeds would be needed to attribute a gap to
the optimizer update alone.

The width-256 reference (seed 1234, `r=1`, Frobenius sphere, 3B tokens) is:

| Width | Method | LR | Matrix momentum | Final val loss | Job |
| ---: | --- | ---: | --- | ---: | ---: |
| 256 | MCSD | 0.01 | beta=0.95 EMA | 3.467720 | 4197465 |
| 256 | NorMuon-Sphere | 0.015 | beta=0.9 Nesterov-style gradient mixture | 3.464757 | 4197478 |

These reference runs came from source commit
`922820e9782d04f374b658be09d7332c55271c45`. The later frozen H20
project used for the new widths is `fc15600a2898d1748c8b34fe8d3945b1036f03ab`.
The added MCSD Nesterov and direction-normalization switches are **disabled**
here; a six-case, ten-update check found its disabled path bitwise identical
to the earlier optimizer snapshot. NorMuon-Sphere's optimizer implementation
did not change in that source diff. Nevertheless, the width-256 anchor is not
a same-commit rerun and must be labelled as such in a paper.

## New runs

| Width | Method | LR | Matrix momentum | Tokens | Seed | Checkpoints |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 384 | MCSD | 0.01 | beta=0.95 EMA | 3B | 1234 | disabled |
| 384 | NorMuon-Sphere | 0.015 | beta=0.9 Nesterov-style | 3B | 1234 | disabled |
| 512 | MCSD | 0.01 | beta=0.95 EMA | 3B | 1234 | disabled |
| 512 | NorMuon-Sphere | 0.015 | beta=0.9 Nesterov-style | 3B | 1234 | disabled |

All runs use the same indexed OLMo-Mix-1124 train and validation prefixes,
28 layers, sequence length 2048, global batch 128, micro-batch 4, BF16
model arithmetic with FP32 master optimizer state, 2% linear LR warmup,
cosine decay to 10% of peak LR, evaluation on 40 batches every 200 updates
and at the end, one H20 each. At 262,144 tokens per update, 3B rounds up to
11,445 updates / 3,000,238,080 processed tokens. The shape-radius policy
fixes `r=1` and `R_F=sqrt(D_out)` per logical matrix; each method uses the
same weight/data initialization seed and auxiliary AdamW settings.

The width override sets FFN width to `3d`, attention heads to `d/64`, and
query groups to `d/128` (6/3 at 384 and 8/4 at 512). The tokenizer and
training/validation indices stay fixed. Larger models see fewer tokens per
parameter under this fixed budget; conclusions about asymptotic scaling or
full convergence must wait for longer-token runs.

Each width/method has a four-update CUDA smoke. A successful smoke gates its
3B run; within each method lane, width 512 waits for width 384. Two lanes
limit this bundle to at most two simultaneous new H20 jobs. The MCSD and
NorMuon lane parents are the last known 6B jobs `4288199` and `4288211`;
their status must be freshly checked before submission. Long jobs request
three days and save no checkpoints, so interruption would require a rerun.

The external submission helper is
`scripts/manifold/submit_width_transfer.py`. It can be placed outside the
frozen project and invokes the frozen project's own `launch.py`, which records
the source commit and file hashes in each `resolved_config.json`. Its register
must also be outside the frozen project to keep the latter clean.

After H20 connectivity returns, place the helper in the remote home directory
and invoke it with the frozen project and an external register:

```sh
scp scripts/manifold/submit_width_transfer.py hpc2021:/home/u3013198/submit_width_transfer_20261022.py
ssh hpc2021 '/home/u3013198/envs/sso_h20/bin/python /home/u3013198/submit_width_transfer_20261022.py --project /home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005 --after-jobs 4288199 4288211 --register /home/u3013198/width_transfer_register_20261022.json --submit'
```

First verify those parent IDs and that the frozen source/dataset still exist;
the helper handles parents that have already completed. Keep the generated
register with the Slurm logs; rerunning the same command reuses registered
jobs instead of duplicating them. A failed smoke leaves its long job pending
on an unsatisfied `afterok` dependency and must be inspected manually.

## What to report

Compare final validation loss and the full validation trajectory at common
token counts, final train loss, wall-clock time, steps/sec, skipped iterations,
constraint residuals and parameter counts. Plot validation loss versus
**tokens** and versus **GPU-hours**. Report best-of-grid vs fixed-LR-transfer
separately; do not claim that a single seed or these two chosen learning rates
establish a universal winner. If the ordering changes, first run a small
nearby LR grid at the affected width before interpreting it as a width effect.

True manifold point extrapolation is a separate mechanism from NorMuon's
Nesterov-style gradient mixture. The existing CPU toy shows promise, but
these new width runs do **not** test that point-extrapolation algorithm.
