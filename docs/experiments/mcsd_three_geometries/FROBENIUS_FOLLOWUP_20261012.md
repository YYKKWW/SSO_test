# Frobenius follow-up: submitted 2026-10-12

## Status

All 12 jobs were submitted and their resolved configurations checked against
the submission register. At 02:05 HKT on 2026-10-12, all were PENDING, not
training. Jobs 4255793/4255794 were waiting for nodes reserved for maintenance;
the other ten were waiting for their preceding lane jobs. Both H20 nodes
SPG-7-1 and SPG-7-2 were in administrator maintenance reservation `root_4650`.
Its configured end date is not a reliable estimate of actual service recovery.

No existing jobs were cancelled or changed. The four older pending jobs
4226869--4226872 remain separate from this batch. This new batch has two serial
lanes, so at most two of its single-H20 jobs can run concurrently. When service
returns, the old and new batches together can occupy at most four H20s under
the observed queues. Submission does not establish training success.

## Fixed protocol

- Width 256, 28 layers, Frobenius sphere, shape radius R = sqrt(D_out), r=1.
- 3B-token training budget: 11,445 updates, 3,000,238,080 processed tokens.
- Same existing 3B indexed train/validation data and tokenizer as the controls.
- Sequence length 2048, global batch 128, micro batch 4, one H20 per job.
- Cosine LR decay to 0.1 of peak; 229 warmup steps; auxiliary AdamW LR 0.001.
- Evaluation every 200 steps with 40 batches, plus final evaluation.
- BF16 Polar Express and FP32 master weights; constrained-matrix weight decay
  stays zero. Optional unit-F direction normalization is OFF.
- Checkpoints OFF, retaining logs, configurations and diagnostics. These jobs
  cannot subsequently be resumed as a 6B continuation without checkpoints.

## Submitted grid

| Job | Method | LR | Momentum | Seed | Predecessor |
| --- | --- | --- | --- | --- | --- |
| 4255793 | MCSD | 0.015 | beta=0.9 EMA | 1234 | None |
| 4255794 | MuonH | 0.015 | beta=0.9 EMA | 1234 | None |
| 4255795 | MCSD-TP | 0.015 | beta=0.9 EMA | 1234 | 4255793 |
| 4255796 | NorMuon-Sphere | 0.015 | beta=0.9 EMA | 1234 | 4255794 |
| 4255797 | MCSD | 0.012 | beta=0.9 EMA | 1234 | 4255795 |
| 4255798 | MuonH | 0.012 | beta=0.9 EMA | 1234 | 4255796 |
| 4255799 | MCSD-TP | 0.012 | beta=0.9 EMA | 1234 | 4255797 |
| 4255800 | NorMuon-Sphere | 0.012 | beta=0.9 EMA | 1234 | 4255798 |
| 4255801 | MCSD | 0.010 | beta=0.95 EMA | 2026 | 4255799 |
| 4255802 | MuonH | 0.015 | beta=0.9 Nesterov | 2026 | 4255800 |
| 4255803 | MCSD-TP | 0.010 | beta=0.95 EMA | 2026 | 4255801 |
| 4255804 | NorMuon-Sphere | 0.015 | beta=0.9 Nesterov | 2026 | 4255802 |

The eight matched-EMA runs extend the earlier LR=0.01 momentum ablation.
The four seed=2026 runs repeat the preselected native-momentum settings;
they are not a matched-momentum comparison. Dependencies use `afterany`, so
a failed predecessor ends its slot without suppressing later independent
configurations. Training failure must still be investigated before analysis.

## Source and verification

Frozen H20 project:
`/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005`.
Commit: `fc15600a2898d1748c8b34fe8d3945b1036f03ab` (clean before and after).
No optimizer code or GitHub branch was changed during this submission.

The previously uploaded helper at
`results/manifold/followup_20261011/submit_frobenius_followup.py` matches the
local `scripts/manifold/submit_frobenius_followup.py` SHA256:
`fde9015523f469d15b8a57780841219966a1800704088aa219d191572556a4ef`.

- Local submission-plan tests: 4 passed.
- H20 preflight: all 12 configurations resolve; source, data, tokenizer and
  training environment are present; no duplicate matching runs were found.
- Post-submit: all 12 config hashes, seeds, LR/momentum choices, GPU counts,
  budgets and checkpoint settings match the register; no inflight submission.
- Remote register: `results/manifold/frobenius_followup_register_20261012.json`.
- Local copy: `results/manifold/status_20261012_followup/`.

## Storage and longer-budget work

The home filesystem had 54,824,796,160 bytes free (about 51.1 GiB) at preflight.
No datasets or results were removed, and no 6B download was started: the user
prioritized submitting this 3B batch first. A future 6B study should separately
specify fresh unique data versus repeating 3B, a matched LR horizon and usable
checkpoint policy. Do not delete the whole old 1B directory: these launchers
still use its tokenizer subdirectory even though training uses the 3B files.
