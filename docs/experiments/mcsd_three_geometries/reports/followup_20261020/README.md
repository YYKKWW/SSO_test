# Frobenius 3B follow-up results (2026-10-20)

All 12 new jobs completed with exit code 0:0, final evaluation at step 11445, and zero logged skipped/NaN iterations. Six earlier momentum runs were re-read to complete the matched comparison.

## Protocol

Width 256, 28 layers, Frobenius sphere R=sqrt(D_out), r=1; 3B tokens (3,000,238,080 processed), sequence length 2048, global batch 128, micro-batch 4, one H20 per run. BF16 training with FP32 optimizer master weights; direction normalization disabled. Cosine schedule, 229 warmup steps, final LR 10% of peak; auxiliary AdamW LR 0.001. Validation: 40 batches every 200 steps plus the final evaluation.

Final loss is the final validation result, not the best intermediate checkpoint. Train loss below averages the last ten logged points (steps 11350-11440). Elapsed time is Slurm wall time including evaluation and startup, not training-only time. Checkpoint saving was disabled. The validation iterator advances: compare corresponding steps within a seed, not arbitrary points on different trajectories.

## Matched momentum: beta=0.9, EMA, seed=1234

| Method | LR 0.010 | LR 0.012 | LR 0.015 |
|---|---:|---:|---:|
| MCSD | 3.473145 | 3.471094 | 3.472291 |
| MCSD-TP | 3.474456 | 3.471926 | 3.472186 |
| MuonH | 3.471362 | 3.467071 | 3.464883 |
| NorMuon-Sphere | 3.470496 | 3.465714 | 3.464984 |

## Frozen configurations: second seed

Different momentum settings and selected LRs are retained here; this is not a matched-momentum comparison. Two seeds are descriptive, not sufficient to establish statistical superiority.

| Method | LR | Momentum | Seed 1234 | Seed 2026 | Mean |
|---|---:|---|---:|---:|---:|
| MCSD | 0.01 | 0.95 EMA | 3.467720 | 3.474090 | 3.470905 |
| MuonH | 0.015 | 0.9 Nesterov | 3.466668 | 3.469006 | 3.467837 |
| MCSD-TP | 0.01 | 0.95 EMA | 3.467858 | 3.472994 | 3.470426 |
| NorMuon-Sphere | 0.015 | 0.9 Nesterov | 3.464757 | 3.468215 | 3.466486 |

## All 12 new jobs

| Job | Method | LR | Beta/mode | Seed | Final val | Tail train | Wall time |
|---|---|---:|---|---:|---:|---:|---|
| 4255793 | MCSD | 0.015 | 0.9/EMA | 1234 | 3.472291 | 3.449286 | 18:35:26 |
| 4255794 | MuonH | 0.015 | 0.9/EMA | 1234 | 3.464883 | 3.442046 | 18:21:47 |
| 4255795 | MCSD-TP | 0.015 | 0.9/EMA | 1234 | 3.472186 | 3.448819 | 18:31:01 |
| 4255796 | NorMuon-Sphere | 0.015 | 0.9/EMA | 1234 | 3.464984 | 3.441936 | 18:28:56 |
| 4255797 | MCSD | 0.012 | 0.9/EMA | 1234 | 3.471094 | 3.447518 | 18:28:42 |
| 4255798 | MuonH | 0.012 | 0.9/EMA | 1234 | 3.467071 | 3.443948 | 18:26:06 |
| 4255799 | MCSD-TP | 0.012 | 0.9/EMA | 1234 | 3.471926 | 3.448620 | 18:33:12 |
| 4255800 | NorMuon-Sphere | 0.012 | 0.9/EMA | 1234 | 3.465714 | 3.442293 | 18:26:08 |
| 4255801 | MCSD | 0.01 | 0.95/EMA | 2026 | 3.474090 | 3.440482 | 18:35:04 |
| 4255802 | MuonH | 0.015 | 0.9/Nest | 2026 | 3.469006 | 3.435059 | 18:25:41 |
| 4255803 | MCSD-TP | 0.01 | 0.95/EMA | 2026 | 3.472994 | 3.439775 | 18:39:58 |
| 4255804 | NorMuon-Sphere | 0.015 | 0.9/Nest | 2026 | 3.468215 | 3.434254 | 18:32:16 |

## Interpretation

- Matched beta=0.9 EMA: both MCSD variants prefer 0.012 among {0.010, 0.012, 0.015}. MuonH and NorMuon-Sphere continue to improve up to 0.015, so that endpoint is not an established optimum.
- At LR 0.012, MCSD trails MuonH by 0.004023 and NorMuon-Sphere by 0.005380. At each method's best observed matched-EMA LR, MCSD trails MuonH by 0.006211. Matching momentum and filling the LR grid did not remove the baseline advantage.
- The frozen-configuration second seed preserves the ranking: NorMuon-Sphere, MuonH, MCSD-TP, MCSD. The MCSD vs MCSD-TP ordering flips between seeds and their differences are small; avoid claiming a robust gain from TP on these two runs.
- The original beta=0.95 EMA MCSD configurations remain better than the tested beta=0.9 EMA variants for seed 1234. Do not replace the default solely to match beta.
- A 6B comparison is a new matched-budget experiment, not proof that extra data will reverse the ranking. Start all methods with the same 6B schedule and frozen seed/data protocol; these runs saved no checkpoints and cannot be directly continued.

Machine-readable curves, arguments, raw-log hashes and accounting are in `results.json`; the compact table is in `results.csv`.
