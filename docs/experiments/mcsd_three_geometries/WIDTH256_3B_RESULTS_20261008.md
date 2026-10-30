# Width-256 3B results: 2026-10-08

Snapshot (HKT): 2026-10-08T15:05:06.730761

This report does not change jobs or optimizer code. Smoke tests are excluded.

## Scope and comparison rules

Frobenius sphere, r=1, width=256, 28 layers, 73,418,496 parameters, seed=1234.
All final results use 11,445 steps and 3,000,238,080 tokens. Single H20 per run.
The validation iterator advances across evaluations: compare matching steps.
Final loss means the final validation at step 11,445, not the minimum over checkpoints.
Original momentum differs: MCSD/TP use beta=.95 EMA; MuonH/NorMuon use beta=.9 Nesterov.
Native-LR, unit-F and momentum variants are kept separate. Single-seed results are not significance tests.

## Best completed native-momentum runs

| Method | LR | Final validation | Runtime | Job |
| --- | --- | --- | --- | --- |
| MCSD | 0.01 | 3.467720 | 18:36:00 | 4197465 |
| MCSD-TP | 0.01 | 3.467858 | 18:32:38 | 4197469 |
| MuonH | 0.015 | 3.466668 | 18:30:16 | 4197474 |
| NorMuon-Sphere | 0.015 | 3.464757 | 18:33:07 | 4197478 |

## New run inventory

| Job | Method | LR | Direction | beta | Nesterov | Status | Step | Final val | Latest val @ step | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4226861 | MCSD | 0.01 | frobenius | 0.95 | False | COMPLETED | 11440 | 3.467961 | 3.467961 @ 11445 | 18:35:24 |
| 4226862 | NorMuon-Sphere | 0.02 | none | 0.9 | True | COMPLETED | 11440 | 3.469132 | 3.469132 @ 11445 | 18:31:51 |
| 4226863 | MCSD | 0.012 | none | 0.95 | False | COMPLETED | 11440 | 3.473471 | 3.473471 @ 11445 | 18:30:16 |
| 4226864 | MCSD | 0.012 | frobenius | 0.95 | False | COMPLETED | 11440 | 3.474321 | 3.474321 @ 11445 | 18:31:38 |
| 4226865 | MuonH | 0.02 | none | 0.9 | True | COMPLETED | 11440 | 3.470523 | 3.470523 @ 11445 | 18:23:59 |
| 4226866 | MCSD-TP | 0.012 | none | 0.95 | False | COMPLETED | 11440 | 3.472968 | 3.472968 @ 11445 | 18:39:59 |
| 4226867 | MCSD | 0.015 | frobenius | 0.95 | False | RUNNING | 10830 | 未完成 | 3.49543 @ 10800 | 17:36:10 |
| 4226868 | NorMuon-Sphere | 0.012 | none | 0.9 | True | RUNNING | 10570 | 未完成 | 3.494304 @ 10400 | 17:07:18 |
| 4226869 | MuonH | 0.012 | none | - | - | PENDING | - | 未完成 | - | 00:00:00 |
| 4226870 | MCSD | 0.02 | frobenius | - | - | PENDING | - | 未完成 | - | 00:00:00 |
| 4226871 | MCSD | 0.02 | none | - | - | PENDING | - | 未完成 | - | 00:00:00 |
| 4226872 | MCSD-TP | 0.02 | none | - | - | PENDING | - | 未完成 | - | 00:00:00 |
| 4227544 | MCSD | 0.01 | none | 0.9 | True | COMPLETED | 11440 | 3.478060 | 3.47806 @ 11445 | 18:29:33 |
| 4227545 | MCSD-TP | 0.01 | none | 0.9 | True | COMPLETED | 11440 | 3.478570 | 3.47857 @ 11445 | 18:39:34 |
| 4227546 | MuonH | 0.01 | none | 0.9 | False | COMPLETED | 11440 | 3.471362 | 3.471362 @ 11445 | 18:30:02 |
| 4227547 | NorMuon-Sphere | 0.01 | none | 0.9 | False | COMPLETED | 11440 | 3.470496 | 3.470496 @ 11445 | 18:27:06 |
| 4227548 | MCSD | 0.01 | none | 0.9 | False | RUNNING | 10930 | 未完成 | 3.475169 @ 10800 | 17:44:17 |
| 4227549 | MCSD-TP | 0.01 | none | 0.9 | False | RUNNING | 10820 | 未完成 | 3.475757 @ 10800 | 17:37:21 |

## Matched-momentum common-step check

All beta=.9 EMA curves are compared through common step 10800 only.

| Method | beta=.9 EMA val |
| --- | --- |
| MCSD | 3.475169 |
| MCSD-TP | 3.475757 |
| MuonH | 3.473564 |
| NorMuon-Sphere | 3.472310 |

## Main conclusions

- Native MCSD/TP: LR=.012 did not improve upon .01.
- Native MuonH/NorMuon: LR=.02 did not improve upon .015; .012 comparison remains incomplete.
- MCSD unit-F at .01/.012 changes final val by +.000241/+.000850; no observed improvement.
- MCSD/TP beta=.9 Nesterov is worse than native beta=.95 EMA, but changes two factors.
- Baselines at beta=.9 benefit from disabling Nesterov at LR=.01 (+same data/radius/schedule).
- At matched beta=.9 Nesterov and LR=.01, MCSD/TP do not beat the baselines.
- Finish beta=.9 EMA MCSD/TP before attributing the separate beta and Nesterov effects.
- Freeze candidates, then add two matched seeds per method before width transfer or longer-budget claims.

## Provenance and deliverables

- Local downloaded logs and Slurm snapshot: `results/manifold/status_20261008/`.
- Full curves, argument checks and stdout SHA256: `reports/width256_3b_20261008/curves_and_results.json`.
- Report: `output/pdf/width256_3b_results_20261008.pdf`.
- Four PNG/SVG figures: `output/figures/width256_3b_20261008/`.
- Rebuild: `scripts/manifold/report_width256_20261008.py` (frozen snapshot, no network calls).
