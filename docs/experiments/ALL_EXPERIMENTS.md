# Experiment Inventory

**Organized October 29, 2026.** Latest saved server snapshot: 2026-10-29T15:01:23.442148+08:00. This is an offline inventory, not a fresh queue check.

**80 completed training runs in this three-geometry project: 60 current 3B + 4 current 6B + 6 historical 3B + 10 historical 100M pilots.** Related legacy spectral runs, smokes, cancellations, local toy studies, and plans are listed separately. No result is removed because it is worse than a later result.

[Interpretation and curves](FROBENIUS_3B_6B_SUMMARY.md) | [All 80 final results, times and job IDs](experiment_inventory/COMPLETED_RUNS.md) | [Machine-readable ledger](experiment_inventory/index.json) | [Source-document catalog](experiment_inventory/SOURCE_CATALOG.md)

## Comparison Boundaries

| Protocol | Completed | Keep fixed when comparing | Evidence |
| --- | --- | --- | --- |
| Width 256, shape radius, Frobenius | 60 x 3B; 4 x 6B | 73.418M parameters, R=sqrt(D_out), r=1, sequence 2048 | [Current study](FROBENIUS_3B_6B_SUMMARY.md) |
| Width 384, initialization-derived radii | 6 x 3B | 126.641M parameters; 3 Frobenius + 3 Stiefel | [Historical main batch](mcsd_three_geometries/MAIN_3B_20260924.md) |
| Width 384, initialization-derived radii | 10 x 100M | 3 Frobenius + 3 Stiefel + 4 spectral; pilot horizon | [Pilot batch](mcsd_three_geometries/PILOT_20260924.md) |

The old width-384 loss near 3.30 and current width-256 loss near 3.46 are **not** matched optimizer results. Width, model size and radius protocol differ. The corresponding 25-hour versus 18-hour runtimes are not an optimizer-only speed comparison. Within a protocol, match budget, LR, momentum, seed and variant before attributing a change to the algorithm.

## Every Tested Learning Rate

All LRs below are **peak matrix learning rates**, not constant per-step rates. The current protocol uses 2% warmup and cosine decay to 10% of peak; auxiliary AdamW uses a separate LR. Cells show final validation loss and link to the job record. `--` means **no completed result in this ledger**, not a failed run. LR=0.015 and LR=0.020 remain visible even when a newer matched comparison uses only 0.010.

### 3B, Seed 1234, Standard Direction

| Method | Momentum | 0.005 | 0.007 | 0.008 | 0.010 | 0.012 | 0.015 | 0.020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCSD | 0.9 EMA | -- | -- | -- | [3.473145](experiment_inventory/COMPLETED_RUNS.md#job-4227548) | [3.471094](experiment_inventory/COMPLETED_RUNS.md#job-4255797) | [3.472291](experiment_inventory/COMPLETED_RUNS.md#job-4255793) | -- |
| MCSD | 0.9 Nesterov | -- | [3.501387](experiment_inventory/COMPLETED_RUNS.md#job-4320103) | -- | [3.478060](experiment_inventory/COMPLETED_RUNS.md#job-4227544) | [3.473377](experiment_inventory/COMPLETED_RUNS.md#job-4320104) | -- | -- |
| MCSD | 0.95 EMA | [3.485531](experiment_inventory/COMPLETED_RUNS.md#job-4197463) | [3.470591](experiment_inventory/COMPLETED_RUNS.md#job-4197464) | [3.469048](experiment_inventory/COMPLETED_RUNS.md#job-4329081) | [3.467720](experiment_inventory/COMPLETED_RUNS.md#job-4197465) | [3.473471](experiment_inventory/COMPLETED_RUNS.md#job-4226863) | [3.489496](experiment_inventory/COMPLETED_RUNS.md#job-4197466) | [3.520329](experiment_inventory/COMPLETED_RUNS.md#job-4226871) |
| MCSD | 0.95 Nesterov | -- | -- | -- | [3.461687](experiment_inventory/COMPLETED_RUNS.md#job-4329085) | -- | -- | -- |
| MCSD-TP | 0.9 EMA | -- | -- | -- | [3.474456](experiment_inventory/COMPLETED_RUNS.md#job-4227549) | [3.471926](experiment_inventory/COMPLETED_RUNS.md#job-4255799) | [3.472186](experiment_inventory/COMPLETED_RUNS.md#job-4255795) | -- |
| MCSD-TP | 0.9 Nesterov | -- | -- | -- | [3.478570](experiment_inventory/COMPLETED_RUNS.md#job-4227545) | -- | -- | -- |
| MCSD-TP | 0.95 EMA | [3.485612](experiment_inventory/COMPLETED_RUNS.md#job-4197467) | [3.470890](experiment_inventory/COMPLETED_RUNS.md#job-4197468) | [3.468235](experiment_inventory/COMPLETED_RUNS.md#job-4329083) | [3.467858](experiment_inventory/COMPLETED_RUNS.md#job-4197469) | [3.472968](experiment_inventory/COMPLETED_RUNS.md#job-4226866) | [3.487949](experiment_inventory/COMPLETED_RUNS.md#job-4197470) | [3.518967](experiment_inventory/COMPLETED_RUNS.md#job-4226872) |
| MCSD-TP | 0.95 Nesterov | -- | -- | -- | [3.461153](experiment_inventory/COMPLETED_RUNS.md#job-4329086) | -- | -- | -- |
| MuonH | 0.9 EMA | -- | -- | -- | [3.471362](experiment_inventory/COMPLETED_RUNS.md#job-4227546) | [3.467071](experiment_inventory/COMPLETED_RUNS.md#job-4255798) | [3.464883](experiment_inventory/COMPLETED_RUNS.md#job-4255794) | -- |
| MuonH | 0.9 Nesterov | [3.529123](experiment_inventory/COMPLETED_RUNS.md#job-4197471) | [3.500046](experiment_inventory/COMPLETED_RUNS.md#job-4197472) | -- | [3.476769](experiment_inventory/COMPLETED_RUNS.md#job-4197473) | [3.470822](experiment_inventory/COMPLETED_RUNS.md#job-4226869) | [3.466668](experiment_inventory/COMPLETED_RUNS.md#job-4197474) | [3.470523](experiment_inventory/COMPLETED_RUNS.md#job-4226865) |
| MuonH | 0.95 EMA | -- | -- | [3.465584](experiment_inventory/COMPLETED_RUNS.md#job-4329084) | [3.464988](experiment_inventory/COMPLETED_RUNS.md#job-4320101) | -- | -- | -- |
| MuonH | 0.95 Nesterov | -- | -- | -- | [3.458490](experiment_inventory/COMPLETED_RUNS.md#job-4346665) | -- | -- | -- |
| NorMuon-Sphere | 0.9 EMA | -- | -- | -- | [3.470496](experiment_inventory/COMPLETED_RUNS.md#job-4227547) | [3.465714](experiment_inventory/COMPLETED_RUNS.md#job-4255800) | [3.464984](experiment_inventory/COMPLETED_RUNS.md#job-4255796) | -- |
| NorMuon-Sphere | 0.9 Nesterov | [3.526564](experiment_inventory/COMPLETED_RUNS.md#job-4197475) | [3.499072](experiment_inventory/COMPLETED_RUNS.md#job-4197476) | -- | [3.476793](experiment_inventory/COMPLETED_RUNS.md#job-4197477) | [3.469203](experiment_inventory/COMPLETED_RUNS.md#job-4226868) | [3.464757](experiment_inventory/COMPLETED_RUNS.md#job-4197478) | [3.469132](experiment_inventory/COMPLETED_RUNS.md#job-4226862) |
| NorMuon-Sphere | 0.95 EMA | -- | -- | [3.466207](experiment_inventory/COMPLETED_RUNS.md#job-4329082) | [3.463500](experiment_inventory/COMPLETED_RUNS.md#job-4320102) | -- | -- | -- |
| NorMuon-Sphere | 0.95 Nesterov | -- | -- | -- | [3.455796](experiment_inventory/COMPLETED_RUNS.md#job-4346666) | -- | -- | -- |

No cross-row bold winner is used: momentum differs between rows. The matched 0.95-Nesterov / 0.010 ordering is NorMuon-Sphere < MuonH < MCSD-TP < MCSD. 0.95 Nesterov has not yet been swept over LR=0.012/0.015.

### 3B, Seed 2026, Standard Direction

| Method | Momentum | 0.005 | 0.007 | 0.008 | 0.010 | 0.012 | 0.015 | 0.020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCSD | 0.95 EMA | -- | -- | -- | [3.474090](experiment_inventory/COMPLETED_RUNS.md#job-4255801) | -- | -- | -- |
| MCSD | 0.95 Nesterov | -- | -- | -- | [3.464190](experiment_inventory/COMPLETED_RUNS.md#job-4346667) | -- | -- | -- |
| MCSD-TP | 0.95 EMA | -- | -- | -- | [3.472994](experiment_inventory/COMPLETED_RUNS.md#job-4255803) | -- | -- | -- |
| MCSD-TP | 0.95 Nesterov | -- | -- | -- | [3.463603](experiment_inventory/COMPLETED_RUNS.md#job-4346668) | -- | -- | -- |
| MuonH | 0.9 Nesterov | -- | -- | -- | -- | -- | [3.469006](experiment_inventory/COMPLETED_RUNS.md#job-4255802) | -- |
| NorMuon-Sphere | 0.9 Nesterov | -- | -- | -- | -- | -- | [3.468215](experiment_inventory/COMPLETED_RUNS.md#job-4255804) | -- |

### 3B, Seed 1234, Extra Unit-F Ablation

| Method | Momentum | 0.005 | 0.007 | 0.008 | 0.010 | 0.012 | 0.015 | 0.020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCSD + unit-F | 0.95 EMA | -- | -- | -- | [3.467961](experiment_inventory/COMPLETED_RUNS.md#job-4226861) | [3.474321](experiment_inventory/COMPLETED_RUNS.md#job-4226864) | [3.490344](experiment_inventory/COMPLETED_RUNS.md#job-4226867) | [3.519835](experiment_inventory/COMPLETED_RUNS.md#job-4226870) |

This extra normalization is a separate MCSD variant. `none` in the standard rows does not disable a baseline's native normalization.

### All 6B Results

| Method | Seed | Peak LR | Momentum | Final val | Elapsed | Job |
| --- | --- | --- | --- | --- | --- | --- |
| MCSD | 1234 | 0.010 | 0.95 EMA | 3.431077 | 36:52:20 | 4288199 |
| NorMuon-Sphere | 1234 | 0.015 | 0.9 Nesterov | 3.426804 | 36:49:13 | 4288211 |
| MCSD | 2026 | 0.010 | 0.95 EMA | 3.428305 | 36:48:51 | 4320105 |
| NorMuon-Sphere | 2026 | 0.015 | 0.9 Nesterov | 3.414900 | 36:57:26 | 4320106 |

Elapsed times are hours:minutes:seconds, including startup and validation. These are fresh 6B trainings, not continuations of the 3B schedule. MCSD and NorMuon-Sphere use different LR/momentum configurations. There is no completed 6B MCSD-TP or MuonH result here, nor a 6B test of the new 0.95-Nesterov configuration.

## Historical Three-Geometry Results

### Width 384, 100M

[Protocol, audits and original register](mcsd_three_geometries/PILOT_20260924.md). All rows use peak LR=0.010, seed=1234 and initialization-derived frozen radii.

| Geometry | Method | Momentum | Final val | Elapsed | Job |
| --- | --- | --- | --- | --- | --- |
| spectral | SSO | 0.9 Nesterov | 4.858260 | 00:56:38 | 4105835 |
| spectral | MCSD | 0.95 EMA | 4.874525 | 01:01:23 | 4105836 |
| stiefel | MCSD | 0.95 EMA | 4.985168 | 00:52:34 | 4105837 |
| stiefel | iMuon | 0.95 EMA | 4.935535 | 00:53:21 | 4105838 |
| spectral | MuonSphere | 0.9 Nesterov | 4.851736 | 00:51:51 | 4105839 |
| frobenius | MuonH | 0.9 Nesterov | 4.842168 | 00:51:29 | 4105840 |
| stiefel | MCSD-TP | 0.95 EMA | 4.981171 | 00:52:45 | 4105841 |
| spectral | MCSD-TP | 0.95 EMA | 4.871354 | 00:53:41 | 4105842 |
| frobenius | MCSD | 0.95 EMA | 4.968240 | 00:51:25 | 4105843 |
| frobenius | MCSD-TP | 0.95 EMA | 4.958734 | 00:51:43 | 4105844 |

### Width 384, 3B

[Protocol, audits and original register](mcsd_three_geometries/MAIN_3B_20260924.md). All rows use peak LR=0.010, seed=1234 and initialization-derived frozen radii.

| Geometry | Method | Momentum | Final val | Elapsed | Job |
| --- | --- | --- | --- | --- | --- |
| frobenius | MCSD | 0.95 EMA | 3.299155 | 25:28:44 | 4106582 |
| stiefel | MCSD | 0.95 EMA | 3.430487 | 26:01:18 | 4106583 |
| frobenius | MCSD-TP | 0.95 EMA | 3.299154 | 25:26:49 | 4106584 |
| stiefel | MCSD-TP | 0.95 EMA | 3.430471 | 26:00:59 | 4106585 |
| frobenius | MuonH | 0.9 Nesterov | 3.309457 | 25:19:02 | 4106586 |
| stiefel | iMuon | 0.95 EMA | 3.428445 | 26:13:50 | 4106587 |

The 100M spectral pilots completed, but their sparse post-update norm defects reached approximately 5-16%; all four spectral methods were deferred from this historical formal 3B batch. Their losses are retained, not presented as exact-constraint results. Frobenius and Stiefel sparse audits are likewise not per-update certificates.

## Smokes, Failures and Cancellations

These are not included in the 80 completed-budget runs. Job groups can overlap where a smoke also supplied a recovery reference.

| Jobs | Category | Recorded outcome | Source |
| --- | --- | --- | --- |
| 4104124 | Startup failure | Python/module binary mismatch; repaired before the successful smoke suite. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4104125 | Startup failure | Required ignored Megatron runtime sources were missing; restored before training checks. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4104136 | Startup failure | Initialization used raw BF16 parameters instead of FP32 masters; initialization/radius routing repaired. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4104139, 4104144-4104152 | Single-H20 smoke | Ten method/geometry checks completed, 2-4 steps; not training-budget results. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4104138, 4104142, 4104148 | Preliminary resume | Endpoint matched but the initial RNG/state comparison needed fixes; not the strict-resume acceptance record. Job 4104148 also appears in the smoke suite. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4104181, 4104182, 4104211 | Strict resume | Passed after fixes: continuous versus resumed model/state/RNG/scheduler agreement. | [record](mcsd_three_geometries/resume_verification_20260924.json) |
| 4104214 | Optional two-GPU check | TIMEOUT after 00:20:24, before a training iteration. Cause unestablished; not a passed DP check. | [record](mcsd_three_geometries/H20_VALIDATION_20260924.md) |
| 4106552 | Checkpoint retention | Completed four steps in 00:01:48; verified step-4 save and removal of the preceding checkpoint. | [record](mcsd_three_geometries/MAIN_3B_20260924.md) |
| 4197448-4197451 | Shape-radius smoke | Four method checks passed; identical component metadata, zero NaN/skips. Metadata equality is not a tensor-byte equality check. | [record](experiment_inventory/width256_smoke_audit_20261001.json) |
| 4226742, 4226743 | Unit-F smoke | Both completed; checked default versus unit-F direction and constraint residual. Training source hashes matched; the submit helper changed later. | [record](experiment_inventory/frobenius_unit_audit_20261005.json) |
| 4288197 | Cancelled by user | 3B MuonH, 0.95 EMA, LR=0.01. No qualified final-budget result. Later replacement: 4320101, completed. | [record](mcsd_three_geometries/reports/width256_6b_transfer_20261020/README.md) |
| 4288198 | Cancelled by user | 3B NorMuon-Sphere, 0.95 EMA, LR=0.01. No qualified final-budget result. Later replacement: 4320102, completed. | [record](mcsd_three_geometries/reports/width256_6b_transfer_20261020/README.md) |
| 4288200 | Cancelled by user | 6B MCSD-TP; no qualified final-budget result, no completed replacement in this ledger. | [record](mcsd_three_geometries/reports/width256_6b_transfer_20261020/README.md) |
| 4288207 | Cancelled by user | 6B MuonH; no qualified final-budget result, no completed replacement in this ledger. | [record](mcsd_three_geometries/reports/width256_6b_transfer_20261020/README.md) |
| 4262916 | Data preparation failure | 6B preparation failed on October 13 before committing a new shard; compute-node download access unavailable. | [record](mcsd_three_geometries/DATA_PREP_RESTART_20261020.md) |
| 4287616 | Data transport diagnosis | Compute-node connection failed; official IO-node access worked. | [record](mcsd_three_geometries/DATA_PREP_RESTART_20261020.md) |
| 4287721 | Data preparation completed | IO-node download plus CPU indexing produced 6,000,011,141 indexed training tokens. Existing 3B validation retained. This is not a GPU training run. | [record](mcsd_three_geometries/reports/width256_6b_transfer_20261020/README.md) |

[Momentum formula/regression/resume checks](experiment_inventory/momentum_validation_20261005.json) also passed on CPU (6 regression cases, 48 formula/resume cases); these are correctness checks, not training results.

## Local Manifold-Nesterov Tests

These are **CPU FP64, 8x8 Rayleigh-quotient problems on the Frobenius sphere**, not LLM or MCSD matrix-sign runs. Both versions use 12 seeds, 1,500 iterations, beta=0.9, R=1, six LRs, and a target objective gap of 1e-8. Each version has 3 methods x 6 LRs x 12 seeds = 216 trials. The revised implementation uses an atan2 log map, one extra matrix per method, and no restart.

| Version | Method | LR | Median target-hit iteration | Median final gap | Full-run CPU ms |
| --- | --- | --- | --- | --- | --- |
| Original log map | ema | 0.16 | 401.5 | 0 | 50.76 |
| Original log map | muon_nesterov | 0.16 | 366 | 0 | 51.76 |
| Original log map | manifold_nesterov | 0.12 | 38 | 1.04e-13 | 82.85 |
| Compact atan2 log map | ema | 0.16 | 401.5 | 0 | 52.14 |
| Compact atan2 log map | muon_nesterov | 0.16 | 366 | 0 | 52.43 |
| Compact atan2 log map | manifold_nesterov | 0.12 | 38 | 0 | 86.19 |

Best LR is selected by target-hit iteration, then final gap. CPU time is for the full run, not time-to-target. The early target-hit improvement is not an LLM speedup claim. Source JSON retains the original `Infinity` sentinel for unreached targets.

<details>
<summary>All local LR settings, including unsuccessful settings</summary>

| Version | Method | LR | Median target-hit iteration | Median final gap | Full-run CPU ms |
| --- | --- | --- | --- | --- | --- |
| Original log map | ema | 0.04 | 486 | 9.33e-15 | 51.36 |
| Original log map | ema | 0.08 | 431 | 0 | 51.09 |
| Original log map | ema | 0.12 | 412 | 0 | 50.65 |
| Original log map | ema | 0.16 | 401.5 | 0 | 50.76 |
| Original log map | ema | 0.24 | Not reached | 1.06 | 51.07 |
| Original log map | ema | 0.32 | Not reached | 2.29 | 52.10 |
| Original log map | manifold_nesterov | 0.04 | 108 | 1.73e-12 | 85.66 |
| Original log map | manifold_nesterov | 0.08 | 57 | 4.02e-13 | 84.55 |
| Original log map | manifold_nesterov | 0.12 | 38 | 1.04e-13 | 82.85 |
| Original log map | manifold_nesterov | 0.16 | Not reached | 0.0565 | 84.38 |
| Original log map | manifold_nesterov | 0.24 | Not reached | 0.635 | 82.70 |
| Original log map | manifold_nesterov | 0.32 | Not reached | 1.36 | 82.33 |
| Original log map | muon_nesterov | 0.04 | 451 | 0 | 52.91 |
| Original log map | muon_nesterov | 0.08 | 395.5 | 0 | 52.54 |
| Original log map | muon_nesterov | 0.12 | 376.5 | 0 | 52.12 |
| Original log map | muon_nesterov | 0.16 | 366 | 0 | 51.76 |
| Original log map | muon_nesterov | 0.24 | Not reached | 1.01 | 52.21 |
| Original log map | muon_nesterov | 0.32 | Not reached | 2.26 | 54.68 |
| Compact atan2 log map | ema | 0.04 | 486 | 9.33e-15 | 53.25 |
| Compact atan2 log map | ema | 0.08 | 431 | 0 | 52.10 |
| Compact atan2 log map | ema | 0.12 | 412 | 0 | 52.39 |
| Compact atan2 log map | ema | 0.16 | 401.5 | 0 | 52.14 |
| Compact atan2 log map | ema | 0.24 | Not reached | 1.06 | 51.88 |
| Compact atan2 log map | ema | 0.32 | Not reached | 2.29 | 51.99 |
| Compact atan2 log map | manifold_nesterov | 0.04 | 108 | 0 | 87.52 |
| Compact atan2 log map | manifold_nesterov | 0.08 | 57 | 0 | 86.45 |
| Compact atan2 log map | manifold_nesterov | 0.12 | 38 | 0 | 86.19 |
| Compact atan2 log map | manifold_nesterov | 0.16 | Not reached | 0.0565 | 92.48 |
| Compact atan2 log map | manifold_nesterov | 0.24 | Not reached | 0.635 | 92.59 |
| Compact atan2 log map | manifold_nesterov | 0.32 | Not reached | 1.57 | 92.60 |
| Compact atan2 log map | muon_nesterov | 0.04 | 451 | 0 | 52.85 |
| Compact atan2 log map | muon_nesterov | 0.08 | 395.5 | 0 | 52.41 |
| Compact atan2 log map | muon_nesterov | 0.12 | 376.5 | 0 | 52.28 |
| Compact atan2 log map | muon_nesterov | 0.16 | 366 | 0 | 52.43 |
| Compact atan2 log map | muon_nesterov | 0.24 | Not reached | 1.01 | 53.09 |
| Compact atan2 log map | muon_nesterov | 0.32 | Not reached | 2.26 | 53.13 |

</details>

[Original summary](experiment_inventory/nesterov_small_original.json) | [Compact summary](experiment_inventory/nesterov_small_compact_20261014.json) | [Implementation](../../scripts/manifold/compare_manifold_nesterov_small.py)

## Planned Is Not Completed

| Item | Evidence status | Missing experiment |
| --- | --- | --- |
| 0.95 Nesterov LR refinement | Only LR=0.010 completed for all four methods | [Prepared LR=0.012/0.015 grid](FROBENIUS_NESTEROV_NORUNIT_20261029.md): MCSD/MuonH/NorMuon-Sphere only; no new MCSD-TP run |
| Matched Nesterov seed replication | MCSD/TP: seeds 1234 and 2026; baselines: 1234 only | MuonH and NorMuon-Sphere seed 2026, beta=0.95 Nesterov, LR=0.010 |
| Shape-radius width transfer | [Width-384/512 plan](mcsd_three_geometries/WIDTH_TRANSFER_MCSD_NORMUON_20261022.md) was dry-run only; no job IDs | Run a matched protocol; old width-384 initialization-radius results do not fill these cells |
| 6B momentum transfer | Four old-configuration completed results above | Freeze equally tuned 3B configurations before a four-method 6B comparison |
| MCSD row adaptation | Local opt-in implementation/tests only | [Prepared MCSD-NorUnit run](FROBENIUS_NESTEROV_NORUNIT_20261029.md); no H20 loss evidence; not included in historical MCSD results |
| Manifold point extrapolation | Small CPU problems only | No LLM implementation result or acceleration claim |

## Related Spectral and Legacy Archives

The following six completed runs belong to the **separate legacy spectral-cluster checkout**, not the new three-geometry comparison. They use sequence length 4096, 5,723 steps, LR=0.015, width 256 and seed 1234; model layout and validation protocol differ. Do not rank these losses against the Frobenius table. The [saved snapshot](frobenius_3b_6b/latest_batch_20261029.json) contains their configurations, hashes and curves.

| Job | Method | Cluster mode | Block iterations | Actual return | Final val | Elapsed | Saved status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4329039_0 | Spectral-cluster variant | control | 20 | approximate_radial_both_branches | 3.444459 | 18:21:23 | COMPLETED |
| 4329039_1 | Spectral-cluster variant | adaptive4 | 20 | approximate_radial_both_branches | 3.443598 | 18:21:58 | COMPLETED |
| 4329039_2 | MuonSphere | none | -- | reference_hard_pre_retraction | 3.442197 | 16:38:08 | COMPLETED |
| 4329039_3 | SSO | none | -- | reference_hard_pre_retraction | 3.442296 | 18:29:01 | COMPLETED |
| 4329039_4 | Spectral-cluster variant | control | 10 | approximate_radial_both_branches | 3.443943 | 17:42:31 | COMPLETED |
| 4329039_5 | Spectral-cluster variant | adaptive4 | 10 | approximate_radial_both_branches | 3.442441 | 17:41:05 | COMPLETED |

The mode names alone are not complete configuration identities; consult the snapshot before combining rows. A later seed-transfer array was RUNNING (4346705_2) / PENDING (4346705_3-5) in that snapshot. No final-budget results for those four tasks are available in this record; this is not a live status assertion.

The older 1B/12B SSO, MuonBall/MuonSphere, MCSD-PGD, precision, gap and cached-spectrum explorations remain in [the inherited README archive](../../README.md#historical-sso-test-experiments), [its detailed records](../../README.md#supplementary-results-and-detailed-records), and [the old 1B sweep note](width256_sso_mcsd_lr_sweep_1b.md). Those historical tables have not been re-audited into this 80-run ledger; their old running/pending labels are dated observations, not today's queue.

## Curves and Source Preservation

- [Original 16-run LR sweep: train and validation curves](mcsd_three_geometries/reports/width256_3b_20261005/curves_and_results.json), [train figure](mcsd_three_geometries/reports/width256_3b_20261005/figures/all_learning_rates_train.png), [validation figure](mcsd_three_geometries/reports/width256_3b_20261005/figures/all_learning_rates_validation.png).
- [Extended 3B curves/results](mcsd_three_geometries/reports/width256_3b_20261008/curves_and_results.json) and [12-run momentum/seed follow-up](mcsd_three_geometries/reports/followup_20261020/results.json).
- [6B validation curves](frobenius_3b_6b/validation_loss_6b.png), [matched EMA curves](frobenius_3b_6b/matched_ema095_validation.png), [matched Nesterov curves](frobenius_3b_6b/nesterov095_matched_20261029.png).

The 64-row Frobenius source ledger is preserved unchanged. Historical 16-row results are transcribed with source links; source hashes are saved in the combined JSON. Small engineering/toy summaries have been copied out of ignored local results so the documentation can stand alone. Raw checkpoints are not required to read this archive. Not every old raw train log or per-step trace is Git-tracked; missing raw traces are not synthesized from endpoint losses.

To update: add verified final rows to the source ledgers, retain partial/failed/cancelled records separately, then run `python scripts/manifold/build_experiment_inventory.py` from the repository root. Use `--check` to detect stale generated documents. Never replace a historical configuration with a later best setting.
