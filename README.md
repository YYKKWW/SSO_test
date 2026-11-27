# MCSD on the Spectral Sphere

Research code and experiment records for **MCSD-TP**, near-region hybrids,
and **MuonSphere / SSO** baselines in Megatron-LM pretraining.

[Results ledger](docs/experiments/mcsd_p2gd_results_summary.md) ·
[November 27 update](docs/experiments/results_update_20261127.md) ·
[New 3B transfer](docs/experiments/warm_p2gd_smallstep_3b_20261126.md) ·
[Latest paired results](docs/experiments/p2gd_replication_recovery_20261125.md) ·
[BF16-matched results](docs/experiments/p2gd_late_controls_20261122.md) ·
[Independent short trajectories](docs/experiments/p2gd_trajectory_panel_20261122.md) ·
[Six-control and local tests](docs/experiments/p2gd_local_and_six_results_20261122.md) ·
[Latest 3B comparison](docs/experiments/warm_radial_controls_3b_20261117.md) ·
[Latest budget controls](docs/experiments/p2gd_budget_matching_20261120.md) ·
[Returned-step replay](docs/experiments/p2gd_returned_elr_replay_20261119.md) ·
[Spectral ELR confirmation](docs/experiments/spectral_elr_confirmation_20261119.md) ·
[Algorithms](docs/algorithms/current_comparison_pseudocode.md) ·
[Reproduction](docs/runbooks/cluster_experiments_reproduction.md) ·
[Related work](docs/related_work.md)

> **November 27:** network restored; three requested small-step warm K10
> 3B arms submitted as **4519937_0--2**, max two concurrent H20s. Cluster/top-1
> passed CUDA tests and four-step smokes; ambient queues. **No new 3B loss yet.**
> The separate 64-update common-AUS study favors MuonSphere over MCSD-TP at
> matched master-angle targets; it does not test P2GD. Existing 3B references
> are reused, not rerun. [Results, status and next decisions](docs/experiments/results_update_20261127.md).
>
> **Completed results checked November 26:** all four later-start paths **4506265_0-3**,
> restore **4506264** and audit **4506266** COMPLETED, exit0:0. Cluster remains
> better than top-1 and shrinkage controls under the BF16 matching rule, but
> costs more. This is two starts of **one seed**, not independent-seed proof.
> No new jobs were submitted. The 49-run 1B/3B ledger is unchanged.
> Original failed jobs remain documented, not relabeled. This is a research
> branch, not a frozen paper release.

## Latest: Later-Start BF16-Matched Paths

Width256, shared step768 checkpoint and optimizer state, same training
batches, 256 updates per policy, 20 fresh fixed held-out batches.

| Near-region policy | Final loss | Last-four mean | Near rate | H20 time |
|---|---:|---:|---:|---:|
| Half-D0 | 3.825426 | 3.843236 | 1.373% | 1:25:14 |
| Top-1 | 3.800925 | 3.823121 | 8.177% | 1:50:58 |
| **Cluster-cone, P2GD-type** | **3.791397** | **3.817367** | 14.738% | 2:17:50 |
| Output-matched half-D0 | 3.812997 | 3.838130 | 2.313% | 1:37:23 |

Cluster is lower at all eight post-update evaluations. Its maximum BF16
budget error is **0.471%**, with strict-tolerance failures disclosed, and it
costs **24.2% more than top-1**. Matching uses each path's own-state budget;
absolute steps are not identical after divergence. This is not a comparison
against MuonSphere/SSO or a full validation benchmark.

Top6 covers all observed clusters in this panel; top4 misses directions in
4.85% of near calls. Oracle-warm block8/K10 looks accurate in 23 snapshots.
Persistent warm K10 training already exists; the new top6/small-step policy
is the untested transfer, and per-call speed is not training throughput.
BF16 budget search still costs about31.5 norm SVDs per cluster call.

**Requested next:** three width256/3B arms (top<=6 cluster, top-1, ambient),
persistent block8/K10, a shared frozen small-step coefficient, no online
budget search. Sparse offline BF16 audits disclose actual step differences;
this is not the exact matched-step protocol. At most two new concurrent H20s.

[Curves, matching/capacity audits, raw evidence and next plan](docs/experiments/p2gd_replication_recovery_20261125.md)

## Previous: Step512-Start Short Trajectories

Width256; same step512 checkpoint, full optimizer state and training batches;
256 updates per policy,20 fixed held-out microbatches. After divergence each
arm matches its own-state reference budget, not an identical absolute step.

| Near-region policy | Final loss | Last-four mean | Near rate | H20 time |
|---|---:|---:|---:|---:|
| Half-D0 | 3.942878 | 3.980456 | 0.769% | 1:26:11 |
| Ambient | 3.935439 | 3.966445 | 1.705% | 1:29:51 |
| Top-1 | 3.929235 | 3.964658 | 3.491% | 1:35:15 |
| **Cluster-cone** | **3.911326** | **3.945195** | 8.530% | 1:53:43 |
| Output-attenuated half-D0 | 3.947058 | 3.966588 | 0.885% | 1:29:50 |

Cluster improves on top-1 by0.017909 at the endpoint, with19.4% higher
diagnostic training time. Its92.4% consecutive-near fraction is not alone a
failure indicator; the resulting trajectory needs clusters up to size7,
with525 events exceeding top4. All matching errors stay below1%, though
strict-tolerance failures are retained. This panel does not contain
MuonSphere/SSO and is not a full validation or1B/3B comparison.

[Curves, capacity audit, verified records and next tests](docs/experiments/p2gd_trajectory_panel_20261122.md)

The step768-start follow-up above is now complete. It uses different held-out
batches; do not concatenate these two curves or compare their absolute losses
as one continuous validation trajectory.

## Previous BF16-Matched Single-Step Result

At steps513-528, cluster-cone minus ambient/top-1/output-attenuated half-D0
held-out loss is **-9.116e-6 / +7.644e-6 / +2.941e-6** over14 informative states.
Maximum cluster BF16 budget error is0.0578%; all28 numerical clusters have
dimension2. This is mixed evidence, not a P2GD winner. The approved independent
paths test cumulative behavior, not a larger hyperparameter grid.

[Full results, precision audit and source records](docs/experiments/p2gd_late_controls_20261122.md)

## Previous Result: No Isolated P2GD Gain in the LLM Replay

Across 24 informative states, cluster-cone minus ambient loss is **+0.000003216**,
and cluster-cone minus top-1 is **+0.000001483**, at matched spectral movement.
Output-attenuated half-D0 reproduces **98.6%** of the cluster arm's mean gain
over the original half-D0. Finite-step repair does not improve it. These are
correlated one-step comparisons, not independent training runs.

Local FP64 sphere tests show a geometric advantage over ambient/top-1 on
constructed matched-step problems, but not universal superiority over half-D0.
Keep P2GD as a mechanism candidate; do not promote it to 3B based on this window.

[Six-way table, local trajectories, integrity checks and next decision](docs/experiments/p2gd_local_and_six_results_20261122.md)

## Previous Result: Budget Matching Changes the Comparison

The follow-up **16-update / 15-candidate** budget replay completed in **16:30**
on one H20. These are counterfactuals on one common trajectory, not 15 training
runs. Each loss difference is against half-D0 at the same returned budget:

| Direction | Matched spectral ELR, 100% | Matched local output, 100% |
|---|---:|---:|
| Finite D0 repair | +0.000081403 | -0.000000290 |
| Cluster-cone momentum | -0.000886338 | -0.000114720 |
| Finite-momentum repair | **-0.000898480** | -0.000100044 |

Negative is better. Both momentum-cone variants win **7/7 informative steps**
under spectral matching, but make only **0.34x the independent-input output
movement** and **0.13x the AUS** of half-D0. Under output matching, the gain
shrinks and spectral movement rises to 2.93x. The finite-D0 signal does not
survive equal spectral movement. FP32 matching also leaves up to 6.33% budget
error after BF16 rounding in the spectral case.

**Relative wins are not monotonic descent:** every candidate reduces held-out
loss from before the update in **4/7**, not 7/7, informative steps. For example,
spectral-matched finite momentum beats half-D0 by 0.000898 on average, with a
complete-update mean change of -0.002005; half-D0 changes it by -0.001107.
Output-matched momentum slightly improves held-out loss but worsens the first
training microbatch. Stochastic updates need not decrease validation loss at
every step; these numbers are diagnostics, not a descent or convergence proof.

**Decision:** retain momentum-cone directions for a short controlled test;
do not yet claim a P2GD geometric benefit. Add spectral-matched ambient/top-1
and output-attenuated half-D0 controls at a later checkpoint before long runs.

[Full results, precision audit, figure and six-way next test](docs/experiments/p2gd_budget_matching_20261120.md) ·
[Verified numerical record](docs/results/p2gd_budget_20261120/h20_summary.json)

## Earlier Replay: Equal AUS Leaves Unequal Movement

The new **16-update same-state replay** completed on one H20 in **12:47**.
Eight counterfactual directions share one MCSD-TP/near-half trajectory;
they are not eight independent training runs. Only **seven steps / 12 matrix
components** enter the near region, all with numerical cluster size two.

| Candidate at matched returned AUS | Mean held-out loss difference from half-D0 | Better / worse steps |
|---|---:|---:|
| Ambient momentum | +0.002099895 | 0 / 7 |
| Cluster-cone momentum | +0.002107545 | 0 / 7 |
| Gap-preserving momentum repair | +0.002112242 | 0 / 7 |
| **Gap-preserving D0 repair** | **-0.000012626** | **6 / 1** |

**Interpretation at this stage:** the momentum family makes about **8.4x** larger spectral
and **2.9x** larger local-output moves despite equal AUS. Finite-D0 repair
has a tiny favorable signal but also **5.6% smaller spectral movement**.
Neither result isolates a geometric benefit. The completed budget controls
above address the next amplitude question; do not pool absolute losses across
the two evaluation sets or count the repeated states as new independent cases.

[All eight directions, raw numerical records, figure and next tests](docs/experiments/p2gd_returned_elr_replay_20261119.md) ·
[Verified H20 summary](docs/results/returned_step_repair_20261119/h20_summary.json)

## Earlier 32-Step Confirmation: Actual Step Control

All **13 width-256, 32-update** ELR confirmation arms completed. MCSD-TP with
AUS calibration and a uniform 0.95 coefficient reached **4.550776**, versus
4.561753 with local-output matching and 4.578351 for the common-return
MuonSphere reference. Uncapped spectral matching degraded SSO/MCSD-TP;
AUS/output caps recovered most of the loss. Missing attenuated Muon controls
and later-checkpoint replication prevent a direction-superiority claim.
These short interventions are separate from the 1B/3B results below.

[Full table, curves, angle diagnostics and next tests](docs/experiments/spectral_elr_confirmation_20261119.md) ·
[Numerical record](docs/results/spectral_elr_confirmation_20261119/summary.json)

## Latest 3B Result: Near-Region Step Attenuation

**Width 256 · 28 layers · 3B processed tokens · seed 2345 · LR 0.015 · eval20.**
The fresh matched pair completed 5723 iterations with no logged NaN/skipped
iterations. Each used one H20, warm block-8 K10 with FP32 QR/Ritz, and no
original-matrix full SVD online.

| Method | Final validation loss | Train + validation | Comparison scope |
|---|---:|---:|---|
| **MCSD-TP, near-only half-D0** | **3.423857** | 17:57:09 | Fresh matched pair |
| MCSD-TP control | 3.429645 | 17:56:55 | Fresh matched pair |
| Ambient-radial hybrid, rho=0.5 | 3.432013 | 17:54:03 | Earlier matching-protocol warm run |
| Exact MCSD-TP | 3.430719 | 25:47:49 | Exact-spectrum reference |
| Exact ambient-radial, rho=0.5 | 3.429565 | 25:47:55 | Exact-spectrum reference |
| MuonSphere | 3.436023 | **16:35:03** | Historical same-protocol baseline |
| SSO | 3.435041 | 18:28:11 | Historical same-protocol baseline |

**Interpretation:** the near-only half-D0 arm improves loss by **0.005788**
over its source-matched control, with essentially unchanged time. It keeps
the MCSD-TP direction and halves it only when the cached relative gap is
at most `2e-4`. It is **not PGD or P2GD**. Its near/repeat rates are higher,
so fewer branch calls or faster gap escape are not sufficient success criteria.

The differences from historical MuonSphere / SSO are -0.012166 / -0.011184;
runtime is about 8.3% longer / 2.8% shorter. These are contextual,
single-seed comparisons, not evidence of consistent superiority.

![Validation curves and same-step differences](docs/results/warm_radial_controls_3b_20261117/validation_comparison_20261118.png)

[Full curves, branch counts, precision audit and provenance](docs/experiments/warm_radial_controls_3b_20261117.md) ·
[Machine-readable evidence](docs/results/warm_radial_controls_3b_20261117/final_summary_20261118.json)

## What Is Supported

| Direction | Current evidence | Decision |
|---|---|---|
| Near-only half-D0 | Best current 3B endpoint; one matched seed | Test all-step matrix-half, then paired replication |
| Single-LMO c0.5 | Small control gain at both 1B seeds; not consistently better than MuonSphere | Retain as a mechanism reference |
| 25% shape blend | Best selected seed-2345 1B endpoint; extra gain does not repeat at seed1234 | Do not claim a confirmed blend advantage |
| Ambient-radial switch | Exact half-step slightly helps; warm variant loses to matched control | Keep as a reference, not the leading candidate |
| A2 / K-LMO and full-projection PGD pilots | Current 100M settings underperform | Do not extend these settings to long runs |
| Finite D0 repair + radial | Tiny AUS signal disappears at equal spectral movement | Do not prioritize long training |
| Momentum-cone + radial | Positive same-state spectral-budget result; much lower output movement | Add ambient/top-1 and output-attenuated controls |
| Anchored cone-face + radial | Local algebra checks; matched-AUS LLM replay does not improve over half-D0 on average | Research candidate, not a demonstrated LLM improvement |

[Complete favorable and unfavorable results](docs/experiments/mcsd_p2gd_results_summary.md) ·
[1B second-seed confirmation](docs/experiments/cluster_lmo_confirmation_1b_20261110.md)

## Theory and Local Diagnostics

The [corner/momentum audit](docs/experiments/corner_momentum_audit_20261117.md)
separates raw direction, momentum error and radial-return effects. Its
constructed quadratic losses are not LLM loss measurements.

The new [anchored cone-face note](docs/experiments/anchored_cone_radial_theory_20261118.md)
contains 324 algebra checks and 360 constructed states. On 72 exact-corner
cases with momentum equal to gradient, all anchored updates decrease the
quadratic objective; with cosine alignment 0.5, four increase it. The
conditional target is Clarke stationarity, not B-stationarity. Exact spectral
information, complete cluster coverage and decreasing-error assumptions are
not automatically satisfied by fixed-K10 training.

## Next Tests, Not Submitted

1. Repeat the bounded256-update cluster/top-1/output-attenuation comparison
   from another starting state and fresh evaluation data; keep half-D0 as a
   reference where affordable. No new gap search or longer horizon yet.
2. Sample identical near states along the cluster trajectory to distinguish
   multi-direction correction from altered near-region residence.
3. Offline, test capacity4/6 versus the complete observed cluster, then
   warm-block K10/K20 and cheaper coefficient calibration. Top4 is not enough
   to reproduce all current exact-oracle updates.

MuonSphere/SSO matched short branches and the older MuonSphere coefficient
controls remain missing. Existing full-budget endpoints are not substitutes
for these comparisons. See the [detailed decision](docs/experiments/p2gd_trajectory_panel_20261122.md#next-tests-not-submitted-by-this-update).

After these short checks, the open 3B controls are **all-step matrix-only
half-D0** and a second-seed control/near-half pair. Preserve the AdamW
learning rate; do not halve the global LR flag. No new long jobs are queued.

Only promote a new cone direction after a matched loss probe. Keep width256,
gap2e-4 and the data/evaluation protocol fixed; use exact spectra for the short
geometry oracle and K10 for subsequent warm implementation checks. Defer wider gap grids,
larger widths and 12B expansion. See the [predeclared comparisons](docs/experiments/mcsd_p2gd_results_summary.md#next).

## Implementation and Reproduction

| Component | Code / instructions |
|---|---|
| Original cluster-radial optimizer | [spectral_cluster_radial.py](Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py) |
| Warm FP32 Ritz spectrum | [warm_block_ritz.py](Megatron-LM/emerging_optimizers/orthogonalized_optimizers/warm_block_ritz.py) |
| Current matched direction adapter | [direct_top_pgd_pk.py](scripts/direct_top_pgd_pk.py) |
| Half-D0 regression tests | [test_warm_radial_controls.py](scripts/test_warm_radial_controls.py) |
| Frozen two-arm launcher | [Slurm configuration](slurm/warm_radial_controls_3b_20261117.sbatch) |
| Local anchored-face checks | [local_anchored_cone_radial.py](scripts/local_anchored_cone_radial.py) |
| Environment and data | [H20 runbook](docs/runbooks/cluster_experiments_reproduction.md) |

MuonSphere (`muon_ball_dist`) and SSO (`spectral_ball_dist`) retain their
reference BF16 PI10 hard pre-step normalization. The current MCSD-TP variants
use post-step approximate radial return and cache the transformed trial
spectrum. Radial normalization is not the Frobenius nearest-point projection.

Published evidence includes configurations, endpoint metrics, curves,
checksums, accounting and audit summaries. Datasets, environments, checkpoints,
matrix snapshots and bulk raw-log archives remain outside Git. Some collectors
require these local archives; the published JSON is sufficient to inspect the
reported comparisons, not to reconstruct missing model checkpoints.

Publication recheck on November25:43 replay/trajectory tests and8 local-sphere
tests passed, including
return/restore and new streak/capacity aggregation checks. Five complete raw
trajectories were reaggregated with paired model/optimizer/data verification.
No new CUDA training or long job was launched for publication.

Earlier publication recheck on November22:23 related CPU tests passed, including five
new aggregation checks separating relative wins from pre-update descent.
The expanded budget summary preserves every historical metric and raw-file
checksum; the H20 files were rechecked unchanged. Online accounting and the
empty queue are recorded in the [status record](docs/results/p2gd_budget_20261120/status_check_20261122.json).
The earlier November19 checks (80 CPU tests and 324 anchored-face algebra
cases) remain recorded separately; no new CUDA training ran in this update.

## Attribution

Built on [Spectral Sphere Optimizer](https://github.com/Unakar/Spectral-Sphere-Optimizer),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and the included optimizer
implementations. MuonSphere is called MuonBall in legacy filenames. Preserve
upstream licenses and attribution. The separate Frobenius-geometry study is
not pooled here. Earlier width sweeps and 12B runs remain in the
[historical record](HISTORICAL_README.md).
