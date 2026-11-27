# Results Update: November 27, 2026

## What Is New

- H20 is reachable again. The requested three-arm warm P2GD 3B transfer was
  **not submitted during the outage** and has no new validation result yet.
  It is now submitted as **4519937_0--2**, at most two concurrent H20s.
- The separate six-arm common-AUS diagnostic batch **4514694 / 4514695_0--1**
  completed successfully. These are 64-update paths, **not P2GD or 3B tests**.
- The latest completed direct P2GD evidence remains the BF16-matched
  step768-start panel. Its positive result does not establish 3B superiority.

## 1. Completed P2GD Evidence

Width256, seed1234, common step768 model/optimizer state, 256 updates, 20
fixed held-out microbatches. Exact spectral diagnostics and per-step BF16
budget calibration make these research measurements expensive.

| Near policy | Final held-out loss | Last-four mean | H20 elapsed |
|---|---:|---:|---:|
| Half-D0 | 3.825426 | 3.843236 | 1:25:14 |
| Top-1 | 3.800925 | 3.823121 | 1:50:58 |
| **Numerical-cluster P2GD-type** | **3.791397** | **3.817367** | 2:17:50 |
| Output-matched half-D0 | 3.812997 | 3.838130 | 1:37:23 |

Cluster improves final loss by 0.009528 versus top-1 and 0.021600 versus
output-matched half-D0, and is lower at all eight post-update evaluations.
It costs 24.2% more than top-1. Cluster/top-1 match a BF16 spectral-step
budget at each path's own state; the output control matches a different
output metric. Paths are not forced to share absolute displacements after
divergence. Cluster's maximum BF16 relative matching error is 0.471%, and
639/9508 calls miss the stricter internal search tolerance.

Top6 covers the clusters observed in this panel, but an earlier panel had
size7 events. Two starting points of the same seed are not independent-seed
replication. These results support a production transfer test, not a claim
that P2GD has beaten MuonSphere/SSO over a full budget.

[Full P2GD results and raw evidence](p2gd_replication_recovery_20261125.md)

## 2. New Common-Angular-Step Diagnostics

All six paths restore the same step252 checkpoint and use the same 64
training batches (updates253--316), width256, seed1234 and 20 fixed held-out
microbatches. Each processes 33,554,432 tokens using the 1B schedule horizon
on the existing 3B data source. **This is not 3B training.**

The matching target is the reference MuonSphere trajectory's per-component,
per-update **FP32-master angular update size (AUS)**:

```text
AUS(X, Xnext) = ||Xnext / ||Xnext||_F - X / ||X||_F||_F
```

Multipliers 1.000, 0.985 and 0.950 apply to this target, not to the learning
rate directly. MCSD-TP uses the same reference trace, while its direction
and momentum evolve on its own trajectory. Radial return is shared.

| Direction | AUS multiplier | Final held-out loss | Last-16 mean | Process time |
|---|---:|---:|---:|---:|
| MuonSphere | 1.000 | 4.499633 | 4.518949 | 31:37 |
| MuonSphere | 0.985 | 4.483395 | 4.515647 | 35:28 |
| **MuonSphere** | **0.950** | **4.476445** | **4.508469** | 35:16 |
| MCSD-TP | 1.000 | 4.502379 | 4.530159 | 35:08 |
| MCSD-TP | 0.985 | 4.500671 | 4.528250 | 35:25 |
| MCSD-TP | 0.950 | 4.511190 | 4.513379 | 35:25 |

Process times include measurement overhead. Scheduler times are 34:48 for
the reference/gate job, 1:11:28 for lane0 and 1:46:44 for lane1; each lane
runs multiple paths sequentially. Do not assign a lane's total to every arm.

**Interpretation:**

1. A 5% smaller angular target improves MuonSphere in this early window:
   final loss changes by -0.023188 and the last-16 mean by -0.010480.
   A modest step-size change alone can matter substantially.
2. MCSD-TP does not beat MuonSphere at any of the three matched master-AUS
   targets. At 1.000, MCSD-TP minus MuonSphere is +0.002746 at the endpoint
   and +0.011211 over the last16 updates.
3. MCSD-TP/0.950 has a better last16 mean than MCSD-TP/1.000, but a worse
   endpoint. This mixed short-window behavior does not establish an optimal
   coefficient or justify selecting a favorable intermediate evaluation.
4. There is **no P2GD branch here**. These results motivate keeping ambient,
   top-1 and shrinkage controls; they neither prove nor refute the cluster
   benefit seen in the separate later-state panel.

The collector independently verifies the logged initial weight/gradient/
Nesterov hashes and optimizer samples, all 64 training-stream hashes, all
20 held-out token/label identities, and 16,128 component targets per arm.
The maximum master-AUS relative error is 4.43e-7. BF16 AUS averages about
0.36%--0.40% above master AUS; **BF16 spectral displacement and output ELR
are not matched by this protocol**. This is one seed and one early checkpoint,
not a significance test or a full-validation benchmark. Absolute losses
must not be ranked against the later P2GD panel or the completed 3B ledger.

[Verified summary](../results/elr_trace_20261127/summary.json) |
[Minimal raw evidence](../results/elr_trace_20261127/evidence.tar.gz) |
[Standard-library collector](../../scripts/summarize_elr_trace.py)

```bash
mkdir -p /tmp/elr-trace-evidence
tar -xzf docs/results/elr_trace_20261127/evidence.tar.gz -C /tmp/elr-trace-evidence
python scripts/summarize_elr_trace.py /tmp/elr-trace-evidence/results /tmp/elr-trace-summary.json
```

## 3. Requested 3B Transfer: Now Submitted

Submitted **November 27 at 12:11:54 UTC+08** after network recovery.

| Job | Near policy | Initial status |
|---|---|---|
| 4519937_0 | Numerical-cluster repair, top<=6 | CUDA/smoke passed; main step10/5723 logged, no skipped/NaN iteration |
| 4519937_1 | Top-1 repair | CUDA/smoke passed; main step10/5723 logged, no skipped/NaN iteration |
| 4519937_2 | Ambient, no correction | Pending behind the two-job concurrency cap |

Width256, seed2345, 5723 updates, LR0.015, gap2e-4, persistent warm block8/K10
and FP32 Ritz; all use approximate radial return. The common raw-momentum
coefficient **0.075** was calibrated from the first32 updates of the previous
diagnostic panel, with no online coefficient search. It does **not** guarantee
equal BF16 returned steps. Checkpoints are disabled; sparse offline audits
record actual movement, gap/radius error and missing cluster capacity.

Persistent warm block8/K10 already has 3B evidence. The new question is the
small-step top6 repair, not whether this spectral backend has ever run.
Completed references are reused, not resubmitted:

| Completed 3B reference, seed2345 | Final loss |
|---|---:|
| Current near-only half-D0 | **3.423857** |
| Current warm MCSD-TP | 3.429645 |
| Current ambient-radial, rho0.5 | 3.432013 |
| Historical MuonSphere | 3.436023 |
| Historical SSO | 3.435041 |

Historical MuonSphere/SSO are contextual, not fresh source-matched arms.
There are no new 3B losses yet. Rough expected runtime is 18--20h per arm,
excluding queueing; three arms require two waves with the concurrency cap.
The first logged segment is about11.1s/update for both arms. Near occupancy
is still zero during early warmup; identical early loss does not compare the
new near policies. Small-Ritz calls equal trial solves plus bootstrap calls
in the current adapter counters; there is no observed full-SVD fallback.

[Full 3B protocol](warm_p2gd_smallstep_3b_20261126.md) |
[Submission receipt](../results/warm_p2gd_smallstep_3b_20261126/submission.json) |
[Startup check at12:17 UTC+08](../results/warm_p2gd_smallstep_3b_20261126/status_20261127.json) |
[Baseline configuration comparison](../results/warm_p2gd_smallstep_3b_20261126/config_comparison.json)

## Next Decisions

1. **Finish this batch first.** No additional gap grid, width or token-budget
   sweep. Compare endpoint, final evaluation tail and wall time against both
   new top-1/ambient and the completed half-D0 control.
2. **Separate geometry from realized step size.** If cluster wins but moves
   much less in the BF16 snapshot audits, run a short matched-step replay at
   a late checkpoint before attributing the gain to multidirectional repair.
   The cheap 3B coefficient is not a replacement for the earlier exact audit.
3. **Diagnose capacity/accuracy before scaling.** Report top>6 and saturated
   block8 events, missed near cases and radius drift. More capacity or a
   different estimator is justified only by measured failures, not by a
   larger gap threshold chosen to force more branch use.
4. **Replicate only a convincing winner.** If the cluster beats both simple
   directions and remains competitive with half-D0, repeat the paired cluster
   and strongest simple control on a second seed. Otherwise keep the simpler
   update and use saved snapshots to isolate the failed mechanism.

High near-branch repetition is not itself failure. Neither approximate
numerical-cluster radial updates nor these empirical gains establish the
stochastic convergence theorem of exact projection P2GD.
