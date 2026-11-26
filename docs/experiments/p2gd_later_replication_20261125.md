# Later-Start Replication and Snapshot Audit

**Recovery completed, checked November 26:** a
[separate recovery batch](p2gd_replication_recovery_20261125.md) reuses the
saved step-768 checkpoint. Jobs 4506264 / 4506265 / 4506266 perform the restore
gate, four original short branches and snapshot audit, respectively; all
completed with exit0:0. Cluster retains the lowest loss in the later panel.
The historical failure below remains unchanged, not relabeled as success.

## Historical Status: Prefix Verification Failed

Checked **November 25, 2026 at 11:02 UTC+08**. There are **zero new branch
results**, and no matrix-snapshot audit has run.

| Job | Recorded outcome | Elapsed | Interpretation |
|---|---|---:|---|
| 4504395 | FAILED, exit 1:0 | 1:25:16 | Training reached step 768 and saved; postprocessing failed |
| 4504396_0-3 | CANCELLED before start | 0:00:00 | Prefix success dependency was not satisfied |
| 4504397 | CANCELLED before start | 0:00:00 | Dependent snapshot audit did not run |

**Cause:** the runner called a verifier whose default expects a state audit
at every iteration. This trajectory intentionally audits the first/last
iterations and multiples of 32. Its valid schedule is
`513, 544, 576, 608, 640, 672, 704, 736, 768`, with five sampled parameter
records per audited iteration. The error is in our validation code, not
evidence that the optimizer diverged. The original scheduler status remains
FAILED; it is not relabeled as COMPLETED.

### What Was Actually Verified

The corrected local check establishes:

- All 256 updates and all 8,192 training microbatch token/label hashes match
  the previously completed half-D0 path.
- Initial full model/optimizer signatures and all 45 sampled weight,
  gradient, EMA and Nesterov records match exactly.
- All nine evaluation records, including individual batch losses, match
  exactly. The endpoint is **3.9428776145**, not a new improvement or seed.
- All committed return checks pass; maximum FP32 master radius error is
  **5.34e-7**. There were 496 near-component decisions.
- The frozen H20 source verifies against its original manifest. The saved
  step-768 checkpoint totals about **959.6 MB**, with individual file hashes
  recorded. The post-update model/optimizer signature is present.

**Still unverified:** checkpoint restoration, fresh evaluation-batch
separation, the four later trajectories, capacity/approximation snapshots.
File existence and matching prefix logs do not replace the restore smoke.

The local correction accepts the explicitly declared sparse schedule while
retaining dense update/stream checks and equal sampled-component counts.
Regression tests reject missing scheduled steps, missing component records,
unexpected audit steps and altered states/data/losses. The default dense
verifier used by earlier experiments is unchanged. **49 selected local tests
pass after this correction**; these do not replace the pending GPU restore
test. Corrected repository code
is **not** the originally executed overlay; the frozen H20 files have not
been overwritten. No jobs were resubmitted in this results-only update.

[Scheduler, source and checkpoint evidence](../results/p2gd_replication_20261125/status_20261125.json) |
[Post-hoc prefix verification and input hashes](../results/p2gd_replication_20261125/prefix_verification_20261125.json) |
[Recomputable collector](../../scripts/summarize_p2gd_replication.py)

## Interpretation and Next Decision

The latest **completed comparison** remains the earlier steps513--768 panel:

| Policy | Endpoint loss | Last-four mean | Scheduler time |
|---|---:|---:|---:|
| Half-D0 | 3.942878 | 3.980456 | 1:26:11 |
| Ambient | 3.935439 | 3.966445 | 1:29:51 |
| Top-1 | 3.929235 | 3.964658 | 1:35:15 |
| **Full numerical cluster** | **3.911326** | **3.945195** | 1:53:43 |
| Output-attenuated half-D0 | 3.947058 | 3.966588 | 1:29:50 |

Cluster is lower than top-1 by **0.017909** at the endpoint and **0.019463**
on the last-four mean. Thus the observed ordering is not just one selected
endpoint. Nevertheless, the paths have diverged: near occupancy rises to
8.530% and consecutive near events to 92.4%. This supports a cumulative
mechanism candidate, not an isolated per-call geometric benefit. The earlier
same-state replays were mixed. There is no MuonSphere/SSO arm in this panel,
and these 20-batch losses cannot be pooled with older full-budget tables.

1. **Finish the interrupted replication first.** Freeze a separate recovery
   overlay, recheck saved checkpoint hashes, run only the two-step restore
   smoke, then release the original four arms at most two H20 at a time.
   Reuse step768; do not repeat the already verified 256-prefix updates.
   Keep the cancelled job IDs and record any replacement IDs separately.
2. **Use its snapshots to test capacity and cost.** The completed cluster
   panel had 525/5,503 near calls above top4 (9.54%), but only nine above top6
   (0.16%). Top6 is a worthwhile candidate, not yet an equivalent substitute.
   Compare top4/top6/full returns first; then oracle-warm block8/12, K10/K20.
   A production proposal must also test a persistent approximate cache.
3. **Only after repeatable gains, add matched baselines.** Compare the chosen
   cheaper variant with MuonSphere/SSO under the same checkpoint/data/evaluation
   protocol, and add an independent seed. Retain both measured loss and total
   compute; the exact cluster diagnostic costs 19.4% more than top-1 and
   uses repeated norm SVDs for budget search. This is not a speed improvement.

Do not enlarge the gap grid, select the best intermediate checkpoint, or
launch a new 1B/3B sweep based on the current single-panel signal.

## Original Submission

Submitted November25,2026 at03:06 UTC+08. This is a bounded follow-up to the
[completed five-path panel](p2gd_trajectory_panel_20261122.md), not a1B/3B
experiment or a second initialization seed. Previous completed projects are
unchanged; source is isolated in `mcsd-p2gd-replication-20261125-v1` on H20.

### Planned Queue and Resources

| Job | Work | Dependency / limit |
|---|---|---|
| 4504395 | Continue shared half-D0 state512 to768; save once; restore smoke | One H20, three-hour cap |
| 4504396_0 | Half-D0,769--1024 | Prefix success |
| 4504396_1 | Top-1,769--1024 | Prefix success |
| 4504396_2 | Complete numerical cluster,769--1024; sparse snapshots | Prefix success |
| 4504396_3 | Output-attenuated half-D0,769--1024 | Prefix success |
| 4504397 | Offline capacity and approximate-return audit | All four branches succeed; one H20,30-minute cap |

The four branch tasks each use one GPU with array throttle `%2` and a six-hour
per-task cap. The prefix, array and audit run sequentially by dependency, so
this study uses at most two H20 GPUs concurrently. At preparation, the user
queue was empty, SPG-7-2 had eight idle H20 GPUs and disk had about15GB free.
No data download/copy or automatic cache cleanup is performed.

Historical startup check: prefix4504395 was RUNNING onSPG-7-1, and all16 H20
CUDA/host preflight tests passed. The four branches and audit were waiting
on their success dependencies. Another account job4504355 belongs to the
separate Frobenius-band study and was left unchanged; the two-GPU limit
here applies to this replication, not every chat's combined account usage.

[Submission receipt](../results/p2gd_replication_20261125/replication_submission.json) |
[Frozen source manifest metadata](../results/p2gd_replication_20261125/replay_preparation.json)

The original estimate, before the gate failure, was roughly1.5 hours for the prefix and two
waves of1.5--2-hour branches, plus audit. A rough5--7-hour total excludes
queueing and could increase if near-region occupancy/search work grew. The
scheduler limits are caps, not promised completion times.

## Fixed Protocol and Restore Gate

Width256,28 layers, seed1234, inherited1908-step LR schedule (peak0.015),
Nesterov0.9, gap/cluster2e-4, FP32 exact spectra, BF16 model and exact-norm
radial return. Keep the previous own-state BF16 budget rules unchanged.
The full numerical cluster is not capped at4. Constrained matrix weight
decay remains disabled; auxiliary AdamW behavior is unchanged.

Reconstruct the earlier half-D0 path513--768 from its saved step512 state.
Verify the whole reproduced evaluation curve, training stream and audited
weight/gradient/momentum hashes against that completed path. Save one full
checkpoint at768 and also record the actual post-update model/optimizer
signature. Restore for a two-step smoke769--770 and require exact equality
to that post-save signature before releasing branches. The source checkpoint
is hash-checked and never overwritten.

Each arm independently commits256 updates, or134,217,728 additional training
tokens. Twenty fresh fixed held-out microbatches and one separate calibration
batch use validation skip160; reject token overlap with the preceding panel.
Evaluate before the first update and every32 updates. Report endpoint and
last-four-evaluation mean, not a selected best intermediate point. Initial
model/optimizer states and all training microbatch hashes must match across
arms. After divergence only the matching *rule* is shared, not an identical
absolute displacement or common momentum trajectory.

This checkpoint is later on the same reference trajectory. It is not an
independent seed or full validation benchmark. Numerical-cluster radial
directions are P2GD-type variants, not standard metric-projection P2GD.

## Bounded Snapshot Design

Only the cluster arm saves matrices. Partition the256 updates into four
64-step windows. In each window select at most two distinct components per
category: first/new near entry, consecutive near use, and cluster dimension
above4 (the large-cluster category has priority). A component is sampled at
most once per window. At most24 snapshots and256MiB are permitted; this is
stratified diagnostic sampling, not an unbiased frequency estimate.

Each snapshot contains FP32 X, actual momentum and D0, spectrum/top16 vectors,
calibration activations, original BF16 forward weights and the selected
returned point. Snapshots stay on H20; they are not uploaded to GitHub.
Capture does not change the chosen update or RNG. Runtime includes this
small collection overhead. Branches save no model checkpoints.

## Offline Comparisons

1. On identical X/M/D0, fix the coefficient actually selected by the full
   cluster arm. Compare top4, top6 and the full numerical cluster: omitted
   directions, returned-point error, FP32/BF16 movement, output change,
   momentum work and radius error. No objective is used to choose coefficients.
2. On the same full-cluster trial Y, compare warm blocks8/12 with10/20 QR/Ritz
   iterations against exact `svdvals`: norm/radius error, gap error, returned
   point error and synchronized kernel time. QR/Ritz uses FP32; there is no
   online fallback or change to the four training policies.

The approximate test starts with the **true current-X right subspace**.
It is an optimistic oracle-warm test, not a persistent approximate-cache
trajectory. Timings are single-call diagnostics, not production throughput.
Momentum work and frozen-input output change are not measured LLM loss
descent; the independently evolving training arms supply the loss evidence.

## Implementation and Checks

45 selected local tests passed before packaging, including later-window
integrity, exact post-save restoration, bounded snapshots/RNG preservation,
rectangular block iteration and the previous trajectory regressions.
Every H20 allocation verifies frozen source hashes and runs CUDA/host tests.

- [Snapshot hooks](../../scripts/p2gd_replication.py)
- [Runner and restore gate](../../scripts/run_p2gd_replication.py)
- [Submission](../../scripts/submit_p2gd_replication.py)
- [Offline audit](../../scripts/audit_p2gd_snapshots.py)
- [Tests](../../scripts/test_p2gd_replication.py)
- [Slurm launcher](../../scripts/p2gd_replication.sbatch)

The submission receipt and startup status are separate from results. Do not
promote a method based only on successful job startup or passing unit tests.
