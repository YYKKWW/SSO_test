# P2GD Later-Start Recovery

## Scope

This is the authorized recovery of the [interrupted replication](p2gd_later_replication_20261125.md),
not another parameter sweep or a 1B/3B run. The failed source project and its
original job states are retained unchanged. A separately frozen recovery
project reuses the saved step-768 checkpoint and prefix logs through links;
it does not replay steps 513--768 or save new training checkpoints.

Submitted **November 25, 2026 at 11:29 UTC+08**:

| Job | Work | Resource/dependency |
|---|---|---|
| 4506264 | Restore checkpoint; verify updates 769--770 | One H20; 45-minute cap |
| 4506265_0 | Half-D0, updates 769--1024 | Restore succeeds |
| 4506265_1 | Top-1, updates 769--1024 | Restore succeeds |
| 4506265_2 | Full numerical cluster, updates 769--1024 | Restore succeeds |
| 4506265_3 | Output-matched half-D0, updates 769--1024 | Restore succeeds |
| 4506266 | Top-4/top-6/full and warm-block snapshot audit | All four branches succeed |

The array is limited to **two simultaneous one-H20 jobs**. The audit uses
one H20 for at most 30 minutes. The unrelated Frobenius-band task is not
modified. At preparation the account had about 14 GB free; the recovery
creates no additional checkpoint or dataset copy.

## Fixed Comparison

Keep width 256, 28 layers, seed 1234, gap/cluster window `2e-4`, Nesterov 0.9,
the inherited 1908-step learning-rate schedule (peak `0.015`), exact FP32
spectra and exact-norm radial return. Constrained matrix weight decay stays
disabled. All four paths start from identical model and optimizer states
and consume the same training batches. Each path commits 256 updates
(134,217,728 training tokens).

The BF16 step-matching rule is unchanged. Top-1/cluster candidates
match the reference spectral displacement at their own current state;
the output-matched half-D0 control matches the cluster candidate's output
displacement. Diverged trajectories do not have identical absolute steps.
Only 20 fresh fixed held-out batches are used; these are not full validation
or directly comparable to older 1B/3B tables.

The numerical cluster is uncapped during training. Cluster snapshots are
bounded by 24 matrices and 256 MiB. The dependent offline audit compares
top-4, top-6 and full-cluster returns at the same selected coefficient, then
oracle-warm block-8/12 with K10/K20 against exact spectra. The approximate
audit does not change the training policy and is not a persistent-cache
training result. No new MuonSphere/SSO arm is launched yet.

## Recovery Safeguards

- Verify every original source-file hash and the previously recorded
  checkpoint file set, sizes and SHA256 hashes before staging.
- Preserve the source checkpoint and prefix; never replace a staged or
  partially run recovery directory.
- Recheck all 256 prefix updates, training streams, evaluations and the
  declared sparse state-audit schedule against the completed reference.
- Run two restored updates; compare the initial full model/optimizer
  signature with the recorded post-save signature at step 768.
- Check fresh held-out/calibration batches do not overlap the prior panel.
- Release branches only when the restore job succeeds. Each branch checks
  the shared checkpoint before and after training and verifies pairing.

Local validation: **56 tests passed**, including recovery idempotency,
changed-checkpoint rejection, dependency/concurrency settings, sparse audit
coverage and prior geometry/trajectory tests. CUDA/host tests run again on
each H20 allocation. Unit-test success is not a restored-checkpoint proof
or evidence of algorithmic improvement.

## Status and Decision

The two-update restore job **4506264 COMPLETED**, exit `0:0`, in **1:35**.
Its gate **passed**: full model/optimizer signatures
exactly match the recorded post-save state, the 45 reference state audits
match, and the new evaluation batches are disjoint. Restored training
finished at 11:30 UTC+08 with no skipped/NaN iterations.

Live check after the gate: **4506265_0 (half-D0)** and **4506265_1 (top-1)**
are RUNNING on SPG-7-1. Arms 2--3 wait on the two-job array limit; the audit
waits on all four branches. Both running arms have committed update 769;
FP32 radius error is below `2.25e-7`, and top-1 BF16 budget error is 0.0189%.
These are startup checks, not comparative performance results.
No final branch comparison is available yet.
The independent restoration proof is recorded in
[PREFIX_READY.json](../results/p2gd_replication_20261125_recovery/PREFIX_READY.json).

Report endpoint and last-four-evaluation mean, actual BF16 matching error,
near occupancy/repetition, required cluster capacity and total runtime.
Only a repeated advantage over top-1 and output-matched shrinkage supports
moving to a cheaper persistent-cache implementation and matched baselines.
Do not widen the gap grid or promote a diagnostic to a 3B result.

[Submission receipt](../results/p2gd_replication_20261125_recovery/replication_submission.json) |
[Recovery provenance](../results/p2gd_replication_20261125_recovery/recovery_preparation.json) |
[Checkpoint hashes](../results/p2gd_replication_20261125_recovery/common_checkpoint_manifest.json)

Implementation: [staging](../../scripts/prepare_p2gd_recovery.py),
[restore/branch runner](../../scripts/run_p2gd_replication.py),
[submission](../../scripts/submit_p2gd_replication.py),
[regression tests](../../scripts/test_p2gd_recovery.py).
