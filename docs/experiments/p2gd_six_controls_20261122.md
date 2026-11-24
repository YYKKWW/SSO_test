# Six-Way P2GD Mechanism Replay

**Submitted November 22, 2026, 17:24 HKT.** Follow-up to the
[spectral/output budget study](p2gd_budget_matching_20261120.md). This is an
authorized one-H20 short replay, not six independent runs or a 3B training job.

**All updates finished; post-hoc integrity checks passed.** Job **4487800**
used one H20 on **SPG-7-2** for 37:16. The training process exited 0 after
step 300. Slurm reports FAILED because the final checker compared 48 steps
of state audits with the prior 16-step log. Comparing their complete common
prefix gives 80 identical records and zero loss difference; checkpoint hashes
are unchanged. The local checker is corrected; training was not rerun.

[Complete results and local sphere tests](p2gd_local_and_six_results_20261122.md) |
[Numeric summary](../results/p2gd_six_20261122/h20_summary.json) |
[Integrity receipt](../results/p2gd_six_20261122/posthoc_integrity.json)

**Result:** 32 measured steps, 24 informative steps and 47 near components,
all cluster size two. Cluster-cone does not beat spectral-matched ambient or
top-1. Output-attenuated half-D0 reproduces 98.6% of its mean gain over the
original half-D0. No new long training is justified by this replay alone.

The account queue was empty before submission, with eight H20 GPUs idle on
SPG-7-2 and about16GB disk free. These are submission-time observations.
The isolated project is `/home/u3013198/projects/mcsd-p2gd-six-20261122-v1`.
The [submission receipt](../results/p2gd_six_20261122/replay_submission.json)
records the frozen source and overlay hashes. All19 selected local regression
checks passed. At elapsed01:33, all **nine CUDA/host checks passed** and the
training launcher had started loading the original checkpoint and data.
At elapsed02:59, step254 had completed with `optimizer_update_successful=true`.
These timestamps describe startup; the final status and results are above.

## Fixed Protocol

Width256, 28 layers, seed1234, base LR0.015 with the existing 1908-step
schedule, Nesterov0.9, gap and cluster windows2e-4. All candidates share the
same FP32 full-spectrum information, safe MCSD-TP components, non-matrix
updates and exact-norm radial return. These are P2GD-type directions with
radial return, not standard metric-projection P2GD.

Load the immutable step252 checkpoint. Replay steps253--268 under the same
near-half MCSD-TP common trajectory; verify its training tokens, losses and
initial optimizer state against job4479762. Then measure steps269--300,
fixed in advance. This obtains later states in memory without saving another
checkpoint. It is not a late-training checkpoint replication.

Use 20 fresh fixed held-out microbatches, one separate calibration microbatch,
and 256 input tokens per linear module; skip the first68 validation batches.
All measured candidates are evaluated at the same state and restored. Only
half-D0 advances the common trajectory. No candidate's loss tunes its step.

## Six Candidates

| Candidate | Matched complete-return quantity | Question |
|---|---|---|
| half-D0 | Reference | Existing near-only shrink control |
| Output-attenuated half-D0 | Output ELR of this component's spectral-matched cluster-cone return | Does smaller output movement explain the gain? |
| Ambient momentum | Spectral ELR of half-D0 | Is changing direction family sufficient? |
| Top-1 momentum repair | Same spectral target | Is a single normal sufficient? |
| Cluster-cone momentum | Same spectral target | Does multi-direction correction add value? |
| Finite-momentum repair | Same spectral target | Does preserving singular-value spacing add value? |

The momentum input before repair is
`Z = -0.5 * ||D0||_F * M / ||M||_F`, inherited from the preceding replay.
Each coefficient is then calibrated using the actual radial return. The
finite-step repair is rebuilt at every coefficient. The attenuation control
matches a per-component output target, not a global0.34 multiplier. Numerical
clusters are not truncated to top4; their full identified capacity is logged.

Matching uses `2e-7 + 2e-4 * target`; unmatchable cases fail explicitly. BF16
forward-weight movement is audited separately. The attenuation control's
BF16 discrepancy is measured against the rounded **cluster-cone** return,
not incorrectly against rounded half-D0.

## Predeclared Comparisons

- Cluster-cone and finite momentum versus ambient, top-1 and half-D0 at equal
  spectral movement.
- Cluster-cone versus output-attenuated half-D0 at equal local-output movement.
- Finite momentum versus cluster-cone; keep the simpler formula if indistinguishable.
- Report both informative-step and all32-step loss differences, before/after
  complete-update losses, first-training-microbatch loss, independent-input
  output movement, BF16 errors, cluster capacity and wall time.

Seven wins over a control would not mean seven decreases from the pre-update
loss. Correlated states and repeated held-out batches are not independent
seeds. No full-budget promotion or stochastic convergence claim follows from
this single experiment. The exact oracle is for mechanism isolation, not
production-speed benchmarking.

## Implementation

[Geometry](../../scripts/p2gd_six_geometry.py) |
[Replay adapter](../../scripts/p2gd_six_replay.py) |
[Runner](../../scripts/run_p2gd_six.py) |
[Tests](../../scripts/test_p2gd_six.py) |
[Summary](../../scripts/summarize_p2gd_six.py)

```bash
PYTHONPATH=scripts:/path/to/frozen/elr/h20_elr:/path/to/frozen/elr \
  python -m unittest test_p2gd_six -v
python scripts/summarize_p2gd_six.py /path/to/results/six_replay/metrics \
  /tmp/six_summary.json
```

The adapter requires the frozen ELR host and existing checkpoint/data. It does
not modify the production optimizer. One H20,80GB host RAM,3-hour scheduler
limit,48 common updates total, no model checkpoint saving, no duplicate job.
