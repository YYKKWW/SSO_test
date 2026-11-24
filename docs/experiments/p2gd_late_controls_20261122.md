# Later-state P2GD mechanism replay

## Status and scope

Single-H20 diagnostic **4487990 COMPLETED**, exit **0:0**, on **SPG-7-2**.
It ran November22, 18:46:33-20:27:55 UTC+08: scheduler time **1:41:22**,
training and diagnostics **1:41:01**. Local validation: **37 tests passed**;
the H20 job also passed **13 CUDA/integrity tests** before training.

### Completed BF16-matched result

**Cluster-cone P2GD has a tiny advantage over ambient, but does not beat top-1
or output-attenuated half-D0.** Matching BF16 movement does not isolate a
multi-direction advantage in this window.

The window513-528 contains **14 informative states** and **28 near matrix
components**, all numerical cluster size2. All14 states remain in the
predeclared <=1% BF16 matching-error subset. Negative differences below favor
P2GD; these are complete-update losses at shared states, not training endpoints.

| P2GD minus control | Mean held-out loss difference | Better / worse states | Mean training-microbatch difference |
|---|---:|---:|---:|
| Ambient | `-0.000009116` | 12 / 2 | `+0.000039220` |
| Top-1 | `+0.000007644` | 5 / 9 | `+0.000006046` |
| Output-attenuated half-D0 | `+0.000002941` | 6 / 8 | `+0.000118988` |
| Unadjusted half-D0 | `-0.000595295` | 13 / 1 | `-0.000335676` |

Every arm lowers held-out loss from its pre-update point in10/14 informative
states, not every state. Relative wins against another update are not a
monotonic-descent or convergence guarantee. States and batches are correlated.

Maximum relative BF16 budget errors: **0.0301% ambient, 0.0295% top-1,
0.0578% cluster, 0.0251% output attenuation**. There were1/1/2/0 violations of
the much stricter search tolerance, respectively, all retained and disclosed;
none exceeds the predeclared1% subset threshold. Maximum candidate master
radius error is `1.79e-7`.

Cluster correction is not simply erased by BF16: its mean correction norm is
3.70% of the ambient direction, the correction has rank2 in25/28 components
and rank1 in3/28, and about8.83% of rounded returned entries differ from ambient
in the affected components. This includes coefficient recalibration as well
as direction repair; it does not alone establish useful geometry. No component
in this window needs capacity above2.

All276 optimizer updates succeeded. All80 prefix state records match, the244
continuation steps passed layout/exact-commit/radius checks, evaluation batches
remained fixed and disjoint from the earlier replay, and the source checkpoint
was unchanged. Local recomputation from downloaded raw records agrees with
the H20 summary after JSON key canonicalization.

[Summary](../results/p2gd_late_20261122/h20_summary.json) |
[Raw losses](../results/p2gd_late_20261122/repair_loss_replay.jsonl) |
[Raw geometry](../results/p2gd_late_20261122/repair_geometry.jsonl) |
[Submission](../results/p2gd_late_20261122/submission.json)

### What the completed earlier replay does and does not show

The following numbers are from **4487800, steps 269-300**, not this later
job. Its 24 informative states were matched in FP32 and only audited after
BF16 rounding. Negative differences favor cluster-cone P2GD.

| Paired contrast | Mean held-out loss difference | Better / worse states |
|---|---:|---:|
| Cluster minus ambient | `+3.216e-6` | 10 / 14 |
| Cluster minus top-1 | `+1.483e-6` | 10 / 14 |
| Cluster minus output-attenuated half-D0 | `-1.225e-5` | 15 / 9 |

Those results do **not** establish a direction advantage: cluster did not
beat ambient/top-1; attenuation alone reproduced 98.6% of its mean benefit
over unadjusted half-D0. Cluster also worsened the training-microbatch loss
relative to the attenuation control by `9.835e-5`. BF16 budget discrepancies
reached 6.89% for cluster and 8.03% for the attenuation control. They cannot
answer whether P2GD wins after directly matching BF16 movement.

The completed later job addresses that precision confound. Its outcome remains
mixed. With the user's subsequent approval, the bounded five-trajectory
[follow-up](p2gd_trajectory_panel_20261122.md) was submitted and has now completed to test cumulative
effects, not because P2GD had established superiority. No1B/3B run was submitted.

- Isolated project: `/home/u3013198/projects/mcsd-p2gd-late-20261122-v1`.
- Source overlay SHA256: `644531334a0d1f21de576fe6b8f861dab6bb125566dfc9509ea51784e79aa9c4`.
- Existing jobs 4487881 and 4487882 were left unchanged; disk had 16 GB free
  before staging. This diagnostic saves no new model checkpoint.

The preceding six-way replay did not isolate a cluster-cone advantage over
ambient or top-1. This follow-up tests whether that conclusion changes later
in training and after directly matching BF16 forward-weight movement.

## Fixed protocol

| Item | Setting |
|---|---|
| Model / data | Width 256, 28 layers, same existing indexed source and 1908-step schedule |
| Initialization | Existing matched checkpoint at step 252; seed 1234 |
| Common path | MCSD-TP outside the near region; half-D0 inside it |
| Spectral information / return | FP32 exact SVD / exact-norm radial return |
| Gap / numerical cluster | Both fixed at `2e-4`; full numerical cluster retained |
| Learning rate / momentum | Inherited peak `0.015` and Nesterov `0.9` |
| Prefix verification | Steps 253-268 against the completed budget replay |
| Common continuation | Steps 269-512, without candidate/loss probes |
| Measurement window | Steps 513-528, four alternative policies at each state |
| Evaluation | 20 fixed held-out microbatches, one separate calibration batch |
| Data separation | Skip 96 validation batches; reject overlap with prior replays |
| Resources | One H20, 12 CPUs, 80 GB host RAM, three-hour maximum |
| Storage | Logs and summaries only; checkpoint saving disabled |

There is no existing step-512 checkpoint. All candidates share one continuation
to that state. They are counterfactual complete updates, not independent
training trajectories. The first 16 steps verify continuity and are excluded
from the new result table. Exact spectral computations are a diagnostic
reference, not a production-speed proposal.

The inherited data prefix is the existing 3B indexed source, as in the prior
replays; `1B` denotes the scheduler horizon, not a newly downloaded dataset.
This job stops at iteration 528 and does not train for either 1B or 3B tokens.

## Four controls plus common reference

Let `D0` be the shared MCSD-TP descent direction and `Z` the normalized negative
momentum scaled to half the Frobenius norm of `D0`. All candidates use the same
radial return. Away from the near region, their updates are identical.

| Label | Near-region direction | Budget |
|---|---|---|
| Reference | Half-D0 | Original step |
| Output-attenuated half-D0 | Half-D0 with adjusted scale | Match cluster arm's BF16 local-output ELR |
| Ambient | Z, no correction | Match reference BF16 spectral ELR |
| Top-1 | Z minus its leading-normal component | Same BF16 spectral ELR |
| Cluster cone | Full numerical-top-cluster cone-style correction of Z | Same BF16 spectral ELR |

The cluster rule is an approximate geometric model for near-equal singular
values; it is not the exact tangent cone unless the selected singular values
are genuinely equal. No convergence conclusion is inferred from this replay.

## BF16 matching and mechanism audit

Matching uses the complete displacement of rounded forward weights, relative
to the actual pre-update BF16 weights. The committed master parameter stays
FP32 and receives the exact radial return. Spectral ELR uses the matrix
spectral norm; output ELR uses frozen calibration activations. Held-out loss
does not participate in scale selection.

Quantization makes some target displacements unattainable. The bounded search
records its best observed candidate and marks it unmatched when outside
`2e-7 + 2e-4 * target`; it never silently accepts an infeasible match. Report:

- All informative states, including mismatches.
- A predeclared subset where every relevant component has relative BF16
  budget error at most 1%, without selecting on loss.
- Correction norm relative to the ambient direction and small-block rank.
- Momentum/cone angles, top-block eigenvalues, and required cluster capacity.
- BF16 rounding displacement and changed-entry fraction versus ambient.
- Complete-update held-out loss, not just a direction-gradient inner product.

States along one path and held-out microbatches are correlated. Their counts
are not independent training seeds, and different validation windows must not
be pooled by absolute loss.

## Decision gate

Only consider 128-256-step independent branches if cluster correction beats
ambient, top-1, and output-attenuated half-D0 at comparable returned budgets,
with a non-negligible correction surviving BF16 rounding. Report the number
of informative/matched states and the distribution of paired differences.
If evidence is sparse or mixed, collect another short window instead of
expanding the hyperparameter grid or escalating to 1B/3B.

## Implementation

- [BF16 budget matching and directions](../../scripts/p2gd_late_geometry.py)
- [Common continuation and replay](../../scripts/p2gd_late_replay.py)
- [Integrity checks](../../scripts/verify_p2gd_late.py)
- [Summary and paired comparisons](../../scripts/summarize_p2gd_late.py)
- [Runner](../../scripts/run_p2gd_late.py) and [one-GPU job](../../scripts/p2gd_late.sbatch)
- [Local/CUDA regression tests](../../scripts/test_p2gd_late.py)

These scripts use the frozen diagnostic host from the earlier H20 confirmation
project. They are study overlays, not standalone LLM training entry points.
