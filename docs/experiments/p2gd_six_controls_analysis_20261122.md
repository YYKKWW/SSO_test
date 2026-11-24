# Six-Control Replay: Analysis and Next-Experiment Decisions

**Historical plan, written November22.** The six-way replay, later-state
BF16 replay and independent trajectories are now complete. See the
[November25 results and current decisions](p2gd_trajectory_panel_20261122.md).
The scheduling statements below describe the initial six-way plan, not the
subsequent separately approved step512 continuation.

**Results:** see the [completed analysis](p2gd_local_and_six_results_20261122.md).
This page preserves the predeclared comparisons. The GPU updates completed;
the postprocessing-only failure was diagnosed and independently verified.

This companion to the [execution protocol](p2gd_six_controls_20261122.md)
organizes interpretation before reviewing the six-way outcomes. It does not
launch another job. The shared submission is **4487800**, one H20 on
SPG-7-2, with a three-hour scheduler limit and no checkpoint saving.

## What Is Actually Being Tested?

Start from the existing width-256, step-252 checkpoint and optimizer state.
Reproduce the previous 16 common updates, then evaluate six counterfactual
full updates at each of steps **269--300**. Only the common near-half-D0
trajectory advances; candidate weights are restored after evaluation.
This is 32 repeated single-step comparisons, not six 32-step branch runs.
It is later than the preceding replay, but not a late-training checkpoint.
The initially considered step-512 continuation is **not submitted**.

Keep the existing seed, data order, learning-rate schedule, Nesterov momentum,
gap threshold, cluster window and radial return fixed. Exact FP32 spectral
information isolates direction design from approximate-spectrum errors.
All six candidates share safe-region and non-matrix updates. Changes apply
only to the identified near-region components.

| ID | Candidate | Complete-return matching target |
|---|---|---|
| A | Half-D0 | Reference: half the MCSD-TP direction in the near region |
| B | Output-matched half-D0 | This component's output movement under E |
| C | Ambient normalized momentum | A's spectral displacement |
| D | Top-1 momentum correction | A's spectral displacement |
| E | Original cluster-cone correction | A's spectral displacement |
| F | Finite-step, gap-preserving correction | A's spectral displacement |

Here D0 includes the descent sign. C uses radial return, so it is **not PGD**.
E and F are numerical-cluster variants, not automatically exact tangent-cone
methods at a multiple singular value. The coefficient search matches
geometry only; it does not inspect loss or perform a loss line search.

## Primary Comparisons

| Comparison | Question | Evidence that would justify further testing |
|---|---|---|
| E versus C and D | Does multi-direction correction help beyond ambient/top-1? | Lower paired held-out loss against both, without a large BF16 budget mismatch explaining the difference |
| E versus B | Is the effect more than output attenuation of D0? | Lower paired held-out loss after matching local output movement |
| F versus E | Does preserving spectral spacing add value? | A repeatable improvement sufficient to justify the extra coefficient-dependent direction computation |
| B versus A | Does simply reducing output movement explain the previous gain? | A comparable improvement to E would favor a simpler amplitude-control explanation |

Report every comparison, including negative or mixed outcomes. A win over A
alone is insufficient evidence for multi-direction geometry. If C or D is
as good as E, retain that simpler control instead of attributing the result
to P2GD. If E and F are indistinguishable, prefer E for the next diagnostic.

## Required Report

1. **Full-return loss:** candidate loss minus control loss, and candidate
   loss minus the pre-update loss. Report held-out and training-microbatch
   quantities separately. Include all 32 steps and the subset with at least
   one near component; zero-effect steps must not disappear from the report.
2. **Actual action:** spectral ELR, AUS, Frobenius displacement, calibration
   output ELR, independent-input output ELR and effective coefficient. Report
   FP32 master-weight and BF16 forward-weight movement separately.
3. **Geometry and capacity:** gap, matrix/layer identity, full identified
   cluster dimension, counts above top-4, and radial-return correction.
   A numerical cluster size is not proof of exact singular-value multiplicity.
4. **Integrity:** source/checkpoint hashes, initial optimizer-state match,
   shared-prefix replay, token pairing, weight restoration, successful
   updates, finite values, matching failures and radius error.
5. **Cost:** total diagnostic time and coefficient-search/SVD work, clearly
   separated from any production-speed claim. This oracle replay is not a
   cheap deployable optimizer benchmark.

Use `comparisons` in the six-way summary for E-versus-B. Generic per-method
differences versus `half_d0` are reference differences, not evidence that B
and A have matched output budgets. B's matching target is E, not A.

Output matching uses frozen local inputs H and the complete matrix return:
`||Delta W H||_F / ||W H||_F`. It is not equality of the full network's output
change: upstream activations can change under the complete model update.
The independent-input metric and actual held-out model loss test whether the
local calibration transfers.

## Interpretation Limits and Follow-Up

- Consecutive states and reused evaluation batches are correlated; do not
  treat 32 states, 20 batches or matrix components as independent seeds.
  Absolute full-update descent also includes shared safe/non-matrix updates
  and does not prove descent of the near branch by itself.
- If restoration, data pairing or prefix checks fail, repair and repeat the
  same protocol. Do not interpret partial failed data as algorithm evidence.
- If few near events occur or BF16 matching differs materially, call the
  mechanism unresolved. Do not tune the gap using these outcomes just to
  manufacture more favorable events.
- If E survives the simpler controls, next repeat the selected paired
  contrasts at a genuinely later checkpoint. Only then run independently
  evolving short branches from the same checkpoint, batches and optimizer
  state; candidate-specific momentum histories must then evolve normally.
- Test warm approximate spectral information only after an oracle-direction
  effect survives these controls. No automatic 1B/3B promotion, new width
  sweep, or convergence claim follows from this single replay.

The six-arm geometry, return matching, finite-step reconstruction and state
restoration passed the local 19-test suite on November 22. Live H20 status
belongs in the execution receipt, not in the scientific conclusions above.
