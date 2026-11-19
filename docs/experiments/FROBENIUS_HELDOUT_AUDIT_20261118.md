# Same-State Held-Out Direction Audit

Submitted November 18, 2026, 11:19 HKT. **Both jobs completed.** Verified
November 19 at 15:17 HKT. The gate took 00:01:44 and the 9,000-step
trajectory took 14:34:50 (Slurm elapsed), both exit 0:0.
[Final paired results, limitations and next tests](FROBENIUS_RESULTS_20261119.md#3-completed-same-state-held-out-audit).

| Stage | Job | Stop / audit steps |
|---|---:|---|
| H20 CUDA gate | 4464031 | Stop at step 6; audit before update 2 |
| MuonH trajectory | 4464032 | Stop after update 9,000; audit before updates 3,000 / 6,000 / 9,000 |

The trajectory uses width 256, seed 1234, beta=0.95 Nesterov, Frobenius
shape radii, and the same model, data order, auxiliary AdamW and full
11,445-step LR horizon as MuonH reference 4427413. Nominal peak LR is
0.010. This is one observational MuonH trajectory, not four full training
runs. One H20 is requested; the long job requires the short gate to pass.
No checkpoints are written. Existing jobs and runtime trees are untouched.

## Same-State Comparison

At each audit step, after gradients are prepared and clipped but before
**any** child optimizer in the chain steps, build candidates from the
same FP32 master weight, current gradient G, and updated Nesterov input S:

| Source | LMO input without projection | LMO input with projection |
|---|---|---|
| Current batch | G | P_W G |
| Nesterov | S | P_W S |

Every candidate uses the same practical BF16 Polar Express kernel, then
Frobenius tangent projection and unit-F direction. Target the common
per-matrix chord from MuonH 4427413 at this step using the sphere's exact
angular formula. Zero tangent rays are recorded as stalls, not as if the
target chord were reached. Check each nonzero FP32 chord within `2e-6`.

Audit seven logical matrix types (Q, K, V, O, MLP gate, up, down) in layers
0, 14 and 27. Perturb one matrix at a time, holding the rest of the model
fixed. This is 21 blocks x 4 candidate rules = 84 records per audit step,
with two held-out microbatch losses per candidate. These are paired
observations, not 168 independent model replications.

## Fixed Diagnostic Data

The microbatches are four sequences of length 2,049, interpreted as two
batches of two 2,048-token predictions. They come from appended 6B source
positions 3,001,012,394; 3,002,012,394; 3,003,012,394; and 3,004,012,394,
outside this trajectory's 3B training and final-validation source. The
original 3B index identity and 6B payload size were checked against the
extension manifest. No new dataset download or large index is required.

Payload SHA256:
`102579a1c9f39e409f9fb343c196e6868c9f5f5ebf1a8b7e22274f1065bdb2c2`.
These early appended positions are DCLM diagnostic samples, not a
representative validation benchmark. They are fixed before the outcomes
are read; they must not replace final reported validation loss.

## Numerical and Trajectory Controls

- Save and restore the exact model-weight rows in a `finally` block.
  Check that the BF16 model rows initially equal the cast of the FP32
  master block, detecting an incorrectly ordered audit.
- Build hypothetical EMA and Nesterov inputs on clones. Do not update
  the stored momentum or training gradients during the audit.
- Preserve CPU/CUDA RNG and restore all module training flags. Use
  no-backward evaluation, without advancing train or validation iterators.
- Repeat the unmodified forward and recheck it after every candidate.
  Reject a discrepancy above `1e-6` rather than accepting a corrupted audit.
- Log both the FP32 master chord and the **effective BF16 model chord**.
  Master chords match exactly as in the previous AUS experiment; BF16
  rounding may make the two effective forward-model displacements differ.
  A paired loss advantage must be interpreted alongside that discrepancy
  and the repeated-forward numerical floor, not treated as exact equality
  of BF16 chords.

The local suite passed 253 checks, including restoration on exceptions
and preservation of weights, gradients, momentum, training flags, and
RNG in the audit component. Full H20 execution passed: 252 candidate
records at steps 3,000 / 6,000 / 9,000, no stalls, no skipped or NaN
updates. All 45 logged validation values reproduce reference 4427413
at printed precision. The maximum master-chord error was 9.314e-10;
effective BF16 chord differences remain visible in the released records.

## Readout and Decision

For each step, source, matrix type and sampled layer, report
`loss(projected input) - loss(ambient input)` on the **same** microbatch.
Negative favors pre-projection. Keep the two batches and the block labels
visible; include ties and loss changes relative to the unmodified model.
Record training-gradient linear predictions separately rather than using
them as the held-out outcome.

The main question is whether the paired advantage changes between steps
3,000 and 9,000 on the same reference trajectory. This is a local
direction diagnostic, not a comparison between training trajectories or
a proof that one optimizer converges faster. The independent no-momentum
pair 4459729/4459730 also completed; its endpoint and the predeclared
six-point late-window comparison are in the final readout.

The completed audit has mixed signs across states and blocks. Do not
select a projection policy from a favorable isolated microbatch. The
next priority is the full-budget Nesterov input ablation, followed by
broader paired diagnostics and a seed-specific replication if warranted.
