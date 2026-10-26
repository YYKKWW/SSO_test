# Normalized momentum MCSD P2GD radial pilots

This four-arm width-256 pilot tests the requested mixed algorithm:
normalize the effective momentum in Frobenius norm, project it onto the
selected top-cluster cone model, and use the same approximate radial return
as the MCSD branch. There is no MCSD length matching, normalization after
cone projection, online accuracy gate, or original-matrix SVD fallback.

## Completed results

All four tasks completed with exit0, no logged NaN or skipped iterations,
and completed offline audits. The last task ended on2026-10-21 at19:05:36
HKT. This array no longer occupies any GPUs. No new jobs were submitted
or existing jobs canceled during this result check.

All losses below are final validation at191 iterations. Branch statistics
are cumulative through the last training log at190, with47880 decisions
including252 initial zero-LR/bootstrap calls. They are component decisions,
not a percentage of training iterations. Repeat percentage is the fraction
of cluster calls continuing a previous cluster call for the same component.

| Job suffix | K | Cluster tolerance | Final val | Cluster % | Repeat % | Job elapsed | Training elapsed | Last s/step |
|---|---:|---:|---:|---:|---:|---|---|---:|
| _0 | 10 | 2e-3 | 4.975846 | 4.482 | 90.40 | 37:07 | 36:22 | 10.797 |
| _1 | 10 | 5e-5 | 4.978305 | 4.860 | 92.05 | 37:00 | 36:14 | 10.767 |
| _2 | 20 | 2e-3 | **4.969451** | 4.680 | 90.81 | 38:09 | 37:35 | 11.196 |
| _3 | 20 | 5e-5 | 4.971009 | 4.839 | 91.37 | 38:14 | 37:41 | 11.215 |

The best new arm is0.022390 ABOVE MuonSphere,0.012581 above SSO,
0.022280 above its K20 MCSD-TP control, and0.008992 above the previous K20
matched-length broad-cluster hybrid. All four new arms are worse than their
own K-matched controls and broad-cluster historical hybrid references.
These one-seed100M pilots do not warrant promotion to1B/3B in their present
form; this is not a general impossibility result for normalized P2GD.

### Branch persistence and capacity

| Suffix | Cluster calls | Burst starts | Calls per burst start | q1/q2/q3/q4 calls | Cap overflow | Step190 cluster % |
|---|---:|---:|---:|---|---:|---:|
| _0 | 2146 | 206 | 10.42 | 0/415/402/1329 | 249 | 31.75 |
| _1 | 2327 | 185 | 12.58 | 0/2123/204/0 | 0 | 34.52 |
| _2 | 2241 | 206 | 10.88 | 0/540/621/1080 | 134 | 38.49 |
| _3 | 2317 | 200 | 11.59 | 0/2154/163/0 | 0 | 36.51 |

Calls per burst start includes bursts still active at the logging boundary;
it is not an estimate from independently completed episodes. Compared with
the earlier matched-length hybrids (cluster1.35--1.45%, repeat61--62%),
normalized momentum is associated with substantially more persistent
near-region visits. A low overall branch percentage hides a late-stage
increase to32--38% of components at the last logged step. Tight clustering
removes detected top-four overflows but does not improve final loss or
eliminate repeated routing.

### Precision and computational checks

All four runs counted48132 small Ritz SVDs for48132 component calls, with
zero original-matrix full-SVD and svdvals calls INSIDE optimizer updates.
Full decompositions remain confined to tests and post-training audits.

| Suffix | Audit samples | Mean absolute gap error | Maximum absolute radius-relative error | Mean relative direction error |
|---|---:|---:|---:|---:|
| _0 | 24 | 3.02e-4 | 1.49e-3 | 2.48% |
| _1 | 24 | 2.28e-4 | 1.15e-3 | 2.57% |
| _2 | 24 | 1.64e-5 | 1.25e-5 | 1.54% |
| _3 | 24 | 1.77e-5 | 3.10e-5 | 1.61% |

Each audit selected shapes/steps and early cluster calls, not a random
population. No false-safe/false-cluster event was found in these24 samples
per run, which does not certify all training decisions. Sampled cluster
direction Frobenius norms were approximately1, as intended. The oracle
direction comparison uses the same near-cluster rule with exact spectral
information; it is not a comparison against the true tangent cone when
the selected roots are merely close rather than exactly multiple.

K20 improves sampled spectral accuracy and final loss relative to K10,
but does not remove the hybrid's regression. The result therefore does not
support attributing the entire problem to approximate gap estimates.
The normalization changes direction scale relative to spectral-unit MCSD;
small effective spectral displacement is a plausible contributor to longer
residence, but these logs alone do not establish causality. First compare
actual displacement norms and post-return gaps on the SAME stored states
before another training-scale sweep. Do not compensate by automatically
expanding the routing threshold or by silently restoring length matching.

Local logs, launch configs and audits:
`output/cluster_momentum_fro_h20_20261021/results/`.
Machine-readable extraction: `output/cluster_momentum_fro_h20_20261021/final_summary.json`.
Historical reference results and their limitations are recorded below.

## Direction and return

Only the small-gap branch changes. With the effective Nesterov direction M,

    Z = -M / max(||M||_F, 1e-12)
    A = sym(U.T @ Z @ V)
    B = O @ diag(0, min(a2, 0), ..., min(aq, 0)) @ O.T
    D = Z + U @ (B - A) @ V.T
    Y = X + effective_lr * D

The momentum buffer is not changed. Eigenvalues ai are ordered descending.
Both branches estimate Y's spectrum using cached warm block-8 FP32 Ritz
iteration, then set X_next=R*Y/estimated_sigma1(Y). The next cache stores the
scaled approximate singular values and paired vectors of the returned point.
No standalone momentum spectral norm is computed. The cluster branch no
longer computes an unused matrix-sign MCSD length reference.

The safe branch remains tangent projection, BF16 NS8 matrix sign, and a
second tangent projection, exactly as in the existing MCSD-TP control.
The matrix-wise spectral-muP effective learning-rate scaling is unchanged.
The extra small-gap coefficient is 1.0; there is no additional matching
factor. Frobenius normalization does NOT match a spectral-unit MCSD step:
smaller cluster updates can follow from this deliberate change in step rule.

For a complete true top eigenspace, D is a Frobenius tangent-cone projection
of normalized momentum. Near-root clustering, the rank cap, approximate
spectra and radial return make these engineering variants, not a proven
stochastic P2GD method. Neither a smaller cluster rate nor a short-run loss
improvement establishes convergence or robustness.

## Four configurations

All arms use adaptive top<=4, routing gap 5e-5 and coefficient 1.0.
The cluster-membership tolerance is separate from the routing threshold.

| Array index | Ritz iterations | Cluster tolerance | Direction |
|---|---:|---:|---|
| 0 | 10 | 2e-3 | Normalize momentum before cone |
| 1 | 10 | 5e-5 | Normalize momentum before cone |
| 2 | 20 | 2e-3 | Normalize momentum before cone |
| 3 | 20 | 5e-5 | Normalize momentum before cone |

The broad-cluster arms isolate the normalization change against the previous
matched-length hybrids. Tight-cluster arms test whether merging fewer nearby
directions helps. All four are mixed methods; controls are reused rather than
launching another control-only tuning campaign.

## Matched historical references

| Method | Job | Final validation loss | Job elapsed |
|---|---|---:|---|
| MCSD-TP warm Ritz K10 | 4291611_0 | 4.946553 | 36:55 |
| Matched-length hybrid K10, tolerance 2e-3 | 4291611_1 | 4.958415 | 36:41 |
| MCSD-TP warm Ritz K20 | 4291611_2 | 4.947171 | 38:04 |
| Matched-length hybrid K20, tolerance 2e-3 | 4291611_3 | 4.960459 | 37:56 |
| MuonSphere | 4290694_0 | 4.947061 | 34:23 |
| SSO | 4290694_1 | 4.956870 | 38:04 |

These are one-seed 100M-token pilots. They are not the historical 1B paper
matrix. Compare K10 and K20 against their own controls, and report training
time separately from tests and offline audits.

## Protocol and diagnostics

Width256, 28 layers, FFN768, head dimension128, sequence length4096,
global batch128, micro-batch4, seed1234, TP=PP=DP=1, one H20 per task.
191 iterations process 100139008 tokens from the SAME existing 3B-indexed
OLMo-Mix source as the previous pilots. Peak LR0.015, minimum LR0.0015,
25 warmup steps, cosine decay, validation every50 steps and at191 with
five batches. No new data is downloaded. Hidden matrix weight decay is0;
auxiliary optimizer settings are unchanged. Checkpoint writing is disabled.

The array caps concurrency at2 GPUs. Existing jobs are not canceled or
modified. At the resource check, SPG-7-1 had5 of8 GPUs allocated; SPG-7-2
was draining and is not counted as available. Home storage had31GB free.

Report validation/train loss, cluster and repeat rates, rank/cap overflow,
training seconds/step, cache reuse, and full/small SVD call counts. Up to24
bounded snapshots per run are audited AFTER training for gap/radius errors,
direction discrepancy, direction Frobenius norm and its inner product with
momentum. These selected samples are not population-level guarantees.

## Validation and submission

Local tests: 8 normalization tests, 8 warm-Ritz tests, and 11 existing radial
regression tests passed. The new tests verify the literal trial, no length
matching or unused matrix sign, momentum-buffer preservation, positive-scale
invariance, zero-momentum behavior, one small Ritz SVD, optimizer dispatch,
offline direction reconstruction, and the launch grid. CUDA tests run in
each allocation before training. Runtime guards reject an accidental full
SVD as a programming error; they do not trigger a numerical fallback.

Launcher: `slurm/cluster_momentum_fro_100m.sbatch`.
Remote source snapshot: `/home/u3013198/projects/cluster-momentum-fro-20261021`.
Submitted array **4292952** on 2026-10-21. Tasks0 and1 entered real training
on SPG-7-1 at17:51 HKT; both passed all27 CUDA tests. The training argument
dump and optimizer configuration both confirm `momentum_fro` is active.
The first logged iteration had zero NaN/skipped iterations. Tasks2 and3
initially waited with reason `JobArrayTaskLimit`, then started at18:27:22.
The table below supersedes the initial running/queued status.

| Job | K | Cluster tolerance | Final verified status |
|---|---:|---:|---|
| 4292952_0 | 10 | 2e-3 | Completed18:27:11, exit0 |
| 4292952_1 | 10 | 5e-5 | Completed18:27:04, exit0 |
| 4292952_2 | 20 | 2e-3 | Completed19:05:31, exit0 |
| 4292952_3 | 20 | 5e-5 | Completed19:05:36, exit0 |

The independent snapshot contains approximately63MB of source, without
copying historical results or training data. Archive SHA256:
`b1e4252208934446e4c09678e228ed24f1329987a07556dc9256421181a2580a`.
All3224 source files are recorded in the remote `source.sha256` and verified
before each task. Receipt: `output/cluster_momentum_fro_h20_20261021/submission.json`.
Results remain at remote `results/4292952_<index>/`. Actual per-task elapsed
time was37:00--38:14, and the two-wave array spanned75:32 from first start
to final completion. Job elapsed includes setup, validation and audits;
separately timed training durations are reported in the results table.
