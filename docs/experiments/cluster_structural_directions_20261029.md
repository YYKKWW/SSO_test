# Algorithmic Improvements Without Hyperparameter Tuning

This follow-up changes the **local direction construction**, not the seed,
learning rate, branch scale, routing threshold, cluster tolerance, subspace
size, or iteration count. No new long training run was submitted. The earlier
seed/parameter launch drafts were withdrawn locally; running training code and
queue concurrency were not changed.

## Structural Question

The current near branch uses the selected near-top singular vectors as if their
singular values were all equal to the sphere radius. This is exact for a true
top plateau with a complete basis, but only a model for a positive-gap cluster.
It can correct directions that would remain below the boundary at the actual
step size. This is a possible structural limitation, not evidence that it caused
the observed validation losses.

We tested a small **gap-aware affine spectral model** while preserving the
existing momentum normalization and approximate radial return.

Let `Z = -M / max(||M||_F, 1e-12)`, let `alpha` include the existing effective
learning rate and c8 multiplier, and let `(U, s, V)` be the selected cached
q≤4 triplets. Define

\[
 A=\operatorname{sym}(U^\top ZV),\qquad
 B=\operatorname{diag}(s)+\alpha A.
\]

Write `B = Q diag(b) Q^T` with descending eigenvalues. Its Frobenius projection
onto the symmetric boundary `lambda_max(B)=R` is

\[
 \bar B=Q\operatorname{diag}\bigl(R,\min(b_2,R),\ldots,\min(b_q,R)\bigr)Q^\top.
\]

The candidate trial step is

\[
 \Delta=\alpha Z+U(\bar B-B)V^\top,\qquad Y=X+\Delta.
\]

Both variants then use the **same K10 warm block-8 Ritz solve and approximate
radial return**. This is a projection of a small symmetric affine model,
**not** a Frobenius projection of the full trial matrix. Coupling to omitted
directions, skew terms, and spectral approximation error remain.

Two consistency checks hold algebraically and were tested on CPU/CUDA:

- If q=1 and s1=R, the step is the usual top-1 tangent correction.
- If all selected s_i=R and alpha>0, it reduces to the existing Frobenius
  tangent-cone correction, provided the supplied basis is the complete top
  space when claiming exact geometry.

The construction does not establish stochastic convergence, exact feasibility,
or an arbitrary-norm LMO guarantee. All cached singular values remain estimates.

## Fixed-Matrix H20 Test

- Job **4360368**: `COMPLETED`, exit `0:0`, elapsed **22 seconds**.
- Four CUDA structural tests passed; all 49 local cluster tests passed.
- 88 saved matrices from the completed seed-1234 K10 control/hybrid 3B runs.
- 19 met the unchanged cached near-region gate; the control contributes one
  hypothetical near-branch evaluation, not an actual training branch call.
- Fixed K10, block-8, top≤4, gap `5e-5`, cluster tolerance `2e-3`, c8 and each
  snapshot's actual effective learning rate. All variants receive the same
  matrix, momentum, cache, and warm basis.
- Full SVD is used only as an offline oracle. It is not an added online fallback.

`<M, X_next-X>` below is a frozen-momentum linear model, **not measured training
or validation loss**. More negative indicates stronger descent in that model.

| Direction | One-step exits, out of 19 | Mean momentum linear change | Median radial correction / trial-step norm | Maximum sampled radius error |
|---|---:|---:|---:|---:|
| Current flat-cluster correction | 9 | -2.274754e-4 | 0.00706 | 3.815e-6 |
| Gap-aware small-model correction | 8 | -2.274528e-4 | 0.00622 | 3.934e-6 |
| Top-1 correction | 18 | -2.267357e-4 | 0.04397 | 2.265e-6 |
| No correction / ambient | 19 | -2.265480e-4 | 0.10800 | 3.576e-6 |

All four variants have negative returned momentum inner products on these
19 cases. The gap-aware candidate changes the returned displacement by about
**0.588% on average** relative to the current cluster rule. It has a more
negative momentum model in 5 cases and a less negative one in 14, but the mean
difference is only about **0.0099%** of the reference magnitude. This is not a
robust optimization win, and these tiny paired differences should not be
interpreted as a training-loss ranking.

Top-1 and ambient directions exit the near region more often but have weaker
average momentum-model descent and larger radial corrections. **Fast exit is
not equivalent to better optimization.** The existing long-run mechanism
controls are still necessary.

The near samples had true cluster sizes 2–4 under the same cluster tolerance.
They contain no top-4 overflow example, despite overflows in full-training
statistics. This bounded sample cannot resolve the capacity problem or
estimate population error rates. No sampled gate misclassification occurred.

## Non-Tuning Research Directions

| Direction | Structural change | Main difficulty / next evidence |
|---|---|---|
| Gap-aware correction | Keep the spectral separations in the local small-matrix model instead of flattening the cluster conceptually | Implemented offline here, but no clear advantage. Do not launch a large training grid on this evidence alone |
| Consistent LMO geometry | Formulate the near direction with the same norm-constrained descent oracle as the MCSD branch, rather than switching from a matrix-sign LMO to a normalized ambient-momentum correction | For a nonlinear tangent cone, projecting an ambient LMO does not automatically preserve its descent identity. Need a feasible-direction construction and analysis of the actual constrained LMO before integration |
| Subspace refresh | Preserve the active cached space while refreshing redundant directions, rather than relying only on a previously aligned warm basis | A small Ritz residual does not exclude a missed leading invariant space. Test controlled missing-direction fixtures and fixed-matrix replay; protect the training RNG and avoid introducing a full-SVD fallback |
| Return-aware direction selection | Compare first-order-feasible candidate directions in a common local model before one chosen trial/return, rather than optimizing branch-exit frequency | Momentum is not the exact objective gradient, and positive-gap cluster constraints are approximate. Requires independent-batch/full-training evaluation and a stochastic argument; no claim of a proven replacement |

**Priority:** finish the already queued top-1 / ambient / cluster mechanism
comparison, then choose which structural hypothesis to pursue. Gap-aware
correction is now a tested prototype, not a recommended production replacement.
Seed replication and parameter tuning remain separate tasks and were not
submitted in this follow-up.

## Evidence and Reproduction

- [Metrics and per-matrix results](../results/cluster_structural_20261029/structural_4360368.json)
- [Slurm completion and CUDA tests](../results/cluster_structural_20261029/completion.json)
- [Source hashes and isolated snapshot](../results/cluster_structural_20261029/deployment.json)
- [Offline implementation](../../scripts/audit_cluster_structural_directions.py)
- [Structural tests](../../scripts/test_cluster_structural_directions.py)
- [One-GPU audit launcher](../../slurm/cluster_structural_offline_20261029.sbatch)

The audit needs saved matrix snapshots; they are not distributed with the public
metrics. Source code alone does not reproduce these exact sampled inputs.
