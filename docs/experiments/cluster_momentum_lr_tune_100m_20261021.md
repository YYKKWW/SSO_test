# Normalized momentum branch step size tuning

Eight width-256, 100M-token pilots isolate the small-gap step coefficient
and then test two global learning rates with matched MCSD-TP controls.
Array **4294039** was submitted on 2026-10-21 at 22:37 HKT, with one H20
per task and at most one task running at a time. Existing jobs were not
changed. These are supplementary experiments, not changes to the frozen
1B paper results or the independent reproduction repository.

## Completed results

All eight tasks completed with exit0; the last ended at04:29:07 HKT on
2026-10-22. All training logs report zero skipped and NaN iterations.
This array no longer occupies GPUs. No additional jobs were submitted or
canceled during this result check.

The final validation losses below are at191 iterations. Cluster statistics
are through the last training log at190:47880 matrix-component decisions,
including252 initial zero-LR/bootstrap calls. Repeat percentage is the
fraction of cluster calls that continue the previous cluster call for that
component, calculated from the integer call and burst-start counts.

| Job suffix | Peak LR | c or control | Final val | Cluster % | Repeat % | Job elapsed | Training elapsed |
|---|---:|---|---:|---:|---:|---|---|
| _0 | 1.5e-2 | c=2 | 4.962942 | 3.663 | 87.69 | 38:20 | 37:32 |
| _1 | 1.5e-2 | c=4 | 4.957274 | 3.158 | 84.72 | 38:15 | 37:40 |
| _2 | 1.5e-2 | c=8 | 4.956690 | 2.174 | 76.27 | 38:07 | 37:32 |
| _3 | 1.5e-2 | c=0.5 | 4.970935 | 4.685 | 91.66 | 38:08 | 37:32 |
| _4 | 1e-2 | MCSD-TP | 5.063725 | 0 | N/A | 38:15 | 37:39 |
| _5 | 1e-2 | c=4 | 5.096617 | 5.689 | 84.43 | 38:09 | 37:33 |
| _6 | 2e-2 | MCSD-TP | **4.828390** | 0 | N/A | 38:13 | 37:38 |
| _7 | 2e-2 | c=4 | **4.834641** | 1.619 | 83.10 | 38:12 | 37:36 |

Total training wall time summed across these single-GPU tasks was5:00:42.
Each task took approximately38 minutes including tests and post-training
audits. Last logged step times were11.18--11.24s. This experimental hybrid
has no material measured overhead over its K20 MCSD-TP controls, but remains
about11% slower in job time than the historical34:23 MuonSphere pilot.

### Coefficient sweep at a fixed learning rate

At LR1.5e-2, the observed validation losses ordered by coefficient are:

| c | Final val | Cluster % | Repeat % | Calls per burst start |
|---:|---:|---:|---:|---:|
| 0.5 | 4.970935 | 4.685 | 91.66 | 11.99 |
| 1, previous run4292952_2 | 4.969451 | 4.680 | 90.81 | 10.88 |
| 2 | 4.962942 | 3.663 | 87.69 | 8.12 |
| 4 | 4.957274 | 3.158 | 84.72 | 6.55 |
| 8 | 4.956690 | 2.174 | 76.27 | 4.21 |

The c=8 arm improves loss by0.012761 over c=1 and reduces persistent
near-region visits. This supports step-size mismatch as a contributor to
the previous regression, not a proof that it is the only cause. The loss
difference between c=4 and c=8 is only0.000584 in one seed. Calls per burst
start includes episodes still active at the logging boundary.

The c=8 result remains0.009519 above matched K20 MCSD-TP (4.947171) and
0.009629 above MuonSphere (4.947061). It is0.000180 below SSO (4.956870),
too small a single-seed difference to establish an advantage. Its final
logged cluster rate is21.43%, despite a cumulative rate of2.174%; repeated
near-region visits are reduced, not eliminated. It also has168 detected
top-four capacity overflows, so this is not complete tangent-cone handling.

### Global learning rate and fair comparisons

At LR1e-2, adding the cluster branch increases loss by0.032892 relative to
its control. At LR2e-2, it increases loss by0.006251. Thus all tested
hybrids still lose to their own same-LR MCSD-TP control. At LR2e-2 the
hybrid is slightly better at validation steps50/100/150, but that advantage
reverses by191: the step150 losses are5.014030 versus5.014371, while final
losses are4.834641 versus4.828390.

The large improvement from LR1.5e-2 to2e-2 occurs for the control as well
as the hybrid. It is not evidence of a P2GD contribution or superiority to
MuonSphere/SSO, which have not been retuned at2e-2 in this matched pilot.
The best tested global LR is at the upper grid boundary; no optimum has
been established. All conclusions remain specific to one-seed100M pilots.

### Numerical audit

Every run recorded48132 small Ritz SVDs for48132 component calls and zero
original-matrix full SVD or svdvals calls inside optimizer updates. Full
decompositions were restricted to tests and offline audits.

| Suffix | Samples | False cluster calls | Mean absolute gap error | Max returned radius-relative error |
|---|---:|---:|---:|---:|
| _0 | 24 | 0 | 1.17e-5 | 4.64e-5 |
| _1 | 24 | 1 | 1.15e-5 | 4.86e-5 |
| _2 | 24 | 0 | 2.01e-5 | 7.51e-5 |
| _3 | 24 | 1 | 1.46e-5 | 2.32e-5 |
| _4 | 16 | 0 | 7.82e-6 | 4.53e-6 |
| _5 | 24 | 0 | 5.77e-6 | 6.68e-6 |
| _6 | 16 | 0 | 1.37e-5 | 4.04e-5 |
| _7 | 24 | 3 | 5.36e-5 | 3.52e-4 |

These are shape/step-selected and early-cluster snapshots, NOT population
error rates. No false-safe event was found in these samples; this does not
certify the entire run. For controls, routing discrepancies are hypothetical.
The LR2e-2 hybrid's step44 FFN sample has estimated gap3.71e-5 versus
FP32 full-SVD gap6.58e-4, falsely selecting the cluster branch. Its larger
mean direction discrepancy partly reflects such branch disagreements,
not solely vector error within the same branch. K20 is not universally
accurate at the larger step sizes.

### Next tests suggested, not submitted

Keep Frobenius normalization as requested. Prioritize LR2e-2 with c=8 and
c=12, and match MuonSphere/SSO at LR2e-2 before claiming an algorithmic
improvement. First replay the misclassified snapshots with more Ritz
iterations or a larger subspace offline; this does not introduce an online
accuracy gate or automatic full-SVD fallback. Do not promote c<=1 or
LR1e-2 to long runs on the present evidence. A longer matched pair and a
second seed are needed before treating a small pilot gain as reproducible.

Local logs/configs/audits: `output/cluster_lr_tune_20261021/results/`.
Machine-readable extraction: `output/cluster_lr_tune_20261021/final_summary.json`.

## Motivation and step definition

The previous K20 normalized-momentum hybrid reached final validation loss
4.969451, versus 4.947171 for its MCSD-TP control. Its cumulative cluster
rate was 4.680%, with 90.81% consecutive repeat calls. Better spectral
accuracy alone did not remove this regression; see the
[completed normalization experiment](cluster_momentum_fro_100m_20261021.md).

Keep the requested normalization and tune a predetermined scalar c:

```text
Z = -M / max(||M||_F, 1e-12)
Q = selected top-cluster cone projection of Z
small-gap trial: Y = X + effective_lr * c * Q
safe-region trial: unchanged MCSD-TP
both returns: X_next = R * Y / estimated_sigma1(Y)
```

There is no normalization after the cone projection, MCSD length matching,
function-value line search, online accuracy gate, or full-SVD fallback.
The implementation stores c in the legacy `SPEL_PGD_PGD_LR_SCALE` variable.
It is a step multiplier, NOT a target branch percentage, and not a guarantee
that the resulting update is c times the MCSD update. Historical PGD scales
0.10/0.25 used different normalization and cannot be transferred literally.

## Fixed state audit

Two downloaded, previously selected small-gap snapshots from job4292952_2
were checked locally: shapes128x256 and1536x256. They are not a random or
representative population sample. Each counterfactual uses the SAME stored
X, momentum, cached subspace and effective learning rate, with K20 return.
Old snapshots lack effective_lr; it is recovered from the scalar radial
relation and checked by a relative reconstruction residual below2e-6.
New snapshots explicitly save effective_lr.

| Direction or coefficient | Mean direction Frobenius norm | Mean returned relative Frobenius step |
|---|---:|---:|
| MCSD-TP counterfactual | 13.353 | 1.9806% |
| c=0.5 | 0.500 | 0.0777% |
| c=1 | 1.000 | 0.1554% |
| c=2 | 2.000 | 0.3108% |
| c=4 | 4.000 | 0.6215% |
| c=8 | 8.000 | 1.2428% |

All checked returns were finite, with negative momentum-linear displacement;
maximum sampled radius-relative error across these cases was5.61e-6.
This motivates testing a wider coefficient range but does not establish
descent in LLM loss, optimality, convergence, or improved gap exits.
The audit never chooses steps during training.
Machine-readable results: `output/cluster_lr_tune_20261021/matrix_step_audit.json`.

## Submitted grid

All mixed arms fix routing gap5e-5, cluster-membership tolerance2e-3,
adaptive top<=4, and warm block-8 FP32 Ritz with20 iterations. The two
thresholds have distinct roles and are not tuned in this round.

| Job | Peak LR | Coefficient c | Method |
|---|---:|---:|---|
| 4294039_0 | 1.5e-2 | 2 | Mixed MCSD-P2GD type |
| 4294039_1 | 1.5e-2 | 4 | Mixed MCSD-P2GD type |
| 4294039_2 | 1.5e-2 | 8 | Mixed MCSD-P2GD type |
| 4294039_3 | 1.5e-2 | 0.5 | Mixed MCSD-P2GD type |
| 4294039_4 | 1e-2 | Not used | MCSD-TP control |
| 4294039_5 | 1e-2 | 4 | Mixed MCSD-P2GD type |
| 4294039_6 | 2e-2 | Not used | MCSD-TP control |
| 4294039_7 | 2e-2 | 4 | Mixed MCSD-P2GD type |

At LR1.5e-2, reuse matched runs4292952_2 (c=1, val4.969451) and4291611_2
(MCSD-TP, val4.947171). Historical MuonSphere4290694_0 is4.947061 and
SSO4290694_1 is4.956870 at the same LR. These are references, not newly
retuned baselines at LR1e-2/2e-2. Changing global LR also changes the
auxiliary optimizer LR; therefore compare each new LR's hybrid to its own
control, not solely to the old LR1.5e-2 control.

## Protocol and evaluation

Width256,28 layers, FFN768, head dimension128, sequence length4096,
global batch128, micro-batch4, seed1234, TP=PP=DP=1. Each run uses191
iterations, processing100139008 tokens from the existing3B-indexed OLMo mix
source, identical to previous100M pilots. This is not a3B-token run and not
the historical1B-sample paper comparison. Warmup25, cosine decay to0.1 times
peak LR, validation every50 iterations and at191, five validation batches.
BF16 training/NS8 matrix sign, FP32 master weights and spectral computation;
constrained matrix weight decay0, auxiliary settings unchanged. No checkpoints.

Record final validation, train loss, branch/repeat rates, late-step branch
rate, burst lengths, cluster-cap overflow, time/step and sampled radius/gap
errors. A lower branch rate alone is NOT success: remaining in a zero-gap
region can be compatible with useful optimization. Promote only promising
matched pairs to longer training and another seed; one-seed100M selection
does not demonstrate statistical superiority or a convergence theorem.

## Checks and launch status

All41 local cluster tests passed, including the literal coefficient-scaled
trial and matched-LR launch grid. The uploaded core algorithms match local
checksums. The remote source snapshot contains3230 files in its checksum
manifest, excluding result directories and Python caches.

All tasks passed their30 CUDA preflight tests and completed training and
offline audits. Saved configurations and logged cluster multipliers confirm
the requested momentum_fro/K20 step coefficients. The completed tables above
supersede the initial running/queued report.

Launcher: `slurm/cluster_momentum_lr_tune_100m.sbatch`.
Remote snapshot: `/home/u3013198/projects/cluster-momentum-lr-tune-20261021`.
Local submission receipt: `output/cluster_lr_tune_20261021/submission.json`.
Patch archive SHA256:
`48f9dc313823dc2fb3e3612ce2f0e5ad6f8fe73cce15146e45528f6e71ee04ed`.
