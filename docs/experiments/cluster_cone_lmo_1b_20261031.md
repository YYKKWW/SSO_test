# Narrow Clusters and a Single-LMO Near Direction

> **Completed:** all six runs finished. See the [2026-11-01 final comparison](cluster_results_20261101.md) for endpoints, curves, runtime, and numerical audits. Execution notes below record the original submission, not live status.

## Motivation

The second-seed 3B cluster hybrid was worse than matched MCSD-TP by 0.004007.
The wide cluster used three or four directions in about 84% of near calls,
with 60.19% consecutive near calls and 13.29% top-4 capacity overflow.
Ambient/top-1 controls performed better in that seed, but do not establish
that multi-direction geometry is useless. See the
[completed results](cluster_results_20261031.md).

This follow-up tests two explanations: a cluster may be too wide, or the
normalized-gradient near direction may be poorly matched to the MCSD LMO.
It does not search for a seed that makes the previous hybrid look better.

## Matched Six-Arm Experiment

| Index | Method | Near mapping | Cluster tolerance | Near multiplier |
|---|---|---|---:|---:|
| 0 | MuonSphere | Reference implementation | Not used | Not used |
| 1 | SSO | Reference implementation | Not used | Not used |
| 2 | MCSD-TP K10 | No near branch | Not used | Not used |
| 3 | Existing hybrid | Frobenius cone correction | `2e-3` | 8 |
| 4 | Narrow-cluster hybrid | Frobenius cone correction | `2e-4` | 8 |
| 5 | Single-LMO hybrid | Cone-face MCSD extension | `2e-4` | 1 |

Common settings: width 256, 28 layers, peak LR `1.5e-2`, seed 2345,
1908 iterations, batch 128 x 4096 tokens, warmup 250, cosine decay to 10%.
The processed budget is 1,000,341,504 tokens from the existing **3B indexed
source**, not the older paper's separate 1B sample. All six use **20 validation
batches** every 250 steps and at the end. Old 1B runs used five batches;
their final losses are not exact evaluation matches to this experiment.

All four experimental arms share warm block-8, K10, FP32 QR/Ritz, returned
spectrum caching, BF16 Newton--Schulz matrix sign (8 steps), and approximate
radial return. Near routing uses `gap <= 5e-5`; the cluster tolerance is a
separate direction-selection parameter. The cluster is capped at four, and
overflow is logged rather than silently certified. Matrix weight decay is
zero. Non-matrix parameter groups retain the existing reference optimizer.
No checkpoint is written, and each experimental run retains at most 48
offline matrix samples. No online accuracy gate or full-SVD fallback is added.

Arm 4 versus arm 3 isolates cluster width. Arm 5 versus arm 4 changes the
direction construction **and its appropriate scale**: c8 is not transferred
to the matrix-sign LMO. This is not an equal-displacement comparison.
The baseline/control arms are rerun to match seed, data order, budget,
evaluation and environment. No existing 3B result is used as a 1B endpoint.

## Single-LMO Construction

Let `Z = -M / max(||M||_F, 1e-12)`, with paired selected bases U,V.
Define `H(D) = sym(U.T @ D @ V)`.

1. Diagonalize `A=H(Z)`. Clip its eigenvalues at zero and set the largest
   one to zero even if all are negative. Call the result B.
2. Set `Q = Z + U @ (B-A) @ V.T`. Let P project onto B's negative eigenspace.
3. Compute **one** matrix-sign LMO, `S = msign(Q / max(||Q||_F,1e-7))`.
4. Set `C = P @ H(S) @ P` and retain its negative semidefinite part `C_-`.
   Return `D = S + U @ (C_- - H(S)) @ V.T`.
5. Use `Y = X + eta*D`, estimate its spectrum with warm block-8/K10,
   return `R*Y/sigma1_hat(Y)`, and cache the scaled Ritz values and vectors.

For an exact, complete multiple top subspace and an exact norm LMO,
`lambda_max(H(D))=0` and `<Z,D> >= N_*(Q)`. For q=1 this reduces to the
tangent-projection / LMO / tangent-projection construction of MCSD-TP.
With the production approximate matrix sign the algebra instead bounds
`<Z,D>` below by `<Q,S>`, not an automatically exact dual norm.

These pointwise statements do **not** prove convergence of the training
algorithm: selected near clusters need not be exact multiple singular
subspaces, the subspace may be incomplete, return norms are approximate,
and stochastic momentum is used. The variant is not standard P2GD and is
not claimed to solve the exact cone-constrained LMO for arbitrary norms.

## Validation and Execution

- Local: 60 related unit/regression tests passed, including rectangular
  cone feasibility, exact-LMO descent, q=1 reduction, unchanged safe branch,
  production/audit agreement, and one small Ritz SVD per cached update.
- Isolated H20 snapshot: `/home/u3013198/projects/cluster-cone-lmo-1b-20261031-v2`.
- Overlay archive SHA256: `b841746cf47ac5ca694428f325b517be7e9249305554cd0d6e280487031ff42b`.
- Complete source-manifest SHA256: `d08971e09f2d7c5d6f4931ccb3c3fbf3416135a4d125f8e0393ef39074e77d94`.
- Successful gate: **`4368330`**. All 60 CUDA tests passed; the 192-sample
  paired replay passed sanity checks; both four-step LLM smokes completed.
  Each smoke recorded 1,008 small Ritz SVDs, zero full-SVD/svdvals calls,
  and no NaN/skipped iterations. Those first four LLM steps did not route
  near; the near branch was forced in CUDA unit tests and offline replay.
- Six 1B jobs submitted on **2026-10-31 at 14:48 HKT**:
  `4368382_2` (MCSD-TP control), `4368382_5` (single-LMO hybrid), then
  `4368383_0` (MuonSphere), `_1` (SSO), `_3` (wide cluster), `_4` (narrow cluster).
  The second array waits for completion of the first (`afterany:4368382`),
  preserving the global two-GPU limit. There are no final 1B results yet.
  At **14:52 HKT**, both priority tasks were RUNNING on SPG-7-1 and had
  entered training (first logged iteration 1/1908, zero NaN/skips). The
  remaining four were pending on their dependency. The local and deployed
  3,125 source-file hashes matched after framework restoration.
- Resource check: 12 of 16 H20 GPUs were allocated to other jobs. The main
  array is capped at **two** concurrent GPUs, leaving two initially free.
  Expected training is approximately 6 hours per arm, about 18--21 hours for
  six arms at concurrency two, excluding scheduler delays. This is an
  estimate from prior runs, not a measured duration for the new algorithm.

The initial gate `4368295` passed CUDA tests and replay but failed before
training because the local source snapshot lacked framework `models`
directories. The broad `models/` ignore rule was corrected to `/models/`.
The 78 missing framework files were restored unchanged from the successful
second-seed 3B snapshot, with per-file SHA256 provenance. No completed
experiment was overwritten. The v2 launchers explicitly check required GPT
source files before training; the complete source manifest includes them.

The offline replay compares wide/narrow/fixed-top-2 cone, single-LMO,
top-1 and ambient directions on the same X/M/learning rate, using both cached
and oracle spectra. It records direction norms, momentum inner products,
return-induced change, post-return gap, capacity and radius errors. These
selected snapshots are not population error rates; momentum inner products
are not measured objective changes. Online decisions never see oracle data.

Of 192 selected matrices, 51 were routed near by the cached gap. Across all
six candidate directions none had a positive recorded momentum inner product
before or after radial return. This is a sanity result, not an objective-loss
or stochastic convergence result.

| Cached-spectrum replay | One-step exits / 51 | Median radial change / step | Maximum relative radius error |
|---|---:|---:|---:|
| Wide cone, c8 | 37 | 0.00538 | `2.06e-5` |
| Narrow cone, c8 | 36 | 0.01002 | `2.80e-5` |
| Single-LMO narrow, c1 | 26 | 0.00257 | `2.18e-5` |
| Top-1, c8 | 47 | 0.06099 | `2.46e-5` |
| Ambient, c8 | 49 | 0.08823 | `2.44e-5` |

Narrow adaptive clusters selected q=2 on all 51 samples, so fixed top-2
gave the same directions and is not allocated a duplicate long run. The LMO
candidate does not have a better one-step exit rate; its motivation is the
MCSD-compatible direction construction and smaller radial distortion. Its
median direction Frobenius norm was 15.64 versus approximately one for the
normalized-momentum directions. Coefficients and exit rates therefore must
not be interpreted as equal displacement or as evidence of superior loss.

[Paired replay](../results/cluster_cone_lmo_1b_20261031/paired_192.json) |
[Submission](../results/cluster_cone_lmo_1b_20261031/submission.json) |
[Framework provenance](../results/cluster_cone_lmo_1b_20261031/framework_completion.json)

## What Would Justify Longer Runs

Compare loss against both matched MCSD-TP and the two baselines, together
with time, actual displacement, near-call repetition and capacity overflow.
A smaller near-call rate is not itself a success. If a candidate is promising,
replicate its paired improvement and then extend to 3B before increasing
width. Retain negative results. Do not silently alter the direction or
threshold during a run.

## Code

- [Single-LMO helper](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cone_lmo.py)
- [Training dispatch](../../Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py)
- [Frozen-matrix replay](../../scripts/audit_cluster_cone_lmo.py)
- [Tests](../../scripts/test_cluster_cone_lmo.py)
- [Main launcher](../../slurm/cluster_cone_lmo_1b_20261031.sbatch)
- [Gate launcher](../../slurm/cluster_cone_lmo_gate_20261031.sbatch)
