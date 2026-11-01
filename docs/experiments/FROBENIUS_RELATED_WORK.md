# Related Work and Testable Directions

[Results overview](FROBENIUS_3B_6B_SUMMARY.md) | [Point-extrapolation pilot](MCSD_POINT_PILOT_20261031.md)

Primary sources checked November 2, 2026. Dates distinguish initial submissions
from revisions. These papers motivate tests; they do not establish the cause
of this project's results or guarantees for our LLM variant.

## Current Optimizer Experiments

| Paper | Date | Relevant result | Consequence for our tests |
|---|---|---|---|
| [Hyperball Optimization](https://arxiv.org/abs/2606.16899), Wen et al. | June 2026; revised October 25 | Fixes Frobenius norms of matrix weights and updates; studies angular LR and scale transfer | Keep radii and direction scaling explicit; distinguish nominal LR from actual movement |
| [Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444), Xiao et al. | July 2026 | Studies angular effective LR including weight/update alignment; schedule matching explains substantial differences in its tested settings | Log angular displacement before claiming a better direction; retain late-stage comparisons |
| [The Best Optimizer Depends on Batch Size](https://arxiv.org/abs/2610.08975), Dang, Wen and Malladi | October 6; revised October 12, 2026 | Optimizer rankings can depend on batch size even after tuning LR and momentum | Freeze batch size for paired tests; later compare two batches with equal tokens and retuned LR |
| [NorMuon](https://arxiv.org/abs/2510.05491), Li et al. | October 2025 | Row-wise second-moment normalization follows orthogonalization | Separate row adaptation from global direction normalization in ablations |

The [official NorMuon implementation](https://github.com/zichongli5/NorMuon)
restores the orthogonalized update's original Frobenius norm after row
normalization. Unit-F MCSD is a different ablation, not an exact copy.

## Manifold Acceleration and Momentum

| Paper | What it establishes | Boundary for this project |
|---|---|---|
| [Huang--Wei, An Extension of FISTA to Riemannian Optimization](https://arxiv.org/abs/1909.05485), 2019; revised 2021 | Riemannian proximal acceleration with restarting for sparse PCA | Our radial-inverse outer extrapolation resembles its structure, but omits the proximal subproblem and objective safeguard |
| [Feng, Jiang, Huang and Ying, A Riemannian Accelerated Proximal Gradient Method](https://arxiv.org/abs/2509.21897), September 2025 | Acceleration under geodesic convexity assumptions; a safeguard for nonconvex convergence | Nonconvex stochastic LLMs with approximate msign do not inherit these rates |
| [Recursive Gradient Retraction Methods](https://arxiv.org/abs/2610.19944), Yuan, Sun and Yuan, October 2026 | Constructs recursive estimators in ambient space before mapping to Stiefel tangent space, avoiding vector transport | Relevant to future stochastic-estimator tests; not proof of equivalence to Muon-style Nesterov or our Frobenius LMO |

## Proposed Diagnostics

1. **Finish the four submitted short follow-ups.** Keep source, data, batch,
   validation and cosine horizon fixed; compare common iterations.
2. **Measure motion rather than infer it from LR.** On sampled layers record
   `||X_new-X||_F/R`, `||Y_new-X_new||_F/R` and their angles in FP32.
   Distinguish accepted-point motion from live-model motion.
3. **Separate evaluation location from optimization.** Evaluate X and Y on
   identical diagnostic batches at sparse checkpoints. Ordinary validation
   stays at Y; do not relabel the metrics after seeing the results.
4. **Only then test damping or restart.** Decreasing caps or periodic restart
   are hypotheses, not proven fixes. Loss-based restart requires comparable
   batches and explicit reporting of extra evaluation cost.
5. **Replicate, then transfer.** Freeze candidates after 3B, add paired seeds
   and the strongest baseline, then move to 6B or a larger width. An early
   crossing is not evidence for a scaling law.

For equal-radius matrices, angular displacement is
`acos(clip(<X_new,X>/R^2, -1, 1))`. For small motions, the chord length
`||X_new-X||_F/R` is often easier to resolve numerically. Neither statistic
alone identifies causality after trajectories diverge.

## Presentation References

- [NorMuon](https://github.com/zichongli5/NorMuon): concise mechanism, minimal
  implementation, usage and citation. Our overview keeps results separate
  from implementation details.
- [Hyperball May Not Be a Free Lunch](https://github.com/mangocrazz/hyperball-may-not-be-a-free-lunch):
  distinguishes CPU figure reproduction from full training, with an experiment
  map. We release scalar data and a CPU plotter separately from runtime provenance.

Published numerical results from these sources are not substitutes for our
own measurements.
