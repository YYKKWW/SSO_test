# Related Work and Research Priorities

[Overview](../README.md) · [Current evidence](experiments/mcsd_p2gd_results_summary.md) · [Algorithm guide](algorithms/current_comparison_pseudocode.md)

**Search checked through 2026-11-01.** This is a focused reading list, not an
exhaustive survey. Links lead to papers or their authors' repositories.
Preprint claims below are attributed to the papers, not independently verified
results of this project. Dates distinguish first publication from revisions.

## Most Relevant Recent Papers

| Paper and date | Connection to this project | Important boundary |
|---|---|---|
| [Convergence of Practical Muon with Finite Newton-Schulz Iterations and Nesterov Momentum](https://arxiv.org/abs/2609.39595), 2026-09-30 | Analyzes finite-step, layerwise NS and Nesterov rather than substituting an exact polar factor. Especially relevant to the actual NS8 direction. | The vanishing rate uses coupled learning-rate/momentum schedules and smoothness/noise assumptions. It does not cover our nonsmooth constraint, fixed 0.9 momentum, approximate radial return, or branch switching automatically. |
| [Spectrally Targeted Muon](https://arxiv.org/abs/2610.10965), 2026-10-07 | Uses shifted-Gram NS iterations to select momentum spectral bands without SVD; motivates testing which update directions matter. | Momentum singular bands are not the top singular subspace of the constrained weight. This is not an accuracy guarantee for our weight-gap detector. |
| [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781), 2026-05-19; revised 2026-10-27 | Data-dependent, layerwise LMO geometry offers an alternative to treating the near branch's Frobenius and matrix-sign geometry as interchangeable. | Selecting an LMO norm does not enforce the spectral-sphere tangent cone or establish convergence of our returned iterates. |
| [Musec: MomentUm SpEctral Clipping for Stable Muon-type Training](https://arxiv.org/abs/2609.11655), 2026-09-10; revised 2026-09-11 | Soft spectral saturation, implemented using coupled NS iterations, suggests a direction-shaping alternative to full flattening. | The paper's nonsmooth stochastic analysis is not a theorem for a nonsmooth spectral-sphere constraint. Our runs have not shown instability that would already justify replacing the main path. |
| [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899), 2026-06-15; revised 2026-10-25 | Explicitly controls weight/update norms and examines learning-rate transfer and longer training. Useful for interpreting effective displacement rather than nominal LR alone. | Frobenius spheres are smooth away from zero; spectral spheres have repeated-top singular strata. Its geometry is not ours. |
| [On MUON optimization: From non-convergence to an error analysis with Polar Express and the Newton-Schulz polynomial from implementations](https://arxiv.org/abs/2608.04607), 2026-08-05 | Studies finite NS error and stochastic optimization; cautions against treating an approximate polar factor as exact. | Its examples and assumptions differ from the constrained hybrid. Its nonconvergence examples do not establish failure of our algorithm. |

## Baselines and Theoretical Foundations

| Reference | Why retain it |
|---|---|
| [Controlled LLM Training on Spectral Sphere](https://arxiv.org/abs/2601.08393), 2026-01-13; revised 2026-03-05 | Direct SSO/MuonSphere baseline and spectral-muP protocol. Compare the pinned local implementation, not presumed defaults of a moving upstream branch. |
| [Muon is Provably Faster with Momentum Variance Reduction](https://arxiv.org/abs/2512.16598), 2025-12-18 | An arbitrary-norm LMO framework with variance-reduced momentum. Relevant to stochastic tracking error, but additional gradient evaluations and different momentum recursions must be costed. |
| [First-order optimization on stratified sets](https://arxiv.org/abs/2303.16040), 2023-03-28 | Explains why approaching lower strata can require more than a projected tangent direction. Its method evaluates lower-stratum candidates and uses backtracking; it is not a proof for our no-backtracking stochastic variant. |

## What to Test Next

These are **our inferences from the papers and local evidence**, not results
claimed by the cited authors for MCSD-P²GD.

1. **Measure the geometry after the return.** Compare direction norm,
   momentum inner product, radial correction, and actual displacement on the
   same saved matrices. A better tangent-cone direction can lose its advantage
   after radial scaling. The current `c=0.5/1/2` single-LMO audit addresses this
   before paying for another training grid.
2. **Preserve a matched MCSD-TP control.** The hybrid's 3B improvement changed
   sign across seeds. The latest 1B advantage is only `0.000129`; neither a
   narrower cluster nor single-LMO c1 improved it. Replicate the same budget
   and evaluation protocol first.
3. **Study softer or consistent LMO geometry only as a separate ablation.**
   Spectral saturation or Schatten-p directions are plausible research
   directions. Derive their cone/return behavior and compare actual step
   lengths before training; do not add multiple changes at once.
4. **Keep theory and implementation claims separate.** A future theorem must
   account for momentum tracking, near-cluster identification, rank-cap
   overflow, approximate radial error, and switching. Fixed K10 plus successful
   loss curves is not a substitute for these assumptions.

## Repository Presentation References

The overview borrows organizational ideas, not experimental claims or code:

- [KellerJordan/Muon](https://github.com/KellerJordan/Muon): short method
  description, installation/use, explicit parameter-group responsibilities.
- [Unakar/Spectral-Sphere-Optimizer](https://github.com/Unakar/Spectral-Sphere-Optimizer):
  separate algorithm, evaluation, and scaling experiments.

This repository consequently keeps one short overview, one consolidated
results page, a code-linked algorithm guide, and dated detailed records.
Historical logs remain accessible but do not masquerade as current status.
