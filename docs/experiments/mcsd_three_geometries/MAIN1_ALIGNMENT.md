# Alignment with `main1.tex` (2026-09-23)

Reviewed against the user-supplied `main1.tex` with SHA256
`6c6354c50b0d8b1917446680cf56fdae191ecf2e3218b7167cec553499407ad0`.
The handoff's earlier SHA refers to a different revision. This is a code and
evidence audit, not a claim that the LLM study has been completed.

## Algorithm correspondence

| Paper requirement | Implementation | Scope |
| --- | --- | --- |
| Current sample updates ambient EMA before the current direction | `ManifoldMCSD.step`: `momentum = beta*momentum + (1-beta)*grad` before `q` | Matches Algorithms 1/2 after the shared gradient clip. |
| Project momentum before spectral-ball LMO | `q = P_W(momentum)`, then `direction = -msign(q)` | Exact SVD mode is a reference; finite BF16 Polar Express is an empirical approximation. |
| MCSD returns ambient trial by metric projection | Frobenius scaling; full polar for Stiefel; singular-value clipping for spectral sphere | Exact modes match locally. Finite-step Stiefel and PI/top-k spectral returns are not exact projections. |
| MCSD-TP projects LMO output, then retracts | `direction = P_W(direction)` before trial; Frobenius/polar/spectral radial return | Exact modes match locally. The projected direction is not renormalized. |
| Spectral smooth-stratum tangent | Leading `u_1 v_1^T` normal | Valid as a tangent normal only for simple `sigma_1`. At a tie, continuing with one vector is an empirical extension, per the user's explicit design choice. |
| Closed-set PGD fallback | Not implemented in this three-geometry branch | Do not identify these runs with Algorithm 3 or its limiting-normal guarantee. |

The fixed component factor `c_l = ||A_l||_F / sqrt(min(shape))` changes the
per-component step from the unscaled notation of Algorithms 1/2. It is an
explicit spectral-muP-style experimental scaling, not a second application
of the LMO radius. The main optimizer uses no decoupled weight decay on
constrained matrices; auxiliary parameters use their own AdamW groups.

## Exact versus practical implementations

The paper's inexact-LMO condition has **two** parts: `||S||_op <= 1` and
`<Q,S> + ||Q||_* <= zeta`. A finite BF16 matrix sign need not satisfy the
first. The optimizer now samples both quantities before MCSD-TP's second
tangent projection, plus the returned constraint defect. Audits are staggered
by component and update count; they are observations, not a per-step
certificate. A small observed average is not a proof of the paper's
`O(T^-1/4)` oracle-error budget.

The local synthetic width-128 PI10/top-8 spectral return had about 1.5%
relative sphere defect. That is too large to describe as the paper's exact
projection/retraction. The practical spectral run must retain a distinct
variant label, report defect distributions and compare against the exact
reference on matched seeds. Finite-step polar Stiefel return likewise needs
the observed orthogonality defect reported, even when it is small.

For exact spectral runs, the trial SVD now supplies the next iterate's chosen
top normal. The first step bootstraps from the initialized matrix; subsequent
steps use one SVD for the LMO and one for the trial return per component,
excluding staggered diagnostics. A cached normal is still only a *choice* at
a repeated leading singular value. Checkpoint state includes that choice.

The paper's finite-time rate specializes to a horizon-dependent **constant**
step and momentum choice. The LLM script uses fixed `beta=0.95`, warmup,
cosine decay, global clipping, BF16 forwards and finite-step returns. These
can be sensible empirical settings but must not be presented as a numerical
verification of the stated rate or its unbiased-oracle assumptions.

## Paper evidence gate

The current `main1.tex` reports stochastic PCA and Brockett experiments on
Stiefel. It does not contain this three-geometry LLM experiment. Its own
discussion states that spectral safe-region certification and projection
costs remain to be evaluated. The new branch is therefore supplementary
research infrastructure, not evidence already present in the manuscript.

Before using an LLM result as a conference claim:

1. Run real H20 model-construction, forward/backward, first-step and
   checkpoint-resume smoke tests. Verify the logical Q/K/V and gate/up
   manifest against actual Megatron tensors, optimizer ownership and FP32
   master/model synchronization. Local CPU tests do not establish this.
2. Fix an independent-token train split, development validation split and
   held-out test split. Disclose whether a 3B budget repeats a 1B sample.
   With about 126.6M parameters, 1B and 3B correspond to about 7.9 and
   23.7 tokens per parameter, respectively.
3. Run geometry-matched strong baselines: MuonH for Frobenius, SSO and
   MuonSphere for spectral, and an explicitly labeled iMuon adaptation for
   Stiefel. The four matched adapters are now implemented in
   `manifold_baselines.py`; formal budget-matched evaluations remain to be run.
   AdamW/Muon application references remain separate from this matched protocol.
4. Tune on the same development budget and LR grid, freeze the winning
   settings, then evaluate at least three paired seeds per method on held-out
   data. Report means, uncertainty, full trajectories and failures, not only
   the best single final loss.
5. Report per-step and total GPU time, peak memory, sampled LMO ball excess,
   LMO model error, return defect, spectral gap/tie incidence and SVD audit
   frequency. Compare both exact and practical variants without conflating
   their approximation costs or theoretical scope.

The highest-value next experiment is a matched 100-step H20 smoke across the
three geometries and both methods. Long 3B runs should wait until the
integration and constraint gates pass; there is no basis yet to predict an
advantage over the strong baselines.
