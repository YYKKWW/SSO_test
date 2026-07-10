# MCSD-PGD FP32 Gap Estimators

This note documents the FP32 gap estimators for `spel_pgd_dist`.

## Motivation

The original MCSD-PGD branch rule estimates

```text
rel_gap = 1 - sigma2 / sigma1
```

by first running ordinary `power_iteration(W)` for `sigma1,u1,v1`, then
deflating `W - sigma1 u1 v1^T`, then running power iteration again on the
residual. That path is cheap, but the shared `power_iteration` helper casts to
`bfloat16`, and deflation quality depends on the accuracy of the first triplet.
This makes very small gap thresholds hard to interpret.

The FP32 estimators are intended to make the MCSD-PGD branch decision more
theoretically meaningful when the top singular values are close. There are
three high-level modes:

- `deflated_fp32_gap_only`: keep the original SpEL `sigma1,u1,v1` path, but
  recompute the original deflated gap rule in FP32 for branch selection.
- `block2_fp32`: use the FP32 block-2 top Ritz pair for both the SpEL tangent
  object and the gap decision.
- `block2_fp32_gap_only`: keep the original SpEL `sigma1,u1,v1` path, and use
  a separate FP32 block-2 estimate only for `sigma2/sigma1` and `rel_gap`.

## Deflated FP32 Gap-Only

`deflated_fp32_gap_only` follows the original branch rule most closely, but
does the gap calculation in FP32:

```text
sigma1_gap, u_gap, v_gap = fp32_power_iteration(W)
R = W - sigma1_gap u_gap v_gap^T
sigma2_gap = fp32_power_iteration(R).sigma
rel_gap = 1 - sigma2_gap / sigma1_gap
```

The SpEL branch still uses the original SpEL path:

```text
sigma1_spel, u_spel, v_spel = original power_iteration(W)
M_projected = project_to_tangent_plane(M, u_spel, v_spel)
```

This mode is the closest test of the paper idea if the intended branch rule is
literally based on rank-one deflation after the leading singular pair. Its weak
point is also inherited from the original rule: when `sigma1` and `sigma2` are
very close, a small error in `u_gap,v_gap` can make the deflated residual noisy.

## Block2 FP32

The block-2 estimator computes the top two Ritz singular values from one FP32
two-dimensional subspace. For one matrix `W`, run deterministic FP32 block
subspace iteration with a pseudo-random sinusoidal right block `V` of width two:

```text
repeat k steps:
    U = qr(W V)
    V = qr(W^T U)

B = U^T W V
Ub, S, Vhb = svd(B)
u1 = U Ub[:, 0]
v1 = V Vhb^T[:, 0]
sigma1 = S[0]
sigma2 = S[1]
```

In `block2_fp32`, the SpEL branch uses the block-2 `sigma1,u1,v1`:

```text
M_projected = project_to_tangent_plane(M, u1, v1)
```

The PGD branch rule uses:

```text
rel_gap = 1 - sigma2 / sigma1
```

`block2_fp32` couples two effects: a more careful gap estimate, and a different SpEL
top-vector path. In near-degenerate spectra, the top two-dimensional subspace
may be stable while the individual first vector inside that subspace can rotate
relative to the original power-iteration vector. That rotation changes the
tangent projection even when PGD is never selected.

In `block2_fp32_gap_only`, the SpEL branch remains:

```text
sigma1_spel, u_spel, v_spel = original power_iteration(W)
M_projected = project_to_tangent_plane(M, u_spel, v_spel)
```

and only the branch rule uses the separate FP32 block-2 estimate:

```text
sigma1_gap, sigma2_gap = block2_fp32(W)
rel_gap = 1 - sigma2_gap / sigma1_gap
```

This is a cleaner theory-facing mode when the goal is to preserve the SpEL
trajectory while improving the precision of the dangerous-region test. Compared
with `deflated_fp32_gap_only`, it avoids explicit rank-one residual deflation and
usually gives a more coherent top-2 estimate near degeneracy, but it is a
slightly different branch-test algorithm.

## How To Enable

Use:

```bash
SPEL_PGD_GAP_ESTIMATOR_MODE=block2_fp32_gap_only
```

or pass the Megatron argument directly:

```bash
--spel-pgd-gap-estimator-mode block2_fp32_gap_only
```

For the FP32 deflated variant, use:

```bash
SPEL_PGD_GAP_ESTIMATOR_MODE=deflated_fp32_gap_only
```

The default remains `deflated_power` for reproducibility of older runs.

## Expected Tradeoff

The FP32 gap-only modes are more expensive than the old deflated estimator
because they keep the ordinary SpEL power iteration and add a separate FP32 gap
test. The tradeoff is interpretability: if `gap=0`, the SpEL update path should
match the original path much more closely, and any change in behavior should
come from the branch decision rather than from replacing the top singular
vectors used by SpEL.
