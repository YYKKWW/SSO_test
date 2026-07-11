# Primary MCSD and MCSD-PGD Implementations

This note defines the two algorithms used in the locked 1B-token primary
matrix. It describes the code that actually runs, including dtype casts that
differ from a few stale comments in the implementation.

## Naming

| Paper-facing label | Repository optimizer | Required mode |
|---|---|---|
| MCSD / SpEL | `spel_dist` | `topk`, rank `8`, post-msign tangent projection disabled |
| MCSD-PGD | `spel_pgd_dist` | `shared_topk`, rank `8`, post-msign tangent projection disabled |

Historical `spel_tp_dist` and MCSD-TP-PGD jobs are separate supplementary
ablations. They are not used to fill the primary matrix.

## Shared Optimizer Wrapper

Both algorithms inherit the same matrix-optimizer wrapper. For each supported
matrix component it performs decoupled weight decay, then updates the momentum
buffer

```text
m_t = beta * m_(t-1) + (1 - beta) * g_t
```

with `beta=0.9`. With the configured Nesterov flag, the tensor passed to the
matrix algorithm is exactly the implementation's interpolation

```text
M_t = (1 - beta) * g_t + beta * m_t.
```

Fused QKV weights are split by attention head, and gated FC1 weights are split
into gate/up components. The spectral operations below run independently for
each resulting 2D matrix.

For a matrix with shape `(n_out, n_in)`, both optimizers use

```text
target radius R = sqrt(n_out / n_in)
update scale  c = sqrt(n_out / n_in)
effective step eta_eff = learning_rate * c.
```

## MCSD / SpEL

The locked MCSD row is `spel_dist` with `projection_mode=topk`, rank `8`, hard
retraction, and no post-msign tangent projection.

For every matrix and optimizer step:

1. Estimate the leading singular triplet of the current weight. The shared
   `power_iteration` forcibly casts `W` to BF16, initializes `v` to all ones,
   and runs 10 bilateral iterations

   ```text
   v <- normalize(W^T (W v)),
   u <- normalize(W v),
   sigma1 <- u^T W v.
   ```

   This is a deterministic cold start; `sigma1/u/v` are BF16-path estimates.

2. Apply hard pre-retraction in place:

   ```text
   W <- R * W / (sigma1 + 1e-8).
   ```

3. Cast the Nesterov momentum and singular vectors to FP32 and remove its
   normal component on the smooth spectral-sphere stratum:

   ```text
   M_tan = M - (u^T M v) u v^T.
   M_tan <- M_tan / ||M_tan||_F.
   ```

4. Compute `D = msign(M_tan)` with 8 Newton-Schulz steps. In the current H20
   code, `msign` receives FP32 `M_tan`, normalizes it, and the forced
   `_small_msign` branch then casts its iterate to BF16.

5. Do not re-project `D` onto the tangent plane after `msign`; the locked
   primary setting has `SPEL_TANGENT_PROJECT_AFTER_MSIGN=0`.

6. Form the FP32 trial point

   ```text
   Z = W - eta_eff * D.
   ```

7. Apply the rank-8 approximate spectral-sphere projection. The projection
   keeps `Y` and its deflation residual in FP32. For each of up to eight
   components it calls the same BF16 power iteration to estimate a leading
   triplet, then applies the rank-one correction in FP32. Singular values above
   `R` are reduced to `R`; if the first estimate is below `R`, the first
   component is increased to `R` so the target is the sphere, not merely the
   interior of the ball.

8. Encode the projected target as the update consumed by the outer optimizer:

   ```text
   update = (W - Project_topk8(Z)) / eta_eff.
   ```

## MCSD-PGD

The locked MCSD-PGD row preserves the MCSD main path above and adds an
independent high-precision region test plus a PGD fallback.

### Main Singular Path

`SPEL_PGD_MAIN_POWER_DTYPE=bf16` calls the original SpEL power iteration for
the main `sigma1/u/v`. The FP32 gap estimator never replaces these vectors.
Warm start is disabled, so this path remains the original all-ones cold start.

### FP32 Gap Test

When a component is scheduled for probing, `block2_fp32_gap_only` casts `W` to
FP32 and constructs a deterministic two-column right subspace from fixed
trigonometric vectors. It runs 10 block subspace iterations:

```text
U <- qr(W V)
V <- qr(W^T U)
```

It then computes the SVD of the projected `2 x 2` matrix

```text
B = U^T W V
```

to obtain independent FP32 decision estimates `gap_sigma1` and `sigma2`. The
branch statistic is

```text
rel_gap = max(0, 1 - sigma2 / max(gap_sigma1, eps)).
```

These block-2 Ritz vectors are discarded. Only the scalar gap statistics enter
the branch decision.

### Adaptive Probe Schedule

The schedule is maintained separately for every parameter/component key:

- The first optimizer step is always probed.
- If the last measured `rel_gap > 10 * 3e-4 = 3e-3`, the next probe is five
  optimizer steps later.
- Otherwise, the component is probed again on the next step.
- A skipped probe forces the MCSD branch; it does not reuse or remember an old
  PGD decision.

### Branches

On a probe step, PGD is selected only when

```text
rel_gap < 3e-4.
```

The MCSD branch computes the same BF16-main-path tangent direction and
Newton-Schulz `msign` direction described above.

For the locked primary configuration, the MCSD branch stops at
`D = msign(M_tan)`: `SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0`, so it does not
project `D` onto the tangent plane again. The PGD branch never calls that
post-msign tangent projection. The later `shared_topk` operation is a
spectral-sphere projection of the trial weight, not a tangent-plane projection.

The PGD branch starts from the FP32 Nesterov momentum. With spectral direction
normalization it divides by an estimated leading singular value:

```text
D_pgd = M / sigma1(M).
```

The current `sigma1(M)` helper is the same forced-BF16 power iteration used by
SpEL, even though `M` is first cast to FP32. Thus the direction tensor and
division are FP32, while this normalization scalar is a BF16-path estimate.

### Shared Projection

`shared_topk` means both branches form a trial point and run the same rank-8
projection. The only branch-specific step-size change is

```text
MCSD: eta_trial = eta_eff
PGD:  eta_trial = 0.5 * eta_eff.
```

The final update is always encoded as

```text
update = (W - Project_topk8(W - eta_trial * D)) / eta_eff.
```

There is no line search, loss reevaluation, sticky PGD flag, or cooldown.

## Precision Summary

| Operation | Actual primary-matrix precision |
|---|---|
| Model training | BF16 (`--bf16`) |
| Main MCSD `sigma1/u/v` | BF16 power iteration, 10 steps, all-ones cold start |
| Tangent projection | FP32 |
| Current small-matrix `msign` iterate | BF16 after FP32 input normalization |
| Top-k residual and rank-one corrections | FP32 |
| Top-k singular-triplet estimates | BF16 power iteration |
| MCSD-PGD block-2 gap decision | FP32 QR/subspace iteration and small SVD |
| PGD direction tensor/division | FP32 |
| PGD spectral-normalization scalar | BF16 power-iteration estimate |

## Locked MCSD-PGD Environment

```text
SPEL_PGD_BRANCH_MODE=auto
SPEL_PGD_PROJECTION_MODE=shared_topk
SPEL_PGD_PROJECTION_RANK=8
SPEL_PGD_MAIN_POWER_DTYPE=bf16
SPEL_PGD_GAP_ESTIMATOR_MODE=block2_fp32_gap_only
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10
SPEL_PGD_GAP_THRESHOLD_REL=3e-4
SPEL_PGD_GAP_PROBE_INTERVAL=5
SPEL_PGD_GAP_PROBE_SAFE_MULTIPLIER=10
SPEL_PGD_WARM_START_UV=0
SPEL_PGD_DIRECTION_NORMALIZATION=spectral
SPEL_PGD_PGD_LR_SCALE=0.5
SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0
```
