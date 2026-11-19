# Kernel-Preserving LMO: Local Evidence and Decision

[Theory and proofs](kernel_lmo_theory_20261111.md) |
[Generated tables](kernel_lmo_local_20261111/computed_tables.md) |
[Machine-readable summary](kernel_lmo_local_20261111/summary.json) |
[Code](../../scripts/local_kernel_lmo_study.py)

## Decision

Retain **A2**, **A2 with input scale 1.5**, and **K-LMO with theta=1** as
local candidates. Retain ordinary MCSD-TP and its matched-length replay as
controls. Do not submit any of them to H20 yet. K-LMO has a cleaner geometric
invariant than A4, but its extra LMO has not established a reliable practical
advantage over the cheaper normalized-momentum model.

The most useful new theoretical result is not a selected winning endpoint:
it is the separation of the **finite-step normal correction**, the **free
MCSD direction**, and the **actual radial-return error**. A second result is
that conditional bounded variance suffices for the new bounded-direction
proof; uniformly bounded individual gradient samples are unnecessary.

## Protocol

- 864 trajectories, 2000 steps each: 16 paired instance seeds, three shapes
  (4x4, 16x12, 12x16), two protocols and nine methods/control variants.
- Each comparison uses the same initial matrix and Gaussian noise sequence.
  Per-sample noise has expected squared Frobenius norm 0.05^2 R^2. Unlike
  the previous bounded-noise test, these samples have no deterministic bound.
- R=1, exact FP64 reference spectra and polar LMOs, no objective line search,
  no full-sphere projection fallback, and no online coupling certificate.
- The conservative EMA schedule and larger-step Nesterov schedule are the
  same as the previous study. The latter uses alpha_peak=0.0075R, 20-step
  warmup, cosine decay to 10%, h=0.0002R and momentum retention 0.9.
- Full selected windows, no top-4/top-8 cap. Of 48 instances per method and
  protocol, 36 have a known constrained optimum; 12 are anisotropic.
- The synthetic families are reused from the earlier study and are not a
  new independent problem distribution. Family, dense-tail choice and initial
  multiplicity are not fully factorial. No LLM validation loss is measured.

The earlier seven-method screening is superseded, not added to the 864-run
count. The completed nine-method run took 127.7 CPU seconds including audits.
Its runtime is not a training-kernel or H20 speed benchmark.

## Geometry and Complete Return

All 15 regression tests passed: 10 existing tests and 5 new tests. The new
tests include 864 model/descent/budget/coupling cases, explicit-dilation
agreement, radius scaling, zero momentum, q=1 equivalence, the bounded-variance
descent inequality with very large momentum errors, and replay repeatability.

In the trajectories, 2592 sampled complete-return audits found no violations
of either the new coupling bound or its corresponding full-objective upper
bound. The maximum recorded model-boundary error was 2.22e-15; sampled radius
error was 2.66e-15. No reference-momentum protection violation was observed.
These are numerical checks, not a proof of asymptotic convergence.

For K-LMO/theta=1, the new overshoot bound averaged 0.266 times the old
2 alpha^2 ||D||_2^2/Gamma bound in the conservative protocol and 0.343 times
it in the stress test, on audits where the old bound was applicable. This
tightens the estimated remainder, but does not by itself establish a smaller
required cluster capacity or certify decrease at all practical steps.

## Endpoints and Amplitude

Mean final f-f* over the 36 known-optimum instances:

| Method | Conservative EMA | Larger-step Nesterov |
|---|---:|---:|
| MCSD-TP | 1.987074 | 1.902510e-4 |
| MCSD-TP replaying A2 trial lengths | 2.987163 | 8.021322e-5 |
| A2 | 2.914992 | 6.895617e-5 |
| A4 | 2.015481 | 1.301111e-4 |
| A4 with pre-correction budget and guard | 2.914992 | 6.895617e-5 |
| K-LMO, theta=0.5 | 2.719439 | 6.834321e-5 |
| K-LMO, theta=1 | 2.512042 | 6.680288e-5 |
| A2, input scale 1.5 | 2.454970 | 6.783832e-5 |
| A2, input scale 2 | 2.170788 | 2.116513e-4 |

The conservative schedule is far from solving many instances within 2000
steps. MCSD-TP/A4 make more progress there partly by taking larger steps.
The stress test is near the optima and measures a different regime.

For the replay control, every trial Frobenius length is taken from the
corresponding A2 trajectory at that iteration. Maximum direction-length
mismatch was 2.22e-15. The two returned lengths need not match because radial
return depends on each evolving state. This is an offline causal-isolation
control, **not a deployable optimizer** and not a proof that A2's entire
benefit comes from a particular percentage of amplitude or geometry.

The A2 advantage over MCSD-TP becomes much smaller after this length match:
the stress-test known-optimum gap changes from 1.212948e-4 to 1.125704e-5.
Across all 48 paired objectives, A2 beats the replay in 19 cases, ties in 25
and loses in 4. This supports retaining the model correction, but not treating
the unmatched result as a pure geometry comparison.

K-LMO/theta=1 versus A2 in the stress test: **12 wins, 28 ties, 8 losses**,
mean paired difference -1.615e-6. The gain is small. A2/input-1.5 is comparably
competitive and avoids an extra near-branch LMO. Increasing the input to 2
is detrimental in the stress test despite being helpful on the conservative
schedule. Larger predicted decrease is not a sufficient selection rule.

The budgeted A4 candidate was rejected in all 25,215 near calls and exactly
reproduced A2, with extra work. It should not be taken to long training in
this form. The projection-optimality obstruction in the theory note explains
why competing with A2 at the same Frobenius budget is intrinsically difficult;
it does not claim that every possible budgeted candidate must be rejected.

## Handling a Corner Is Not the Same as Leaving It

In the stress test, A2 uses the near branch on 4.745% of steps; K-LMO/theta=1
on 5.730%. Conditional consecutive-near proportions are 75.83% and 97.27%,
respectively. The new algorithm **does not eliminate near-region repetition**.

Some test targets have several singular values above R. Their closest
spectral-sphere point has several singular values exactly R. Remaining near
this multiple-top optimum is legitimate; forcing a positive top gap can
actually move away from it. For the plateau-target family, stress-test mean
f-f* is 3.38186e-5 (A2), 2.72295e-5 (K-LMO/theta=1), and 4.61843e-4
(A2/input-2). These synthetic values do not establish the same mechanism in
LLM training.

Near-branch uphill rates are still 31.79% for A2, 33.50% for K-LMO/theta=1
and 56.07% for A4 under noisy Nesterov. Momentum protection does not protect
the full returned objective. These near-only rates are reported separately
from the overall rates in the generated tables.

## Capacity and Next Work

Eight stress-test sampled states per model method require q>8, reaching the
full smaller dimension 12. **All eight are initialization states with large
top multiplicity.** They are not a population estimate for LLM layers and
do not show frequent steady-state overflow. The conservative protocol has
20 such sampled states for the principal model variants. A fixed top-4 cap
cannot be claimed to satisfy the exact complete-window theorem on these
cases.

Recommended next work, still without new H20 long jobs:

1. Use saved real width-256 X/M/learning-rate snapshots to compare A2,
   A2/input-1.5 and K-LMO/theta=1 on **the same matrices**, using exact
   reference spectra initially. Record complete-return changes, residual
   coupling, required q and actual lengths. Objective claims need a replay
   loss or a justified local model, not just a momentum inner product.
2. Add paired length-matched controls for K-LMO versus A2/input-1.5 and
   stratify by true multiplicity, near-distinct roots, and whether the target
   optimum itself lies on a multiple-top stratum. Do not optimize solely
   for a small branch rate or a large exit gap.
3. Only then replace spectral data with the cached warm block implementation,
   separating direction/model error from return feasibility. Fixed K10 and
   top-4 still need empirical validation and an inexact-theory extension.
4. Keep the existing H20 five-arm confirmation separate. This work neither
   refreshed its status nor submitted, canceled, or changed any job.

## Reproduce

From the repository root, with a Python environment providing PyTorch:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover \
  -s scripts -p test_local_kernel_lmo_study.py -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/local_kernel_lmo_study.py \
  --steps 2000 --seeds 16 \
  --output docs/results/local_kernel_lmo_20261111/trajectories_with_amplitude_controls.json

python -B scripts/report_kernel_lmo_study.py \
  docs/results/local_kernel_lmo_20261111/trajectories_with_amplitude_controls.json \
  docs/experiments/kernel_lmo_local_20261111
```

The versionable summary includes source hashes and the raw-data hash. Raw
trajectories stay under the local ignored results directory. Production code,
the handoff and `main1.tex` remain unchanged.
