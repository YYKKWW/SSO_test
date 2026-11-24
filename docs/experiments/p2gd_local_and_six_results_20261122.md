# P2GD After Matching the Actual Returned Step

**Historical November22 report.** The single-step and local results below
remain unchanged. The subsequently approved
[independent short trajectories](p2gd_trajectory_panel_20261122.md) completed
and provide a favorable, single-start cumulative signal for cluster-cone;
their November25 analysis supersedes this report's scheduling proposals.

## Decision

**Keep P2GD as a mechanism candidate, but do not promote it to 1B/3B training.**
Synthetic matched-step tests show a geometric benefit over ambient/top-1
momentum. The completed LLM replay does not reproduce that advantage.
Simple output attenuation reproduces 98.6% of the cluster arm's mean gain
over half-D0 in this window. This ratio is descriptive, not a causal decomposition.

No new H20 training job was submitted for this analysis. No 3B run was started.
The existing 49 full-budget runs are unchanged.

## H20: Six Controls

Existing job **4487800**, one H20 on SPG-7-2, ran all 48 updates. Training
exited successfully after step 300 in **36:14**. Total scheduler elapsed time
was **37:16**, including startup and tests. The scheduler reports **FAILED**
because a final checker compared the entire new 48-step state audit with the
old 16-step audit, rather than their common prefix.

The corrected post-hoc check verifies all **80 state records at steps
253--268**, including weight, gradient and momentum hashes. Prefix losses
match exactly, all 48 optimizer updates succeeded, the initial optimizer state
matches, the 20 fixed evaluation batches are fresh, and checkpoint hashes are
unchanged. Frozen H20 training code and data were not modified or rerun.
The local and remote numeric summaries agree exactly.

[Numeric summary](../results/p2gd_six_20261122/h20_summary.json) |
[Integrity receipt](../results/p2gd_six_20261122/posthoc_integrity.json) |
[Protocol](p2gd_six_controls_20261122.md)

### Protocol and Results

Width 256, seed 1234, Nesterov 0.9, gap/cluster windows `2e-4`, the inherited
LR schedule, FP32 full-SVD reference spectra and common radial return are
fixed. Only half-D0 advances. The six alternatives are evaluated and restored
at each common state; these are **not six independent training trajectories**.

The predeclared window contains **32 steps**, of which **24 are informative**,
with **47 near components**, all numerical cluster size **two**. This batch
provides no evidence for increasing capacity beyond top-4.

Mean held-out loss differences below are relative to half-D0 on the 24
informative steps. Negative is better. All 32-step differences and raw values
are available in the numeric summary.

| Arm | Returned-step matching | Held-out difference | Better/worse than half-D0 |
|---|---|---:|---:|
| A: half-D0 | Reference | 0 | -- |
| B: attenuated half-D0 | Output ELR of E | -0.000858292 | 23 / 1 |
| C: ambient momentum | Spectral ELR of A | -0.000873760 | 23 / 1 |
| D: top-1 momentum | Spectral ELR of A | -0.000872028 | 22 / 2 |
| E: cluster-cone momentum | Spectral ELR of A | -0.000870545 | 23 / 1 |
| F: gap-preserving momentum | Spectral ELR of A | -0.000866211 | 23 / 1 |

The decisive contrasts, rather than the comparison with A alone, are:

| Contrast | Held-out difference | Better/worse steps | Interpretation |
|---|---:|---:|---|
| E minus C | +0.000003216 | 10 / 14 | No advantage over ambient |
| E minus D | +0.000001483 | 10 / 14 | No advantage over top-1 |
| E minus B | -0.000012252 | 15 / 9 | Tiny residual benefit after output matching |
| F minus E | +0.000004334 | 11 / 13 | No gain from the more complex finite-step repair |

E versus B worsens the training microbatch by **0.000098348**, despite the
small favorable held-out difference. This is mixed evidence, not a confirmed
direction improvement. Consecutive states and reused evaluation batches are
correlated; neither 24 states nor 47 components are independent training seeds.

### Complete Return and Precision

- A decreases held-out loss from the pre-update point in **15/24** informative
  steps. Each of B--F decreases it in **17/24**, not 23/24. Relative wins over A
  do not mean every full update decreases loss.
- Mean spectral ELR is about **0.008909** for A/C/D/E/F, versus **0.003179**
  for B. Mean calibration-output ELR is **0.011818** for A and about
  **0.004275** for B/E. Equal spectral movement is not equal output movement.
- E's independent-input output movement is **0.004161**, versus **0.004273**
  for B. Local calibration is not exact matching of the full network output.
- FP32 matching has **zero violations**. BF16 forward-weight rounding leaves
  maximum budget discrepancies of **6.89%** for E and **8.03%** for B.
  These are movement discrepancies, not measured loss-error bars.
- Maximum FP32 radius error is **1.79e-7**. Matching uses about 12--14 norm
  evaluations per modified component; runtime is a diagnostic cost, not a
  deployable optimizer benchmark.

## Local Spectral-Sphere Tests

All tests use CPU FP64, radius one and exact decompositions up to numerical
roundoff. Runtime was **206 seconds**. There are **5,760 trajectory records**
across six methods and **1,120 matched states** across six controls.
These include repeated noise seeds and related problems, not 5,760 independent
problem families. Zero-noise diagonal-square seeds are identical replicas.

### Theory-Aligned Trajectories

- Diagonal 2x2 square boundary: four targets, including a smooth-edge optimum,
  a corner optimum, an interior target and a limiting-stationary corner that
  still has a feasible descent direction; horizons 256 and 1024.
- Full 16x16 and 32x32 spheres: four quadratic target families, including
  optima with top multiplicity two and six; 512 updates.
- Twenty paired seeds, Gaussian noise Frobenius RMS `0 / 0.05 / 0.2`;
  identical noise samples and initial states across methods.
- Current-sample EMA, `beta=1-T^(-1/2)`, deterministic
  `alpha=min(0.1/(4*sqrt(n)), 0.5, T^(-3/4))`, safe absolute gap `0.1`.
  The toy gap is deliberately separate from the LLM threshold.
- References: exact PGD; MCSD-TP/exact-PGD hybrid; the same hybrid with its
  **safe** LMO direction multiplied by 0.5. The latter is not the LLM
  near-only half-D0 control.
- Exploratory radial branches: normalized ambient momentum, numerical-cluster
  cone momentum and true-multiplicity cone momentum. The last arm was added
  after the initial run to diagnose numerical-cluster stagnation. Its tie
  tolerance is `1e-12`; it is an oracle diagnostic, not a production estimator.

Maximum sphere error is **4.44e-15**. The safe trial's Frobenius displacement
is at most **0.25 times the certified gap**, inside the manuscript's local
domain. The diagonal problem uses the actual-edge limiting-normal residual:
a near-corner point is not rounded into a corner. Matrix problems use an
explicitly labeled projection-mapping diagnostic, not an asserted formula for
the limiting residual.

Illustrative mean final objective gaps, noise RMS 0.05, four target families:

| Method | 16x16, 512 steps | 32x32, 512 steps |
|---|---:|---:|
| Exact PGD | 0.011532 | 0.066576 |
| MCSD-TP / exact PGD | 0.002784 | 0.008806 |
| Safe-direction half-step / exact PGD | 0.003399 | 0.061922 |
| Numerical-cluster momentum / radial | 0.058431 | 0.096292 |
| True-multiplicity momentum / radial | 0.011955 | 0.038106 |
| Ambient momentum / radial | 0.105359 | 1.429991 |

These trajectories use prescribed, untuned steps and are **not ELR-matched**;
they diagnose behavior, not isolate direction superiority. Constant-factor
attenuation can delay progress at a fixed horizon. Two horizons do not validate
an asymptotic convergence rate.

An important negative case is the square problem with target `2I` and start
`-I`. At noise 0.05 and step 1024, the numerical-cluster radial arm has mean
objective gap **7.9401**, versus **0.002536** for the true-cone radial control
and **0** for exact PGD and the exact hybrid. The true-cone arm's final actual-
edge limiting residual is still about **1.0025** despite its small objective
gap. A nearby point is not the same as the stationary corner. This illustrates
the risks of treating separated singular values as exact ties; it does not
prove divergence or predict the LLM outcome at a much smaller threshold.

For the diagonal interior-target case, exact PGD can approach a nonglobal
stationary point. This is consistent with a stationarity theorem, not a
failure to meet a global-optimality guarantee that the manuscript never claims.

### Matched Complete Updates

The same six-way geometry code as H20 is applied to 16x16/32x32 matrices with
top clusters of size 1/2/4/6, true ties and nearby distinct values, coefficients
`0.005 / 0.015`, and momentum/gradient cosines `1 / 0.5`. Calibration and
independent inputs share an anisotropic covariance. No objective value enters
the coefficient search. The smooth q=1 sanity cases are reported separately.

For each alignment, the table contains **480 near states**. Entries are
cluster-cone loss minus control loss after the complete radial return.

| Control and matching | M=G | cosine(M,G)=0.5 |
|---|---:|---:|
| Half-D0, equal spectral movement | +0.00356171 | +0.00170964 |
| Attenuated half-D0, equal local-output movement | -0.00290860 | -0.00145722 |
| Ambient, equal spectral movement | -0.00044594 | -0.00021700 |
| Top-1, equal spectral movement | -0.00043143 | -0.00019354 |

There are **zero matching failures**. The exact quadratic identity
`loss_change = <gradient, returned_displacement> + 0.5*||returned_displacement||_F^2`
holds to **2.08e-16**. Every arm decreases the objective on these selected near
states; this is not a universal descent claim. The complete returned step,
not the pre-return direction alone, determines the loss change.

Thus a local geometric effect exists on these synthetic problems, but depends
on the budget definition and has not transferred to the measured LLM window.
Finite-step momentum improves over cluster momentum by only about **1.1e-6**
locally; that extra complexity is not supported by the H20 comparison.

[Configuration](local_paper_sphere_20261122/config.json) |
[Paired analysis and seed-group intervals](local_paper_sphere_20261122/analysis.json) |
[All trajectory summaries](local_paper_sphere_20261122/trajectory_summary.json)

## Next H20 Decision

1. **Do not submit another six-arm grid or 3B run.** Deprioritize finite-step
   repair and larger cluster capacity for now. The LLM point estimates do not
   justify selecting P2GD as a winner.
2. First prepare one later-checkpoint, 16--32-state diagnostic, at most one H20:
   keep gap, momentum, radial backend and data pairing fixed. Retain cluster,
   ambient/top-1 and output-attenuated D0 as controls, not just cluster versus A.
3. Add correction-strength, active correction rank, momentum/top-block angle
   and BF16 returned-displacement differences. Match/audit the **rounded forward
   weights**, not only FP32 masters. If rounding prevents an accurate match,
   report the attainable error rather than silently declaring equality.
4. Only if cluster beats both the simpler direction and the shrink control at
   the new state window, run 128--256 independent paired updates. Otherwise,
   focus on bounded amplitude control and retain P2GD as a negative mechanism
   result. A later-checkpoint confirmation is not submitted by this report.

Keep the paper's exact-projection hybrid as the theoretical reference. The
local radial variants and current Nesterov LLM replay are not automatically
covered by that theorem. Low final loss alone is not a stationarity certificate.

## Reproduce

```bash
python -m unittest discover -s scripts -p test_local_paper_sphere.py
python -m unittest discover -s scripts -p test_verify_p2gd_six.py
python scripts/local_paper_sphere.py --output /tmp/paper-sphere \
  --seeds 20 --square-horizons 256 1024 --matrix-steps 512
python scripts/report_local_paper_sphere.py /tmp/paper-sphere
python scripts/summarize_p2gd_six.py /path/to/six/metrics /tmp/six_summary.json
python scripts/verify_p2gd_six.py /path/to/six/metrics \
  /path/to/prior/metrics /tmp/six_integrity.json
```

The local scripts need PyTorch and NumPy, no Megatron/CUDA or plotting packages.
Source hashes, per-seed outcomes and per-state comparisons are retained.
The corrected checker is covered by tests for changed/missing prefix records;
it does not relax state equality to make a failed comparison pass.
