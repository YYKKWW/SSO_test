# Protocol decisions after the handoff

1. The user chose empirical continuation at small spectral gap, matching the
   operational behavior of the existing spectral baselines. A gap at or below
   `1e-4` is a diagnostic warning, not a stop or a PGD trigger. This **does
   not** extend the smooth-stratum theorem to repeated leading singular
   values. Report the frequency and loss of such runs separately.
2. Practical language-model runs use BF16 Polar Express for the LMO. Stiefel
   initialization and returns use full finite-step `msign` plus three FP32
   polishing steps; exact SVD polar is retained for small-matrix reference
   checks. The finite-step map must be labelled approximate and audited.
3. Frobenius-sphere returns are FP32 norm scaling and require no SVD. We do
   not adopt the extra unit-Frobenius tangent-direction normalization from
   the cited hyperspherical optimizer, since it would change the MCSD step.
4. The PI/top-8 spectral return is an exploratory practical variant, not the
   exact metric projection in the handoff. On a seeded 128x128 synthetic
   matrix, BF16 PI10/top-8 gave relative post-return spectral-norm defect
   about 0.015. It is **not approved for a theorem-aligned main table**.
   Compare against the exact SVD control and either improve the approximate
   return or keep the exact reference despite its runtime cost.

The source `experiment_spec.json` is kept unchanged so the original contract
can be audited. These changes require a new protocol/method identifier in
every result table, not silent relabeling of historical MCSD experiments.
