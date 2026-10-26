# H20 pilot: near-cluster directions with unified radial returns

Date: 2026-10-21. This is an exploratory short LLM experiment requested after
a local stage-1 calibration (a preliminary study not included in this
publication bundle).
The local adaptive variants did not robustly beat MCSD-TP. This submission
tests transfer of the mechanism, not an established winning configuration.

## Completion update, 2026-10-21

All four arms completed successfully. Final step191 validation losses are
4.947731 (control), 4.954692 (fixed4/c1), 4.948502 (adaptive4/c1), and
4.957767 (adaptive4/c0.5). Wall times are 53:20, 53:21, 52:43, and 53:24.
The initial advantage and absence of repeated cluster calls were early-stage
observations only: fixed4/c1 ultimately had a61.6% repeat fraction among its
698 recorded cluster calls through step190. Full results, matched baselines,
second-seed comparison and follow-up grid are in the
[completed follow-up report](cluster_radial_followup_100m_20261021.md).

## Submission

Final array **4290663**, maximum concurrency **two single-H20 tasks**:

| Array task | Method | Matched-length coefficient c |
|---|---|---:|
| 4290663_0 | MCSD-TP control | Not used |
| 4290663_1 | Fixed top-four near-cluster direction | 1.0 |
| 4290663_2 | Adaptive cluster, capped at four | 1.0 |
| 4290663_3 | Adaptive cluster, capped at four | 0.5 |

The first submission, array 4290599, was canceled after 8 minutes 35 seconds
to correct a logging-only defect: the generic statistics collector discarded
the matching-multiplier sum before the logger computed its mean. Updates were
unaffected, but the printed multiplier mean was incorrectly zero. At step20,
both running arms had no NaN/skips, took about 16.1 seconds/update, and fixed4
had already used its cluster branch once. Those truncated runs are debugging
artifacts, not final experiment results.

The collector now computes the mean before dropping raw sums; a new test
executes the real collector and checks a known nonzero mean. Array 4290663
restarts all four arms with unchanged algorithm settings. The original
`overlay.tgz` and canceled logs are retained; `overlay_v2.tgz` is the final
upload. Each final arm also saves its own source-hash manifest.

### Verified final-array progress

At **01:46:56 HKT**, tasks 4290663_0 and 4290663_1 were both at **20/191**
updates on SPG-7-1; tasks 2 and 3 were waiting for the two-task array limit.
Both running tasks passed all nine CUDA checks and had zero skipped/NaN
iterations. Their actual launch settings differ only in the cluster mode
and task-specific output/cache paths and IDs; their source manifests match.

| Metric at step20 | MCSD-TP control | Fixed top-four |
|---|---:|---:|
| Last logged training loss | 7.856270 | 7.856270 |
| Seconds/update, last interval | 16.1187 | 16.1303 |
| Cumulative cluster decisions | 0/5040 | 1/5040 |
| Current cache hits / bootstraps | 252/0 | 252/0 |
| Maximum radial correction / trial displacement | 0.1924 | 0.1924 |

These are early training aggregates, not validation results or evidence of
equal final performance. Fixed-four's first branch has a correctly recorded
matching multiplier of approximately 5.174e4. This rescales raw momentum
to the reference direction length; it does not mean the intended trial
displacement is 5.174e4 times the MCSD displacement. Its flattened-cluster
distance is 0.04677R, illustrating why a fixed top-four cluster need not be
a genuinely near-multiple-root model even when the first gap is tiny.

At submission SPG-7-1 had three free H20 GPUs. This pilot takes two, leaving
one available. SPG-7-2 was draining and is not used. The two existing user
training tasks were not modified. Home storage had approximately 32 GiB free.
No checkpoints are saved.

The prior same-size pilot took approximately 53 minutes per method. Allow
roughly one hour per arm, or about two hours for two waves, excluding queue
delays. New branch costs may change this estimate; it is not a measured
speedup. Each task has a three-hour wall-time limit.

## Matched training protocol

- Width 256, 28 transformer layers, FFN width 768, head dimension 128.
- Sequence length 4096, global batch 128, microbatch four, seed 1234.
- 191 updates, exactly 100,139,008 scheduled training tokens.
- Peak LR 1.5e-2, 25 warmup updates, cosine decay to 1.5e-3.
- Validation every 50 updates and at the end; five validation batches.
- Nesterov momentum 0.9, spectral-muP radii and component LR scaling.
- BF16 model arithmetic and matrix-sign iterations; FP32 master weights,
  tangent arithmetic, spectral cache and full SVD (`gesvd` on CUDA).
- Eight Newton--Schulz iterations, tangent correction before and after msign.
- Constrained matrix weight decay zero; auxiliary weight decay 0.1.
- The current-point cache is used for every branch decision. No PI gap
  estimate, periodic gap-skipping, online error gate, line search or metric
  projection fallback is used in this pilot.

The data source is the existing **3B indexed corpus**, but each run consumes
about **100M tokens**. The old 1B indexed training source is no longer present
on the server. This experiment must not be inserted into the historical
1B-source / 1B-token main table.

The routing threshold and near-cluster selection threshold are both 5e-5
in this first H20 comparison. No cluster-threshold sweep is included yet.

## Exact implementation

The new opt-in module is
`Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cluster_radial.py`.
The host optimizer defaults to `radial_cluster_mode=none`, preserving the
old update path. Its experimental values are `control`, `fixed2`, `fixed4`
and `adaptive4`; fixed2 is tested in unit checks but not submitted here.

For the current cached paired singular bases, write Z=-M and
A=sym(U_q^T Z V_q). An eigendecomposition of A replaces the largest
eigenvalue by zero and all other positive eigenvalues by zero to form B.
The near-cluster direction is

$$D_{\rm cluster}=Z+U_q(B-A)V_q^\top.$$

Fixed-four uses the first four pairs (or all pairs if a component is
smaller). Adaptive-four selects the values within the cluster tolerance,
capped at four. Cluster size is deliberately **not** logged as an exact
top multiplicity. A nonzero small gap is not an algebraic multiple root.

The safe direction is the same tangent / BF16 msign / tangent pipeline
as the cached MCSD-TP reference. On a cluster step the trial coefficient is

$$
\alpha_{\rm cluster}
 = c\,\eta_{\rm effective}
 \frac{\|D_{\rm MCSD}(X,M)\|_F}{\|D_{\rm cluster}(X,M)\|_F}.
$$

Here eta_effective already includes the component's spectral-muP scaling.
The scalar c is not the old raw PGD step multiplier. The extra reference
MCSD evaluation on a cluster step is part of this diagnostic and must be
included in any speed comparison.

Both branches then use

$$Y=X+\alpha D,\qquad X^+=RY/\sigma_1(Y).$$

One FP32 trial SVD supplies U, s and V. The next cache stores paired vectors
and transformed singular values `R*s/s1`, so the next gap belongs to the
returned matrix. The initial zero-LR step builds the first cache with one
current-point decomposition. Thereafter the tested update has one full SVD
per component per update and no separate `svdvals` call. Offline unit-test
audits are not counted as optimizer calls.

In the launch configuration, `projection_mode=tp_radial_pgd_exact` selects
the existing cache infrastructure. The explicit `radial_cluster_mode`
dispatch overrides the update and both returns; update diagnostics identify
the actual mode as `cluster_radial`, and `launch_config.json` records
`actual_return=radial_both_branches`. There is no full-SVD metric projection
in this opt-in path.

As in the local experiment, fixed-threshold near-cluster directions plus
radial returns do not automatically inherit standard P2GD or exact Clarke
convergence results. FP32 cached algebraic values also differ slightly from
the independently decomposed rounded master matrix.

## Diagnostics and verification

43 local unit/regression tests passed before the final packaging. The nine
CPU/CUDA tests cover rank choice, complete-return caches, sphere feasibility,
literal matched-step/radial formulas, one trial SVD, unchanged safe directions,
bootstrap behavior, cap overflow, optimizer dispatch and preservation of the
matching-multiplier mean through training-statistics aggregation.
The running H20 environment is PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA H20.

Training logs retain the existing branch, repeat/burst and cache statistics,
plus:

- cumulative near-cluster rank 1/2/3/4 counts;
- cluster-cap overflow count, without automatic expansion or fallback;
- mean applied matching multiplier;
- maximum radial correction divided by intended trial displacement;
- maximum normalized distance to the flattened near-cluster model.

The host logger still labels the non-MCSD branch `pgd`. For this opt-in run
that counter means **near-cluster/radial branch**, not ambient PGD or metric
projection. Rank overflow checks inspect the retained leading eight values;
the fifth value suffices to detect overflow of the four-dimensional cap.
The complete active cluster count above eight is not reported.

Assess final and matched-step validation, training loss, per-update time,
late branch rates and bursts together. Do not select a method just because
it uses fewer cluster steps or produces a larger gap. There is only one LLM
seed, and no matched MuonSphere/SSO arm in this short mechanism comparison.

## Provenance and artifacts

An isolated 68 MiB source snapshot was copied from the previously validated
`/home/u3013198/projects/p2gd-cached-pair-20261020/Megatron-LM` into
`/home/u3013198/projects/cluster-radial-20261021`.
Only the explicit overlay was applied. Running projects and the frozen
paper repository were not overwritten.

- Launcher: `slurm/cluster_radial_100m.sbatch`, array 0-3 with concurrency 2.
- Local upload and hashes: `output/cluster_radial_h20_20261021/`.
- Downloaded launch/CUDA receipt: `launch_receipt_4290663.tgz` in that local directory.
- Remote overlay checks: `overlay.sha256`, verified before each task.
- Remote logs: `logs/cluster_4290663_{0,1,2,3}.{out,err}`.
- Remote per-arm results: `results/4290663_<task>/`.
- Each arm saves its complete launch configuration and `train.log`.
- A `COMPLETED` marker is written only after a successful training process.

All result paths above are relative to the isolated remote snapshot unless
explicitly marked local. No final validation result was available at this
initial submission record.
