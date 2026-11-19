# Ambient-Radial versus MCSD-TP: 3B Follow-Up

## Latest: Matched Warm Controls Completed, November 18

Checked 10:44 UTC+08: `4453017_0-1` both completed with exit0:0 at step5723,
44 CUDA tests and smoke passed, and no logged NaN/skipped iterations.

| Warm block-8 K10 arm | Final loss | Main train + validation |
|---|---:|---:|
| Current MCSD-TP control | 3.429645 | 17:56:55 |
| **Near-only half-D0** | **3.423857** | 17:57:09 |
| Completed ambient rho=0.5 | 3.432013 | 17:54:03 |

Keeping D0 and attenuating it near small gap beats both switching to ambient
momentum and doing nothing. The gain over control is 0.005788 in this seed.
The winning arm has MORE near calls (14.067%) and repeats (54.771% of near
calls), so fast escape from near regions is not a sufficient optimization goal.
This is MCSD-TP step attenuation, not PGD/P2GD. Next test an all-step
matrix-only half-D0 control, paired replication, and same-state full-return
loss probes before promoting a new cone direction. No new tasks were added.
[Full results, configuration differences and next plan](warm_radial_controls_3b_20261117.md).

<a id="latest-warm-k10-completed-november-17"></a>

## Previous: Warm-K10 Completed, November 17

**Checked 2026-11-17 10:44 UTC+08.** Job `4442918_2` completed at 07:43
with exit 0:0, both completion markers, and zero NaN/skipped iterations in
all 573 regular training reports. Final evaluation is at step 5723. Main
training plus validation took **17:54:03**; the full allocation took 17:56:22,
including preflight, smoke and offline audits. The account queue is empty;
this does not imply that other users' GPUs are free. Home storage has about
20 GB available. That results-only check added no jobs; the subsequent
11:24 submission is recorded under Next Experiments below.

All rows use width256, seed2345, LR0.015, 3B processed tokens and eval20.
Warm versus exact rho=0.5 changes only spectral estimation and its mode
marker among the recorded training settings, excluding path metadata.
Historical methods share the data/model/schedule protocol but not all
algorithm details. Times below exclude preflight, smoke and offline audits.

| Method | Final validation loss | Train + validation | Job |
|---|---:|---:|---|
| **New warm block-8 K10, rho=0.5** | **3.432013** | **17:54:03** | `4442918_2` |
| Exact ambient-radial, rho=0.5 | 3.429565 | 25:47:55 | `4434107_2` |
| Exact MCSD-TP control | 3.430719 | 25:47:49 | `4434107_0` |
| Exact ambient-radial, rho=1.0 | 3.436268 | 25:44:49 | `4434107_3` |
| Historical MuonSphere | 3.436023 | 16:35:03 | `4346705_0` |
| Historical SSO | 3.435041 | 18:28:11 | `4346705_1` |
| Historical MCSD-TP, warm block-8 K10 | 3.428613 | 17:39:34 | `4346705_2` |
| Historical ambient-radial, K10/c8 | 3.427854 | 17:40:58 | `4346705_4` |

### Interpretation

- Replacing full SVD saves **7:53:52 (30.6%)** against the exact half-step
  hybrid, but increases loss by **0.002448**. The warm method is 7.9% slower
  than MuonSphere and 3.1% faster than SSO. These are end-to-end historical
  runs, not isolated optimizer-kernel benchmarks.
- Its loss is 0.004010 below MuonSphere and 0.003028 below SSO in this seed.
  It is nevertheless **0.003400 worse than historical K10 control** and
  0.004159 worse than historical ambient/c8. It is a viable low-cost
  implementation, not the strongest observed method or a replicated winner.
- The exact-to-warm change affects initialization, tangent vectors, routing
  and radial normalization together. Similar aggregate routing rates do not
  show that the same matrices took the same branch. Neither the loss gap nor
  the small sampled gap errors identify the causal mechanism.
- This is **MCSD-TP plus an ambient-radial near branch**, not PGD or standard
  P2GD. There is no near-branch tangent-cone correction or sphere projection.

| Iteration | Exact control | Exact rho=0.5 | Warm rho=0.5 |
|---:|---:|---:|---:|
| 4750 | 3.492759 | 3.488971 | 3.495223 |
| 5000 | 3.469515 | 3.467204 | 3.470299 |
| 5250 | 3.460761 | 3.460162 | 3.462290 |
| 5500 | 3.457448 | 3.455646 | 3.458169 |
| 5723 | 3.430719 | 3.429565 | 3.432013 |

### Routing, Returns and Numerical Audit

| Diagnostic | Exact rho=0.5 | Warm rho=0.5 |
|---|---:|---:|
| Component updates | 1,441,944 | 1,441,944 |
| Ambient-near calls | 81,948 (5.683%) | 80,422 (5.577%) |
| Consecutive repeats / near calls | 13,130 (16.022%) | 12,688 (15.777%) |
| Near returned/trial length, ratio of sums | 102.983% | 102.927% |
| Near returned/D0-trial length, ratio of sums | 51.491% | 51.463% |
| Original-matrix full SVDs online | 1,442,196 | 0 |
| Small Ritz SVDs online | 0 | 1,442,196 |

Both counts include 252 bootstrap solves plus one trial solve per positive-LR
component update. Warm K10 uses no online SVD fallback. Its selected q never
exceeds three, but this alone cannot exclude omitted directions. The two
branches deliberately do **not** match ELR: near trial Frobenius length is
half of same-state D0; returned and spectral ELR are different quantities.
The measured 51.463% is not proof that half-length is optimal.

The offline FP64 audit of eight saved FP32 states reports mean/max current
gap error **1.48e-6 / 5.60e-6**, maximum returned relative radius error
**4.85e-6**, and no false/missed near decisions in those samples. However,
only **two samples are truly near**, both early (steps 31 and 37); this is
not an unbiased estimate of training-wide routing accuracy. No tangent-angle
accuracy is established. At step 2 the *current* radius error is 1.24e-4,
before that step's return reduces it to 4.85e-6. Returned tensors are saved
before host update rounding. That initial report audited eight of 32 snapshots.

**Expanded local audit, November17:** all32 are now checked. Mean/max gap
errors are 1.08e-4 / 2.93e-3, with the largest error at step2 on a safely
separated 1536x256 matrix. Nine samples are truly inside the near window;
their maximum gap error is 8.20e-7. No sampled false/missed branch is found.
Maximum returned radius error is 1.50e-4, and one startup correction gives
5.51 times the nominal trial displacement. This supersedes any inference
that the initial eight-sample error bound covered every saved state.
[Cone angles, complete-return diagnostics and limitations](corner_momentum_audit_20261117.md).

### November 17 Plan (Controls Now Completed Above)

**Submission update, November17 11:24 UTC+08:** controls 2 and 3 below are
now submitted as **`4453017_0` / `4453017_1`** and are pending (Priority).
Each uses one H20, at most two concurrently, with checkpoints disabled.
44 local/server CPU checks passed; CUDA checks and smoke await allocation.
[Frozen configurations, implementation and receipt](warm_radial_controls_3b_20261117.md).
The other proposed audits/conditional runs have not been newly executed.

1. **Audit the existing snapshots first.** Extend the audit to all 32 and
   compare warm/exact top-1 normals (up to sign), leading subspaces, D0,
   actual returned displacement and momentum pairing on the same matrix.
   Accurate singular values do not establish accurate tangent directions.
   For future runs, sample late near-boundary states as well as early calls.
2. **Current-backend warm MCSD-TP control.** Keep width256, seed2345, 3B,
   LR0.015, K10, FP32 Ritz, initialization and radial return identical; always
   use D0 and retain the 2e-4 window only for diagnostics. Historical K10
   control is useful context but is not a source-matched current-backend arm.
3. **Warm near-step attenuation control.** Use the same routing; safe D=D0,
   near D=0.5*D0, with the same radial return. Compare against the completed
   ambient half-step arm. This isolates a direction change from halving
   trial length without rerunning the existing exact or baseline methods.
4. **Conditional follow-up.** If ambient beats both matched controls, repeat
   the winning pair at a second seed. If half-D0 matches it, prioritize step
   attenuation and do not credit ambient/cone geometry. If all new arms trail
   historical K10, audit the initialization/cache/update path before a broader
   hyperparameter search. An exact-bootstrap-only ablation is warranted only
   if the snapshot audit implicates startup error.

Defer a larger gap grid, K40, wider models and 12B runs. Retain rho=0.5 and
window2e-4 as the current reference, not a proven optimum. A 1B screen cannot
replace the 3B mechanism comparison: prior method rankings depended on budget.

Evidence: [final summary, curves, health and checksums](../results/warm_ambient_radial_3b_20261116/final_summary_20261117.json),
[compact log evidence](../results/published_log_summaries_20261119.json),
[branch statistics](../results/warm_ambient_radial_3b_20261116/pilot_statistics.json),
[FP64 audit](../results/warm_ambient_radial_3b_20261116/warm_spectrum_audit.json),
[server status](../results/warm_ambient_radial_3b_20261116/server_status_20261117.json).

## Previous Exact Results: November 16, 11:16 UTC+08

**All three runs completed successfully at step 5723.** Slurm exit codes
are 0:0, both completion markers exist, and all 573 regular training reports
per arm have zero NaN/skipped iterations. Each allocation passed 32 checks
and its separate four-step smoke. This update adds no jobs.

All rows below are width256, seed2345, LR0.015, 3B processed tokens and
eval20. Times are main-process training plus validation, excluding queue,
preflight/smoke and the post-training audit. Historical baselines reuse the
previously checked data/model/schedule protocol; historical experimental
methods differ in spectrum estimation, routing and near-direction scaling.

| Method | Final validation loss | Train + validation | Job |
|---|---:|---:|---|
| Exact MCSD-TP control | 3.430719 | 25:47:49 | `4434107_0` |
| **Exact ambient-radial, rho=0.5** | **3.429565** | 25:47:55 | `4434107_2` |
| Exact ambient-radial, rho=1.0 | 3.436268 | 25:44:49 | `4434107_3` |
| Historical MuonSphere | 3.436023 | **16:35:03** | `4346705_0` |
| Historical SSO | 3.435041 | 18:28:11 | `4346705_1` |
| Historical MCSD-TP, warm block-8 K10 | 3.428613 | 17:39:34 | `4346705_2` |
| Historical ambient-radial, K10/c8 | **3.427854** | 17:40:58 | `4346705_4` |

### What the Completed Runs Show

- rho=0.5 improves on its matched exact control by **0.001154**. Its final
  loss is 0.006458 below MuonSphere and 0.005476 below SSO in this seed.
  This is a single-seed result, not a demonstrated reproducible advantage.
- It is still **0.001711 worse and 45.9% slower** than the old K10/c8 ambient
  reference, and 0.000952 worse than old K10 MCSD-TP. Changing spectrum,
  window and scaling together prevents attributing this difference to SVD
  alone. The exact hybrid costs 55.6% more wall time than MuonSphere.
- rho=1.0 is **0.005549 worse than exact control**, although it has fewer
  near calls and repeats. Reducing repeats is not a sufficient selection
  criterion. Do not promote this full-length setting to longer training.
- Both hybrid arms use normalized ambient momentum followed by **radial
  return**, not sphere projection or a P2GD tangent-cone correction.

The late evaluations retain the rho=0.5 advantage over exact control, but
its endpoint advantage is smaller than the earlier 3500-step difference.

| Iteration | Exact control | rho=0.5 | rho=1.0 |
|---:|---:|---:|---:|
| 4750 | 3.492759 | 3.488971 | 3.500251 |
| 5000 | 3.469515 | 3.467204 | 3.476301 |
| 5250 | 3.460761 | 3.460162 | 3.467038 |
| 5500 | 3.457448 | 3.455646 | 3.463989 |
| 5723 | 3.430719 | 3.429565 | 3.436268 |

### Final Branch and Return Diagnostics

Denominator: **1,441,944 matrix-component updates per arm**, not 5723
whole-model decisions. For control, eligibility/repeats are diagnostics;
the actual ambient branch never runs. Repeat fraction is repeated near calls
divided by near calls, not by all component updates.

| Diagnostic | Exact control | rho=0.5 | rho=1.0 |
|---|---:|---:|---:|
| Near-window eligible calls | 135,112 (9.370%) | 81,948 (5.683%) | 56,570 (3.923%) |
| Actual ambient calls | 0 | 81,948 | 56,570 |
| Consecutive eligible/near calls | 54,932 (40.657%) | 13,130 (16.022%) | 5,244 (9.270%) |
| Maximum selected q | 4 | 3 | 3 |
| Mean near direction norm / norm(D0,F) | 1.000 | 0.500 | 1.000 |
| Near returned/trial length, ratio of sums | 99.814% | 102.983% | 105.405% |
| Near returned/D0-trial length, ratio of sums | 99.814% | 51.491% | 105.405% |

The final 5.683% / 16.022% half-step statistics supersede the interim
1.366% / 4.002% numbers below. Each run records **1,441,944 trial SVDs plus
252 bootstrap SVDs**, consistent with one trial decomposition per component
update. No q>4 is recorded in this batch. Capacity requirements from the
earlier finite-step repair experiments are a different measurement.
The PGD-only zero-plateau counter is inactive and cannot certify positive
returned gaps. All three post-training snapshot audits also completed.

### Status and Next Decision

At the check, these three jobs and ELR prefix `4434317` / eight short branches
`4434318_0-7` have completed. ELR branches are only 32-step continuations;
their training-microbatch probes are not final validation results and are
excluded from this table. The account queue contains one separate Frobenius
width384 job (`4428784`, one H20). Home storage has approximately 22 GB free.

Retain rho=0.5 as the exact-spectrum mechanism reference, alongside the
faster K10 control/ambient references. Before another long run, distinguish
the effects of changing direction and halving the near trial length; a
matched `0.5 * D0` near control would isolate the latter. Reuse the existing
baselines. No further training or resource changes are made by this update.

Evidence: [final statistics](../results/direct_ambient_radial_3b_20261113/final_summary_20261116.json),
[full curves, health, configurations and Slurm records](../results/direct_ambient_radial_3b_20261113/completed_20261116/results/results_20261116.json),
and the locally retained raw-log / separate ELR archives. Bulk archives are
not published; the linked structured records contain the reported results.

## Warm-K10 Protocol and Launch Record: November 16

**Historical launch record; final results are above.** Submitted at13:42
UTC+08 as `4442918_2`, one H20. After a short resource
wait, it started on SPG-7-2 at13:47. Local, H20 CPU and H20 CUDA suites each
passed all39 tests; the separate four-step smoke completed successfully.
At13:51, fresh main training reached10/5723 with zero logged NaN/skipped
iterations and about11.06seconds/step in the first ten-step timing window.
This is startup timing, not a matched steady-state speed comparison. The
server-side comparison against `4434107_2` confirms that only
`KERNEL_SOLVER` and `SPEL_PGD_CLUSTER_SPECTRUM_MODE` change among recorded
training options, excluding output-path metadata. Other queued/running
experiments are untouched. Home storage is approximately21GB free.

[Submission receipt](../results/warm_ambient_radial_3b_20261116/submission.json),
[server status and configuration comparison](../results/warm_ambient_radial_3b_20261116/status_20261116.json),
[CUDA, smoke and main-launch evidence](../results/warm_ambient_radial_3b_20261116/launch_check_20261116.json).
Frozen project: `/home/u3013198/projects/direct-ambient-radial-warm8k10-3b-20261116`.
Source manifest: `9617f0aa4c6df6219115789ac295248dda30dcbbd449308d47d858f192f479d6`.
An initial archive included macOS metadata and was rejected before tests or
submission; that unused staging copy is archived separately. Only this one
Slurm job was submitted; its completed result is now included above.

Smoke counters: 756 positive-LR component updates, 756 trial Ritz solves,
252 bootstrap Ritz solves, zero full SVDs, and one natural ambient-near call.
The main step10 report shows252/252 current cache hits and no solver fallback.
Legacy host labels such as `post projections` describe this adapter's radial
return; they do not imply sphere clipping. A large correction/trial ratio at
tiny startup LR is not itself a radius-error measurement; the offline audit
measures radius error and actual ELR directly.

The user authorized **one missing 3B configuration**, not a new grid: width256,
seed2345, LR0.015, rho=0.5, window2e-4, warm block-8 K10 with FP32 QR/Ritz.
The exact half-step job `4434107_2`, exact control and historical baselines
are reused, not resubmitted. A read-only scan of 158 saved server launch
configurations found no matching warm direct-PK radial 3B run.

The follow-up replaces only spectral estimation, including initialization,
trial norm and next-point cache. It keeps the same direction formula,
5723-step schedule, data order, 250-step warmup, eval20/250, Nesterov0.9,
BF16 NS8, matrix weight decay0, and disabled checkpoints. Every positive-LR
component update uses one at-most-8x8 Ritz SVD; no original-matrix full SVD,
online residual certificate, or automatic full-SVD fallback is added.
The next cache contains the scaled approximate spectrum of the returned point.
Accuracy and radius feasibility are not certified by fixed K10.

The adapter's `KERNEL_SOLVER` and decomposition counters describe the actual
solver. Legacy host gap/projection option names remain unchanged because
this research adapter replaces that component-update path. Tests explicitly
reject full SVD/svdvals calls in the warm update and check host cache reuse.
The route still uses the estimated second singular value; the block stores
eight directions, and a fully occupied window is recorded as censored.

Local verification: 39 tests passed, including seven warm-specific tests.
Each H20 allocation reruns the same CUDA tests, executes a separate four-step
smoke, and then starts fresh main training. The existing eight-snapshot replay
is retained; an additional offline FP64 audit measures true current/returned
gaps, missed/false near decisions, radius error, cluster capacity and actual
ELR on the stored states without pre-normalizing them. These decompositions
are outside the timed training process and cannot affect its trajectory.

### ELR Is Not Fully Matched

For nonzero momentum and D0, the near direction is

```text
D_near = -rho * ||D0||_F * M / ||M||_F
Y       = X + alpha * D_near
X_next  = R * Y / estimated_sigma1(Y)
```

Thus at the **same current state and alpha**, its trial Frobenius ELR is
rho times the D0 trial Frobenius ELR. rho=0.5 deliberately halves this length;
it does not equalize the two branches. Neither actual returned Frobenius
ELR `||X_next-X||F/||X||F`, spectral ELR, nor AUS is matched by this formula.
The zero-D0 fallback reference remains `sqrt(min(matrix shape))`.
Online displacement/R statistics are not mislabeled as Frobenius ELR.

Writing stable rank as `sr(A)=||A||F^2/||A||2^2`, the same-state trial
spectral-length ratio is

```text
||D_near||2 / ||D0||2 = rho * sqrt(sr(D0) / sr(M)).
```

Consequently, halving Frobenius length need not halve spectral length when
the momentum and matrix-sign directions have different singular-value
profiles. This identity explains a possible mechanism, not a measured causal
explanation of the completed loss differences.

The exact half-step run's near returned length is 51.491% of its same-state
D0 **trial** length; the full-step run gives 105.405%. These are ratios of
sums, not an exact counterfactual returned-step comparison. Half-step improves
loss despite more repeated near calls. A plausible explanation is reduced
perturbation by the unprojected momentum direction; the evidence does not
separate direction benefits from attenuation. A future `0.5*D0` near control
would isolate that distinction, but is not part of this authorized submission.

The older K10/c8 reference also changes routing and amplitude, so it cannot
replace this solver-only comparison. The completed comparison above shows a
speed improvement but a loss regression relative to exact rho=0.5.

<details>
<summary>Historical interim snapshot: November 14, 10:48 UTC+08</summary>

## Interim Progress: November 14

**Interim, not final:** both running arms have reached step 3550/5723
(62.03%, approximately 1.861B processed tokens). Their latest completed
validation is at 3500. Across 356 regular training reports per arm, skipped
and NaN iteration counters remain zero. No job was submitted, canceled or
rescheduled in this results check.

| Job | Method | Progress | Latest validation | State |
|---|---|---:|---:|---|
| 4434107_0 | Exact MCSD-TP | 3550/5723 | 3.653660 @3500 | Running |
| 4434107_2 | Exact ambient-radial, rho=0.5 | 3550/5723 | 3.649616 @3500 | Running |
| 4434107_3 | Exact ambient-radial, rho=1.0 | Not started | -- | Array limit |

### Same-Step Comparisons

All rows below use seed 2345 and the same 3B training schedule, data/model
configuration and eval20 protocol. These shared fields were checked again
against the saved launch configurations. Historical rows are aligned at the
same step, not their 5723-step endpoints. The old experimental algorithms
retain their different spectral estimators and near-branch definitions.

| Method | Val @2000 | Val @3000 | Val @3500 | Training seconds/step |
|---|---:|---:|---:|---:|
| Exact MCSD-TP | 3.837266 | 3.709355 | 3.653660 | 15.857 |
| Exact ambient-radial, rho=0.5 | 3.833585 | 3.708151 | 3.649616 | 15.868 |
| Historical MuonSphere | 3.868907 | 3.736480 | 3.673319 | 10.097 |
| Historical SSO | 3.839826 | 3.715383 | 3.654132 | 11.266 |
| Historical MCSD-TP K10 | 3.836037 | 3.711278 | 3.650006 | 10.779 |
| Historical ambient K10/c8 | 3.830257 | 3.707478 | 3.649574 | 10.791 |

Timing uses the mean of 20 logged training timing reports from steps 3310
through 3500, excluding validation. Main-process wall time to the 3500 training
report, including earlier evaluations/startup but before the 3500 evaluation,
is 15:45:28 / 15:45:48 for exact control/hybrid, 10:07:20 for MuonSphere,
11:22:06 for SSO, and 10:46:45 / 10:47:39 for old K10 control/ambient.
These are historical hardware runs, not isolated optimizer-kernel benchmarks.

At 3500, rho=0.5 minus control is **-0.004044**, minus MuonSphere **-0.023703**,
and minus SSO **-0.004516**. It is only 0.000390 below K10 control and
0.000042 above old ambient/c8. The last three validation differences versus
exact control are -0.001204, -0.000577, -0.004044, but earlier differences change
sign. There is no final or multi-seed superiority claim.

The near-direction change adds only about 0.07% to matched exact-control
training seconds/step, but the exact hybrid costs about 47.0% more than old
ambient/c8 and 57.1% more than MuonSphere. Full trial SVD remains the common
cost. Current evidence supports finishing this comparison, not concluding
that exact spectra improve practical efficiency over K10.

### Branches and Actual Displacement

Counters come from the latest adapter diagnostic during the 3550 update;
they cover 894,349 component updates, not 3550 full-model decisions. The host
`pgd branches` counter is zero by design for this radial variant; it does
not mean that no ambient near branch ran.

| Diagnostic | Exact control | Ambient-radial rho=0.5 |
|---|---:|---:|
| Window-eligible calls | 13,105 (1.465%) | 12,220 (1.366%) |
| Actual ambient-radial calls | 0 | 12,220 (1.366%) |
| Consecutive eligible/near calls | 551 (4.205%) | 489 (4.002%) |
| Largest selected q | 3 | 3 |
| Mean near direction norm / norm(D0,F) | 1.000 | 0.500 |
| Near returned/trial length, ratio of sums | 99.416% | 105.299% |
| Mean returned displacement / R, all updates | 0.144696 | 0.143849 |

Most updates remain MCSD-TP, and this snapshot does not show the previous
high repeat fractions. Radial return can enlarge the displacement relative
to the ambient trial: half D0 trial length does not mean exactly half the
returned displacement. No singular values are clipped. The PGD-only
zero-plateau counter is inactive here; its zero value must not be treated as
a measurement proving that all radial returned gaps are positive.

### Timing, Resources and Next Check

Each active run has spent about 16 hours in main training. Using the observed
whole-run rate, about 9.8 hours remain at this snapshot, approximately
November 14 around 20:30-21:00 UTC+08 if throughput stays stable. The rho=1.0
arm starts when an array slot frees; its completion estimate remains open.
The account uses four H20s (two here and two unrelated chains). ELR prefix/
branch jobs also appear as dependency-pending; no results from them are
included. Home storage has approximately 24 GB free.

Keep the current matched comparison running through 5723. Reuse old baseline
results; do not add duplicate baseline runs or expand the grid from this
interim advantage. Compare the final loss and late trajectory, then decide
whether the near direction warrants an approximate-spectrum version.

Evidence: [progress, curves, configuration checks and same-step timing](../results/direct_ambient_radial_3b_20261113/progress_20261114.json).
The completed long-run registry remains 43 runs; these three are not yet
completed results.

</details>

## Scope

Requested 2026-11-13: directly test the whole-matrix radial return at 3B,
reusing existing historical comparisons instead of repeating them. Only
three missing configurations are selected. Submitted **2026-11-13 18:44
UTC+08:00** as array **4434107**, with at most two concurrent tasks.

Historical launch check, **November 13, 18:50 UTC+08:00**: arms 0 and 2 passed all 32 per-allocation
tests and separate four-step smokes, then started main 3B training (step-10
diagnostics visible). Arm 3 waits for the array concurrency limit. Four H20s
are allocated to the account in total, including two unrelated serial chains.
The smoke had no natural near calls; CUDA unit tests explicitly exercise
the ambient-radial branch. At that launch check there was no main validation yet.

| Job | Configuration | Status at check |
|---|---|---|
| 4434107_0 | Exact MCSD-TP control | Running, SPG-7-2 |
| 4434107_2 | Ambient-radial rho=0.5 | Running, SPG-7-2 |
| 4434107_3 | Ambient-radial rho=1.0 | Pending, JobArrayTaskLimit |

[Submission receipt](../results/direct_ambient_radial_3b_20261113/submission.json).
Frozen source: `/home/u3013198/projects/direct-ambient-radial-3b-20261113`.
The 3,161-file source manifest SHA256 is
`b6d9a49bfb5e4c4f750e62427b3f9c75ed63d27f95800c4a63bd002810bd7ed2`.
The source document inside that snapshot remains frozen; this local document
records subsequent launch observations. Home storage had approximately 27 GB
free before submission.

| Array arm | New experiment | Near direction | Return |
|---:|---|---|---|
| 0 | Exact-cache MCSD-TP control | D0 | Radial |
| 2 | Ambient-radial, rho=0.5 | -0.5 * norm(D0,F) * M / norm(M,F) | Radial |
| 3 | Ambient-radial, rho=1.0 | -norm(D0,F) * M / norm(M,F) | Radial |

All three use the same exact-cache MCSD-TP safe direction D0. The near region
contains at least two singular values within relative width 2e-4 of sigma1.
The full spectrum is retained without a rank-four/eight cap. The zero-D0
reference is sqrt(min(matrix shape)); zero momentum gives zero direction.
rho measures trial length relative to D0, not a historical fixed c or a
percentage of branch calls.

```text
Y = X + effective_lr * D
U, s, Vh = full_FP32_SVD(Y)
a = R / s[0]
X_next = a * Y
next_cache = (U, a * s, Vh)
```

No singular-value clipping, tangent correction of the ambient direction,
post-return length matching or objective backtracking is added. The returned
relative gap equals the trial gap; radial scaling does not split an existing
zero-gap plateau. This is not Euclidean PGD. The old projection results and
their convergence interpretation are not silently relabeled as radial.

The exact spectral estimator is retained to isolate the return operation.
After bootstrap there is one full trial SVD per component per update. Thus
this test does not claim to remove the SVD cost. Counters distinguish actual
ambient-radial calls from projected PGD; expected `pgd_calls` is zero.

## Historical Comparisons: Do Not Rerun

All listed endpoints are width256, seed2345, 3B/5723 steps, LR 1.5e-2,
250-step warmup, cosine decay to 1.5e-3, and eval20 every 250 steps plus final.

| Existing experiment | Final loss | Main training + validation | Job |
|---|---:|---:|---|
| MuonSphere | 3.436023 | 16:35:03 | 4346705_0 |
| SSO | 3.435041 | 18:28:11 | 4346705_1 |
| MCSD-TP, warm block-8 K10 | 3.428613 | 17:39:34 | 4346705_2 |
| Ambient-radial, K10 / c8 | 3.427854 | 17:40:58 | 4346705_4 |

Sources: [historical study](cluster_seed_mechanism_3b_20261028.md),
[result summary](mcsd_p2gd_results_summary.md#cross-seed-3b-evidence).

The historical saved data/model/seed/schedule/evaluation configurations were
checked on H20. The training shell, MuonSphere and SSO source, and optimizer
integration file have identical SHA256 hashes in the old and new base
snapshots. The reference baseline runs are reused. The two old
experimental rows are contextual, not exact matched controls: they use
approximate warm block-8 K10, a 5e-5 switch, and (for ambient) fixed c8 rather
than D0 length matching. Their positive results do not validate the new
2e-4/exact/rho configuration. A fresh exact-cache MCSD-TP control was needed
to distinguish benefits of the near direction from changes in spectral
accuracy. At submission, no completed equivalent 3B exact control existed in
the confirmed result registry; the new control is now reported above.
No MuonSphere, SSO or old K10/c8 repetition is
submitted; the failed third-seed study is not counted as a completed result.

## Fixed Training Protocol

- Width256, 28 layers, FFN768, head dimension128, sequence length4096.
- Seed2345, one H20 per run, global batch128, microbatch4, TP=PP=1.
- Fresh initialization; no continuation from the 100M runs.
- Existing 3B indexed OLMo-Mix data and original validation/tokenizer paths.
- 5723 iterations: 3,000,500,224 processed tokens; 250 warmup steps.
- Peak LR0.015, minimum0.0015; cosine decay through step5723.
- Validation: 20 batches every250 steps and at the endpoint.
- BF16 training / NS8 LMO, FP32 master and full-SVD geometry, TF32 off.
- Existing Nesterov0.9; constrained matrix weight decay0; auxiliary AdamW unchanged.
- Checkpoints off; at most48 matrix snapshots per experimental run.

Local and H20 CPU regression checks: 32 passed, including returned spectra, one trial
SVD, identical trials for projected/radial alternatives, host integration,
3B schedule, seed, disabled checkpoints and audit return selection. Each
GPU allocation reruns the CUDA tests, performs a separate four-step smoke,
then starts a fresh 3B training process. An eight-snapshot FP64 audit follows
training; these counterfactual momentum scores are not minibatch losses.

## Resource Budget and Evaluation

Three jobs, one GPU each, at most two concurrent. At the initial queue check
there were two other single-GPU serial chains; this preserves the requested
account-wide budget of four H20s for the known workload without canceling or
rescheduling those jobs. Recheck immediately before submission.

The 100M exact runs took about57 minutes each. Linear extrapolation gives
roughly28.5 hours per 3B run, excluding queue and with uncertainty from the
changed evaluation frequency. The Slurm allocation allows48 hours. No
automatic expansion beyond these three runs is scheduled.

Judge final validation and the whole curve against the exact control first,
then compare to historical MuonSphere/SSO. Track actual displacement, near
call/repetition rates, post-return gap and wall time. Do not declare success
solely because radial return reduces zero-gap cache plateaus. The prior
100M full-projection losses (5.210238/5.341342) are a motivation for changing
the return, not evidence that these new 3B variants will improve.

Implementation: [adapter](../../scripts/direct_top_pgd_pk.py),
[launcher](../../slurm/direct_ambient_radial_3b_20261113.sbatch),
[budget tests](../../scripts/test_direct_radial_3b.py).
