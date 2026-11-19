# Step-Aware A2 and Kernel-LMO: H20 Pilot

[Theory](kernel_lmo_theory_20261111.md) |
[Local evidence](kernel_lmo_results_20261111.md) |
[Adapter](../../scripts/kernel_lmo_h20.py) |
[Launcher](../../slurm/kernel_lmo_pilot_100m_20261112.sbatch)

## Scope

**Results checked 2026-11-12: all eight configurations now have completed
endpoints.** Six come from the original batch; the two Kernel-LMO endpoints
come from successful fresh reruns **`4426353_3` and `4426353_5`**, completed
at 12:32 and 12:16 HKT respectively. Their original v2 attempts failed on
`KeyError: 5` and remain excluded. Both reruns passed nine CUDA tests and
the smoke, with no logged NaN/skipped iterations. Neither Kernel-LMO variant
closes the large validation-loss gap to MCSD-TP or MuonSphere.

**Original batch: `4424416_0-7%2`, submitted 2026-11-12 00:47 HKT.**
This precision-isolated version supersedes the stopped startup batch
`4424403`. [Final submission receipt](../results/kernel_lmo_h20_20261112/submission_v2.json).

Eight fresh width-256 runs, **100M tokens each**, one H20 per arm and at most
two concurrent GPUs. No checkpoint and no new 1B/3B long run. Existing jobs
are left unchanged. Each allocation must pass CUDA unit checks and a separate
four-step smoke before main training. CPU preflight is also mandatory before
submission. Runtime and status will be recorded from the submission receipt.

| Arm | Method | Spectral solve | Purpose |
|---:|---|---|---|
| 0 | MCSD-TP control | Cached full FP32 SVD | Matched exact-spectrum control |
| 1 | A2 | Cached full FP32 SVD | Finite-step normalized-momentum correction |
| 2 | A2, input scale 1.5 | Cached full FP32 SVD | Amplitude control before correction |
| 3 | Kernel-LMO, theta=1 | Cached full FP32 SVD | Preserve corrected normal block; add MCSD in kernel |
| 4 | A2 | Warm block-8, K10 FP32 Ritz | Approximate deployable candidate |
| 5 | Kernel-LMO, theta=1 | Warm block-8, K10 FP32 Ritz | Isolate spectral approximation |
| 6 | MuonSphere | Reference BF16 cold PI10 | Matched practical baseline |
| 7 | SSO | Reference BF16 cold PI10 | Matched spectral-sphere baseline |

## Completed Results

[Original endpoint record](../results/kernel_lmo_h20_20261112/final_4424416_summary.json).
Values were read directly from Slurm accounting and the remote training logs,
timing files and dedicated statistics. Raw logs remain in the frozen v2
directory; a stalled bulk download was stopped and its incomplete archive
removed locally. The reruns' compact raw evidence was downloaded successfully:
[compact exact/warm log evidence and checksums](../results/published_log_summaries_20261119.json),
[exact counters](../results/kernel_lmo_h20_20261112/final_4426353/exact/pilot_statistics.json),
[warm counters](../results/kernel_lmo_h20_20261112/final_4426353/warm/pilot_statistics.json).

Final validation is at step 191 (100,139,008 processed tokens), seed 1234.
Time below is **main training plus validation**, excluding the CUDA checks,
smoke, offline audits and queue time. The table combines six original v2
endpoints and two counter-fix v3 reruns; the failed attempts are not averaged
or resumed. All eight completed runs have Slurm exit 0:0. The six original
last regular reports and all regular rerun logs show zero NaN/skipped iterations.

| Arm | Method / spectrum | Final loss | Difference vs. control | Time (mm:ss) | Near calls / positive-LR calls |
|---:|---|---:|---:|---:|---:|
| 0 | MCSD-TP / full SVD | **4.918329** | 0 | 57:31 | 0 / 47,880 |
| 1 | A2 / full SVD | 5.073903 | +0.155574 | 56:56 | 14,876 / 47,880 (31.07%) |
| 2 | A2 input x1.5 / full SVD | 5.069594 | +0.151265 | 57:29 | 13,196 / 47,880 (27.56%) |
| 3 (v3 rerun) | Kernel-LMO / full SVD | 5.078964 | +0.160635 | 57:05 | 14,736 / 47,880 (30.78%) |
| 4 | A2 / warm block-8 K10 | 5.087419 | +0.169090 | 40:36 | 14,066 / 47,880 (29.38%) |
| 5 (v3 rerun) | Kernel-LMO / warm block-8 K10 | 5.082469 | +0.164140 | 40:33 | 14,197 / 47,880 (29.65%) |
| 6 | MuonSphere | 4.919347 | +0.001018 | **38:06** | Not applicable |
| 7 | SSO | 4.927444 | +0.009115 | 41:43 | Not applicable |

These are A2/Kernel-LMO near-branch frequencies, **not PGD/full-projection rates**. All
experimental arms use radial return. The final dedicated counters exclude
the zero-LR initialization; the legacy step-190 report uses a different
endpoint and includes zero-LR calls, so its percentages differ slightly.

| Method | Step 50 | Step 100 | Step 150 | Step 191 |
|---|---:|---:|---:|---:|
| MCSD-TP exact | 6.468626 | 5.623314 | 5.104645 | 4.918329 |
| A2 exact | 6.480133 | 5.644739 | 5.192223 | 5.073903 |
| A2 input x1.5 exact | 6.484708 | 5.656663 | 5.194430 | 5.069594 |
| A2 warm K10 | 6.521814 | 5.658011 | 5.204269 | 5.087419 |
| Kernel-LMO exact (v3) | 6.481397 | 5.659883 | 5.203477 | 5.078964 |
| Kernel-LMO warm K10 (v3) | 6.512057 | 5.663931 | 5.205884 | 5.082469 |
| MuonSphere | 6.485603 | 5.632350 | 5.105802 | 4.919347 |
| SSO | 6.457970 | 5.627861 | 5.112372 | 4.927444 |

### Interpretation

- MCSD-TP and MuonSphere are close in this one-seed short run. MCSD-TP is
  0.001018 lower but takes 51.0% longer; no robust superiority is established.
- Exact-spectrum A2 is already substantially worse. Spectral approximation
  cannot be the sole cause. Scaling its normalized input by 1.5 improves only
  0.004309, leaving a 0.151265 deficit to control.
- A2 warm saves 28.7% versus exact A2 and is 6.6% slower than MuonSphere,
  but loses another 0.013516 versus exact A2. It is not ready for long training.
- Near-branch use grows to about 79% of components at step 190 in all three
  A2 runs. Their cumulative repeat rates at that report are 95.2%-96.0%.
  Repetition alone is not failure, but here it accompanies worse loss.
- The aggregate radius-normalized returned displacement is 80.3% of control
  for exact A2, 84.6% for input x1.5, and 82.7% for warm A2. These are sums
  along different trajectories, **not same-state causal comparisons**. They
  motivate a matched actual-step audit before a broader parameter search.
- Kernel-LMO exact is 0.005061 worse than exact A2; warm Kernel-LMO is
  0.004950 better than warm A2. This inconsistent small difference does not
  rescue either hybrid: both remain over 0.16 worse than MCSD-TP control.
- Kernel-LMO near use reaches 201/252 exact and 200/252 warm at step 190;
  cumulative repeat rates at that report are 96.6% and 96.3%. Full-run
  returned-displacement sums are 80.8% and 82.6% of control respectively.
  These trajectory aggregates identify a step-size confound, not a causal
  explanation or a reason to equate repeated near steps with failure.
- Warm Kernel-LMO saves 29.0% versus its exact reference and is 6.4% slower
  than MuonSphere, but its final loss is 0.163122 higher than MuonSphere.

### Capacity, Return and Failure Audit

Completed exact A2, input x1.5 and warm A2 reach q=4, with respectively
67, 157 and 121 q=4 calls. Neither completed A2 run exhausts a warm eight-vector
cache. Both original Kernel-LMO attempts encounter q=5 after their last regular step-40
report and before validation at step 50. The direction/return code completes
that component, then the host recorder fails at
`cluster_rank_counts[info.cluster_dimension] += 1`: its dictionary was
initialized only for ranks 1-4. Those attempts have no final validation;
their fresh v3 reruns supply the endpoints above.

The fix is local and opt-in: `pilot_dispatch` initializes missing rank counters
without truncating the selected cluster, and restores both dispatch and
recorder afterward. The initial **eight local CPU regression tests passed**, including
full host integration with q=16 exact / q=8 warm and exception restoration.
The subsequent subset-submission test brings the suite to nine; all pass both
locally, on the server CPU and on CUDA in both allocations. Both v3 reruns
complete successfully. Exact Kernel-LMO records 60 q=5 calls and warm
Kernel-LMO 176; neither selects q>5 or reports a censored eight-vector window.
All near calls accept the kernel addition (14,736 exact / 14,197 warm).
Frozen v2 source and failed-job logs are retained unchanged.

The six completed experimental runs each performed 47,880 trial solves plus
252 bootstrap solves, exactly one solve per positive-LR component update.
Exact arms used 48,132 full SVDs; each warm arm used 48,132 small Ritz SVDs and no
original-matrix full SVD in training.

Offline replay tested eight sampled matrices from each completed experimental
arm, four directions per matrix (192 counterfactuals total including reruns). All returned
displacements had nonpositive momentum inner products. This is **not actual
loss decrease or a convergence proof**. The sampled full-window rank was at
most two, but the sufficient external-gap capacity bound reached 16 for
control snapshots and eight for A2 snapshots on the tested capacity grid.
These are conservative sufficient ranks, not minimal requirements.

Maximum sampled input radius deviations before audit normalization were
5.72e-6 (control), 8.94e-6 (A2 exact), 2.15e-6 (A2 input x1.5), and
**1.70e-3 (A2 warm)**. The reruns add 4.53e-6 (Kernel-LMO exact) and
1.70e-3 (Kernel-LMO warm). The warm deviation exceeds the 2e-4 routing window;
do not equate an uncensored eight-vector cache with accurate geometry.

### Next Decision

Do not extend the current A2 or theta=1 Kernel-LMO settings to 1B/3B on this
evidence. The two missing endpoints are now available and do not support a
long-run promotion. Next, use identical saved matrices/momenta to compare
complete returns at matched actual lengths; measure objective changes where
a common minibatch replay is available, not merely momentum inner products.
If changing a direction's amplitude or effective step, rebuild the step-aware
model rather than post-scaling its corrected direction and assuming its
normal-block condition still holds. Capacity above eight is not the first
priority given the observed maximum q=5. No new jobs were submitted in this
results check, and no general convergence claim is disproved by this pilot.

## Counter-Fix Follow-Up

On 2026-11-12 the user authorized the two interrupted Kernel-LMO arms to be
retested. The submitted frozen v3 copy selects **only arms 3 and 5**, at most
two H20 GPUs concurrently. All mathematical update functions are AST-identical
to v2; only process-local variable-rank diagnostics, their tests and the
subset submission interface change. No A2 long runs are added. Nine local
and server CPU tests pass. Each scheduled allocation must pass the same tests on CUDA
and a separate four-step smoke before the fresh 191-step main run.

The resource check before upload found all 16 H20 GPUs allocated and about
29 GB free in home storage. The follow-up queued without interrupting
existing jobs. Checkpoints remained disabled; each rerun saved 24 snapshots.
The [submission receipt](../results/kernel_lmo_h20_20261112/submission_v3_counterfix.json)
records submission at **2026-11-12 11:25:35 HKT**, array **4426353**:

| Job | Method | Spectrum | Final state | Start / end HKT | Slurm elapsed |
|---|---|---|---|---|---|
| `4426353_3` | Kernel-LMO theta=1 | Full FP32 SVD | COMPLETED, 0:0 | 11:33:02 / 12:32:35 | 59:33 |
| `4426353_5` | Kernel-LMO theta=1 | Warm block-8 K10 | COMPLETED, 0:0 | 11:34:39 / 12:16:51 | 42:12 |

Both request one H20, 12 CPU cores, 80 GB RAM and a four-hour limit. They
start fresh rather than resuming the interrupted step-40 states. Existing
Frobenius jobs and pending dependencies are unchanged. Both jobs ran on
SPG-7-1 and passed all nine CUDA tests and separate smokes before main training.
Slurm elapsed includes checks/smoke/audit and differs from main-training time.

Frozen source:
`/home/u3013198/projects/kernel-lmo-stepmodel-100m-20261112-v3-counterfix`.
Source manifest SHA256:
`f14796e287243ee24c4bb0e0c57f49e048ba0f194a0f933c8bc367896aaf00ea`.
Overlay SHA256:
`75b69d0fe2ed6862fd568510405a7a063363500dd20afe652358fc019ee6927b`.

## Matched Protocol

Seed 1234; 28 layers, width 256, FFN 768, head dimension 128, sequence 4096;
global batch 128, microbatch 4, TP=PP=1. Training lasts 191 steps, processing
100,139,008 tokens from the existing **3B indexed source**. This is neither a
3B training budget nor the historical paper's original 1B sample.

Peak LR 0.015, 25 warmup steps, cosine decay to 0.0015 at step 191. Evaluate
20 batches at steps 50/100/150 and 191. BF16 training, FP32 master weights,
Nesterov 0.9, NS8 matrix-sign implementation. Constrained matrix weight decay
is zero; auxiliary parameters retain the reference AdamW settings.

The new hybrids use a single **relative spectral window 2e-4**: q is the
number of cached singular values in that window, and q>1 selects the near
branch. This is deliberately the local step-aware model, **not** the old
independent route=5e-5 / cluster=2e-4 configuration. No gap sweep is added.
The actual spectral-muP effective learning rate is passed into the small
model and into the trial without post-correction rescaling. A2/input-1.5
scales the normalized momentum before model projection, not the output.

The exact arms keep the full paired spectrum; the host's default top-eight
cache is bypassed only by this opt-in adapter. There is no top-four cap.
The warm arms retain at most eight directions and explicitly count when
the selected window touches that limit. They are not called certified or
complete in that case. No online accuracy gate or full-SVD fallback is added.

Both branches in arms 0-5 use radial return and cache the returned spectrum.
After cache initialization, there is exactly one trial spectral solve per
positive-LR matrix-component update. Bootstrap solves are counted separately.
The FP32 cache pins its leading value to the requested radius and clips
roundoff-sized overshoots in cached values; actual radius error must still be
audited against the returned matrix. "Exact" means the full-SVD numerical
reference, not exact real arithmetic or a formal floating-point certificate.

MuonSphere/SSO keep their existing hard pre-step radial return rather than
being silently modified to match our post-step return. These baselines answer
practical performance, while arms 0-5 isolate our direction/solver choices.

## Measurements and Interpretation

- Primary: final validation loss at 191, with all intermediate values and
  training-plus-validation time. Do not compare a 100M endpoint with a 1B loss.
- Per-component q histogram, window sizes above 4/8, censored warm windows,
  branch use, actual trial/returned Frobenius lengths and radial correction.
- Momentum inner products with the trial and full returned displacement.
  These are **not actual objective changes** or full-return descent proofs.
- One trial solve contract; exact/bootstrap/small-Ritz counts are separate.
- Bounded snapshots, at most 32 per experimental run; no model checkpoints.
- Arm 0 replays 12 existing real snapshots from the completed 1B control and
  single-LMO runs using common feasible-normalized X and original momentum.
  The audit compares all new directions, the coupling return bound and
  reconstructed capacity on a finite grid. It is offline and cannot alter
  the training branches. All experimental arms audit eight of their own
  snapshots after training, outside the reported training time.

Do not promote a candidate based solely on low near-branch repetition or a
large positive post-return gap. A multiple-top optimum may require remaining
near the corner. The first decision is whether the direction/model is useful
on H20; the next is whether the warm approximation preserves that behavior.
No long-run continuation is automatic.

## Initial Startup and Precision Isolation

Submitted **2026-11-12 00:35 HKT**, array **`4424403_0-7%2`**. The frozen
source is `/home/u3013198/projects/kernel-lmo-stepmodel-100m-20261112`, manifest
SHA256 `d00d04cb1ecec82e5f0924a52bd8038cd166032e8d18406b7ce1e738ed9c5e6b`.
The local and server CPU preflight both passed all five pilot tests.

Arm 0 started on SPG-7-1, passed all five **CUDA** tests, completed the real
snapshot audit and separate four-step smoke, and entered fresh main training.
Step 1/191 was logged at **00:38:17 HKT**, without NaN or skipped iterations.
The other seven arms were pending resources at the first startup check.
This was successful startup, not a completed 100M result. A final audit found
that the parent optimizer wraps updates in its `medium` precision context.
Although `NVIDIA_TF32_OVERRIDE=0` was already set, the revised adapter
explicitly enters FP32 `highest` for geometry and restores the caller's
setting afterward; BF16 NS8 is unchanged. A sixth regression test verifies
this behavior. Array 4424403 is superseded, not an optimization result.
Only this new pilot batch is restarted; other account jobs are unchanged.

The first logged step took 17.3 seconds including initialization. Do not use
that alone as a steady-state timing estimate. Approximate/reference pilots
should be shorter than the original 1B runs, but exact full-spectrum timing
must be measured. Each allocation has a four-hour limit; queue time is unknown.

[Submission receipt](../results/kernel_lmo_h20_20261112/submission.json).
Disk availability was about 31 GB before the new isolated copy; checkpointing
is disabled and snapshot counts are bounded.

The final source directory is
`/home/u3013198/projects/kernel-lmo-stepmodel-100m-20261112-v2`, with manifest
SHA256 `ef2a5728c83f5b6692b217e0069b779c7be955d67f3a36d9ca538a58779cf61c`.
All six local and server CPU checks passed before its submission. The old
training log is retained for provenance but excluded from endpoint analysis.

**Historical final-version startup check:** `4424416_0` was running on SPG-7-1. All
six CUDA tests passed, followed by the repeated snapshot replay and four-step
smoke. Fresh main training logged step 1/191 at **00:49:53 HKT** with zero
NaN/skipped iterations. Arms 1-7 were still pending resources at that check.
There was no final validation endpoint at that check; see completed results above.

### First Real-Matrix Replay

The first startup completed 48 counterfactual directions on 12 real matrices
from the previous 1B control and single-LMO runs. Their full-SVD window sizes
were at most two. The conservative external-gap sufficient condition, with
directions rebuilt on a capacity grid, reached 16 for some alternatives;
this is not a proof that 16 is the minimal or practically necessary rank.
All 48 full-return **momentum** inner products were nonpositive. No actual
LLM loss decrease can be inferred from that statistic alone.

The inputs were normalized once to a common exact-norm reference; the largest
relative pre-normalization radius deviation was 6.45e-4. Their original
momentum is held fixed, not recomputed at that adjusted point. This is a
controlled direction audit, not a faithful extra training step. See
[recorded real-matrix comparisons](../results/kernel_lmo_h20_20261112/real_snapshot_audit.json).
The final batch repeats this audit in arm 0.

## Source Isolation

The base is the completed November 10 confirmation source, verified against
its manifest. Only new research scripts and this protocol are overlaid into
a **new directory**. Original training optimizer files and the paper's
reproduction repository remain unchanged. A manifest and an idempotent
submission receipt prevent silent source drift and duplicate submissions.

The original host dispatch setting says `adaptive4`; the opt-in entrypoint
replaces that dispatch in its own process. Actual routing, capacity and solver
are therefore recorded by `KERNEL_METHOD`, `KERNEL_SOLVER` and the dedicated
pilot statistics. The legacy q1/q2/q3/q4 recorder caused the two q=5 crashes
documented above. The local corrected entrypoint supports dynamic counters;
use the complete `q_hist` from `pilot_statistics.json` when interpreting ranks.

The exact arms are full-SVD **spectral** references, not fully exact versions
of the mathematical algorithm: their matrix-sign LMO still uses BF16 NS8.
The near-branch scalar guard protects the added kernel direction, but this
does not certify every safe-branch inexact-LMO condition from the paper.
