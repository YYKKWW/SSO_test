# Direct Top-Block Repair: Same-D0 Paired Study

[Algorithm](../../scripts/direct_top_repair.py) |
[Local audit](../../scripts/local_direct_top_repair.py) |
[Tests](../../scripts/test_direct_top_repair.py) |
[Previous pilot](kernel_lmo_h20_pilot_20261112.md)

## H20 Results: Complete

Checked 2026-11-12 at 23:32 UTC+08:00. All three jobs completed with exit
0:0, passed both CUDA suites (18 tests per arm), completed separate smokes,
and passed the post-training snapshot audit. All 20 regular training reports
per arm record zero skipped and zero NaN iterations. The three saved launch
configurations differ only in repair mode and snapshot output path.

| Method | Final validation loss | Difference vs. none | Main train + validation | Repair calls |
|---|---:|---:|---:|---:|
| None: MCSD-TP | 4.918329 | 0 | 57:13 | 0 |
| Multiple-root repair | 4.918715 | +0.000386 | 57:14 | 6,048 / 47,880 (12.63%) |
| Finite-step spectral-slack repair | **4.912478** | **-0.005851** | 57:03 | 6,382 / 47,880 (13.33%) |
| MuonSphere, previous matched-budget pilot | 4.919347 | +0.001018 | 38:06 | N/A |
| SSO, previous matched-budget pilot | 4.927444 | +0.009115 | 41:43 | N/A |

The last two rows are reused references from the preceding pilot, not fresh
jobs in this three-arm array. The fresh uncorrected control reproduces all
four displayed validation values of the previous exact MCSD-TP run. Times
exclude checks, smoke, offline audit and queueing; Slurm totals for arms 0/1/2
are 59:49 / 59:19 / 59:07. Finite-step repair is 0.006869 below MuonSphere but
still takes about 49.7% longer because all experimental arms use full SVD.
This is one seed and 100M tokens, not evidence of a long-run or significant gain.

| Method | Step 50 | Step 100 | Step 150 | Step 191 |
|---|---:|---:|---:|---:|
| None | 6.468626 | 5.623314 | 5.104645 | 4.918329 |
| Multiple-root | 6.465553 | 5.624090 | 5.105925 | 4.918715 |
| Finite-step | 6.474305 | 5.632930 | **5.099160** | **4.912478** |

### Actual Step and Capacity

| Mode | Mean trial / R | Mean returned step / R | Mean correction / D0 norm, near calls | Max q | q>4 / near calls |
|---|---:|---:|---:|---:|---:|
| None | 0.1017025 | 0.1007933 | 0% | 4 | 0 / 3,680 |
| Multiple-root | 0.1018037 | 0.1008888 | 1.600% | 6 | 459 / 6,048 (7.59%) |
| Finite-step | 0.1017953 | 0.1008820 | 1.499% | 7 | 669 / 6,382 (10.48%) |

Finite-step repair's aggregate returned step is only 0.088% larger than the
control's. This argues against a large overall step enlargement, but does
not establish the causal mechanism: independently trained states differ.
None has 3,680 window-eligible calls (7.69%) but performs zero corrections.
The finite-model residual averaged over near calls is 6.04e-4 / 1.98e-5 /
7.81e-8 for none / multiple / finite-step. This is not the true trial norm
error or a certificate of full-return loss descent.

Each arm records 47,880 trial SVDs plus 252 bootstrap SVDs. No extra full SVD
is used for the correction. No q cap is imposed, so q>4 counts are observed
capacity requirements, not missed directions in these runs. A fixed top-four
implementation would not reproduce the tested correction.

The post-run audit replays the same stored D0 in all three modes on eight
states per arm. All 72 returned momentum inner products are nonpositive;
the audits do not evaluate the LLM loss and do not establish per-step descent.
The FP64 replay radius accuracy must not be presented as FP32 training accuracy.

Evidence: [compact log extracts for all three arms, with checksums](../results/published_log_summaries_20261119.json),
[finite-step counters](../results/direct_top_repair_20261112/h20/results/4427402_2/snapshots/pilot_statistics.json),
[finite-step replay](../results/direct_top_repair_20261112/h20/results/4427402_2/offline_audit.json).

### Interpretation and Next Comparison

- Pause promotion of the current Kernel-LMO configuration: exact / warm
  endpoints are 5.078964 / 5.082469 in the previous same-budget study.
  Exact spectra do not remove its loss deficit. This rejects the tested
  configuration, not every possible kernel-preserving LMO construction.
- Preserve the finite-step D0 repair as a candidate. It changes only the
  selected top block and does not replace the free part or its amplitude.
  It is not standard P2GD of the negative momentum.
- Add a matched MCSD-TP-PGD reference before further broad searches. It
  should share the exact cache, safe branch, data and schedule, but use
  ambient momentum and full sphere projection in the near branch. Specify
  its momentum normalization and actual step independently of old c values.
- A P2GD comparison must distinguish the true top-multiplicity cone from
  a widened numerical cluster; a widened cluster is a surrogate. Likewise,
  tangent-cone direction plus radial return is not standard projected P2GD.
  Stochastic momentum and fixed numerical tolerances need separate analysis.
- Replicate the paired finite-step gain before 1B/3B promotion. No new job
  was submitted in this result check; the completed array uses no GPUs now.

## Question and Controls

Can we repair the top block of the existing MCSD-TP direction without replacing
its overall scale by a normalized-momentum direction? Three modes share the
same D0 generator, spectral window, actual effective step and radial return.
There is no extra LMO, post-repair normalization, acceptance test or line search.

For paired orthonormal U,V define H(D)=sym(U^T D V). D0 is the existing
top-1 MCSD-TP direction: project negative momentum, apply the matrix-sign LMO,
then project its output back onto the top-1 plane. Let S contain the selected
current singular values and alpha be the actual spectral-muP matrix step.

| Arm | Mode | Near-region operation |
|---:|---|---|
| 0 | `none` | D = D0 |
| 1 | `multiple` | Closest direction to D0 with lambda_max(H(D))=0 |
| 2 | `finite_step` | Closest direction to D0 with lambda_max(S+alpha H(D))=R |

For the finite-step operation, A=H(D0)-(R I-S)/alpha. Diagonalize A, clamp
positive eigenvalues to zero and force the largest eigenvalue to zero even
if all are negative. Calling the resulting matrix B, set
D=D0+U(B-A)V^T. The multiple-root variant uses S=R I in this same formula.
Both preserve the free component P_ker(H)(D0). A low-rank correction need
not be small in norm; neither construction guarantees full-return descent.

**Terminology:** `multiple` is the full selected-cone repair of D0. It is not
the historical selected-face Single-LMO/c0.5 pipeline, nor an exact sphere
projection. Its true tangent-cone interpretation requires a complete exact
top multiplicity. Near clusters are a surrogate. The finite-step constraint
is a local model, not the true singular-value constraint on the whole trial.

q counts cached singular values within relative distance 2e-4 of the largest.
Only q>1 invokes a repair; q=1 is exactly the same path in all modes. There is
no top-four cap. Both repaired modes coincide on an exact selected plateau.

## Meaning of Same D0

**Local same-state tests compute D0 once**, then pass the same tensor, X, M,
spectrum and alpha to all three modes. The input tensor is not mutated.
The returned point is formed separately for each mode with the same exact
radial-return procedure. True quadratic objective changes are evaluated.

**Independent training trajectories share the formula, not the numerical D0
at every later step.** Initialization, seed, data order and schedule match;
after trajectories separate, their weights, gradients and D0 naturally differ.
Saved H20 snapshots include the actual D0. Offline replay reuses that stored
D0 for all three counterfactuals on each state. A full FP64 spectrum and a
common feasible-normalized X are used in that audit, with the original
momentum held fixed. It is not a faithful additional training step.

## Local Results

[Machine-readable results](../results/direct_top_repair_20261112/local_audit.json).
Nine new tests and nine existing pilot tests passed on CPU. Tests cover the
same D0 call, no mutation, free-block preservation, retained spectral slack,
exact-plateau equivalence, q=5 host integration, one trial solve, and equality
of the uncorrected adapter to the existing MCSD-TP control. Shell validation
also passed. CUDA checks must run again inside each allocation.

### Eight Real LLM Matrices

Eight bounded snapshots from the completed exact Kernel-LMO job 4426353_3;
only three selected states are near-region (q=2). These snapshots predate D0
storage, so each uses one freshly computed BF16 NS8 D0 shared by all repairs.
These samples are not representative estimates of all layers or late training.

| Mode | Mean trial length / R | Mean returned length / R | Max finite-model error / R |
|---|---:|---:|---:|
| None | 0.09389948 | 0.09321633 | 7.08e-4 |
| Multiple-root | 0.09389876 | 0.09327735 | 9.75e-5 |
| Finite-step | 0.09389880 | 0.09326799 | 1.90e-10 |

Thus the finite-step repair satisfies the intended selected-block model while
largely retaining step magnitude in these samples. All returned momentum
inner products are nonpositive; **there is no actual LLM loss oracle here**.
The radial returns use FP64 full SVD in this offline audit; the measured
radius errors below 1.4e-15 do not claim FP32 training accuracy.

### Quadratic Objectives and Trajectories

The audit includes 54 same-state/step combinations at shapes 256x256,
128x256 and 768x256 with BF16 NS8 D0, plus 12 aligned near-plateau stress
combinations with an exact-polar D0. All three returns are evaluated on each
state (198 synthetic counterfactuals, plus 24 real-snapshot counterfactuals).
In the broad shape cases, every method decreases the objective, but the
uncorrected method has the largest average decrease. Repair is not uniformly
better. In the aligned stress family, full-return uphill cases change from
12/12 uncorrected to 4/12 with either repair; the remaining uphill cases are
the exact zero-gap examples. This is an explicit limitation, not a rounding
certificate or a convergence claim.

Independent paired trajectories: 16 objectives, three methods, 2000 steps
each; identical initial states and stochastic noise with practical Nesterov
0.9. These small reference trajectories use exact-polar LMO and exact radial
return, not the BF16 production LMO. Twelve objectives have known optima.

| Mode | Mean final f-f* (12 known optima) | Wins / ties / losses vs none (all 16) | Mean returned step / R |
|---|---:|---:|---:|
| None | 2.03277e-4 | Reference | 0.01378880 |
| Multiple-root | 2.04579e-4 | 2 / 11 / 3 | 0.01379012 |
| Finite-step | 1.34649e-4 | 5 / 11 / 0 | 0.01379636 |

This supports a small H20 comparison, not promotion to a long run. Most
paired cases tie and this selected synthetic suite is not an LLM benchmark.

## H20 Protocol

Three fresh width-256 / 100M runs (191 iterations; 100,139,008 actual tokens),
seed 1234, one H20 each, at most two concurrent. This uses the same existing
3B indexed source as the previous 100M pilot, not 3B training and not the
historical paper's original 1B sample. No checkpoints; at most 32 matrix
snapshots per arm. No existing job is canceled or changed.

28 layers, FFN 768, head dimension 128, sequence 4096, global batch 128,
microbatch 4, TP=PP=1. Peak LR .015, minimum .0015, 25 warmup iterations,
cosine decay through step 191. Validation: 20 batches at 50/100/150/191.
BF16 forward/backward, FP32 master weights and geometry, BF16 NS8 LMO,
Nesterov .9. Constrained matrix WD=0; reference auxiliary AdamW settings.

**All three fix the spectral solver to full FP32 SVD** and retain the full
paired spectrum. Warm approximation is not varied in this mechanism test.
Both branches use post-step radial return and returned-point cache. Each
positive-LR component update uses exactly one trial solve after a separate
bootstrap. No post-repair scale, gap-dependent step multiplier, clipping
projection or fallback is added. None of these are ambient PGD runs.

Main training logs distinguish trial, returned and radial lengths; additional
statistics record D0 trial length, correction size and direction ratios.
For `none`, `near` means window eligibility, **not a performed correction**.
Legacy host labels `pgd branches` and `post projections` must not be read as
actual PGD or Frobenius sphere projection. Use the dedicated statistics.

Each allocation checks the frozen source, runs both CUDA test suites, and
completes a separate four-step smoke before fresh main training. An eight-
snapshot same-D0 audit runs after training and is excluded from training time.
The matched H20 final validation losses, correction frequency and actual step
profiles will decide the next experiment; no 1B/3B continuation is automatic.

## Files and Reproduction

- `scripts/direct_top_repair.py`: three repairs and the opt-in training adapter.
- `scripts/local_direct_top_repair.py`: local objectives, paired trajectories
  and real-snapshot counterfactuals; no training-loss claims from momentum.
- `scripts/test_direct_top_repair.py`: geometry and host-integration tests.
- `slurm/direct_top_repair_100m_20261112.sbatch`: three arms, at most two GPUs.
- Existing pilot launcher/observer are reused with `--study direct_top`.
  Existing production optimizer files and the paper reproduction repo are unchanged.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/local_direct_top_repair.py \
  --snapshots docs/results/direct_top_repair_20261112/input_snapshots \
  --steps 2000 --output docs/results/direct_top_repair_20261112/local_audit.json
```

## Submission and Completion

Submitted on 2026-11-12 at 16:19 HKT. All three were pending resources at
submission and are now complete. Times below are UTC+08:00 on November 12.

| Job | Mode | Start | End | State |
|---|---|---|---|---|
| 4427402_0 | No correction | 17:34:57 | 18:34:46 | Completed, 0:0 |
| 4427402_1 | Multiple-root correction | 18:34:57 | 19:34:16 | Completed, 0:0 |
| 4427402_2 | Finite-step spectral-slack correction | 18:37:38 | 19:36:45 | Completed, 0:0 |

[Submission receipt](../results/direct_top_repair_20261112/submission.json).
The array explicitly limits concurrency to two GPUs. Both CPU suites passed
again on H20's software environment (18 tests total), followed by the same
CUDA tests in every allocation. All smokes and main runs passed. Home storage
had approximately 29 GB free at submission; this is not a new storage reading.

Frozen remote project:
`/home/u3013198/projects/direct-top-repair-100m-20261112`.
Its 3,154-file source manifest SHA256 is
`79faea9f656d9ceb3a7f1cc5c233b0098b751dec27e0d6eaa91c5668b74e4a19`.
The frozen launch document is intentionally not edited after submission;
this local report records the later submission receipt and status.
