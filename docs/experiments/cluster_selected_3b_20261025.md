# Width-256 selected MCSD-P2GD candidates at 3B tokens

> Verified on **2026-10-26 15:16 HKT**: SSH access recovered. K20 control,
> K20/c8 hybrid and MuonSphere completed all 5723 steps with exit code 0:0.
> SSO is at step 3590; both K10 arms are at step 3230. The startup section
> below is historical. Full results and same-step comparisons are in the
> [dated result summary](cluster_results_20261026.md).

Array **4329039** was submitted on 2026-10-25 at 10:54 HKT. Six matched
runs test whether the small 1B benefit of the normalized-momentum hybrid
persists over longer training, and whether K10 can replace K20 without
losing that benefit. Each run uses one H20, with at most three concurrent
tasks. Existing jobs and the frozen reproduction repository are unchanged.

## Latest results

| Task | Method | Status / latest training step | Latest validation (step) | Training + validation time |
|---|---|---|---|---|
| 0 | MCSD-TP K20 | Completed / 5723 | 3.444459 (5723) | 18:20:33 |
| 1 | Hybrid K20/c8 | Completed / 5723 | 3.443598 (5723) | 18:21:08 |
| 2 | MuonSphere | Completed / 5723 | **3.442197 (5723)** | **16:37:29** |
| 3 | SSO | Running / 3590 | 3.652187 (3500) | Pending completion |
| 4 | MCSD-TP K10 | Running / 3230 | 3.713605 (3000) | Pending completion |
| 5 | Hybrid K10/c8 | Running / 3230 | 3.713485 (3000) | Pending completion |

The K20 hybrid improves its control by 0.000861 at 35 seconds extra runtime,
but remains 0.001401 worse and 10.39% slower than MuonSphere. No logged NaN
or skipped iteration was found in any arm. Do not rank partial validation
values against final ones. At the common step 3000, K10 hybrid/control are
3.713485/3.713605 and SSO is 3.713976; these single-seed differences are small.

At step 5720, K20 hybrid used 25,490/1,441,440 cluster calls (1.7684%), with
58.69% repeats, mean burst 2.42, and 2,931 top-four-cap overflows. Each completed
experimental arm recorded 1,442,196 small Ritz SVD calls and zero original-matrix
full SVD/svdvals calls. The hybrid's 52 selected offline samples had mean gap
error 4.34e-6, maximum returned relative radius error 2.85e-5, and no observed
false-safe/false-cluster decisions. This is not a population error guarantee.
K10 audits are not available until completion.

The [machine-readable snapshot](../results/cluster_selected_3b_20261025/status_20261026_1516.json)
contains all validation points, source metadata, sampled audits and Slurm
status. This read-only update did not modify, cancel or submit jobs.

## Verified startup

At 11:00 HKT on 2026-10-25, tasks4329039_0/_1/_2 had each reached
step20/5723 after passing all38 CUDA preflight tests. They reported no
NaN/skipped iterations or startup errors. K20 control/hybrid were about
11.2s/step and MuonSphere10.1s/step. Tasks_3/_4/_5 were pending because
of the three-task concurrency limit. There is no validation result yet.
The two unrelated running jobs remained unchanged, for five user GPUs
occupied in total at this check.

## Selection from completed experiments

The [completed 1B comparison](cluster_candidates_1b_20261024.md) found:

- At LR1.5e-2, K20/c8 hybrid loss was 3.587659 versus 3.588559 for MCSD-TP,
  3.582948 for MuonSphere and 3.591530 for SSO. The hybrid improved its
  control, but did not beat MuonSphere at its best tested LR.
- At LR2e-2, K20/c8 improved its control by 0.001681. The best hybrid
  losses at the two LRs differed by only 0.000209; higher LR was not
  demonstrated to transfer better to a longer horizon.
- K40 improved sampled numerical accuracy but worsened final loss and
  added 6.83% runtime. c12 reduced repeated routing but worsened loss.
- K10/c8 has not yet been tested at 1B. Earlier 100M K10/c1 was less
  accurate than K20/c1, whereas the K10 MCSD-TP control was slightly
  better than its K20 control. K10 remains an unproven speed candidate,
  not a rejected method or an established replacement.

The selected main candidate is therefore K20/c8 at LR1.5e-2. K10 is
included with its OWN MCSD-TP control to distinguish changes in the
spectral solver from the effect of adding the near-cluster branch.
MuonSphere and SSO are rerun under the same 3B protocol. No K40, c12,
additional widths or LR sweep are added in this round.

## Submitted grid

| Job | Method | Warm block iterations | Near-branch coefficient |
|---|---|---:|---:|
| 4329039_0 | MCSD-TP control | 20 | N/A |
| 4329039_1 | Mixed MCSD-P2GD type | 20 | 8 |
| 4329039_2 | MuonSphere | Reference BF16 PI10 | N/A |
| 4329039_3 | SSO | Reference BF16 PI10 | N/A |
| 4329039_4 | MCSD-TP control | 10 | N/A |
| 4329039_5 | Mixed MCSD-P2GD type | 10 | 8 |

K10/K20 are iteration counts of the same warm block-8 FP32 QR/Ritz
solver, not block ranks or switches to BF16 spectral estimation.
The baselines retain their reference NS8/PI10 implementations. Cluster
flags in their environment are inactive.

## Fixed algorithm

Routing gap5e-5; cluster-membership tolerance2e-3; adaptive top<=4;
Frobenius normalization of effective momentum BEFORE cone-model projection;
c8 scales only the near-branch trial step. No post-cone normalization or
MCSD length matching. Both branches share the approximate radial return
and cache the scaled singular-value estimates of the returned point.

The four experimental arms have no original-matrix full SVD/svdvals in
optimizer updates, no online error gate and no automatic SVD fallback.
The runtime guard only catches unintended full-SVD code paths. Exact
numerical checks are performed offline after training. The cached gap is
approximate; the capped near-root cluster is not necessarily the complete
true top eigenspace. This is an engineering P2GD-type variant, not a
claim of stochastic Clarke convergence.

The legacy host flags `exact_svd_gap_only`, `tp_radial_pgd_exact` and
`warm_start_uv=0` are overridden by cluster dispatch, as in the 1B runs.
Effective solver/return choices are recorded explicitly in launch metadata.

## Training and evaluation

All runs start from scratch with seed1234; no 1B checkpoint is resumed.
Width256, 28 layers, FFN768, head dimension128, sequence length4096,
global batch128, micro-batch4, TP=PP=DP=1. Peak LR1.5e-2, 250 warmup
iterations, cosine decay to1.5e-3 at iteration5723. Actual processed tokens
are **3,000,500,224**, the rounded-up 3B budget. The existing 3B OLMo indexed
source and tokenizer are reused, without downloading or copying the corpus.

The warmup duration stays at250 steps rather than being tripled. This is
a new 3B schedule, not continuation of a finished 1B cosine schedule.
BF16 training/NS8 and FP32 master weights/Ritz arithmetic are unchanged.
Experimental constrained matrices have weight-decay multiplier0; auxiliary
parameters retain the existing0.1 weight-decay protocol.

Validation runs every250 iterations and at5723, with **20 batches** per
evaluation instead of5 in the 1B grid. This reduces sampling noise but is
not a fixed-subset evaluation: the existing validation iterator advances.
All six runs use the same schedule. Compare their same-step validation
losses; do not splice these values into the old 1B/3B result tables with
different evaluation/data protocols. No checkpoint is saved, so retrospective
reevaluation or restart from the last checkpoint will not be available.

## Diagnostics and decision criteria

Log every10 steps. Retain at most56 matrix snapshots per experimental run:
scheduled steps2/50/250/1000/2000/3000/4000/5000/5700, eight early cluster
calls and up to eight additional cluster calls after step3000. Full-SVD
audits run only after training. Sampling remains bounded and nonrandom.

Compare final validation/train loss, training-only wall time, cumulative
and late-stage branch rates, consecutive repeat rates, burst lengths,
cluster-cap overflow, and sampled gap/radius/direction errors. In particular,
K10 must not be promoted solely because it is faster: assess its FP32
subspace approximation and returned radius drift as well.
Neither fewer cluster calls nor better spectral accuracy alone establishes
better optimization. Small one-seed gains need another seed before a
paper-level superiority claim.

## Resources and provenance

At submission, both H20 nodes were healthy with four GPUs free in total.
This array caps concurrency at three, leaving one of those slots available.
Two unrelated Frobenius-manifold jobs already occupied two user GPUs, with
two dependent successors; they were left untouched. The combined currently
planned occupancy is approximately five user GPUs, not eight.

The home filesystem had about30GB free. The isolated source snapshot adds
about63MB; no corpus copy or checkpoint is written. Logs, data indices and
bounded snapshots are retained. Estimate17--20h per run and about36--42h
for the six-job grid at three-way concurrency, subject to allocation and
measured throughput. Each task requests a30h walltime limit.

Local49 tests, shell syntax and whitespace checks passed before submission.
Each allocation runs38 CUDA preflight tests before training. The source
manifest contains3164 files and is checked before every run. Algorithm files
are unchanged from the completed1B snapshot; only the launcher and its
focused configuration test are new.

Remote snapshot: `/home/u3013198/projects/cluster-selected-3b-20261025`.
Launcher: `slurm/cluster_selected_3b_20261025.sbatch`.
Patch SHA256: `9573dcdf3a6629e2a8fd86e63a3e5ca89f83c7956053436b07b647e32e40da72`.
Local receipts and startup checks: `output/cluster_selected_3b_20261025/`.
