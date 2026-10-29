# Third-Seed Replication and Offline Startup/Capacity Audit

Checked on **2026-10-29 at 17:04 HKT**. This batch implements the requested
replication plan, without retuning the candidate or duplicating existing jobs.

## Training Queue

| Job | Method | Seed | Budget | Status at this check |
|---|---|---:|---:|---|
| `4346705_2` | MCSD-TP K10 control | 2345 | 3B | Running, step 1570/5723 |
| `4346705_3` | Hybrid K10/c8, top<=4 | 2345 | 3B | Running, step 20/5723 |
| `4346705_4` | Ambient near-direction control | 2345 | 3B | Running, step 20/5723 |
| `4346705_5` | Top-1 near-direction control | 2345 | 3B | Running, step 20/5723 |
| `4360583_0` | MuonSphere | 3456 | 3B | Dependency wait |
| `4360583_1` | SSO | 3456 | 3B | Dependency wait |
| `4360583_2` | MCSD-TP K10 control | 3456 | 3B | Dependency wait |
| `4360583_3` | Hybrid K10/c8, top<=4 | 3456 | 3B | Dependency wait |

The third-seed array depends on successful completion of **all of array
4346705 and audit 4360582**. A failed dependency blocks it rather than starting
an unchecked run. Every training run uses one H20. The existing array limit
was raised from one to four after checking available resources; the new array
also has a limit of four and does not overlap the previous batch.

At the check, this user occupied four GPUs, other users occupied two, and ten
of sixteen H20 GPUs were unallocated. Home storage had approximately 28 GB
free. No checkpoints are saved. Based on preceding runs, the current batch
needs roughly 17-19 more hours and the third-seed batch another 17-19 hours,
excluding scheduler delays. These are estimates, not reservations.

## Matched Protocol

- Width 256, 28 layers, sequence length 4096, global batch 128, microbatch 4.
- Same indexed 3B OLMo-Mix source and tokenizer; fresh initialization, 5723
  iterations (3,000,500,224 processed tokens), no checkpoint resume.
- Peak LR `1.5e-2`, 250 warmup steps, cosine decay to `1.5e-3`; 20 validation
  batches every 250 steps and at completion.
- Experimental methods: warm block-8, FP32 QR/Ritz, K10, gap `5e-5`, cluster
  tolerance `2e-3`, adaptive top<=4, Frobenius-normalized Nesterov momentum.
  The hybrid uses c8 without post-correction normalization or length matching.
- Both branches use the same approximate radial return. No online accuracy
  gate or original-matrix full-SVD fallback is introduced.

Seed **3456** was selected before seeing any third-seed outcomes. The four
launch descriptions match seed 2345 except for seed. All runtime files under
`Megatron-LM/` are hash-identical to the second-seed snapshot; the existing
remote source was not edited. The shared launcher now accepts an optional
seed with the original default unchanged.

[Launcher](../../slurm/cluster_seed3_3b_20261029.sbatch) |
[Submission](../results/cluster_seed3_3b_20261029/submission.json) |
[Source hashes](../results/cluster_seed3_3b_20261029/deployment.json) |
[Status and validation points](../results/cluster_seed3_3b_20261029/status_20261029.json)

## Completed Offline Audit

Job **4360582** completed successfully in **22 seconds**. Three CUDA tests
passed; all 52 local cluster regression tests also passed. Training and audit
are separate processes. The audit inspected 88 stored matrices from the
completed first-seed K10 control/hybrid: 21 at steps <=50 and 19 near cases.
Matrices, momentum, trial steps, and initial bases were held fixed within each
paired comparison. Full SVD was used only as an offline oracle.

| Early-sample metric | K10 | K20 |
|---|---:|---:|
| Warm trial mean absolute gap error | `8.775e-5` | `1.157e-5` |
| Warm trial maximum absolute gap error | `5.127e-4` | `6.527e-5` |
| Warm trial mean absolute radius error | `2.175e-5` | `1.853e-6` |
| Warm trial maximum absolute radius error | `1.040e-4` | `9.775e-6` |
| Fresh current-point mean absolute gap error | `3.622e-3` | `1.051e-3` |

The fresh estimates restart the solver on stored matrices; they are **not**
the actual cached training estimates. The warm comparisons reuse the same
saved basis on the same reconstructed trial. They are **not** a replay of a
different K20-to-K10 training trajectory. K20 improves these local errors,
but no training-loss improvement is established. The worst sampled K20 warm
gap error still exceeds the routing threshold of `5e-5`.

For capacity, top<=4 and top<=6 produced identical directions on all 19 near
samples, both with cached bases and with full-SVD oracle bases. Both returned
outside the near region in 9/19 cases. **None of these samples had a true
cluster larger than four.** This is a coverage gap, not evidence that top-4
is sufficient: the full training log recorded 2,952 capacity-overflow calls.
A synthetic six-direction plateau test verifies that the audit detects a
top-4/top-6 difference, but cannot substitute for real overflow samples.

[Audit implementation](../../scripts/audit_cluster_startup_capacity.py) |
[Tests](../../scripts/test_cluster_replication.py) |
[Full metrics](../results/cluster_seed3_3b_20261029/startup_capacity_4360582.json) |
[CUDA checks and completion](../results/cluster_seed3_3b_20261029/completion.json)

## Decisions

1. Complete the already-running mechanism controls and the queued four-method
   third seed. Report paired within-seed differences, including unfavorable
   outcomes. Three seeds alone do not establish statistical significance.
2. Retain K20-to-K10 as a possible matched control/hybrid follow-up, but do
   not submit it yet. Numerical improvement alone does not imply better loss.
3. Capture a bounded set of actual capacity-overflow matrices in a future
   instrumented run before deciding on top<=6 long training. Existing saved
   matrices do not cover this question.
4. No broad gap sweep, width-512 expansion, or longer token budget was added.

The prior [structural audit](cluster_structural_directions_20261029.md) remains
separate: it tested a different small-model direction, not this paired
startup/capacity replay. Neither audit proves stochastic convergence.
