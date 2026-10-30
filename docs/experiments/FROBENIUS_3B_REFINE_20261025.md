# Frobenius Sphere: 3B Refinement Before Further 6B Runs

Status checked on **2026-10-25 11:05 HKT**. All six jobs have been submitted;
only the first MCSD run has started. There are no validation results yet.

## Submitted Experiments

All runs use width 256, 3B tokens, Frobenius shape radius `r=1`, and seed 1234.
The first-order momentum coefficient is 0.95 in every run.

| Job | Method | Peak LR | Momentum | Dependency | Checked status |
|---|---|---:|---|---|---|
| 4329081 | MCSD | 0.008 | EMA | None | Running, step 30/11445 |
| 4329082 | NorMuon-Sphere | 0.008 | EMA | None | Pending: Resources |
| 4329083 | MCSD-TP | 0.008 | EMA | afterany:4329081 | Pending: Dependency |
| 4329084 | MuonH | 0.008 | EMA | afterany:4329082 | Pending: Dependency |
| 4329085 | MCSD | 0.010 | Nesterov | afterany:4329083 | Pending: Dependency |
| 4329086 | MCSD-TP | 0.010 | Nesterov | afterany:4329084 | Pending: Dependency |

The two serial lanes cap **this batch** at two concurrent single-H20 jobs.
They do not cap unrelated jobs on the account. At the check, the account had
six running single-GPU jobs: this batch's first job, two older Nesterov runs,
and three tasks in a separately submitted array. All 16 platform GPUs were
allocated. The second lane will start when the scheduler grants resources.

MCSD's last logged step took approximately 5.37 seconds, with zero skipped
iterations and zero NaN iterations. This confirms startup only, not training
quality or final stability. Based on previous runs, each job takes about 18.5
hours; each lane takes about 55.5 hours plus queueing, and the full six-run
batch costs approximately 111 GPU-hours.

## Questions and Existing Controls

The EMA group adds one matched learning-rate point for all four methods. It
does not add direction normalization or modify their optimizer algorithms.
The existing complete 3B controls at `beta=0.95 EMA, LR=0.010, seed=1234` are:

| Method | Existing job | Final validation loss |
|---|---|---:|
| MCSD | 4197465 | 3.467720 |
| MCSD-TP | 4197469 | 3.467858 |
| MuonH | 4320101 | 3.464988 |
| NorMuon-Sphere | 4320102 | 3.463500 |

These complete controls are reused rather than rerun. Each new EMA run
differs from its control only in matrix peak LR and the corresponding
minimum LR (fixed minimum-to-peak ratio 0.1).

The two Nesterov runs differ from the respective MCSD/MCSD-TP EMA controls
only in enabling the existing Muon-style Nesterov gradient combination at
the same beta and LR:

```text
M_t = beta * M_(t-1) + (1 - beta) * G_t
EMA:       S_t = M_t
Nesterov:  S_t = beta * M_t + (1 - beta) * G_t
```

This is not manifold point extrapolation and does not evaluate a gradient at
a separate extrapolated point. A comparison between these runs and a
baseline with different momentum is not a matched-momentum ablation.

## Fixed Protocol

- 28 layers, width 256, FFN width 768, four attention heads, two KV groups,
  head dimension 64, sequence length 2048; 73,418,496 model parameters.
- Frobenius sphere with component radius `R = sqrt(D_out)`; fixed `r=1`.
- Existing OLMo-Mix-1124 **3B training index**, not the 6B index or repeated
  1B data; unchanged 3B validation index and OLMo-2 tokenizer.
- Global batch 128, micro batch 4, one H20 per run, from initialization.
- 11,445 iterations, 3,000,238,080 processed tokens; 229 warmup iterations,
  cosine decay to 10% of peak LR.
- BF16 model training, FP32 optimizer master weights, existing matrix-sign
  Newton--Schulz path; no numerical-kernel changes.
- Auxiliary AdamW LR remains 0.001. NorMuon second-moment beta remains 0.95
  and epsilon remains `1e-10`; these are distinct from first-order momentum.
- Evaluation every 200 iterations and at the final iteration, 40 batches.
- Checkpoints disabled; logs, resolved configurations, source hashes and
  submission records retained. No existing runs or data were deleted.

## Reproducibility and Checks

Frozen training project:
`/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005`.
Training commit: `fc15600a2898d1748c8b34fe8d3945b1036f03ab`; the remote
worktree was clean before and after submission. No local optimizer edits
were uploaded to this frozen project.

- [Six-run plan](../../configs/manifold/frobenius_3b_refine_20261025.json)
- [Slurm registration and configuration hashes](frobenius_3b_6b/refine_register_20261025.json)
- [Existing experiment results](FROBENIUS_3B_6B_SUMMARY.md)
- Local pre-submission checks: `tests/test_frobenius_3b_refine_plan.py`
  (historical local test, not included in this documentation-only publication).

Both local plan checks passed. Remote checks verified the same model, data,
initialization, numerical options, validation and auxiliary optimizer
settings against the previous 3B protocol. All six submitted configurations
match the registration hashes; duplicate runs were excluded before submission.

The existing submission helper, outside the frozen project, was reused:
`/home/u3013198/submit_frobenius_followup.py`, SHA256
`c7069a38ce0c651b8b423d4483e3d7182853a475cac4d67b4a7d48fb10a9dcf7`.
Plan SHA256:
`74f93c6917d02a636090c62953bd03b7f131684f2a52dbe46442e86bba67e4e4`.
The remote registration is
`/home/u3013198/frobenius_3b_refine_register_20261025.json`.

No new 6B jobs were submitted. Existing pending 6B jobs 4320105 and 4320106
have not been cancelled, held or otherwise modified in this submission.

## Decision After Completion

1. Compare the four EMA methods at both 0.008 and 0.010 with the same seed.
2. Compare MCSD and MCSD-TP Nesterov against their respective EMA controls
   at beta 0.95 and LR 0.010; do not mix this change with the LR comparison.
3. Use final validation at step 11445, and aligned validation curves, rather
   than a selected minimum intermediate value. Treat single-seed differences
   as screening evidence and repeat selected configurations on paired seeds.
4. Select a small 6B grid only after this screening. A configuration losing
   at 3B is not proved to lose at 6B; the budget-dependent LR hypothesis
   remains untested until a matched 6B comparison is completed.
