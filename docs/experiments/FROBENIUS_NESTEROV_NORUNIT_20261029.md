# Frobenius Follow-Up: Nesterov LR Grid and MCSD-NorUnit

**November 5, 15:10 HKT:** the six standard LR-grid cells and MCSD-unitF
are complete. MCSD-unitF reaches 3.460665; NorUnit is running with
3.471547 at step 10600. Both seed-2026 baselines remain pending.
[Latest table and curves](FROBENIUS_RESULTS_20261105.md).

## November 4 Submission

**Submitted at 18:45 HKT; all four pending at the submission check.** These
are full 3B runs from initialization, not shortened pilots or point extrapolation.
Width=256, peak LR=0.010, beta=0.95 Nesterov, Frobenius radius
`R=sqrt(D_out)` and the matched protocol below remain fixed.

| Job | Method | Seed | Purpose | Initial queue status |
|---:|---|---:|---|---|
| 4390468 | MCSD-unitF | 1234 | Global normalization only | Pending: Priority |
| 4390469 | MCSD-NorUnit | 1234 | Row adaptation plus unit-F | Pending: Priority |
| 4390470 | MuonH | 2026 | Paired second seed | After 4390468 ends |
| 4390471 | NorMuon-Sphere | 2026 | Paired second seed | After 4390469 ends |

Two serial lanes cap this batch at two single-H20 jobs. Existing jobs are
unchanged. Checkpoints are disabled; each job has a 36-hour limit. Earlier
matched runs took about 18.5 hours per job, excluding queue time; this is an
estimate, not a measured runtime for the new adaptive variant.

Compare 4390468 against standard MCSD 4329085 to test global normalization,
and 4390469 against 4390468 to test the added row adaptation under unit-F
scaling. Compare both seed-2026 baselines against existing MCSD 4346667;
report paired seeds, not a single baseline seed against an MCSD mean.

The active frozen runtime remains `8250d00f9a36832bd6fd7198fe2e2b5d3787b7b3`.
No optimizer/model files were overwritten. A search of 134 prior resolved
configurations found no matching runs. Local checks passed 278 tests; six
framework-free equation/resume cases passed using the H20 environment's
PyTorch 2.6.0+cu124 on **CPU**. GPUs were fully allocated at preflight;
those checks alone were not CUDA integration or long-training validation.
The subsequent training evidence is reported in the dated update above.

[Four-run plan](../../configs/manifold/frobenius_norunit_seeds_20261104.json) |
[Submitted job register](frobenius_norunit_seeds_20261104/norunit_seeds_20261104_register.json) |
[H20-environment CPU checks](frobenius_norunit_seeds_20261104/cpu_preflight.json)

## Earlier Plan

**November 2 update:** the six standard LR=0.012/0.015 configurations were
submitted as jobs 4375562--4375567 in the
[two-lane point/LR batch](FROBENIUS_POINT_LR_BATCH_20261102.md).
MCSD-NorUnit was not part of that batch; it was submitted separately above.

**Historical status, October 30, 2026: prepared locally, not submitted.** The H20
upload failed and subsequent SSH connections timed out. No Slurm job IDs have
been obtained for this batch. Existing tasks were not changed.

## Original Requested Runs (October 29)

| Method | Peak matrix LR | Variant | Serial lane |
|---|---:|---|---:|
| MCSD | 0.010 | NorMuon row adaptation + unit-F direction | 0 |
| MCSD | 0.012 | Standard | 1 |
| MuonH | 0.012 | Standard | 0 |
| NorMuon-Sphere | 0.012 | Standard | 1 |
| MCSD | 0.015 | Standard | 0 |
| MuonH | 0.015 | Standard | 1 |
| NorMuon-Sphere | 0.015 | Standard | 0 |

All seven use **width 256, 3B tokens, Frobenius radius r=1, beta=0.95
Nesterov, seed=1234**, and fresh initialization. No MCSD-TP job is added.
Two serial single-GPU lanes limit this batch to **two simultaneous H20 GPUs**;
jobs in a lane start after the previous job ends. This does not constrain other
already-submitted batches. Checkpoints are disabled; retain logs and configurations.

The matched protocol remains 11,445 iterations, sequence length 2048, global
batch 128, micro-batch 4, 229 warmup steps, cosine decay to 10% of peak LR,
auxiliary AdamW LR=0.001, and 40 validation batches every 200 steps plus final
validation. The four 0.95-Nesterov/LR=0.010 references remain in the
[completed-results summary](FROBENIUS_3B_6B_SUMMARY.md#latest).

## What MCSD-NorUnit Changes

For each matrix component, retain MCSD's tangent projection **before** the
matrix sign, then apply the NorMuon row scaling and unit-F normalization:

```text
M = 0.95 * M + 0.05 * G
S = 0.95 * M + 0.05 * G
H = MSIGN(TP_W(S))
v = 0.95 * v + 0.05 * row_mean(H^2)
A = H / (sqrt(v) + 1e-10)
D = UNIT_F(A)
Y = W - eta * R * D
W_new = R * Y / norm_F(Y)
```

`UNIT_F` uses a tiny-value guard and returns zero for zero input. It does not
restore the old reference direction length or apply an additional
`1/sqrt(rank)` factor. This is a **heuristic adaptive MCSD variant**, not the
unchanged spectral-norm LMO or a proven accelerated method. It differs from
NorMuon-Sphere because MCSD projects `S` before the matrix sign.

The original plan's adaptive run changes both row adaptation and unit-F normalization relative
to standard MCSD. It evaluates their combined effect, not a separate causal
contribution from each. Old unit-F results used 0.95 EMA and cannot serve as
a matched Nesterov unit-only control. The November 4 submission adds that
missing matched control as job 4390468.

## Original Validation and Provenance (October 30)

- Explicit [seven-run plan](../../configs/manifold/frobenius_nesterov_norunit_20261029.json).
- Local frozen deployment commit: `a3f73b135cbd58dffcbe54da83eec9d9d3ae0e5c`,
  based on the proven H20 commit `fc15600a2898d1748c8b34fe8d3945b1036f03ab`.
  This deployment snapshot is separate from the documentation publication;
  its hash is not a claim that it is available on this GitHub branch.
- 166 local tests passed on the isolated snapshot, including update equations,
  row state, resume, argument routing, and the two-lane plan. These are **not**
  an H20 CUDA or full-training validation.
- After connectivity returns: transfer the isolated snapshot, run a short
  single-H20 integration check for MCSD-NorUnit, then submit the seven main
  runs and record their IDs. Do not report prepared jobs as running.

Report final-budget validation loss and total elapsed time. Compare the six
standard runs against their same-method LR=0.010 references, and the adaptive
run against all three matched LR=0.010 methods. Keep unsuccessful runs in the
ledger; do not claim an optimum or a gain before complete results are available.
