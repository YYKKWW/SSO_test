# Frobenius Follow-Up: Nesterov LR Grid and MCSD-NorUnit

**Status checked October 30, 2026: prepared locally, not submitted.** The H20
upload failed and subsequent SSH connections timed out. No Slurm job IDs have
been obtained for this batch. Existing tasks were not changed.

## Requested Runs

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

This one run changes both row adaptation and unit-F normalization relative
to standard MCSD. It evaluates their combined effect, not a separate causal
contribution from each. Old unit-F results used 0.95 EMA and cannot serve as
a matched Nesterov unit-only control.

## Validation and Provenance

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
