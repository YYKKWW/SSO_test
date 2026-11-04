# Point Extrapolation and Matched 3B LR Grid

**November 4, 17:07 HKT update:** 4375560--4375565 completed; 4375566 and
4375567 are running. [Results, times and curves](FROBENIUS_RESULTS_20261104.md).
The table below preserves the original submission plan, not current live status.

**Submitted November 2, 2026.** Eight single-H20 jobs, at most two concurrent
in this batch. The two point pilots started on SPG-7-2; six full-budget jobs
wait for their serial lanes. This is a submission snapshot, not live status.

[Results overview](FROBENIUS_3B_6B_SUMMARY.md) |
[Actual submission register](frobenius_point_20261101/point_lr_batch_20261102_register.json)

## Jobs

| Job | Method | Peak LR | Momentum / cap | Training interval | Dependency |
|---:|---|---:|---|---|---:|
| 4375560 | MCSD point extrapolation | 0.00250 | cap=0.95; no gradient EMA | Resume 1000 to 3000 | None |
| 4375561 | MCSD point extrapolation | 0.00125 | cap=0.95; no gradient EMA | Fresh 0 to 3000 | None |
| 4375562 | MCSD | 0.012 | beta=0.95 Nesterov | Fresh full 3B | 4375560 |
| 4375563 | MuonH | 0.012 | beta=0.95 Nesterov | Fresh full 3B | 4375561 |
| 4375564 | NorMuon-Sphere | 0.012 | beta=0.95 Nesterov | Fresh full 3B | 4375562 |
| 4375565 | MCSD | 0.015 | beta=0.95 Nesterov | Fresh full 3B | 4375563 |
| 4375566 | MuonH | 0.015 | beta=0.95 Nesterov | Fresh full 3B | 4375564 |
| 4375567 | NorMuon-Sphere | 0.015 | beta=0.95 Nesterov | Fresh full 3B | 4375565 |

Dependencies use `afterany` to serialize independent experiments; success of
one configuration is not a scientific gate for another. The already completed
source of 4375560 is job **4373197**, whose checkpoint is at step 1000.
No MCSD-TP, extra-EMA point variant, MCSD-NorUnit, larger width or 6B job was added.

## Matched Protocol

- Width 256, 28 layers, FFN 768, 4 query heads / 2 KV groups, head dimension 64.
- Frobenius radius `R=sqrt(D_out)`, r=1; constrained-matrix weight decay zero.
- Same frozen OLMo 3B train/validation indices and tokenizer; seed 1234.
- Sequence length 2048, global batch 128, micro-batch 4, one H20 per run.
- BF16 model, FP32 master/state; existing matrix kernels unchanged.
- Auxiliary AdamW LR=0.001; validation uses 40 batches every 200 steps.
- Every run retains the **11445-step horizon, 229-step warmup and cosine decay
  to 10% of peak LR**. The two point runs only stop early at step 3000.
- Full 3B grid: beta=0.95 Nesterov, no additional MCSD direction normalization
  or adaptation. NorMuon retains its existing row adaptation (beta2=0.95).

Step 3000 represents 786,432,000 processed tokens, not a completed 3B run.
Scheduled full-budget runs process 3,000,238,080 tokens after batch rounding.
The resume restores accepted X, live Y, auxiliary optimizer, scheduler and RNG;
it does not restart warmup or alter the extrapolation rule. Point validation
still evaluates live Y, not accepted X. No new X/Y diagnostics were enabled.

## Resources and Verification

At the preflight check, 3 H20 GPUs were unallocated and the account had 2
unrelated mechanism jobs running. This batch uses at most 2 additional GPUs,
does not modify those jobs, and leaves one of the then-free GPUs unrequested.
Scheduler availability can subsequently change. Home storage had about 34 GiB free.

The six full-budget runs disable checkpoints, retaining logs/configurations.
The two pilots save at step 3000; the source step-1000 checkpoint is preserved.
Approximate run times excluding queueing: 3.3 hours for the continuation,
5 hours for the fresh pilot, and 18--20 hours per full 3B run. Two serial lanes
give roughly 2.5--3 days for this batch if GPU allocations remain available.

Preflight checked all eight resolved configurations, unchanged runtime source,
available data/tokenizer, strict resume compatibility and duplicate configurations
inside the frozen project. Local plan tests passed. The first queue readback
confirmed both pilots running and the other six waiting on their recorded dependencies.

Runtime commit: `8250d00f9a36832bd6fd7198fe2e2b5d3787b7b3`.
The submission helper is deployed outside the frozen training checkout, so no
optimizer or training source was changed for this batch.

## Readout

1. Compare the point pilots at step 3000 with the completed same-code Nesterov
   control (4373195: 3.778030) and cap=0.90/LR=0.0025 (4373196: 3.814502).
2. Compare each full-budget LR against the same-method beta=0.95 Nesterov
   LR=0.010 reference, then compare methods with equal tuning budgets.
3. Keep failed and unfavorable runs in the ledger. No automatic 6B promotion
   or new diagnostic/normalization ablation is scheduled.
