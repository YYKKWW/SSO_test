# H20 integration verification, 2026-09-24

This is an engineering verification report, **not a 3B optimizer comparison**.
Two/four-step losses and startup-inclusive times cannot establish which method
trains best or give steady-state optimizer overhead.

## Scope and source

- Git branch: `exp/mcsd-three-geometries` in `YYKKWW/SSO_test`.
- Isolated H20 checkout: `/home/u3013198/projects/SSO_test-three-geometries`.
- Existing environment and indexed data are reused read-only. The original
  `SSO_test` source and its four active long jobs were not modified/cancelled.
- Algorithm checks below ran source `9ca16b6`. Recovery fixes are in `92b9ed5`;
  automated GPU verification ran the same kernels under launcher `bd76e91`.
- No new upstream framework was substituted. Missing model, training-model
  configuration and tokenizer source is now tracked in the experiment branch.
- At most two extra H20 GPUs were used concurrently, keeping this user's total
  at six while the four historical jobs were running.

## Real model and data

The actual GPT model has **126,641,024 trainable parameters**, 28 layers,
width 384, FFN 1152, 6 query heads / 3 KV groups, head dimension 64, and
sequence length 2048. There are **196 logical constrained matrices** in
112 physical matrix parameters; auxiliary parameters belong to AdamW.
Checkpoint inspection found 227 FP32 master tensors and BF16 model weights.
Constrained groups have both `weight_decay=0` and `wd_mult=0`.

The existing indexed corpus contains:

| Split | Indexed sequences | Stored tokens |
| --- | ---: | ---: |
| Train | 2,234,668 | 3,000,012,394 |
| Validation | 22,200 | 30,067,753 |

Main training processes `11445 * 128 * 2048 = 3,000,238,080` tokens. This
slightly exceeds the stored training-token count; describe it as a processed
token budget, not as that many distinct tokens. Counts do not prove document
deduplication or split independence. A final held-out evaluation split remains
to be specified before making a publication claim.

## Successful single-GPU integration checks

All rows completed with Slurm exit `0:0`, finite train/validation losses, zero
NaN iterations and zero skipped iterations. Each run constructed the real
model, read the real indexed data, and executed forward/backward and updates.

| Constraint | Method | Job | Steps | Slurm elapsed |
| --- | --- | ---: | ---: | --- |
| Frobenius | MCSD | 4104149 | 2 | 00:00:57 |
| Frobenius | MCSD-TP | 4104148 | 4 | 00:01:38 |
| Frobenius | MuonH adaptation | 4104150 | 2 | 00:00:56 |
| Stiefel | MCSD | 4104144 | 2 | 00:01:05 |
| Stiefel | MCSD-TP | 4104139 | 2 | 00:00:58 |
| Stiefel | iMuon adaptation | 4104145 | 2 | 00:01:10 |
| Spectral | MCSD, PI/top-8 | 4104146 | 2 | 00:01:02 |
| Spectral | MCSD-TP, PI radial | 4104147 | 2 | 00:00:55 |
| Spectral | SSO adaptation | 4104151 | 2 | 00:01:10 |
| Spectral | MuonSphere adaptation | 4104152 | 2 | 00:01:02 |

Frobenius MCSD-TP above includes checkpoint I/O; other rows do not. Peak
allocated memory in the four-step reference was approximately 16.8 GB.
The small number of updates is a startup/integration check, not a stability
test over a learning-rate schedule.

## Checkpoint investigation

Jobs 4104138 -> 4104142 saved at step 2 and resumed to step 4; job 4104148
ran the same four-step schedule without interruption. Final validation loss
was `10.74630` in both. Model weights, numeric optimizer state and schedule
matched, but the first strict state comparison found differences:

1. PyTorch 2.6 converts nested string labels in optimizer state to generator
   representations. `ManifoldMCSD.load_state_dict` now reconstructs labels
   from the model's validated component layout, validates row mappings, and
   clears device row-index caches. Inherited baseline states use the same fix.
2. Recreating a DataLoader consumes global CPU RNG for worker base seeds.
   The new manifold protocol now gives the deterministic indexed GPT loader
   its own seeded generator. Old optimizer entry points are unaffected.

The matching verifier is `scripts/manifold/check_resume.py`. It checks model,
optimizer, scheduler, iteration and all saved RNG states, not just final loss.
Only use it with trusted checkpoints because Megatron legacy torch checkpoints
require pickle deserialization.

Six small CPU round-trip checks (three geometries, both MCSD variants) passed
in the **H20 environment's PyTorch 2.6.0**, including exact restored state and
next-update equality. The private DataLoader generator preserves global CPU
RNG in that environment. These checks do not replace GPU model-level recovery.

The corrected GPU verification **passed**:

| Role | Job | Slurm elapsed | Result |
| --- | ---: | --- | --- |
| Save and stop at step 2 | 4104181 | 00:01:05 | Completed |
| Continuous four-step control | 4104182 | 00:01:41 | Completed |
| Resume to step 4 and compare | 4104211 | 00:01:31 | Completed, exact match |

Model weights, optimizer state (including all component metadata), scheduler,
iteration and all saved RNG states have **zero mismatches**. See the actual
machine-readable [verification result](resume_verification_20260924.json).
This demonstrates clean-boundary single-GPU recovery in this protocol, not
mid-step failure recovery or arbitrary changes in world size.

An additional two-GPU Stiefel MCSD-TP smoke, **4104214**, is queued behind the
successful verifier. Its status is separate from the completed single-GPU gate.

## Resolved startup failures

- 4104124: module environment supplied `python3`, not `python`, before venv
  activation. The Slurm script now uses `python3` at that point.
- 4104125: framework training-model Python files were ignored as model weights;
  restored source and added a regression check on ignore rules.
- 4104136: Megatron constructed raw weights directly in BF16. The manifold
  GPT path now constructs in FP32, captures radii/master initialization, then
  applies the ordinary BF16 model wrapper. Historical methods are unchanged.

## Remaining gates

- Complete the additional DP>1 smoke; single-GPU strict resume passed.
- Run longer practical-accuracy/stability pilots, including Stiefel
  orthogonality defect, spectral sphere defect and sampled LMO error.
- Freeze LR/momentum/precision settings using equal development budgets.
- Run paired-seed 3B comparisons and an independent held-out evaluation.
- Check checkpoint disk usage before parallel long runs: the current launcher
  retains saved checkpoints and does not automatically prune them.

The 3B launcher is implemented and uses 3B data; **no new 3B main experiment
has been submitted as part of this engineering verification**.
