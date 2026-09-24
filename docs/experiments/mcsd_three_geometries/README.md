# Three-geometry MCSD study

This branch implements a fixed-width dense decoder experiment based on the
handoff in this directory. The original four files are retained verbatim as
the initial research contract. `PROTOCOL_DELTA.md` records later decisions.

`EXPERIMENT3_PUBLICATION_PLAN.md` records the 2026-09-24 scope audit and
publication-oriented plan: reuse Megatron, compare the two methods and matched
baselines on three constraints, and use a roughly 127M dense model with 3B
tokens per main run. It distinguishes implemented components from missing
formal evaluations and H20 validation. Baseline integration and full-budget
launch support are now implemented. It is
a plan, not a report of completed language-model experiments.

See [H20 validation](H20_VALIDATION_20260924.md) for actual job IDs and the
exact GPU recovery comparison, and [environment loading](ENVIRONMENT.md) for the
verified runtime and portable path overrides.

The first longer, single-H20-per-run comparison is registered in
[100M pilot, 2026-09-24](PILOT_20260924.md): ten method/constraint pairs,
three concurrent jobs at most, with shared model/data settings and disclosed
method-specific differences. All ten pilots have completed. The
[first 3B batch](MAIN_3B_20260924.md) contains six Frobenius/Stiefel runs;
the spectral group is deferred for return-accuracy investigation. These are
not completed 3B results.

## Method IDs

| Geometry | Optimizer | Main direction | Return |
| --- | --- | --- | --- |
| Frobenius sphere | `manifold_mcsd` | ambient EMA, tangent-projected momentum, BF16 Polar Express LMO | FP32 Frobenius normalization |
| Frobenius sphere | `manifold_mcsd_tp` | same, plus tangent projection after LMO | FP32 Frobenius normalization |
| scaled Stiefel | same two IDs | same respective directions | BF16 Polar Express full `msign`, then three FP32 polar-polish steps |
| spectral sphere, exact reference | same two IDs with `--manifold-spectral-solver exact` | exact SVD tangent normal or practical LMO | MCSD: full-SVD clipping; MCSD-TP: full-SVD norm radial return |
| spectral sphere, practical pilot | same IDs with `--manifold-spectral-solver pi_topk` | BF16 PI tangent normal | MCSD: top-8 deflation; MCSD-TP: PI radial scaling |

The practical spectral pilot is **not** a certified projection, and the
Stiefel finite-step return is **not** an exact polar projection. The exact
reference modes are controls, not the default performance configuration.
The sub-`1e-4` spectral gap is logged but does not stop a run. At a repeated
top singular value the chosen rank-one normal has no unique smooth-stratum
tangent interpretation, so such a run is empirical rather than a direct
validation of the smooth-region theorem.
For the practical spectral mode, exact gap and constraint checks are audits
every 200 component updates by default (`--audit-interval 20` for denser checks),
not per-step decisions; events between audits may be missed.
The same interval staggers LMO feasibility and return-defect audits across
logical components. The reported LMO ball excess and model error are measured
before MCSD-TP projects the direction for the second time. Exact spectral
returns cache the chosen top normal for the next step, avoiding a repeated
SVD of the current master matrix after the first step.

## Local checks

```bash
PYTHONPATH=Megatron-LM python -m pytest tests/test_manifold_mcsd.py tests/test_manifold_baselines.py tests/test_manifold_launch.py -q
python scripts/manifold/launch.py --geometry frobenius --method manifold_mcsd_tp
```

The launcher prints the resolved configuration by default; only `--submit`
consumes cluster resources. It submits one job, not a full sweep. The smoke
stage is limited to 100 steps; pilot/tune/main budgets are 100M/1B/3B tokens.
All stages default to the existing **3B** dataset, not repeats of the old 1B
sample. At sequence 2048 and global batch 128, main training uses 11445 steps
and 3,000,238,080 processed tokens. Verify the indexed corpus's actual unique
tokens and split provenance before publishing a claim about distinct tokens.

## Matched baseline adapters

| Optimizer ID | Geometry | Direction and return |
| --- | --- | --- |
| `manifold_muonh` | Frobenius | Hyperball equation: normalize the Muon direction in Frobenius norm, trial step `lr * R`, radial normalization |
| `manifold_imuon` | Stiefel | Separate vertical/horizontal polar directions, common ambient EMA, scaled polar return |
| `manifold_sso` | spectral | Calls vendored SSO PI/hard pre-return/bisection/NS kernel |
| `manifold_muonsphere` | spectral | Calls vendored MuonBall PI/hard pre-return/NS kernel, without bisection |

These adapters use the same logical matrices, initialized radii and auxiliary
AdamW routing as MCSD/TP. They are explicitly matched **adaptations**, not the
original papers' complete recipes. Spectral baselines preserve upstream
pre-return then update order; no additional post-step projection is hidden.
Their fixed step scale is the common `c_l`, not the legacy native shape scaler.
SSO/MuonSphere use native BF16 NS8/PI10 and SSO tolerance `2e-4`, maximum 20
bisections. MuonH uses the common Polar Express implementation. Default beta
is 0.9 with Nesterov for MuonH/SSO/MuonSphere; MCSD/TP and iMuon use beta 0.95
without Nesterov. These are configurable and must be included in fair tuning.

iMuon uses `K=skew(Q.T M)`, `H=(I-Q Q.T)M`, and direction
`-Q partial_polar(K)-partial_polar(H)` for `Q=W/R`, with row-Stiefel handled by
transposing wide matrices. Numerically negligible blocks are zeroed at
`8 * eps(FP32) * ||M||F`; square matrices have zero horizontal block. This
FP32 guard and finite-step polar are disclosed practical adaptations.

## H20 environment and usage

Reuse the original environment: Python 3.12.1 module, CUDA 12.4 module, and
`$HOME/envs/sso_h20`. Do not recreate or upgrade the environment of active old
jobs. The new checkout has the model source snapshot that was missing from Git;
no new upstream checkout is required on each invocation. See `RUNTIME_SOURCE.md`.

Paths may be set via `--data-root`, `--train-prefix`, `--valid-prefix`,
`--tokenizer`, `--env-dir` (or the documented environment defaults in launch.py).
The model and optimizer module paths are verified on the compute node to catch
accidental imports from another checkout. Submission records all relevant
tracked source hashes, and execution fails if those sources changed in queue.

```bash
# Start with a real-model short check on one H20.
python scripts/manifold/launch.py --geometry frobenius --method manifold_mcsd_tp --steps 4 --time-limit 00:45:00 --submit

# Full 3B run, after validation and parameter selection.
python scripts/manifold/launch.py --geometry stiefel --method manifold_mcsd_tp --stage main --lr 1e-2 --seed 2027 --submit

# Same model/data protocol for the spectral baseline.
python scripts/manifold/launch.py --geometry spectral --method manifold_muonsphere --stage main --lr 1e-2 --seed 2027 --submit

# Inspect a smaller/larger model without consuming GPUs.
python scripts/manifold/launch.py --geometry frobenius --method manifold_muonh --stage main --width 256
```

Main runs save every 2000 steps by default; smoke does not save unless requested.
Use `--save-interval` to change this. For a space-bounded 11445-step main run,
add `--save-interval 2000 --save-retain-interval 12000`: Megatron retains the
latest successful checkpoint and removes its predecessor after the next save
succeeds. Since milestone 12000 lies beyond this horizon, only the latest
checkpoint remains. Reserve space for the old and new checkpoint during saves.
This option only affects checkpoints in that run's own directory; it does not
clean historical results. Without this option, all checkpoints are retained.
Checkpoints use Megatron's replicated
`torch` format because the optimizer holds component-shaped nested states;
generic `torch_dist` parameter-shaped optimizer sharding is not used. TP=PP=1,
no distributed optimizer. DP may use 1/2/4/8 GPUs without changing global batch.

For a controlled resume check, retain `--steps 4 --save-interval 2`, run with
`--exit-interval 2`, then resubmit the same model/method/horizon with
`--resume /absolute/path/to/checkpoints` and no early exit. Compare to a fresh
uninterrupted four-step run. Do not resume with `--no-load-optim` or change
geometry, method, radii, data or schedule. Runtime records distinguish early
exit from full horizon using the training logs; a zero exit code alone is not
proof that all target tokens were processed.

If the checkpoint producer is still queued, a resume may be submitted with
`--dependency afterok:<producer-job-id>`; the launcher checks that this is the
exact job owning the source run. It refuses to silently start afresh when the
checkpoint is missing on the compute node. For a single-GPU smoke add
`--verify-against-run /path/to/continuous-run` and depend on both producer and
control jobs (`afterok:<producer>:<control>`). This writes
`resume_verification.json` and fails the job if the model, optimizer, schedule
or RNG states differ. It is not a speed or final-quality comparison.

Each submission writes a new `results/manifold/<run-id>/` with configuration,
source hashes, command, module provenance, parameter layout, runtime and exit
status. Slurm stdout/stderr are in `logs/`. Keep logs and configs even if
checkpoint storage must later be reduced. No training data or checkpoints
belong in Git.

The launcher submits one job but does not enforce an account-wide GPU cap.
Check existing use before submission and use `--dependency afterok:<job-id>`
to serialize runs. Keep total concurrent use within the agreed 4--6 H20 GPUs
(absolute maximum eight), leaving public capacity available. Checkpoint pruning
is opt-in; budget disk space before starting parallel 3B runs.

## Reproducibility boundaries

The model has 126,641,024 trainable parameters as instantiated on H20. It is
a 28-layer, width-384, FFN-1152 decoder with 6 query heads,
3 KV groups, head dimension 64, sequence length 2048, SwiGLU, QK norm,
RMSNorm and RoPE. Each transformer block is split into logical Q/K/V/O and
gate/up/down matrices. The optimizer stores a fixed radius and shape scale
per logical component, with FP32 master weights and momentum. The forward
model remains BF16. Auxiliary parameters are routed to AdamW with their own
learning rate; constrained matrices receive no weight decay.

Baseline adapters and full-budget launch support are implemented. All ten
method/constraint pairs passed real-model single-H20 short integration checks.
Single-GPU model/optimizer/scheduler/RNG recovery passed exact comparison.
Formal comparisons still require longer stability and precision pilots,
multiple seeds, validation-only selection, held-out evaluation and measured
timing. Short smoke losses are not paper results. Do not compare a practical
variant with an exact reference as if only the algorithmic direction changed.

`MAIN1_ALIGNMENT.md` compares this implementation with the current paper and
lists the evidence still needed before using the new study as a publication
result. The paper's current experiment section contains Stiefel PCA/Brockett
tests, not this LLM study.

Primary implementation references:
[Polar Express msign](https://github.com/thinking-machines-lab/manifolds/blob/main/src/msign.py),
[hyperspherical descent](https://github.com/thinking-machines-lab/manifolds/blob/main/src/hyperspherical_descent.py).
The latter normalizes its tangent direction; MCSD-TP intentionally does not,
because that would change the algorithm specified in `IMPLEMENTATION.md`.
