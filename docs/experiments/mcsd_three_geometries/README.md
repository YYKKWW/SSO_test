# Three-geometry MCSD study

**Consolidated records, October 29:** [complete experiment inventory](../ALL_EXPERIMENTS.md),
[80 completed training runs](../experiment_inventory/COMPLETED_RUNS.md), and
[current Frobenius 3B/6B analysis](../FROBENIUS_3B_6B_SUMMARY.md).
All dated entries below retain their original observation times, including
old pending states. They are historical records, not current queue status.

## Historical follow-up: 2026-10-24

The [six-run missing-experiment batch](FROBENIUS_MISSING_20261024.md) is submitted
as 4320101--4320106: two beta=0.95 EMA baseline controls, two MCSD Nesterov LR
tests, and a paired second-seed MCSD/NorMuon 6B replication. All use width 256,
shape radius r=1, frozen code, no checkpoints and at most two new concurrent
H20 GPUs. At 01:22 HKT, 4320101 had started; the other jobs were pending.
The existing three-GPU candidate array was left unchanged.

Completed evidence is in the [3B momentum/seed report](reports/followup_20261020/README.md)
and [6B final report](WIDTH256_6B_RESULTS_20261022.md). The dated entries below
are historical snapshots, not the current queue or dataset-readiness status.

## Data extension: 2026-10-13

[6B data preparation](DATA_6B_20261013.md) was submitted as CPU job 4262916
and started on GPI-2-1 at 14:46 HKT (8 cores, 32 GB, no GPU). It preserves
the current 3B and adds approximately 3B new, exact-token-deduplicated samples,
excluding both existing validation sets. The first status check showed
exclusion-index construction in progress, not completed data. The 1B training
files and redundant partitions were removed after checking all queued jobs;
tokenizer and validation data were retained. Existing training jobs still use
3B data. See the linked record for readiness, storage, logs and resume details.

## Latest submission: 2026-10-12

The [12-run Frobenius follow-up](FROBENIUS_FOLLOWUP_20261012.md) is submitted
as jobs 4255793--4255804: eight matched-EMA LR runs and four second-seed
replications, all width=256, 3B tokens, r=1. The new batch uses two serial
single-H20 lanes and no checkpoints. At 02:05 HKT all were PENDING because
the H20 nodes were reserved for maintenance or lane dependencies were unmet;
none had started training. Old jobs and datasets were unchanged. Available
home storage was about 51.1 GiB; no 6B dataset download was started.

## Latest results: 2026-10-08

The [width-256 3B update](WIDTH256_3B_RESULTS_20261008.md) combines 26 completed
runs (the original 16 plus 10 follow-ups), with four running and four waiting
at the 15:05 HKT snapshot. Its four-page briefing is
`output/pdf/width256_3b_results_20261008.pdf`; standalone validation PNG/SVG
figures are under `output/figures/width256_3b_20261008/`.

The best completed native-momentum settings remain MCSD/TP at LR=0.01 and
MuonH/NorMuon-Sphere at LR=0.015. LR=0.012 did not improve MCSD/TP, and
LR=0.02 did not improve the baselines. Unit-F normalization at 0.01/0.012
also did not improve MCSD. At matched beta=0.9 Nesterov and LR=0.01,
MCSD/TP do not beat the baselines; the beta=0.9 EMA MCSD/TP runs are not
complete. The report separates native-momentum tuning, matched-momentum
comparisons and partial results. No statistical superiority is claimed.

## Study scope

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
the spectral group is deferred for return-accuracy investigation. As of
2026-10-01, all six runs have completed 3B: Frobenius MCSD/TP about 3.29915
versus MuonH 3.30946; Stiefel MCSD/TP about 3.43048 versus iMuon 3.428445.
The Stiefel endpoint reverses its earlier partial ranking. These are single-seed,
untuned results, not statistical superiority claims.

The separate [width-256 fixed-radius LR search](WIDTH256_3B_RADIUS_LR_20260926.md)
uses the confirmed four-point grid `{5e-3, 7e-3, 1e-2, 1.5e-2}` and a new
opt-in shape-radius configuration. Four H20 smokes passed and all 16 main runs
were submitted on 2026-10-01 as 4197463--4197478, with a four-GPU bundle cap.
Historical default initialization and experiment sources are unchanged.
As verified on 2026-10-05, all 16 runs completed. Best grid endpoints are
MCSD 3.467720 / MCSD-TP 3.467858 at LR 1e-2, versus MuonH 3.466668 /
NorMuon-Sphere 3.464757 at LR 1.5e-2. The small-LR advantage for MCSD/TP
does not persist after each method selects its best grid LR. Baseline minima
are at the upper boundary; this remains single-seed evidence, not a global
LR optimum or a significance claim. No further runs were submitted in this check.

The subsequent [unit-Frobenius direction ablation and LR extension](WIDTH256_3B_RADIUS_LR_20260926.md#unit-direction-ablation-and-grid-extension-2026-10-05)
adds an opt-in MCSD step-length normalization. After 187 local tests and two
successful H20 smokes, 12 matched-budget follow-ups were submitted as
4226861--4226872, separately from the completed
results above. It preserves the old update when disabled and records actual
relative trial/return lengths. Startup diagnostics show about 0.1--2.3%
step-magnitude correction, not evidence that normalization explains all LR
differences or improves final loss.
At the user's subsequent request on 2026-10-05, the existing queue was changed
from one to two concurrent single-H20 lanes. Jobs 4226861 and 4226862 are
running; training configurations and the original running job are unchanged.

The [figure/table bundle](reports/width256_3b_20261005/) now contains all
training/validation curves, an LR sweep, common-LR and best-LR comparisons,
vector figures, LaTeX tables and raw parsed data. The
[Chinese results and scaling-plan PDF](reports/width256_3b_20261005/width256_3b_results_scaling_plan.pdf)
explains measurement definitions and the experiments needed before a scaling claim.

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
PYTHONPATH=Megatron-LM python -m pytest tests/test_manifold_mcsd.py tests/test_manifold_baselines.py tests/test_manifold_normuon.py tests/test_manifold_launch.py tests/test_manifold_radius.py -q
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
| `manifold_normuon_sphere` | Frobenius | NorMuon row-wise FP32 second moment after Polar Express, unit-Frobenius direction, trial step `lr * R`, radial normalization |
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

NorMuon-Sphere is a matched adaptation measured in the new width-256 3B LR
grid, not in the old width-384 batch. It defaults to MuonH's beta 0.9/Nesterov and shared Polar Express
10, with row EMA beta2 0.95 and epsilon 1e-10. See
[NorMuon-Sphere equations, provenance and checks](NORMUON_SPHERE.md).

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
# Matched momentum follow-up (2026-10-05)

The width-256 Frobenius grid has a momentum confound: MCSD/TP uses beta=0.95
EMA, whereas MuonH/NorMuon-Sphere uses beta=0.9 Nesterov. The six-run
[matched-momentum ablation](FROBENIUS_MOMENTUM_20261005.md) reuses four
LR=0.01 controls and holds radius, data, LR schedule and direction scaling
fixed. Its MCSD Nesterov variant is opt-in and leaves the original default
unchanged. See that document for submission and validation status.
