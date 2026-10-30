# Width-256 matched-momentum ablation

## Question and fixed protocol

The completed four-method LR grid used beta=0.95, EMA-only for MCSD/MCSD-TP
and beta=0.9, Nesterov for MuonH/NorMuon-Sphere. Its results cannot isolate
geometry/direction construction from momentum. This follow-up changes only
the constrained-matrix momentum; auxiliary AdamW remains unchanged.

All main runs use Frobenius sphere, r=1, width=256, 28 layers, the same 3B
training data and validation protocol, 11,445 updates, seed=1234, LR=0.01,
the original cosine decay, BF16 Polar Express and FP32 master matrices.
MCSD's optional unit-F direction normalization remains OFF. Matrix weight
decay remains zero. Each run uses one H20 and writes no checkpoints.

## Six new main runs, four reused controls

| Method | beta=0.95, EMA | beta=0.9, EMA | beta=0.9, Nesterov |
| --- | --- | --- | --- |
| MCSD | Existing 3B: 3.467720 | NEW | NEW |
| MCSD-TP | Existing 3B: 3.467858 | NEW | NEW |
| MuonH | Not scheduled | NEW | Existing 3B: 3.476769 |
| NorMuon-Sphere | Not scheduled | NEW | Existing 3B: 3.476793 |

Every value above is at LR=0.01, NOT each method's individually best LR.
The source table is `reports/width256_3b_20261005/curves_and_results.json`.
This is a minimal matched ablation, not a complete beta x Nesterov factorial.
Beta=0.95 with Nesterov is deferred to avoid six or more extra long runs.

- MCSD/TP beta=0.95 EMA vs beta=0.9 EMA isolates the coefficient.
- For all four methods, beta=0.9 EMA vs beta=0.9 Nesterov isolates Nesterov.
- At either beta=0.9 momentum mode, compare all four methods under matched
  momentum, data, radius and nominal LR. Actual update norms can still differ.
- These are single-seed fixed-LR mechanism tests, not evidence of globally
  optimal tuning or automatic extension of the original MCSD theorem.

## Update definition and compatibility

For the current matrix gradient G, first update the ambient state:

    M_t = beta * M_(t-1) + (1 - beta) * G_t
    S_t = M_t                              # EMA-only
    S_t = beta * M_t + (1 - beta) * G_t     # Nesterov variant

MCSD/TP applies its existing tangent projection to S_t before the polar LMO.
MCSD-TP retains the optional second tangent projection after the LMO. The
existing scale and radial return are unchanged. The extra Nesterov source
does not overwrite the EMA buffer or the gradient. The default is still
beta=0.95, EMA-only. Existing baseline updates are unchanged; their FP32 EMA
uses `lerp_`, while historical MCSD uses multiply/add, so algebraic matching
does not imply bitwise identical intermediate rounding across methods.

The launcher sends a separate `--manifold-mcsd-nesterov` flag for MCSD/TP;
the original baseline flag remains baseline-only. Resume rejects changed
beta/Nesterov settings. Legacy MCSD checkpoints without the flag imply EMA.

## Scheduling and reproducibility

Use a separate frozen H20 source directory; do not update running jobs.
`scripts/manifold/submit_frobenius_momentum.py` creates two dependency lanes,
each after the tail of one existing single-GPU lane. Two four-update MCSD/TP
Nesterov smokes precede six main runs. Every main job also requires its lane's
smoke to succeed. A failed smoke blocks that lane's long runs. Main failures
do not suppress other configurations in the same lane.

The helper defaults to dry-run and requires two explicit lane-tail job IDs.
It keeps an atomic, locked submission register for resumability. Logs and
resolved configs retain momentum settings, source hashes and dependencies.
Successful submission is not successful training or a numerical result.

Compare full train/validation curves at common steps, final loss, wall time,
NaN/skipped steps and existing per-component update/radius diagnostics.
At the previous approximately 18.5 h/run, six main runs add about 111 GPU-h,
or 55.5 h on the two lanes after the current queue; this is only an estimate.

## Validation and submission status

Submitted on 2026-10-05 in the isolated H20 worktree
`/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005`,
snapshot `fc15600`. No GitHub push was performed. The existing unit-F/LR
snapshot and its running jobs were not modified.

- Local complete suite: **259 tests passed**.
- H20 training environment: PyTorch 2.6.0+cu124; independent CPU checks
  passed **48 formula/resume cases** without installing pytest or changing
  the shared environment.
- With Nesterov disabled, **6 cases x 10 updates** were bitwise identical
  to the frozen prior optimizer in the unit-F snapshot.
- CUDA end-to-end smokes are **queued, not yet completed**. The main jobs
  wait for them with `afterok`; CPU checks are not GPU training results.

| Job | Phase | Method | Momentum | Prerequisite |
| --- | --- | --- | --- | --- |
| 4227542 | 4-step smoke | MCSD | 0.9 + Nesterov | Existing tail 4226871 |
| 4227543 | 4-step smoke | MCSD-TP | 0.9 + Nesterov | Existing tail 4226872 |
| 4227544 | 3B main | MCSD | 0.9 + Nesterov | Smoke 4227542 succeeds |
| 4227545 | 3B main | MCSD-TP | 0.9 + Nesterov | Smoke 4227543 succeeds |
| 4227546 | 3B main | MuonH | 0.9 EMA | 4227544 ends; lane smoke succeeds |
| 4227547 | 3B main | NorMuon-Sphere | 0.9 EMA | 4227545 ends; lane smoke succeeds |
| 4227548 | 3B main | MCSD | 0.9 EMA | 4227546 ends; lane smoke succeeds |
| 4227549 | 3B main | MCSD-TP | 0.9 EMA | 4227547 ends; lane smoke succeeds |

The final queue check shows all eight jobs PENDING on these dependencies.
Only the existing jobs 4226861/4226862 are RUNNING, so the combined queue
still occupies two H20s, not four. Approximately 52 GB was free before the
small source snapshot was created. All six new main runs disable checkpoints.
Local submission and numerical-validation records are under
`results/manifold/status_20261005_momentum/`.

## Scheduling update: 2026-10-06

At the user's request, two unallocated H20 GPUs were used to start the
momentum ablation earlier. At 08:19 HKT, only the external dependencies of
smokes 4227542 and 4227543 were removed. The original two LR/unit-F lanes
were unchanged. The two momentum lanes now run independently, giving a
maximum of **four simultaneous single-H20 jobs** across both bundles.
All main-job smoke-success dependencies and within-lane dependencies remain
unchanged; no runs were resubmitted or training configurations changed.

- Smokes 4227542/4227543 both COMPLETED successfully (exit 0:0, 78 seconds
  each on SPG-7-2), including all four updates and final validation.
- Main jobs 4227544 (MCSD) and 4227545 (MCSD-TP) were RUNNING on SPG-7-1
  when checked at approximately 08:21 HKT. Both use beta=0.9 with Nesterov,
  LR=0.01, width=256 and the matched 3B protocol; unit-F normalization is OFF.
- The original submission register retains the historical dependencies.
  The scheduler override is recorded separately in
  `results/manifold/status_20261005_momentum/concurrency_change_20261006_momentum.json`
  and in the remote snapshot's `results/manifold/` directory.

This supersedes the earlier two-GPU scheduling snapshot above. The short
tests establish launch viability, not the numerical performance of the 3B runs.
