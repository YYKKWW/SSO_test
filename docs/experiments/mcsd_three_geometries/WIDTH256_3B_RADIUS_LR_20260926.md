# Width-256, fixed-radius 3B learning-rate search

Status (2026-10-05, 12:03 HKT): **all 16 runs completed**, IDs
4197463--4197478, all with Slurm exit code 0:0 and final validation at step
11445. The last job finished on October 4 at 17:16 HKT. The account has no
running or pending jobs at this snapshot; home storage has about 52 GiB free.
Local implementation passes 144 focused CPU tests. All four real-model H20
smokes passed; the remote environment lacks pytest, so the 144-test suite is
not reported as rerun there. Existing experiment sources are unchanged.

## Final LR-grid results, 2026-10-05

All entries are final validation loss after 3,000,238,080 processed tokens,
not trajectory minima. Model: width 256, 73,418,496 parameters; Frobenius
shape-radius protocol with fixed `r=1`, seed 1234, one H20 per run. This is
not the old width-384 initialization-derived-radius experiment.

| Method | LR 5e-3 | LR 7e-3 | LR 1e-2 | LR 1.5e-2 |
| --- | ---: | ---: | ---: | ---: |
| MCSD | 3.485531 | 3.470591 | **3.467720** | 3.489496 |
| MCSD-TP | 3.485612 | 3.470890 | **3.467858** | 3.487949 |
| MuonH | 3.529123 | 3.500046 | 3.476769 | **3.466668** |
| NorMuon-Sphere | 3.526564 | 3.499072 | 3.476793 | **3.464757** |

Bold denotes each method's best LR within this four-point grid.

| Method | Best grid LR | Final loss | Job | Slurm elapsed | Recent training seconds/step |
| --- | ---: | ---: | ---: | --- | ---: |
| MCSD | 1e-2 | 3.467720 | 4197465 | 18:36:00 | 5.397 |
| MCSD-TP | 1e-2 | 3.467858 | 4197469 | 18:32:38 | 5.379 |
| MuonH | 1.5e-2 | 3.466668 | 4197474 | 18:30:16 | 5.366 |
| NorMuon-Sphere | 1.5e-2 | 3.464757 | 4197478 | 18:33:07 | 5.381 |

### Interpretation

- MCSD/TP lead at each of 5e-3, 7e-3 and 1e-2, but this does not establish
  superiority after equal-budget LR selection. At their respective best grid
  LRs, NorMuon-Sphere leads MCSD by 0.002963 and MCSD-TP by 0.003101;
  MuonH leads MCSD by 0.001052. These are observations from one seed, not
  significance claims.
- MCSD and MCSD-TP are nearly tied at their best LR (difference 0.000138).
  At 1.5e-2 both worsen, with TP slightly less affected; there is no demonstrated
  general TP benefit from this batch.
- The baseline minima lie at the upper grid boundary and their losses decrease
  throughout the tested grid. Their true best LR is not yet located. A shared
  additional LR of 2e-2 would test that boundary; no extra jobs were submitted
  during this results check.
- Before a paper claim, repeat selected configurations with paired seeds and
  retained checkpoints, and separate the momentum/Nesterov effects described
  below. Keep radius fixed while making that comparison.

### Full runtime and health audit

Times below are Slurm elapsed (hours:minutes:seconds), including initialization,
evaluation and shutdown, but excluding queue waiting. Recent seconds/step above
average the last 100 logged training intervals and exclude evaluation overhead.

| Method | LR 5e-3 | LR 7e-3 | LR 1e-2 | LR 1.5e-2 |
| --- | --- | --- | --- | --- |
| MCSD | 18:35:22 | 18:29:48 | 18:36:00 | 18:30:10 |
| MCSD-TP | 18:33:00 | 18:32:30 | 18:32:38 | 18:34:16 |
| MuonH | 18:24:36 | 18:24:40 | 18:30:32 | 18:30:16 |
| NorMuon-Sphere | 18:25:12 | 18:26:41 | 18:32:22 | 18:33:07 |

All 16 logs contain the final step-11445 validation and report zero skipped
and NaN iterations. The maximum persisted sparse Frobenius constraint defect
across the logs is 1.515e-7, not a per-update feasibility certificate. Differences
in the best-run total times are below 0.6%; the MCSD pair ran on SPG-7-2 and
the two baselines on SPG-7-1, so tiny timing differences are not isolated
optimizer overhead. Full local log copies: `results/manifold/status_20261005/`.

## Figures and tables, 2026-10-05

The reproducible figure bundle is in
[`reports/width256_3b_20261005/`](reports/width256_3b_20261005/).
The Chinese [PDF report](reports/width256_3b_20261005/width256_3b_results_scaling_plan.pdf)
includes the complete LR table, online-training tail table, selected-run
runtime table, five figures and a staged scaling-study proposal. No further
training jobs were submitted while making this report.

- [Method-wise best-LR train/validation curves](reports/width256_3b_20261005/figures/best_lr_learning_curves.png), with full trajectories and late-stage detail.
- [Train/validation LR sweep](reports/width256_3b_20261005/figures/learning_rate_sweep.png), with stars at each within-grid minimum.
- [All 16 training curves](reports/width256_3b_20261005/figures/all_learning_rates_train.png).
- [All 16 validation curves](reports/width256_3b_20261005/figures/all_learning_rates_validation.png).
- [Common LR 1e-2 comparison](reports/width256_3b_20261005/figures/common_lr_learning_curves.png), kept separate from method-wise best-LR selection.
- [LaTeX tables](reports/width256_3b_20261005/paper_tables.tex) and
  [raw curves, metrics and log hashes](reports/width256_3b_20261005/curves_and_results.json).

Each figure also has a same-name vector SVG. The parser checks all 18,320
training records and 928 validation records, grid completeness, final-step
presence, zero logged skipped/NaN counts, and agreement with the table above.
Training curves show a causal 100-update weighted mean of stdout interval
means. The terminal training metric covers updates 11341--11440, not the
unlogged final five updates and not a full training-set evaluation. Validation
curves are unsmoothed; markers/lines represent real evaluations, not repeated
seeds or confidence bands. Validation sample iteration advances over training.

Rebuild with Python plus numpy, matplotlib, reportlab (and a CJK-capable TTF):

```bash
python scripts/manifold/plot_width256_3b.py --font /path/to/cjk-font.ttf
```

The script's default font is the local macOS Arial Unicode font. It only
parses local snapshots and creates reports; it cannot submit jobs or alter
optimizer/model code. Raw stdout stays in ignored `results/`; the parsed
JSON preserves source paths and SHA256 hashes for audit.

### Scope of future scaling claims

Prioritize: (1) four matched LR 2e-2 runs to check the boundary; (2) two extra
paired seeds at selected settings; (3) width 384/512 at the LR selected on
width 256, then an equal-budget LR search to measure transfer regret.
The old width-384 initialization-radius experiment is not reusable in the
new shape-radius scaling table. Radius, momentum, head dimension, layer count,
data, training horizon and auxiliary optimizer conventions must be explicit.

For a separate initial model/data-budget study, consider widths 256/384/512
and 0.75B/1.5B/3B tokens. Train with a schedule matched to each budget; a point
partway through a 3B schedule is not an independently optimized short-horizon
endpoint. Three nearby widths are insufficient for a convincing general
scaling exponent or large-model extrapolation. Distinguish empirical
[hyperparameter transfer](https://arxiv.org/abs/2203.03466) from
[compute-optimal scaling](https://arxiv.org/abs/2203.15556), and treat data
repetition as its [own experimental axis](https://arxiv.org/abs/2305.16264).

## Historical snapshot, 2026-10-02

These results use width 256, 3B tokens, shape-based Frobenius radius `r=1`,
seed 1234, and the new protocol. Do not merge them into the old width-384
initialization-derived-radius table.

### Completed LR 5e-3

| Method | Job | Final validation at step 11445 | Slurm elapsed | Recent training seconds/step |
| --- | ---: | ---: | --- | ---: |
| MCSD | 4197463 | 3.485531 | 18:35:22 | 5.394 |
| MCSD-TP | 4197467 | 3.485612 | 18:33:00 | 5.382 |
| MuonH | 4197471 | 3.529123 | 18:24:36 | 5.337 |
| NorMuon-Sphere | 4197475 | 3.526564 | 18:25:12 | 5.341 |

All four completed with exit code 0. MCSD's endpoint is lower than MuonH by
0.043592 and NorMuon-Sphere by 0.041033. MCSD versus MCSD-TP differs by only
0.000081, so there is no evidence of a meaningful TP advantage. MCSD takes
about 0.97% longer than MuonH in total elapsed time; MCSD-TP about 0.76%.
NorMuon-Sphere improves on MuonH by 0.002559 at this LR. These are single-seed,
single-LR observations, not comparisons of each method's tuned optimum.

| Common validation step | MCSD | MCSD-TP | MuonH | NorMuon-Sphere |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 3.872688 | 3.873368 | 3.911266 | 3.908163 |
| 4000 | 3.705842 | 3.706580 | 3.726355 | 3.723981 |
| 6000 | 3.620216 | 3.620251 | 3.642067 | 3.639439 |
| 8000 | 3.556336 | 3.556530 | 3.588171 | 3.585789 |
| 10000 | 3.500014 | 3.500689 | 3.540756 | 3.538080 |
| 11445 | 3.485531 | 3.485612 | 3.529123 | 3.526564 |

### Running LR 7e-3

| Method | Job | Latest logged training step | Validation at common step 1000 |
| --- | ---: | ---: | ---: |
| MCSD | 4197464 | 2800 | 4.113669 |
| MCSD-TP | 4197468 | 2810 | 4.111646 |
| MuonH | 4197472 | 1200 | 4.154740 |
| NorMuon-Sphere | 4197476 | 1190 | 4.151285 |

Each run targets 11445 steps. The differing progress reflects start times,
not a 2x throughput advantage. Incomplete training is not a final-LR ranking.
At the current measured whole-run pace, the two MCSD runs have roughly
14 hours left and the baseline runs roughly 16--17 hours; later queued runs
depend on resource availability. LR 1e-2 and 1.5e-2 have not started.

All eight started runs report zero NaN/skipped iterations at the snapshot.
Maximum persisted sparse Frobenius constraint defect is about 1.52e-7;
this is an audit sample, not a guarantee for every update. Training speed
uses the latest 100 logged intervals, excluding evaluation overhead, whereas
Slurm elapsed includes it. Local log snapshots: `results/manifold/status_20261002/`.
The batch occupies four H20s, respecting its cap; disk has about 52 GiB free.

Continue the existing grid without adding runs or selecting a winner from
the partial results. Report each method's best full-budget LR, then repeat
the selected configurations with paired seeds. Momentum differences remain
disclosed below and prevent interpreting this as a direction-only ablation.

## Search scope

The user confirmed **four** peak learning rates:

`{5e-3, 7e-3, 1e-2, 1.5e-2}`.

Use the reference paper only to guide the LR range, not to replace this
project's model architecture. Its Table 3 uses
`{3e-3, 5e-3, 7e-3, 1e-2, 3e-2}`. Radius is also tuned there; the selected
MACRO-fro and MACRO-spec values are respectively `r=0.8` and `r=1.2`.
Neither is a universal default for our methods. See
[Appendix D.1, Tables 6-7](https://arxiv.org/html/2605.04418v1#A4.SS1).

The submitted first batch is Frobenius MCSD, MCSD-TP, MuonH and
NorMuon-Sphere at all four LRs: **16 runs**, one H20 per run. Fix `r=1` as a controlled
starting point rather than jointly tuning radius and LR. If a method's best
LR is an endpoint, report it as best *within this grid*, then consider extending
that side before claiming an optimum. Do not prune solely on 100M results:
the previous Frobenius batch reversed its pilot ranking later in training.

## Radius and step convention

New opt-in protocol: `dense-three-geometries-v4-shape-radius`, configured in
`configs/manifold/width256_3b_shape_radius.json`. The old default remains
`init_energy`; none of its logged results are relabeled as new-protocol runs.

For each logical weight of shape `(m, n) = (D_out, D_in)`:

| Constraint | Fixed radius | Fixed MCSD trial multiplier `c` |
| --- | --- | --- |
| Frobenius sphere | `R=r*sqrt(m)` | `R/sqrt(min(m,n))` |
| Spectral sphere | `R=r*sqrt(m/n)` | `R` |
| Scaled Stiefel | `a=r*sqrt(m/n)` | `a` |

The Stiefel choice follows the user's spectral-matching decision; it does
not assert equal Frobenius energy across geometries. Only Frobenius is in
this first LR batch. Oblique constraints are not added.

Fresh FP32 initialization is mapped to each target constraint before BF16
model conversion. Radii are then frozen. Q/K/V and gate/up are split before
computing dimensions; a fused tensor's total row count is never used for
all its component radii. For width 256, the logical shapes are Q/O `(256,256)`,
K/V `(128,256)`, gate/up `(768,256)` and down `(256,768)`.

MCSD uses `Y=W-lr*c*O`; MCSD-TP uses its tangent-projected direction with the
same fixed multiplier. Neither gains an extra per-step direction normalization.
MuonH and NorMuon-Sphere instead use a unit-Frobenius direction and
`Y=W-lr*R*D`. With an ideal full-rank polar LMO, the MCSD Frobenius trial length
before the optional tangent projection is `lr*R`; finite BF16 Polar Express
and rank deficiency need not preserve that equality. This convention must
be stated with the LR, not treated as interchangeable with any native recipe.

## Matched settings

| Setting | Value |
| --- | --- |
| Model | 28 layers, width 256, FFN 768, 4 query heads, 2 KV groups, head dimension 64 |
| Architecture | Existing GQA/SwiGLU/RMSNorm/QK-norm model, unchanged |
| Data | Existing 3B OLMo-Mix indexed train/validation split, not recycled 1B data |
| Horizon | 11,445 updates; 3,000,238,080 processed tokens |
| Batch | 128 sequences, sequence length 2048, micro-batch 4, one GPU |
| Schedule | 229 warmup updates, cosine decay to 10% of peak LR |
| Auxiliary optimizer | Existing AdamW route; LR `1e-3`, unchanged across the matrix-LR grid |
| Seed | 1234 for all runs |
| Evaluation | Every 200 updates, 40 batches, plus final evaluation |
| Numerics | Existing BF16 model/Polar Express with FP32 master weights and Frobenius return |
| Checkpoints | Disabled for initial search to avoid filling the disk; logs/configs/evaluations retained |

Default momentum remains method-specific: MCSD/TP EMA `0.95` without
Nesterov; MuonH/NorMuon-Sphere `0.9` with Nesterov. NorMuon-Sphere additionally
uses row EMA `0.95`, epsilon `1e-10`. Thus this is a matched-model/data/radius
comparison, not a direction-only ablation. Equal LR search budgets are used.
Checkpoint-free runs cannot resume or be re-evaluated from saved weights;
selected configurations should be repeated with paired seeds and retained
checkpoints before publication.

## Prepare without submitting

From the project root, this resolves all 16 configurations without using GPUs:

```bash
for method in manifold_mcsd manifold_mcsd_tp manifold_muonh manifold_normuon_sphere; do
  for lr in 5e-3 7e-3 1e-2 1.5e-2; do
    python scripts/manifold/launch.py \
      --config configs/manifold/width256_3b_shape_radius.json \
      --geometry frobenius --method "$method" --stage main --lr "$lr" --gpus 1
  done
done
```

Do not simply add `--submit` to this loop: the single-job launcher does not
cap account-wide GPU occupancy. Freeze a separate validated source snapshot
and submit bounded dependency chains after an H20 smoke check, preserving
the active experiment directory. Proposed budget: four concurrent cards
account-wide, including older jobs, not four additional cards.

At 2026-09-26 10:59 HKT, Slurm reported 8/8 GPUs allocated on each of
SPG-7-1 and SPG-7-2: **zero unallocated H20s**. The user's three running jobs
were 4106583/4106585/4106587 (Stiefel MCSD/MCSD-TP/iMuon); other jobs held
the remaining 13 cards. This is a time-stamped allocation snapshot, not a
reservation or a promise of immediate availability.

## H20 submission, 2026-10-01

Frozen source: `922820e9782d04f374b658be09d7332c55271c45` in the independent
worktree `/home/u3013198/projects/SSO_test-three-geometries-w256-lr-20261001`.
No GitHub push and no edits to the historical training source were performed.

| Method | LR 5e-3 | LR 7e-3 | LR 1e-2 | LR 1.5e-2 |
| --- | ---: | ---: | ---: | ---: |
| MCSD | 4197463 | 4197464 | 4197465 | 4197466 |
| MCSD-TP | 4197467 | 4197468 | 4197469 | 4197470 |
| MuonH | 4197471 | 4197472 | 4197473 | 4197474 |
| NorMuon-Sphere | 4197475 | 4197476 | 4197477 | 4197478 |

Each method is one sequential chain in increasing LR order, giving at most
four simultaneous one-GPU main jobs. Main jobs require successful completion
of all four smoke jobs. Within each method, `afterany` preserves the cap without
suppressing another LR if a preceding main run fails. This cap applies to this
bundle; no other user jobs were active at submission. Actual initial occupancy
was two GPUs, with the other two method chains awaiting cluster resources.

The four smoke jobs (4197448--4197451) ran four steps at LR 1.5e-2 and all
completed with exit code 0 and no NaN/skipped iterations. Their 196-component
manifests match exactly, SHA256
`bdab6b93a50769a8cda655b4a3d6aa34c5985efdc055bcb015a17cb9c5da6270`.
The audit independently checks every radius equals `sqrt(logical D_out)`.
This verifies layout/radii/scales, not a hash of initial weight tensors.
The smoke horizon is not used to claim final optimizer performance.

The register and audit are stored under this worktree's `results/manifold/`
as `width256_lr_register_20261001.json` and
`width256_smoke_audit_20261001.json`, with local copies in
`results/manifold/status_20261001/`.

## Validation

The 144-test CPU suite includes explicit radii for tall/wide/square matrices,
logical fused layout, unchanged legacy initialization, all four Frobenius
methods' updates and restart equivalence, rejection of mismatched radius
protocols, and all 16 launch configurations. H20 integration smokes passed;
all 16 full-budget runs have now completed, as recorded above. This is still
a single-seed validation-based selection, not independent test-set evidence.

## Unit-direction ablation and grid extension, 2026-10-05

The four-point grid does **not** establish a common optimal LR. MuonH and
NorMuon-Sphere both attain their lowest observed loss at the upper boundary
0.015. MCSD/TP prefer the interior point 0.01. The ratio of these selected
LRs is 1.5, and the best-loss differences are about 0.001--0.003, with only
one seed. Additional measurements, not a change in the interpretation of
the old curves, are needed.

### What the new normalization changes

Let k = min(D_out, D_in), M be the ambient EMA, and D = polar(P_T M),
with an additional P_T on D for MCSD-TP. For this shape-radius protocol:

- Existing MCSD/TP: Y = W - eta (R / sqrt(k)) D.
- Opt-in unit-Frobenius variant: Y = W - eta R D / (||D||_F + 1e-12).
- Both return W_new = R Y / ||Y||_F in FP32, with R = sqrt(D_out).

The new variant is called **MCSD-unitF** in this experiment. In the
implementation D is stored with a negative sign, so the trial is written
with addition. The switch is `--direction-normalization frobenius`, and is
allowed only for Frobenius MCSD/TP. It is disabled by default. Momentum,
LMO precision, tangential projection and the return map do not change.
The option applies per logical matrix after QKV and gate/up splitting,
not to a concatenation of all model parameters.

For an exact full-rank polar factor, ||D||_F = sqrt(k), so the two MCSD
trial steps already agree up to epsilon. Rank deficiency, approximate
BF16 Polar Express, and post-LMO TP can change this norm. Unit normalization
removes that magnitude variation; it does not turn MCSD into MuonH.
MCSD uses tangent-projected momentum and EMA 0.95 without Nesterov, whereas
MuonH/NorMuon use ambient-source directions and momentum 0.9 with Nesterov.
Both baselines already normalize the final direction in Frobenius norm.
Thus similar baseline LR preferences are plausible, but no causal explanation
or guaranteed improvement follows just from their best grid points.

This is an algorithmic step-scaling ablation, not an unchanged implementation
of the original fixed-scale spectral-ball LMO theorem. Also, eta sets the
**trial** relative length; after radial return the displacement need not
equal eta R. The raw norm ratio ||D||_F/sqrt(k), trial displacement/(eta R),
and returned displacement/(eta R) are sampled with the existing audit
schedule and logged initially and every 50 updates. The samples are not
population-wide bounds or confidence intervals. These diagnostics add
Frobenius reductions, not a new SVD or power iteration. Near-zero D maps
to a near-zero update; there is no division by zero.

### Chosen follow-up grid

| Methods | Additional peak LRs | Number of new 3B runs |
| --- | --- | ---: |
| Original MCSD, MCSD-TP, MuonH, NorMuon-Sphere | 0.012, 0.02 each | 8 |
| MCSD-unitF | 0.01, 0.012, 0.015, 0.02 | 4 |

0.012 refines the interval between the MCSD minimum and the deterioration
at 0.015. 0.02 tests the baseline's unresolved upper boundary more widely
than 0.017. Add 0.017 later if the 0.015--0.02 interval contains a useful
turning point. This is exploratory validation-based selection; MCSD-unitF
has not yet received the two smallest LRs from the old grid, so its four-point
search must not be described as a fully matched six-point benchmark.

All other settings match the table above, including seed, data split,
evaluation schedule, auxiliary AdamW and 3B horizon. Historical low-LR
results are reused, not rerun. No checkpoints are stored in this exploratory
batch. A single run was about 18.5 H20-hours previously; the 12-run plan is
about 222 GPU-hours before queueing, with no guarantee of an improvement.

`scripts/manifold/submit_frobenius_unit_grid.py` defaults to one concurrent
single-H20 job, supports at most two, and records every job/configuration.
The main phase requires two successful real-model integration smokes (old
MCSD and MCSD-unitF). Both use four updates; they verify execution, not
quality or the 3B convergence ordering. Successful submission and numerical
audit outcomes are recorded separately below when available.

Local validation includes the explicit update equations, tall/wide/square
matrices, BF16/exact LMO, both TP modes, zero/rank-deficient directions,
strict normalization-aware resume and legacy disabled defaults. A separate
regression against the remote frozen `922820e` optimizer gave bitwise-identical
FP32 weights for 6 cases x 10 updates with normalization disabled.

### Verified H20 deployment and registered jobs

The full local `tests/test_manifold*.py` suite passed **187 tests**. H20 smoke
jobs **4226742** (original) and **4226743** (MCSD-unitF) both completed four
updates without NaN/skipped iterations. Their logical parameter manifests
match the historical width-256 grid exactly. This checks layouts and radii,
not a hash of all initial weight tensors.

Across 27 startup samples per smoke, the original raw direction ratio was
0.977497--0.999088; its trial length had the same ratio to eta R. The new
variant's actual trial ratio was 0.999999920--1.000000044. The largest sampled
post-return relative radius defect was 1.20e-7. These startup samples show
only a roughly 0.1--2.3% magnitude correction; they do not establish the
cause of a 1.5x difference in selected LR or long-run performance. Both
short-run losses are omitted from the 3B results tables.

Training snapshot: `972b623`; final submission snapshot: `9c78efb`. The latter
changes only the batch submission tool: it verifies completed smoke artifacts
instead of referencing completed job IDs that Slurm rejected. No long job
was created by the rejected submission, and the two snapshots have identical
training code. Independent remote directory:
`/home/u3013198/projects/SSO_test-three-geometries-w256-unitfro-20261005`.

| Job | Method | Peak LR |
| --- | --- | ---: |
| 4226861 | MCSD-unitF | 0.01 |
| 4226862 | NorMuon-Sphere | 0.02 |
| 4226863 | MCSD | 0.012 |
| 4226864 | MCSD-unitF | 0.012 |
| 4226865 | MuonH | 0.02 |
| 4226866 | MCSD-TP | 0.012 |
| 4226867 | MCSD-unitF | 0.015 |
| 4226868 | NorMuon-Sphere | 0.012 |
| 4226869 | MuonH | 0.012 |
| 4226870 | MCSD-unitF | 0.02 |
| 4226871 | MCSD | 0.02 |
| 4226872 | MCSD-TP | 0.02 |

The bundle was initially capped at one GPU. On 2026-10-05, the user requested
both available H20s, and the existing jobs were rewired into **two concurrent
single-GPU lanes** without resubmission, restart, or training-config changes.
Jobs 4226861 and 4226862 are confirmed RUNNING on SPG-7-1. Each later job
waits with `afterany` for the job two positions earlier; failed main runs
do not suppress other LRs. At this check, other users held 14 GPUs and this
bundle used the remaining two. Availability can change. At the historical
18.5 h/run timing, six runs per lane take about 111 hours (4.6 days), plus
queueing, rather than roughly nine days for the original serial schedule.

`concurrency_change_20261005.json` records the scheduler-only changes. The
original `resolved_config.json` files retain their submission-time dependency
values; the register records the current ones and the previous values as
`submitted_dependency`. Use `--concurrency 2` with the submission helper for
this existing register. Do not edit the frozen training source to change
scheduling.
The register is `results/manifold/frobenius_unit_lr_register_20261005.json`
in the remote snapshot. Local copies of the register and numerical audit
are in `results/manifold/status_20261005_unitfro/`. A registered or running
job is not a completed numerical result. No GitHub push was performed.
