# H20 follow-up: isolate cluster rank, routing, and matched step length

Date: 2026-10-21. Follow-up to
[the first four radial-cluster pilots](cluster_radial_h20_100m_20261021.md).
These are exploratory 100M-token runs, not a claim of convergence or an
established optimal method.

## Completed results, checked 2026-10-21 11:14 HKT

All **18 arms** (four original plus fourteen follow-ups) completed with
Slurm exit0 and `COMPLETED` markers. Every run has validation at steps
50/100/150/191; no logged NaN or skipped iterations were found. The downloaded
launch configurations match in architecture, token budget, LR schedule,
batch sizes, tokenizer, train/validation prefixes and evaluation settings.
This batch no longer occupies GPUs; the two other user training jobs remain
unmodified. These are width256 / 100M-token results, not 1B results.

Final validation is at **step191**. Branch frequency and repeat statistics
below are the latest logged cumulative values at **step190**, not exact
end-of-run totals. The denominator is 47880 component decisions and includes
the initial zero-LR step. A repeat is a branch invocation immediately
following the same component's cluster invocation. It does not mean an
exactly zero spectral gap. Wall time is Slurm elapsed time, including startup,
unit checks and validation, but excluding queue wait.

| Job | Method | tau | epsilon | c | Seed | Final val | Cluster % at190 | Repeat % | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4290694_0 | MuonSphere | n/a | n/a | n/a | 1234 | **4.947061** | n/a | n/a | 00:34:23 |
| 4290694_1 | SSO | n/a | n/a | n/a | 1234 | 4.956870 | n/a | n/a | 00:38:04 |
| 4290663_0 | MCSD-TP control | n/a | n/a | n/a | 1234 | 4.947731 | 0 | n/a | 00:53:20 |
| 4290663_1 | Fixed top4 | 5e-5 | 5e-5 | 1 | 1234 | 4.954692 | 1.458 | 61.6 | 00:53:21 |
| 4290663_2 | Adaptive top<=4 | 5e-5 | 5e-5 | 1 | 1234 | 4.948502 | 0.854 | 29.6 | 00:52:43 |
| 4290663_3 | Adaptive top<=4 | 5e-5 | 5e-5 | 0.5 | 1234 | 4.957767 | 1.579 | 67.2 | 00:53:24 |
| 4290694_2 | Fixed top2 | 5e-5 | 5e-5 | 1 | 1234 | 4.948502 | 0.854 | 29.6 | 00:53:14 |
| 4290694_3 | Fixed top2 | 5e-5 | 5e-5 | 0.5 | 1234 | 4.957570 | 1.562 | 66.0 | 00:52:49 |
| 4290694_4 | Fixed top4 | 5e-5 | 5e-5 | 0.5 | 1234 | 4.959533 | 2.650 | 78.8 | 00:53:20 |
| 4290694_5 | Fixed top4 | 5e-5 | 5e-5 | 0.25 | 1234 | 4.959508 | 2.682 | 82.9 | 00:53:22 |
| 4290694_6 | Adaptive top<=4 | 5e-5 | 5e-4 | 1 | 1234 | 4.949439 | 1.452 | 61.9 | 00:52:41 |
| 4290694_7 | Adaptive top<=4 | 5e-5 | 5e-4 | 0.5 | 1234 | 4.960218 | 2.366 | 79.9 | 00:53:20 |
| 4290694_8 | Adaptive top<=4 | 5e-5 | 2e-3 | 1 | 1234 | **4.947553** | 1.429 | 63.9 | 00:53:15 |
| 4290694_9 | Adaptive top<=4 | 5e-5 | 2e-3 | 0.5 | 1234 | 4.956583 | 2.237 | 78.2 | 00:52:44 |
| 4290694_10 | Fixed top4 | 1e-4 | 1e-4 | 1 | 1234 | 4.993051 | 4.246 | 64.4 | 00:53:25 |
| 4290694_11 | Adaptive top<=4 | 1e-4 | 5e-4 | 1 | 1234 | 4.964408 | 3.947 | 62.9 | 00:53:19 |
| 4290694_12 | MCSD-TP control | n/a | n/a | n/a | 2345 | 4.921256 | 0 | n/a | 00:52:47 |
| 4290694_13 | Fixed top4 | 5e-5 | 5e-5 | 1 | 2345 | 4.920918 | 1.408 | 60.5 | 00:53:23 |

### Interpretation

1. The best hybrid observed at seed1234 is adaptive top<=4, tau5e-5,
   epsilon2e-3, c1. Its loss is only0.000178 below MCSD-TP and0.000492 above
   MuonSphere. This single-seed, many-candidate selection does not establish
   improvement. It is0.009317 below SSO in this particular short run.
2. Fixed top4 is not robustly better: its paired difference from control is
   +0.006961 at seed1234 and -0.000338 at seed2345 (two-seed mean +0.003312).
   Do not compare seed2345 losses directly with seed1234 baselines.
3. Every tested matched c1 configuration outperformed its c0.5 counterpart
   at seed1234. Fixed4/c0.25 was also worse than c1. Smaller matched steps
   increased branch occupancy and repeats; this is evidence against making
   the cluster step uniformly smaller in this protocol, not proof of a
   universal best coefficient.
4. Increasing routing tau from5e-5 to1e-4 degraded the paired fixed4 and
   adaptive/epsilon5e-4 comparisons. The wider **cluster** tolerance2e-3
   is a different parameter and did not itself widen the routing condition.
5. The early "no consecutive repeats" observation no longer holds. Fixed4/c1
   ends with698 cluster calls and430 repeated calls through step190; at
   step190 alone it uses32/252 cluster directions (12.7%), versus1.46%
   cumulative. The smallest-step fixed4/c0.25 has82.9% repeated calls and
   5.86 calls per recorded burst. Radial return avoids forced top clipping
   but does not prevent lingering at small positive gaps.
6. Tight adaptive/c1 selected q2 for all409 recorded calls and matches the
   fixed-top2/c1 final loss exactly. The broader best-loss adaptive candidate
   selected q2/q3/q4 in116/209/359 calls. It recorded97 cap overflows out of
   684 cluster calls (14.2%): more than four values met its cluster tolerance,
   but only four were used. It is not a complete-cluster/convergence-certified
   method despite its favorable numerical ranking.
7. The best hybrid took53:15 versus34:23 for MuonSphere (+54.9%) and38:04 for
   SSO (+39.9%). Its last logged update took16.085 seconds versus10.133 and
   11.117. All hybrid arms still use one full trial SVD per component/update;
   infrequent cluster branches do not remove that baseline cost.

### Next decision

Do not expand this grid or launch long runs automatically. If continuing,
the cheapest useful confirmation is seed2345 for the best adaptive candidate
and MuonSphere, reusing the existing seed2345 MCSD-TP control. That tests
whether the very small best-of-grid difference transfers across seeds.
Keep tight adaptive/fixed2 c1 as the lower-repeat, no-observed-overflow
mechanism reference. Only after a reliable gain should a matched longer run
or approximate-spectrum acceleration be promoted. The current results do
not yet justify claiming superiority to MuonSphere or standard P2GD theory.

Raw logs/configs and accounting are downloaded to the two corresponding
`output/cluster_radial*_h20_20261021/` directories. Structured extraction is
`output/cluster_radial_followup_h20_20261021/final_summary_20261021.json`.
No new job or cancellation was issued during this results check.

## Submission receipt

Accepted at **02:28:47 HKT**, array **4290694**, task indices 0-13, with
**two concurrent single-H20 tasks**. Slurm confirms the dependency
`afterok:4290663_*`, so all four predecessor arms must finish successfully
before any new arm starts. Invalid dependencies cancel the affected queue
instead of leaving permanently blocked jobs.

The array is **pending**, not yet training, at this receipt. No additional
GPU was occupied by the submission. Other jobs were not canceled or changed.
The new isolated source directory is
`/home/u3013198/projects/cluster-radial-followup-20261021`.
The algorithm files were byte-compared against the original snapshot.

Eleven local CPU tests and the same eleven tests in the H20 Python/PyTorch
environment passed. These include independent routing/cluster tolerances and
uniqueness of the 14 launch configurations. New-arm CUDA tests run inside each
allocated job; they have not run at submission time. The frozen manifest
contains 1524 Python/C++/header/CUDA/build-config/launcher source files.

## Motivation and latest observed evidence

At 02:15 HKT, the first control and fixed-four runs had reached step120/191.
Their step100 validation losses were 5.617080 and 5.617097 respectively.
The fixed-four advantage at step50 (6.453842 versus 6.472949) did not persist
at step100. Fixed-four had used its cluster branch 20/30240 times, without
recorded consecutive repeats. Its maximum flattened-cluster distance was
0.04677R, so selecting four directions is not equivalent to identifying four
nearly equal leading singular values.

A subsequent submission-time snapshot showed step150 validation losses of
**5.112296 (control)** and **5.112962 (fixed4)**, a difference of +0.000666
for fixed4. These remain intermediate, single-seed observations.

This motivates a controlled rank/length/cluster-tolerance comparison, rather
than treating the first validation point or a low branch rate as success.
The current array is not canceled or modified.

## Additional grid

All new runs use width256, 28 layers, the same 3B indexed data source, 191
updates (100,139,008 tokens), LR 1.5e-2, 25 warmup steps, cosine decay to
1.5e-3, global batch128, microbatch4, and one H20. Evaluate at steps50,
100,150,191 with five validation batches. Save no checkpoints.

Here tau is the **routing** gap threshold, epsilon is the **cluster
selection** tolerance, and c matches the trial Frobenius displacement to
the MCSD reference direction. c is not the old raw PGD learning-rate scale.
Fixed-rank methods ignore epsilon for rank selection; it remains a diagnostic.

| Array index | Method | tau | epsilon | c | Seed | Main comparison |
|---:|---|---:|---:|---:|---:|---|
| 0 | MuonSphere reference | n/a | n/a | n/a | 1234 | Same-budget baseline |
| 1 | SSO reference | n/a | n/a | n/a | 1234 | Same-budget baseline |
| 2 | Fixed top2 | 5e-5 | 5e-5 | 1.0 | 1234 | Rank2 versus rank4 |
| 3 | Fixed top2 | 5e-5 | 5e-5 | 0.5 | 1234 | Matched step length |
| 4 | Fixed top4 | 5e-5 | 5e-5 | 0.5 | 1234 | Compare original c=1 |
| 5 | Fixed top4 | 5e-5 | 5e-5 | 0.25 | 1234 | Smaller matched step |
| 6 | Adaptive top<=4 | 5e-5 | 5e-4 | 1.0 | 1234 | Broader cluster, same routing |
| 7 | Adaptive top<=4 | 5e-5 | 5e-4 | 0.5 | 1234 | Cluster/step interaction |
| 8 | Adaptive top<=4 | 5e-5 | 2e-3 | 1.0 | 1234 | Broader cluster, same routing |
| 9 | Adaptive top<=4 | 5e-5 | 2e-3 | 0.5 | 1234 | Cluster/step interaction |
| 10 | Fixed top4 | 1e-4 | 1e-4 | 1.0 | 1234 | Routing threshold only |
| 11 | Adaptive top<=4 | 1e-4 | 5e-4 | 1.0 | 1234 | Routing only, compare index6 |
| 12 | MCSD-TP control | 5e-5 | 5e-5 | unused | 2345 | Paired second seed |
| 13 | Fixed top4 | 5e-5 | 5e-5 | 1.0 | 2345 | Paired second seed |

Existing array4290663 supplies seed1234 control, fixed4/c1 and
adaptive4/epsilon5e-5/c1,c0.5; these are not duplicated.

## Scope and fairness

All cluster variants use unchanged production optimizer code from array4290663:
cached FP32 full-SVD spectrum, MCSD tangent/msign/tangent, near-cluster
direction when routed, then radial return on either branch. This grid changes
configuration only. No approximate spectral solver, full-SVD metric projection,
online error gate, line search, or artificial branch cooldown is added.

MuonSphere and SSO use their reference PI10, NS8 and hard pre-retraction
implementations, with the same training protocol. SSO retains bisection with
20 maximum iterations and tolerance2e-4. They are training-performance
baselines, not precision/geometry-matched direction ablations. Constrained
matrix weight decay is disabled by the existing parameter-group rules;
auxiliary decay is0.1.

The first and second seed pairs share settings except seed. A second-seed
fixed4 test is a robustness diagnostic, not a declaration that fixed4 is the
winner. Other candidates require paired confirmation before promotion.

## Resource and submission policy

Submit a dependent array with concurrency2, one H20 per task, after successful
completion of all four tasks in array4290663. This adds no concurrent GPU use
while the first array runs. Other jobs and projects remain untouched.
Use an isolated snapshot, share existing data read-only, and retain only logs,
launch configurations, source hashes and dataset-index caches.

The 14 additional runs are estimated to use roughly 13-16 GPU-hours based on
the initial ~16.1-second/update measurements; SSO and startup overhead can
change this. At two concurrent GPUs this is roughly 7-8 hours of execution,
plus the predecessor array and queue delays. This is a planning estimate,
not a measured runtime.

## Evaluation and next gate

Rank by final step191 validation loss within the common seed/protocol, and
report step150 alongside it to expose late instability. Also report training
loss, wall time, seconds/update, branch frequency, repeat rate, cluster-size
counts, cap overflow and radial-change diagnostics. Do not choose a variant
because it forces a larger gap or minimizes the branch count.

Select at most two candidates for a subsequent matched 250M or 1B test after
reviewing these results. Do not automatically launch longer training. A
candidate should be numerically stable and competitive with MCSD-TP; a claim
of improvement requires paired seeds, not selection from many single-seed
pilots. No minimum-loss threshold or branch-rate target is being tuned on
the fly.

The present logger records mean post-return gap and branch repeats, but not
the minimum per-event gap or a paired counterfactual direction comparison.
Zero repeats must not be reported as a proven 100% one-step exit rate.

## Artifacts

- Launcher: `slurm/cluster_radial_followup_100m.sbatch`.
- `--describe` with an explicit `SLURM_ARRAY_TASK_ID` prints one grid row
  without importing CUDA or starting a run.
- Each arm verifies the immutable snapshot manifest, runs CUDA unit checks,
  saves `launch_config.json` and `source.sha256`, and writes `COMPLETED` only
  after training succeeds.
- Submission ID, snapshot location and verification receipts are appended
  above after scheduler acceptance. Local upload and receipt directory:
  `output/cluster_radial_followup_h20_20261021/`.
- Remote source manifest: `source.sha256`; accepted ID: `submission.job_id`.
- Remote logs: `logs/followup_4290694_<index>.out` and `.err`.
- Remote per-arm results: `results/4290694_<index>/`.
