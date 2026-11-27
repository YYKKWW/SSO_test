# MCSD Fixed Sphere Results and Active Tests

**Snapshot: November 27, 2026, 12:16 UTC+8.** The two missing fixed-sphere
3B arms are now running on one H20 each. Both CUDA/Megatron gates passed.
The main logs reach update **10 / 11,445** and have no NaNs or skipped
updates; neither has a new validation point or a completed endpoint yet.
Yesterday's blocked upload was not a completed experiment.

The norm-band exploration is paused. All new updates return to the fixed
Frobenius sphere; no annulus or active-boundary correction is used.

[Completed evidence](#completed-evidence) | [Active tests](#active-tests) |
[Next decisions](#next-decisions) | [Verified job snapshot](mcsd_sphere_status_20261127/status.json) |
[Overview](FROBENIUS_3B_6B_SUMMARY.md)

## Completed Evidence

These are **historical full 3B endpoints**, not new results from today's
jobs. Width 256, seed 1234, beta=0.95 Muon-style Nesterov, nominal peak
matrix LR=0.010 and R=sqrt(D_out) are shared. Auxiliary AdamW and the
data/model protocol are matched. They are not per-method LR optima or
multi-seed superiority claims. The replay row additionally matches MuonH's
matrix-wise returned AUS; the other rows retain their nominal schedules.

| Practical recipe | Final validation | Difference from MuonH |
|---|---:|---:|
| NorMuon-Sphere | **3.455796** | **-0.002694** |
| MuonH | 3.458490 | 0 |
| MCSD + unit-F + row adaptation | 3.460299 | +0.001809 |
| MCSD + unit-F | 3.460665 | +0.002175 |
| Standard MCSD-TP | 3.461153 | +0.002663 |
| Standard MCSD | 3.461687 | +0.003197 |
| MCSD-TP + unit-F, MuonH AUS replay | 3.462119 | +0.003629 |

**What improved:** unit-F reduces MCSD's seed-1234 endpoint by 0.001022;
row adaptation adds a further 0.000366 improvement. **What did not transfer:**
the selected normalized/adaptive recipe reaches 3.464919 at seed 2026,
behind standard MCSD's 3.464190 in that seed. The engineering gain is not
established as robust, and neither baseline is surpassed.

**Why isolate projection next:** the controlled 3,000-step Nesterov pair
favored preprojection by 0.010182, but the existing projected full-budget
replay still trails native MuonH. Native MuonH also lacks the explicit
post-sign projection, so that endpoint comparison cannot attribute the
gap to preprojection alone. The new arms complete that distinction.

Sources: [full-budget projection and momentum results](FROBENIUS_RESULTS_20261119.md),
[normalization and seed coverage](FROBENIUS_3B_6B_SUMMARY.md#comparisons),
[full LR grids](FROBENIUS_3B_6B_HISTORY.md#comparisons).
The completed [annulus AUS study](FROBENIUS_BAND_AUS_RESULTS_20261126.md)
remains archived separately; its near-match to MuonH is not an MCSD win.

## Active Tests

For each common Nesterov source S, omit the descent sign below and apply
the same unit-F convention, fixed-sphere return and actual FP32 master-AUS
targets. P_W is the tangent projector at the current matrix.

| Arm | Direction before unit-F | Full job | State at snapshot |
|---|---|---:|---|
| MuonH reference | msign(S) | 4427413 | Completed historical reference |
| MCSD + unit-F | msign(P_W S) | **4519941** | Running, logged step 10 |
| Post-projection only | P_W msign(S) | **4519942** | Running, logged step 10 |
| MCSD-TP + unit-F | P_W msign(P_W S) | 4427415 | Completed historical replay |

MCSD's polar direction need not be tangent. Job 4519941 therefore solves
the **actual general ray** before the ordinary sphere return. Job 4519942
preserves the old tangent-replay arithmetic, checked bitwise in the local
CPU regression. Both match the same reference scalar chord per matrix,
not just the same nominal LR. This preserves the intended update directions
while isolating their angular step size.

### Verified Gates

| Real six-update Megatron gate | Job | State / exit | Max master-AUS error | RMS error |
|---|---:|---|---:|---:|
| MCSD + unit-F | 4519939 | Completed / 0:0 | 6.101e-8 | 6.340e-9 |
| Post-projection only | 4519940 | Completed / 0:0 | 6.150e-8 | 6.341e-9 |

Each gate covers 6 x 196 = **1,176 logical-matrix updates**. Both have zero
NaNs/skips, complete AUS traces and a passed framework-free CUDA solver
test. These are integration checks, not short-run optimizer rankings.
Returned-AUS tolerance is 2e-6; it applies to FP32 masters, not identical
rounded BF16 forward-weight displacements.

### Frozen Protocol and Records

Width 256, 28 layers, FFN 768, 4 query heads / 2 KV groups; sequence 2048,
global batch 128, microbatch 4, seed 1234. Each main run processes
**3,000,238,080 tokens** in 11,445 updates, with 229 warmup updates and the
original cosine horizon. Nominal matrix peak LR is 0.010; auxiliary AdamW
peak LR remains 0.001. Hidden-matrix weight decay is zero. BF16 model,
FP32 masters/states and the existing practical 10-step polar kernel remain.

The batch uses at most **two GPUs**, with 30-hour main limits and no
checkpoints. Existing other-project jobs were not changed. Runtime snapshot
`746d3f1aa4e16d3c23eeea9b4e338ad2c05fcb3c` is frozen on H20; this hash is
not asserted to be a public GitHub commit. Shared protocol/protected-source
checks passed before submission. Config/reference hashes and Slurm receipts
are in the released JSON; private data paths and payloads are excluded.

Both arms record actual matrix AUS and sample G/M/S angles plus source,
polar-output and selected-direction radial cosines. The general-ray arm
also records every before/after norm and solved coefficient. Completed
trace counts remain zero for the running jobs until their final metadata
is written; that does not mean they have performed zero updates.

## Next Decisions

1. **Finish these two 3B arms before expanding the search.** Use the final
   validation and the original direction-study late window,
   **9,000:200:11,400 plus 11,445**. Those 14 evaluations are correlated,
   not independent seeds. The earlier annulus report used a different
   nine-point window; do not silently substitute it or select a favorable
   checkpoint. Current startup data cannot rank these methods.
2. **Read the four-arm contrasts.** MCSD versus MuonH tests preprojection
   without postprojection; MCSD-TP versus post-only tests it with
   postprojection. Within each preprojection setting, compare the effect
   of postprojection. Match validation steps and audit returned AUS before
   attributing a difference to direction. Here P_W S projects a momentum
   mixture, not the current Riemannian gradient P_W G.
3. **Replicate a stable full-budget effect, not an early lead.** If an effect
   has the same favorable sign at the endpoint and late mean, test its
   matched pair at seed 2026 using a new seed-specific AUS reference. If
   it vanishes or reverses, retain the result and prioritize the cheap
   same-state normal-extension/held-out diagnostic over another LR, width,
   6B, point-extrapolation or adaptation grid. Exact-LMO geometric claims
   do not by themselves prove superiority after unit normalization and
   practical BF16 polar computation.

This update completes the previously authorized two-arm submission. It
does not launch additional tuning grids. The status exporter is read-only
and operates outside the frozen training tree.
