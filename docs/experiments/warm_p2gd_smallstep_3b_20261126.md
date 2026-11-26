# Small-Step Warm MCSD-TP/P2GD: 3B Transfer

## Question

Does the cluster-direction benefit seen in the
[BF16-matched short paths](p2gd_replication_recovery_20261125.md) survive a
cheap, fixed-step, full-budget implementation?

**Persistent warm block8/K10 is already tested**, including completed 3B
MCSD-TP and ambient-radial runs. The new part is the **small-step numerical
cluster direction, top<=6**, compared with top-1 and ambient at the same raw
momentum coefficient. This is not another claim that the spectral backend is
new, and not a rerun of the old `c8` cluster experiment.

Status: local tests passed and the overlay was uploaded; H20 source
preparation/submission awaits a successful SSH reconnection. No new job ID
has been confirmed yet. The independent ELR jobs are left untouched.

## Three New Arms

All use the same safe MCSD-TP direction `D0`. In the near region, first form

```text
Z = -0.075 * ||D0||_F * M / ||M||_F       # zero if M or D0 is zero
```

| Arm | Near direction | Return |
|---|---|---|
| Cluster, top<=6 | Numerical-cluster cone-type repair of Z | Approximate radial |
| Top-1 | Z - <u1 v1^T, Z> u1 v1^T | Same |
| Ambient | Z, without top correction | Same |

For the cluster arm, let `A=sym(U^T Z V)=Q diag(a) Q^T`, with ascending
eigenvalues. Clamp them to nonpositive values and set the largest to zero,
forming `B`. The direction is `Z + U(B-A)V^T`. Do **not** renormalize after
repair. This is exact cone projection only for a complete true top eigenspace;
a thresholded/truncated numerical cluster is an engineering approximation.

Every positive-LR component update performs one warm block8/K10 Ritz solve
on its trial. Both branches return `R*Y/sigma1_hat(Y)` and cache the scaled
Ritz values for the next point. Only the 8x8 Ritz matrix is SVD-factorized.
Bootstrap uses the same approximate solver. There is no full-matrix SVD,
norm-SVD, online accuracy gate, or per-step BF16 coefficient search.

### Why 0.075?

Use the **first 32 updates (769--800)** of the completed diagnostic panel,
not its final validation ranking, to estimate a plausible transfer scale.
That panel starts with `Z=-0.5*||D0||_F*M/||M||_F` and then multiplies the
trial coefficient by `alpha_ratio`. Hence its effective raw ratio is
`0.5*alpha_ratio`, not `alpha_ratio` alone.

| Prefix | Near calls | Mean effective ratio | Median |
|---|---:|---:|---:|
| Cluster | 461 | 0.075295 | 0.073341 |
| Top-1 | 238 | 0.070274 | 0.064163 |

Freeze **0.075 for all three new arms**. This is a hypothesis based on a
late-stage seed1234 prefix; the 3B experiment starts fresh with seed2345.
It is neither an optimal coefficient nor a guarantee of equal actual steps.
The existing old ambient `rho=0.5` and `c8` runs use different step rules.

**Returned BF16 spectral/output ELR is not matched online.** Sparse offline
checks report BF16 spectral movement relative to same-state half-D0, sphere
error, true versus estimated gap and capacity. No activation samples are
collected, so this audit does not measure output ELR. A gain in this panel
would support practical viability, not isolate a fully step-matched effect.

## Frozen Protocol

| Setting | Value |
|---|---|
| Width / layers / head dimension | 256 / 28 / 128 |
| Training budget | 5723 updates, 3,000,500,224 processed tokens |
| Data | Existing indexed 3B training source; no new download or repeated 1B source |
| Seed | 2345 |
| Sequence / global batch / microbatch | 4096 / 128 / 4 |
| Peak / minimum LR | 0.015 / 0.0015 |
| Schedule | 250 warmup updates, cosine over 5723 updates |
| Validation | 20 batches every 250 updates and at the end |
| Momentum | Reference Nesterov, 0.9 |
| Main weights / geometry | FP32 master; FP32 QR and Ritz; BF16 model/NS8 |
| Near gap and cluster window | 2e-4, from the returned approximate cache |
| Basis / iterations / direction cap | 8 / 10 / 6 |
| Matrix weight decay | Zero; auxiliary optimizer unchanged |
| Resources | One H20 per arm, at most two concurrent new jobs |
| Checkpoints | Disabled; small bounded matrix snapshots only |

Top>6 and saturated block8 windows are counted, not certified as complete.
The same eight-vector cache is propagated throughout training; offline
audits never replace it with exact vectors. New runs share data, seed and
configuration but diverge after their first different update.

## Reused Completed References

No duplicate jobs are submitted for these results. Lower validation loss is
better. All are width256, seed2345, 3B; old MuonSphere/SSO are contextual
references, not fresh source-matched arms.

| Reference | Final loss | Train + validation time | Job |
|---|---:|---:|---|
| Current warm MCSD-TP | 3.429645 | 17:56:55 | 4453017_0 |
| Current near-only half-D0 | **3.423857** | 17:57:09 | 4453017_1 |
| Current ambient-radial, rho=0.5 | 3.432013 | 17:54:03 | 4442918_2 |
| Historical cluster K10/c8, top<=4 | 3.432620 | 17:41:00 | See results ledger |
| Historical MuonSphere | 3.436023 | 16:35:03 | 4346705_0 |
| Historical SSO | 3.435041 | 18:28:11 | 4346705_1 |

The new project is based on the verified source-only snapshot of
`warm-radial-controls-3b-20261117`, manifest SHA256
`ec80d212d45bf1c7dfcad269e38a515c533874c8eb712fc0e57dd03a363cd186`.
All shared recorded training fields are checked against completed control
4453017_0 before submission. Source changes are opt-in and frozen separately.

## Checks and Decision

- Local: 40 regression tests passed (17 warm-backend/transfer and 23 direct
  return/control tests); offline BF16 audit
  smoke passed. Tests reject original-matrix SVD or any `svdvals` online,
  verify cache reuse, zero inputs, top-cap overflow and paired launch fields.
- H20: CUDA tests and an independent four-step smoke must pass before each
  fresh 3B run starts. Smoke is not part of the reported training trajectory.
- Report final loss, last-four evaluation mean, wall time, near occupancy,
  repetition and capacity; disclose approximate-radius/gap/BF16-step audits.
- Promote the cluster only if it improves over **both** new top-1 and ambient
  arms, remains competitive with completed half-D0, and retains useful speed.
  A numerical-cluster radial result is not standard P2GD convergence evidence.
- Do not expand width, gap grid or token budget before these results.

## Code

- [Direction and cached update](../../scripts/warm_p2gd_transfer.py)
- [Launch configuration](../../scripts/run_kernel_lmo_pilot.py)
- [Process-local optimizer adapter](../../scripts/pretrain_kernel_lmo_pilot.py)
- [Offline BF16 step audit](../../scripts/audit_warm_p2gd_transfer.py)
- [Regression tests](../../scripts/test_warm_p2gd_transfer.py)
- [Safe source preparation](../../scripts/prepare_warm_p2gd_transfer.py)
- [Three-arm Slurm array](../../slurm/warm_p2gd_transfer_3b_20261126.sbatch)

```bash
python -B -m unittest discover -s scripts -p 'test_warm*.py' -v
```

The H20 script reuses the existing SSO environment. Expected train/validation
time is roughly 18--20 hours per arm from the completed warm controls, not a
measured runtime for this new direction. With two concurrent slots, three
arms need roughly two waves plus scheduler delay.

Resume preparation and submission after connectivity returns (the preparer
is idempotent after a successful submission; never bypass an existing intent
record after an ambiguous `sbatch` response):

```bash
module purge
module load python/3.12.1 cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"
python -B "$HOME/prepare_warm_p2gd_transfer.py" \
  --base "$HOME/projects/warm-radial-controls-3b-20261117" \
  --target "$HOME/projects/warm-p2gd-smallstep-3b-20261126" \
  --overlay "$HOME/warm_p2gd_transfer_20261126.tar.gz" --submit
```
