# NorMuon-Sphere: a matched Frobenius-sphere baseline

Status, 2026-10-05: implemented, including launcher/optimizer registry and
checkpoint state. The expanded local suite passes 144 tests. NorMuon-Sphere
H20 smoke 4197451 completed four steps normally under the new shape-radius
protocol; the four 3B LR jobs 4197475--4197478 all completed. Final losses
are 3.526564 / 3.499072 / 3.476793 / 3.464757 at LRs
5e-3 / 7e-3 / 1e-2 / 1.5e-2. The lowest value is at the grid boundary;
see the matched [full LR results](WIDTH256_3B_RADIUS_LR_20260926.md).
Historical jobs and their sources remain untouched.

## Sources and scope

Primary references are [NorMuon, Algorithm 1](https://arxiv.org/html/2510.05491v1)
and the [authors' implementation](https://github.com/zichongli5/NorMuon/blob/main/normuon.py).
The source method applies row-wise second-moment adaptation after matrix
orthogonalization. Its unconstrained update is not a Frobenius-sphere optimizer.
Our explicit adaptation combines that idea with MuonH's unit-Frobenius
direction and radial sphere return. Call it **NorMuon-Sphere (matched
adaptation)**, not an untouched reproduction or a theorem-covered MCSD special case.

## Implemented update

For each logical matrix `W` of shape `(D_out, D_in)`, use the same frozen
radius `R`, initialization, auxiliary optimizer, learning-rate schedule and
Q/K/V and gate/up splitting as MuonH. Let `G` be its gradient.

```text
M <- beta1 M + (1-beta1) G
B  = beta1 M + (1-beta1) G        # matched Nesterov; B=M if disabled
O  = PolarExpress(B)             # BF16 products, FP32 output
v <- beta2 v + (1-beta2) mean_columns(O^2)
Z  = O / (sqrt(v) + epsilon)     # FP32 row-wise operations
D  = Z / ||Z||_F                # D=0 when Z=0
Y  = W - learning_rate * R * D
W_next = R * Y / ||Y||_F
```

No tangent projection is applied to B or D. This matches MuonH's ambient
direction construction; MCSD/TP are separate methods. `v` has shape
`(D_out, 1)`, remains FP32, and has no bias correction. Rows always mean the
original logical output rows, even for tall matrices whose Polar Express
implementation transposes internally. Interleaved fused QKV components have
independent states. Gradient buffers are not modified by the optimizer.

The only directional difference from our MuonH baseline is the row adaptation.
Both normalize the resulting direction in Frobenius norm and take relative
trial length `learning_rate * R`. The author's global magnitude restoration
would cancel under this unit normalization, so it is not redundantly computed.
Do not additionally apply the original unconstrained method's 0.2 RMS or
aspect-ratio magnitude factor. Constrained weights receive zero weight decay;
auxiliary weights retain the common AdamW settings.

## Matched defaults versus the original recipe

| Setting | This adapter | Reason |
| --- | --- | --- |
| beta1 / Nesterov | 0.9 / on | Match the measured MuonH configuration |
| beta2 / epsilon | 0.95 / 1e-10 | Author-code convention |
| Matrix sign | Shared BF16 Polar Express, 10 iterations | Same kernel as MuonH, not original NS5 |
| Row moments and sphere return | FP32 | Same master-weight arithmetic as the study |
| Radius | Existing initialization-derived Frobenius radius | Do not silently change the running protocol |
| Extra state | D_out FP32 values per logical matrix | Saved and restored with momentum |

The reference paper uses beta1=0.95; that is an explicit tuning option,
not this matched comparison's default. The new shape-based radius proposal
(`R_fro=r*sqrt(D_out)`, Stiefel spectral matching) requires a separate matched
batch for all methods. The later opt-in implementation is documented in
[the width-256 LR protocol](WIDTH256_3B_RADIUS_LR_20260926.md); it does not
retroactively change the measured MuonH results.

## Execution

Optimizer ID: `manifold_normuon_sphere`. Other geometries are rejected.
Commands below are dry runs, not job submissions:

```bash
python scripts/manifold/launch.py \
  --geometry frobenius --method manifold_normuon_sphere --steps 4

python scripts/manifold/launch.py \
  --geometry frobenius --method manifold_normuon_sphere \
  --stage main --width 384 --lr 1e-2 --seed 1234 \
  --save-interval 2000 --save-retain-interval 12000
```

Optional parameters: `--normuon-beta2`, `--normuon-eps`,
`--momentum-beta`, `--nesterov` / `--no-nesterov`. Normal launcher controls
still apply. Resume validation rejects changed row-EMA hyperparameters or
missing trained row state rather than resetting it. A previously uninitialized
optimizer may initialize zero moments normally.

Before a long run, use the existing H20 environment and perform a single-GPU
short training plus strict checkpoint-resume comparison. Source freezing and
disk-capacity checks are still required. Do not overwrite the code under
running experiments to conduct this test.

## Tests and limits

- Independent exact-polar equations over multiple updates and tall/wide/square shapes.
- FP32 row orientation, non-contiguous QKV-like component state and zero directions.
- BF16 practical path, unchanged gradients and no SVD in practical updates.
- Exact checkpoint continuation, rejection of missing/malformed/mismatched state.
- Full launch configuration matches MuonH except method and new row parameters.
- Existing MCSD, MCSD-TP, iMuon, SSO and MuonSphere regression tests still pass.

The implementation lives in `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/manifold_baselines.py`.
Practical BF16 polar remains approximate; exact mode is only a correctness
reference. Frobenius radial normalization does not need singular values.
Small constraint defects alone do not certify the paper's LMO error assumptions.
The matched four-LR 3B experiment now provides single-seed observations;
multi-seed confirmation and independent held-out evaluation remain necessary.
