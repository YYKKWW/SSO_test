"""Fixed-state step-size audit of saved training matrices, not a loss line search.

Only first-cluster samples are used. Old snapshots lack effective_lr, so infer
it from returned = a * (current + effective_lr * direction) and verify the fit.
This cannot evaluate language-model loss or establish a useful training rate.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz


def infer_effective_lr(x, returned, direction):
    basis = torch.stack((x.double().flatten(), direction.double().flatten()), dim=1)
    target = returned.double().flatten()
    ab = torch.linalg.solve(basis.T @ basis, basis.T @ target)
    error = float(torch.linalg.vector_norm(basis @ ab - target) / torch.linalg.vector_norm(target))
    alpha = float(ab[1] / ab[0])
    if error > 2e-6 or alpha <= 0 or not torch.isfinite(ab).all():
        raise ValueError(f"cannot reconstruct a positive trial step: {alpha=}, {error=}")
    return alpha, error


@torch.no_grad()
def audit_sample(sample, scales, iterations):
    x, m, radius = sample["current"], sample["momentum"], sample["radius"]
    u, s, v = sample["current_u"], sample["current_s"], sample["current_v"]
    q = max(1, min(4, int(((s[0] - s) / s[0] <= sample["cluster_threshold"]).sum())))
    d = host.project_spectral_sphere_cone(
        -m / torch.linalg.vector_norm(m).clamp_min(1e-12), u[:, :q], v[:, :q])
    if "effective_lr" in sample:
        alpha, fit_error = float(sample["effective_lr"]), None
    else:
        alpha, fit_error = infer_effective_lr(x, sample["returned"], sample["coefficient"] * d)
    tangent = host.project_to_tangent_plane(m, u[:, :1], v[:, :1])
    tangent /= torch.linalg.vector_norm(tangent).clamp_min(1e-7)
    control = -host.project_to_tangent_plane(host.msign(tangent, steps=8).float(), u[:, :1], v[:, :1])
    rows = []
    for label, step_direction in [("control", control)] + [(str(c), c * d) for c in scales]:
        y = x + alpha * step_direction
        _, sy, _ = warm_block8_ritz(y, iterations=iterations, initial_v=v)
        returned = y * (radius / sy[0])
        actual_s = host.accurate_svdvals_fp32(returned)
        displacement = returned - x
        rows.append(dict(
            scale=label, direction_fro=float(torch.linalg.vector_norm(step_direction)),
            trial_relative_fro=float(alpha * torch.linalg.vector_norm(step_direction) / torch.linalg.vector_norm(x)),
            actual_relative_fro=float(torch.linalg.vector_norm(displacement) / torch.linalg.vector_norm(x)),
            radial_change_ratio=float(torch.linalg.vector_norm(returned - y) /
                                      (alpha * torch.linalg.vector_norm(step_direction)).clamp_min(1e-30)),
            momentum_linear_change=float((m * displacement).sum()),
            returned_true_gap=float(1 - actual_s[1] / actual_s[0]),
            returned_radius_rel_error=float(actual_s[0] / radius - 1),
            finite=bool(torch.isfinite(returned).all()),
        ))
    return dict(step=sample["step"], shape=list(x.shape), q=q, effective_lr=alpha,
                inferred_lr_fit_error=fit_error, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[.5, 1., 2., 4., 8.])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if any(c <= 0 for c in args.scales):
        parser.error("scales must be positive")
    torch.set_num_threads(4)
    samples = []
    for path in sorted(args.directory.glob("sample_*.pt")):
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if sample["branch"] != "pgd" or sample.get("direction_mode") != "momentum_fro":
            continue
        result = audit_sample(sample, args.scales, args.iterations)
        result["source"] = path.name
        samples.append(result)
    if not samples:
        raise RuntimeError("no normalized-momentum cluster snapshots found")
    summary = {}
    for scale in ["control"] + [str(c) for c in args.scales]:
        rows = [next(r for r in s["rows"] if r["scale"] == scale) for s in samples]
        summary[scale] = dict(
            mean_direction_fro=sum(r["direction_fro"] for r in rows)/len(rows),
            mean_actual_relative_fro=sum(r["actual_relative_fro"] for r in rows)/len(rows),
            max_radius_error=max(abs(r["returned_radius_rel_error"]) for r in rows),
            mean_radial_change_ratio=sum(r["radial_change_ratio"] for r in rows)/len(rows),
            negative_momentum_linear_change=sum(r["momentum_linear_change"] < 0 for r in rows),
            all_finite=all(r["finite"] for r in rows),
        )
    result = dict(samples=len(samples), summary=summary, rows=samples,
                  note="Selected fixed-state matrix diagnostics; not LLM loss, stationarity, convergence, or a population estimate. No online step selection.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(samples=len(samples), summary=summary), indent=2))


if __name__ == "__main__":
    main()
