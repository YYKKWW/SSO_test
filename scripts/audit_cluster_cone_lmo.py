"""Frozen-matrix screening of narrower clusters and a single-LMO direction.

No training loss is evaluated. Exact decompositions are offline oracles only;
all trial returns under test use the production warm block-8/K10 solver.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import \
    spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.spectral_cone_lmo import \
    matrix_sign_cone_direction
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import \
    warm_block8_ritz

ARMS = (
    ("cone_wide_c8", "cone", .002, 8., False),
    ("cone_narrow_c8", "cone", .0002, 8., False),
    ("cone_fixed2_c8", "cone", .002, 8., True),
    ("cone_lmo_narrow_c1", "cone_lmo", .0002, 1., False),
    ("top1_c8", "tangent1", .002, 8., False),
    ("ambient_c8", "ambient", .002, 8., False),
)
LMO_SCALES = (
    ("cone_wide_c8", "cone", .002, 8., False),
    ("cone_narrow_c8", "cone", .0002, 8., False),
    ("cone_lmo_narrow_c05", "cone_lmo", .0002, .5, False),
    ("cone_lmo_narrow_c1", "cone_lmo", .0002, 1., False),
    ("cone_lmo_narrow_c2", "cone_lmo", .0002, 2., False),
)


def gap(s):
    return float(1-s[1]/s[0])


@torch.no_grad()
def audit(sample, arms=ARMS):
    x, m = sample["current"].float(), sample["momentum"].float()
    u, s, v = (sample[k].float() for k in ("current_u", "current_s", "current_v"))
    ou, os, ovh = host.accurate_svd_fp32(x)
    radius, eta = float(sample["radius"]), float(sample["effective_lr"])
    threshold = 5e-5
    row = dict(step=sample["step"], shape=list(x.shape), selection=sample["selection"],
               estimated_gap=sample["current_gap"], true_gap=gap(os),
               estimated_near=sample["current_gap"] <= threshold,
               true_near=gap(os) <= threshold, arms=[])
    if not row["estimated_near"]:
        return row
    z = -m / torch.linalg.vector_norm(m).clamp_min(1e-12)
    # Paired spectra separate direction errors from approximation of the return.
    for source, uu, ss, vv in (("cached", u, s, v), ("oracle", ou, os, ovh.T)):
        for name, kind, tolerance, coefficient, fixed2 in arms:
            active = int(((ss[0]-ss)/ss[0] <= tolerance).sum())
            q = min(2, len(ss)) if fixed2 else max(1, min(4, active))
            if kind == "cone_lmo":
                d = matrix_sign_cone_direction(z, uu[:, :q], vv[:, :q], msign=host.msign)
            elif kind == "cone":
                d = host.project_spectral_sphere_cone(z, uu[:, :q], vv[:, :q])
            elif kind == "tangent1":
                d = host.project_to_tangent_plane(z, uu[:, :1], vv[:, :1])
                q = 1
            else:
                d, q = z, 0
            step = eta * coefficient * d
            trial = x + step
            _, es, _ = warm_block8_ritz(trial, iterations=10, initial_v=v)
            ts = host.accurate_svdvals_fp32(trial)
            returned, exact = trial * (radius/es[0]), trial * (radius/ts[0])
            displacement = returned-x
            h = uu[:, :q].T @ d @ vv[:, :q] if q else None
            row["arms"].append(dict(
                method=name, spectrum=source, q=q, active=active,
                cap_overflow=kind in ("cone", "cone_lmo") and active > q,
                direction_fro=float(torch.linalg.vector_norm(d)),
                direction_momentum_inner=float((m*d).sum()),
                direction_is_descent=float((m*d).sum()) <= 1e-6 * float(torch.linalg.vector_norm(m)*torch.linalg.vector_norm(d)) + 1e-12,
                selected_cone_residual=abs(float(torch.linalg.eigvalsh(.5*(h+h.T))[-1])) if q else 0.,
                returned_momentum_inner=float((m*displacement).sum()),
                exact_return_momentum_inner=float((m*(exact-x)).sum()),
                momentum_cosine=float((m*displacement).sum() / (torch.linalg.vector_norm(m)*torch.linalg.vector_norm(displacement)).clamp_min(1e-30)),
                relative_displacement=float(torch.linalg.vector_norm(displacement)/torch.linalg.vector_norm(x).clamp_min(1e-30)),
                radial_change_ratio=float(torch.linalg.vector_norm(returned-trial)/torch.linalg.vector_norm(step).clamp_min(1e-30)),
                radius_error=float(ts[0]/es[0]-1), post_gap=gap(ts), exits_near=gap(ts)>threshold,
            ))
    return row


def summarize(rows, arms=ARMS):
    result = dict(samples=len(rows), near_samples=sum(r["estimated_near"] for r in rows),
                  false_safe=sum(r["true_near"] and not r["estimated_near"] for r in rows),
                  false_near=sum(r["estimated_near"] and not r["true_near"] for r in rows), arms={})
    for source in ("cached", "oracle"):
        for name, *_ in arms:
            entries = [a for r in rows for a in r["arms"] if a["spectrum"]==source and a["method"]==name]
            if not entries:
                continue
            result["arms"][f"{source}/{name}"] = dict(
                count=len(entries), direction_non_descent=sum(not a["direction_is_descent"] for a in entries),
                positive_returned_inner=sum(a["returned_momentum_inner"]>0 for a in entries),
                exits_near=sum(a["exits_near"] for a in entries),
                cap_overflow=sum(a["cap_overflow"] for a in entries),
                q_counts={str(q):sum(a["q"]==q for a in entries) for q in range(5)},
                median_direction_fro=statistics.median(a["direction_fro"] for a in entries),
                median_radial_change=statistics.median(a["radial_change_ratio"] for a in entries),
                mean_momentum_cosine=statistics.mean(a["momentum_cosine"] for a in entries),
                max_abs_radius_error=max(abs(a["radius_error"]) for a in entries),
                max_selected_cone_residual=max(a["selected_cone_residual"] for a in entries),
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", choices=("structure", "lmo_scales"), default="structure")
    args = parser.parse_args()
    arms = ARMS if args.preset == "structure" else LMO_SCALES
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob("sample_*.pt")):
            row = audit(torch.load(path, map_location=args.device, weights_only=True), arms)
            row.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
    if not rows:
        raise RuntimeError("No snapshots found")
    summary = summarize(rows, arms)
    report = dict(summary=summary, rows=rows, preset=args.preset, arms=arms,
                  early=summarize([r for r in rows if r["step"] < 1000], arms),
                  late=summarize([r for r in rows if r["step"] >= 1000], arms),
                  torch_version=torch.__version__,
                  note="Selected real-training snapshots, not population rates; momentum linear models, not loss measurements. A narrow/fixed cluster need not contain all true top directions. Single-LMO coefficients differ intentionally; this is not an equal-step-length comparison.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
