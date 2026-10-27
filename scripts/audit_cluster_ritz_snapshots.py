"""Post-training oracle audit. Never imported by or fed back into training."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host


def direction(m, u, s, v, *, mode, threshold, cluster_threshold, coefficient,
              direction_mode="matched_fro", near_projection="cone"):
    normal = host.project_to_tangent_plane(m, u[:, :1], v[:, :1])
    normal /= torch.linalg.vector_norm(normal).clamp_min(1e-7)
    ref = -host.project_to_tangent_plane(host.msign(normal, steps=8).float(), u[:, :1], v[:, :1])
    gap = float(1-s[1]/s[0])
    if mode == "control" or gap > threshold:
        return ref
    cap = 2 if mode == "fixed2" else 4
    active = int(((s[0]-s)/s[0] <= cluster_threshold).sum())
    q = max(1, min(cap, len(s), active if mode == "adaptive4" else cap))
    if direction_mode == "momentum_fro":
        normalized = m / torch.linalg.vector_norm(m).clamp_min(1e-12)
        if near_projection == "ambient":
            return -coefficient * normalized
        if near_projection == "tangent1":
            return coefficient * host.project_to_tangent_plane(-normalized, u[:, :1], v[:, :1])
        return coefficient * host.project_spectral_sphere_cone(-normalized, u[:, :q], v[:, :q])
    d = host.project_spectral_sphere_cone(-m, u[:, :q], v[:, :q])
    return coefficient * torch.linalg.vector_norm(ref) / torch.linalg.vector_norm(d).clamp_min(1e-30) * d


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = []
    for path in sorted(args.directory.glob("sample_*.pt")):
        sample = torch.load(path, map_location=args.device, weights_only=True)
        x, returned, m = sample["current"], sample["returned"], sample["momentum"]
        u, s, vh = host.accurate_svd_fp32(x)
        post_s = host.accurate_svdvals_fp32(returned)
        gap, post_gap = float(1-s[1]/s[0]), float(1-post_s[1]/post_s[0])
        opts = dict(mode=sample["cluster_mode"], threshold=sample["gap_threshold"],
                    cluster_threshold=sample["cluster_threshold"], coefficient=sample["coefficient"],
                    direction_mode=sample.get("direction_mode", "matched_fro"),
                    near_projection=sample.get("near_projection", "cone"))
        estimated_d = direction(m, sample["current_u"], sample["current_s"], sample["current_v"], **opts)
        reference_d = direction(m, u, s, vh.T, **opts)
        estimated_near = sample["current_gap"] <= sample["gap_threshold"]
        true_near = gap <= sample["gap_threshold"]
        rows.append(dict(
            file=path.name, step=sample["step"], shape=list(x.shape), selection=sample["selection"],
            branch=sample["branch"], estimated_gap=sample["current_gap"], true_gap=gap,
            current_gap_abs_error=abs(sample["current_gap"]-gap),
            returned_gap_abs_error=abs(sample["returned_gap"]-post_gap),
            current_radius_rel_error=float(s[0]/sample["radius"]-1),
            returned_radius_rel_error=float(post_s[0]/sample["radius"]-1),
            false_safe=bool(not estimated_near and true_near),
            false_cluster=bool(estimated_near and not true_near),
            direction_relative_error=float(torch.linalg.vector_norm(estimated_d-reference_d) /
                                           torch.linalg.vector_norm(reference_d).clamp_min(1e-30)),
            direction_mode=opts["direction_mode"],
            near_projection=opts["near_projection"],
            estimated_direction_fro=float(torch.linalg.vector_norm(estimated_d)),
            reference_direction_fro=float(torch.linalg.vector_norm(reference_d)),
            momentum_inner_product=float((m * estimated_d).sum()),
        ))
    if not rows:
        raise RuntimeError("no samples were captured")
    summary = dict(
        samples=len(rows), false_safe=sum(r["false_safe"] for r in rows),
        false_cluster=sum(r["false_cluster"] for r in rows),
        max_returned_radius_abs_error=max(abs(r["returned_radius_rel_error"]) for r in rows),
        mean_current_gap_abs_error=sum(r["current_gap_abs_error"] for r in rows)/len(rows),
        mean_direction_relative_error=sum(r["direction_relative_error"] for r in rows)/len(rows),
        note="Bounded, shape/early-cluster selected samples; not population error rates. FP32 full-SVD oracle. Control route discrepancies are hypothetical.",
    )
    (args.directory / "offline_audit.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
