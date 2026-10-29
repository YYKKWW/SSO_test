"""Offline paired K10/K20 and top-4/top-6 replay, never imported by training.

Recorded matrices are selected, not a random population sample. Fresh current-
point estimates measure initialization sensitivity; warm trial estimates start
from the identical saved basis. Neither is a K20->K10 training trajectory.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from audit_cluster_ritz_snapshots import direction
from emerging_optimizers.orthogonalized_optimizers import \
    spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import \
    warm_block8_ritz


def relative_error(actual, reference):
    return float(torch.linalg.vector_norm(actual-reference) /
                 torch.linalg.vector_norm(reference).clamp_min(1e-30))


def spectral_gap(s):
    return float(1-s[1]/s[0])


def cone_direction(m, u, s, v, cap, tolerance):
    active = int(((s[0]-s)/s[0] <= tolerance).sum())
    q = max(1, min(cap, len(s), active))
    z = -m / torch.linalg.vector_norm(m).clamp_min(1e-12)
    return host.project_spectral_sphere_cone(z, u[:, :q], v[:, :q]), q, active


@torch.no_grad()
def replay(sample):
    x, m = sample["current"].float(), sample["momentum"].float()
    u, s, v = (sample[k].float() for k in ("current_u", "current_s", "current_v"))
    radius = float(sample["radius"])
    threshold, tolerance = sample["gap_threshold"], sample["cluster_threshold"]
    ou, os, ovh = host.accurate_svd_fp32(x)
    true_gap = spectral_gap(os)
    opts = dict(mode=sample["cluster_mode"], threshold=threshold,
                cluster_threshold=tolerance, coefficient=sample["coefficient"],
                direction_mode=sample.get("direction_mode", "momentum_fro"),
                near_projection=sample.get("near_projection", "cone"))
    d = direction(m, u, s, v, **opts)
    od = direction(m, ou, os, ovh.T, **opts)
    y = x + float(sample["effective_lr"]) * d
    true_y = host.accurate_svdvals_fp32(y)
    row = dict(step=sample["step"], selection=sample["selection"],
               shape=list(x.shape), source_mode=sample["cluster_mode"],
               estimated_near=sample["current_gap"] <= threshold,
               true_near=true_gap <= threshold, true_gap=true_gap,
               cached_gap_error=abs(sample["current_gap"]-true_gap),
               cached_direction_error=relative_error(d, od),
               startup=[], capacity=[])
    # Same input matrix and initial basis for each K; only the iteration count changes.
    if sample["step"] <= 50:
        for k in (10, 20):
            cu, cs, cvh = warm_block8_ritz(x, iterations=k)
            _, ys, _ = warm_block8_ritz(y, iterations=k, initial_v=v)
            row["startup"].append(dict(
                iterations=k,
                fresh_current_gap_error=abs(spectral_gap(cs)-true_gap),
                fresh_current_radius_error=float(os[0]/cs[0]-1),
                fresh_current_direction_error=relative_error(direction(m, cu, cs, cvh.T, **opts), od),
                fresh_false_safe=spectral_gap(cs) > threshold and true_gap <= threshold,
                fresh_false_near=spectral_gap(cs) <= threshold and true_gap > threshold,
                warm_trial_gap_error=abs(spectral_gap(ys)-spectral_gap(true_y)),
                warm_trial_radius_error=float(true_y[0]/ys[0]-1),
            ))
    if not (row["estimated_near"] or row["true_near"]):
        return row
    # q<=4 and q<=6 share X/M/alpha; use both cached and oracle bases to
    # separate truncation error from missing/poorly estimated directions.
    row["true_cluster_size"] = int(((os[0]-os)/os[0] <= tolerance).sum())
    alpha = float(sample["effective_lr"]) * 8.
    for basis, bu, bs, bv in (("cached", u, s, v), ("oracle", ou, os, ovh.T)):
        q4, _, _ = cone_direction(m, bu, bs, bv, 4, tolerance)
        caps = (4, 6) if basis == "cached" else (4, 6, len(os))
        for cap in caps:
            qd, q, active = cone_direction(m, bu, bs, bv, cap, tolerance)
            trial = x + alpha*qd
            exact_s = host.accurate_svdvals_fp32(trial)
            _, approx_s, _ = warm_block8_ritz(trial, iterations=10, initial_v=v)
            returned = trial * (radius/approx_s[0])
            delta = returned-x
            row["capacity"].append(dict(
                basis=basis, cap=cap, q=q, active=active,
                truncated=active > cap, direction_change_vs_q4=relative_error(qd, q4),
                returned_momentum_change=float((m*delta).sum()),
                post_gap=spectral_gap(exact_s), exits_near=spectral_gap(exact_s) > threshold,
                radius_error=float(exact_s[0]/approx_s[0]-1),
                radial_correction_ratio=float(torch.linalg.vector_norm(returned-trial) /
                    (alpha*torch.linalg.vector_norm(qd)).clamp_min(1e-30)),
            ))
    return row


def summarize(rows):
    near = [r for r in rows if r["capacity"]]
    out = dict(samples=len(rows), startup_samples=sum(bool(r["startup"]) for r in rows),
               near_samples=len(near), true_capacity_over4=sum(r["true_cluster_size"] > 4 for r in near),
               true_capacity_over6=sum(r["true_cluster_size"] > 6 for r in near),
               true_capacity_over8=sum(r["true_cluster_size"] > 8 for r in near), startup={}, capacity={})
    for k in (10, 20):
        entries = [a for r in rows for a in r["startup"] if a["iterations"] == k]
        if entries:
            metrics = [key for key in entries[0] if key.endswith("error")]
            out["startup"][str(k)] = {
                key: dict(mean=statistics.mean(abs(a[key]) for a in entries),
                          maximum=max(abs(a[key]) for a in entries)) for key in metrics}
            out["startup"][str(k)].update(
                fresh_false_safe=sum(a["fresh_false_safe"] for a in entries),
                fresh_false_near=sum(a["fresh_false_near"] for a in entries))
    for basis in ("cached", "oracle"):
        for cap in (4, 6):
            entries = [a for r in near for a in r["capacity"] if a["basis"] == basis and a["cap"] == cap]
            if entries:
                out["capacity"][f"{basis}_q{cap}"] = dict(
                    exit_count=sum(a["exits_near"] for a in entries),
                    mean_direction_change_vs_q4=statistics.mean(a["direction_change_vs_q4"] for a in entries),
                    mean_returned_momentum_change=statistics.mean(a["returned_momentum_change"] for a in entries),
                    max_radius_error=max(abs(a["radius_error"]) for a in entries))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob("sample_*.pt")):
            row = replay(torch.load(path, weights_only=True, map_location=args.device))
            row.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
    if not rows:
        raise RuntimeError("No snapshots found")
    report = dict(summary=summarize(rows), rows=rows, torch_version=torch.__version__,
                  note="Selected stored matrices, not population rates or loss changes. Startup is paired frozen-matrix replay, not a changed training trajectory. Capacity trials use fixed c8, K10/block8 returns, and cached or oracle bases; full SVD is offline only. Missing overflow samples cannot establish that top-4 is sufficient.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
