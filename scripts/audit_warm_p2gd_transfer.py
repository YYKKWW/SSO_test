"""Offline BF16 returned-step comparisons; no training state is modified."""
import argparse
import json
from pathlib import Path

import torch

from kernel_lmo_h20 import warm_block8_ritz
from warm_p2gd_transfer import MODES, near_direction


@torch.no_grad()
def audit(data):
    x, m, d0 = (data[k].float() for k in ("current", "momentum", "d0"))
    u, s, v = (data[k].float() for k in ("cached_u", "cached_s", "cached_v"))
    record = data["record"]
    radius, alpha = float(data["radius"]), float(data["alpha"])
    q = min(int(((s[0]-s)/s[0] <= 2e-4).sum()), 6)
    directions = {mode: near_direction(m, d0, u[:, :q], v[:, :q], mode,
                                       record["raw_momentum_ratio"]) for mode in MODES}
    directions["half_d0"] = .5*d0
    points = {"actual": data["returned"].float()}
    for mode, d in directions.items():
        y = x + alpha*d
        _, values, _ = warm_block8_ritz(y, iterations=10, initial_v=v)
        points[mode] = y*(radius/values[0])
    xb = x.to(torch.bfloat16).float().double()
    nx = torch.linalg.svdvals(xb)[0]
    sx = torch.linalg.svdvals(x.double())
    rows = {}
    for name, point in points.items():
        db = point.to(torch.bfloat16).float().double()-xb
        rows[name] = dict(bf16_spectral_elr=float(torch.linalg.svdvals(db)[0]/nx),
                          bf16_frobenius_elr=float(db.norm()/xb.norm()),
                          master_displacement_over_R=float((point-x).norm()/radius),
                          radius_relative_error=float(torch.linalg.svdvals(point.double())[0]/radius-1))
    target = rows["half_d0"]["bf16_spectral_elr"]
    for row in rows.values():
        row["spectral_elr_over_half_d0"] = row["bf16_spectral_elr"]/max(target, 1e-30)
    return dict(step=int(data["step"]), shape=list(x.shape), selection=data["selection"],
                actual_near=record["near"], policy=record["near_mode"],
                cached_gap=record["current_gap"], exact_gap=float(1-sx[1]/sx[0]),
                true_cluster=int(((sx[0]-sx)/sx[0] <= 2e-4).sum()), results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [audit(torch.load(p, map_location="cpu", weights_only=True))
            for p in sorted(args.snapshots.glob("sample_*.pt"))]
    if not rows:
        raise RuntimeError("no snapshots")
    report = dict(rows=rows, samples=len(rows),
        note="Offline CPU audit of computed returns before host arithmetic rounding. BF16 round-trip does not reproduce all forward kernels. Counterfactuals share the saved approximate cache; no exact reinitialization. No activations: output ELR is not measured. Early-near and scheduled-shape samples are biased. Fixed coefficients do NOT enforce matched returned steps.")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(samples=len(rows), actual_near_samples=sum(r["actual_near"] for r in rows))))


if __name__ == "__main__":
    main()
