"""Offline actual-snapshot spectral/ELR audit; never modifies training states."""
import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def audit_snapshot(data, window=2e-4):
    x, y, returned, d0 = (data[k].double() for k in ("current", "trial", "returned", "d0"))
    radius, alpha = float(data["radius"]), float(data["alpha"])
    sx, sy, sr = (torch.linalg.svdvals(v) for v in (x, y, returned))
    delta, trial = returned-x, y-x
    gx, gr = (float(1-s[1]/s[0]) for s in (sx, sr))
    record = data["record"]
    estimated_near = bool(record["near"])
    true_near = gx <= window
    return dict(
        step=int(data["step"]), shape=list(x.shape),
        selection=data.get("selection", "unknown"),
        cached_gap=float(record["current_gap"]), exact_gap=gx,
        current_gap_abs_error=abs(float(record["current_gap"])-gx),
        cached_returned_gap=float(record["returned_gap"]), exact_returned_gap=gr,
        returned_gap_abs_error=abs(float(record["returned_gap"])-gr),
        current_radius_relative_error=float(sx[0]/radius-1),
        returned_radius_relative_error=float(sr[0]/radius-1),
        estimated_near=estimated_near, exact_near=true_near,
        false_near=estimated_near and not true_near,
        missed_near=true_near and not estimated_near,
        exact_cluster_size=int(((sx[0]-sx)/sx[0]<=window).sum()),
        trial_elr_frobenius=float(trial.norm()/x.norm()),
        actual_elr_frobenius=float(delta.norm()/x.norm()),
        trial_elr_spectral=float(torch.linalg.svdvals(trial)[0]/sx[0]),
        actual_elr_spectral=float(torch.linalg.svdvals(delta)[0]/sx[0]),
        trial_over_d0=float(trial.norm()/(alpha*d0.norm()).clamp_min(1e-30)),
        returned_over_d0=float(delta.norm()/(alpha*d0.norm()).clamp_min(1e-30)),
        trial_exact_norm=float(sy[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args=parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    files=sorted(args.snapshots.glob("sample_*.pt"))
    if not files:
        raise RuntimeError("no snapshots to audit")
    torch.set_num_threads(1)
    n=min(args.limit, len(files))
    indices=sorted({round(k*(len(files)-1)/max(1,n-1)) for k in range(n)})
    rows=[]
    for i in indices:
        data=torch.load(files[i], map_location=args.device, weights_only=True)
        rows.append(dict(source=str(files[i]), **audit_snapshot(data)))
    summary=dict(samples=len(rows),
        mean_gap_abs_error=sum(r["current_gap_abs_error"] for r in rows)/len(rows),
        max_gap_abs_error=max(r["current_gap_abs_error"] for r in rows),
        max_abs_returned_radius_error=max(abs(r["returned_radius_relative_error"]) for r in rows),
        false_near=sum(r["false_near"] for r in rows),
        missed_near=sum(r["missed_near"] for r in rows),
        max_exact_cluster_size=max(r["exact_cluster_size"] for r in rows))
    report=dict(summary=summary,rows=rows,
        note="FP64 audit of actual stored FP32 current/trial/returned tensors; no pre-normalization. Returned is the optimizer's computed return before host update rounding. Samples are scheduled shapes and early near calls, not an unbiased sample. Offline only; no online certificate or fallback. ELR uses ||X|| in the corresponding norm, unlike online Frobenius displacement divided by R.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
