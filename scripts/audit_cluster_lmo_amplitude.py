"""Offline screening of a shape-scaled Frobenius cap; no GPU or new training."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def screen(source, coefficient=.5):
    data = json.loads(source.read_text())
    rows = []
    for row in data["rows"]:
        if row["branch"] != "pgd":
            continue
        # The recorded direction already includes the near-step coefficient.
        ratio = row["estimated_direction_fro"] / (coefficient * math.sqrt(min(row["shape"])))
        rows.append(dict(file=row["file"], step=row["step"], shape=row["shape"],
                         normalized_fro=ratio, cap_factor=min(1., 1./max(ratio, 1e-30))))
    if not rows:
        raise ValueError("no near-branch samples")
    return dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), coefficient=coefficient,
        cap_rule="D * min(1, sqrt(min(m,n)) / ||D||_F), before the c multiplier",
        samples=len(rows), minimum_normalized_fro=min(r["normalized_fro"] for r in rows),
        maximum_normalized_fro=max(r["normalized_fro"] for r in rows),
        affected_samples=sum(r["cap_factor"] < 1. for r in rows),
        maximum_fraction_removed=max(1-r["cap_factor"] for r in rows),
        mean_fraction_removed=sum(1-r["cap_factor"] for r in rows)/len(rows),
        note="Selected snapshots only, not population rates. No loss-based cap selection. "
             "A much lower cap would mostly be another global step-scale change.",
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = screen(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
