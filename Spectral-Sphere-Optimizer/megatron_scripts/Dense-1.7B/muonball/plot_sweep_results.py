#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if not row.get("best_val_loss"):
                continue
            row["width"] = int(row["width"])
            row["lr_float"] = float(row["lr"])
            row["best_val_loss"] = float(row["best_val_loss"])
            rows.append(row)

    by_optimizer = defaultdict(list)
    for row in rows:
        by_optimizer[row["optimizer"]].append(row)

    names = {
        "adam": "AdamW",
        "dist_muon": "Muon",
        "spectral_ball_dist": "Spectral Sphere",
        "muon_ball_dist": "Muon Sphere",
        "spel_dist": "True SpEL",
    }
    preferred_order = ["adam", "dist_muon", "spectral_ball_dist", "muon_ball_dist", "spel_dist"]
    optimizers = [opt for opt in preferred_order if opt in by_optimizer]
    if not optimizers:
        optimizers = sorted(by_optimizer)

    fig, axes = plt.subplots(1, len(optimizers), figsize=(5 * len(optimizers), 4), squeeze=False)
    for ax, optimizer in zip(axes[0], optimizers):
        group = by_optimizer[optimizer]
        widths = sorted({row["width"] for row in group})
        for width in widths:
            points = sorted([row for row in group if row["width"] == width], key=lambda r: r["lr_float"])
            xs = [row["lr_float"] for row in points]
            ys = [row["best_val_loss"] for row in points]
            ax.plot(xs, ys, marker="o", label=f"width={width}")
            if ys:
                best_i = min(range(len(ys)), key=ys.__getitem__)
                ax.scatter([xs[best_i]], [ys[best_i]], marker="*", s=140, zorder=5)
                ax.annotate(f"{ys[best_i]:.4f}", (xs[best_i], ys[best_i]), textcoords="offset points", xytext=(4, 6))
        ax.set_xscale("log")
        ax.set_title(names.get(optimizer, optimizer))
        ax.set_xlabel("LR")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Best Validation Loss")
    axes[0][-1].legend(loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
