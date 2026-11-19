"""Plot completed validation curves and same-step differences, without smoothing."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text())
    rows = report["rows"]
    styles = [("MCSD-TP control", "#355d8a", "o"),
              ("Near-only half-D0", "#188577", "s"),
              ("Ambient-radial, rho=0.5", "#c85c48", "^")]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    control = {p["step"]: p["loss"] for p in rows[0]["vals"]}
    for row, (label, color, marker) in zip(rows, styles):
        steps = [p["step"] for p in row["vals"]]
        losses = [p["loss"] for p in row["vals"]]
        axes[0].plot(steps, losses, color=color, marker=marker, markersize=3,
                     linewidth=1.4, label=f'{label} ({row["final_loss"]:.6f})')
        if row is not rows[0]:
            axes[1].plot(steps, [p["loss"]-control[p["step"]] for p in row["vals"]],
                         color=color, marker=marker, markersize=3, linewidth=1.4,
                         label=label)
    axes[0].set_title("Validation loss: all recorded evaluations")
    axes[0].set_ylabel("Validation loss")
    axes[1].set_title("Paired difference from current control")
    axes[1].set_ylabel("Loss difference (negative is better)")
    axes[1].axhline(0, color="#777777", linewidth=.8, linestyle="--")
    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.grid(color="#e2e5e8", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Width 256 / 3B tokens / seed 2345 / warm block-8 K10", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(args.report.parent / f"validation_comparison_20261118.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
