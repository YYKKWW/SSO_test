"""Plot verified 3B curves without smoothing or synthetic uncertainty bands."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/results/cluster_selected_3b_20261025/final_summary_20261027.json"
OUTPUT = ROOT / "docs/experiments/cluster_summary"


def main():
    rows = {row["index"]: row for row in json.loads(SOURCE.read_text())["rows"]}
    curves = {i: {v["step"]: v["loss"] for v in row["vals"]} for i, row in rows.items()}
    for i in (2, 3, 4, 5):
        assert rows[i]["completed"] and rows[i]["accounting"]["ExitCode"] == "0:0"
        assert len(curves[i]) == 23 and curves[i][5723] == rows[i]["final"]
        assert curves[i].keys() == curves[2].keys()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titleweight": "medium"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = [(2, "MuonSphere", "#d58b14", "s"), (3, "SSO", "#d64c5e", "o"),
              (4, "MCSD-TP K10", "#188b87", "^"), (5, "Hybrid K10/c8", "#345cb0", "D")]
    for i, label, color, marker in styles:
        steps = sorted(curves[i])
        axes[0].plot(steps, [curves[i][s] for s in steps], color=color,
                     marker=marker, markersize=3.6, linewidth=1.8, label=label)
        late = [s for s in steps if s >= 4000]
        axes[1].plot(late, [curves[i][s] - curves[2][s] for s in late],
                     color=color, marker=marker, markersize=4.5, linewidth=1.8)
    axes[0].set(title="Full training", ylabel="Validation loss", xlim=(200, 5800))
    axes[1].set(title="Late-stage difference vs. MuonSphere",
                ylabel="Validation loss difference", xlim=(3950, 5800))
    axes[1].set_xticks([4000, 4500, 5000, 5500, 5723])
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.grid(color="#d8dce1", linewidth=.65, linestyle="--", alpha=.75)
        ax.set_axisbelow(True)
    fig.suptitle("Spectral sphere | width 256 | 3B tokens | seed 1234", fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.84, wspace=.28)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"validation_3b_k10.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT / "validation_3b_k10.png")


if __name__ == "__main__":
    main()
