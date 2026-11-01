"""Plot the six completed 1B arms, using recorded validation points only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "docs/results/cluster_cone_lmo_1b_20261031/final_20261101.json"
    rows = json.loads(path.read_text())["rows"]
    styles = [("MuonSphere", "#b47c12", "s"), ("SSO", "#ca4a60", "o"),
              ("MCSD-TP K10", "#13857f", "^"), ("Hybrid wide c8", "#315bb7", "D"),
              ("Hybrid narrow c8", "#8c589f", "v"), ("Single-LMO c1", "#606873", "P")]
    control = {v["step"]: v["loss"] for v in rows[2]["vals"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row, (label, color, marker) in zip(rows, styles):
        assert row["completed"] and row["vals"][-1]["step"] == 1908
        curve = {v["step"]: v["loss"] for v in row["vals"]}
        assert sorted(curve) == [250, 500, 750, 1000, 1250, 1500, 1750, 1908]
        assert curve.keys() == control.keys()
        steps = sorted(curve)
        axes[0].plot(steps, [curve[s] for s in steps], label=label, color=color,
                     marker=marker, linewidth=1.6, markersize=4)
        late = [s for s in steps if s >= 1250]
        axes[1].plot(late, [curve[s]-control[s] for s in late], color=color,
                     marker=marker, linewidth=1.6, markersize=4)
    axes[0].set(title="Full training", ylabel="Validation loss", xlim=(200, 1950))
    axes[1].set(title="Late-stage difference vs. MCSD-TP", ylabel="Validation loss difference",
                xlim=(1210, 1950), xticks=[1250, 1500, 1750, 1908])
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.grid(color="#d8dce1", linewidth=.65, linestyle="--", alpha=.75)
        ax.set_axisbelow(True)
    fig.suptitle("Spectral sphere | width 256 | 1B tokens | seed 2345", y=.98, fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.84, wspace=.3)
    output = ROOT / "docs/experiments/cluster_summary/validation_1b_mechanisms_20261101"
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("."+extension), dpi=180, facecolor="white")
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
