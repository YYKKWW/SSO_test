"""Plot recorded local spectral-sphere optimization runs, without recomputing them."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = (
    ("pgd", "Projected EMA", "#222222", "-"),
    ("mcsd_project", "MCSD + projection", "#cc6677", "-"),
    ("hybrid_mcsd_lmo_project", "MCSD + cone-LMO + projection", "#0072b2", "-"),
    ("hybrid_lmo_project", "MCSD-TP + cone-LMO + projection", "#009e73", "-"),
    ("hybrid_p2gd_project", "MCSD-TP + P2GD + projection", "#e69f00", "-"),
    ("guarded_hybrid_lmo_project", "Model-safeguarded hybrid", "#7b3294", "--"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["experiment"]["rows"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True)
    for col, schedule in enumerate(("constant", "diminishing")):
        for method, label, color, style in SERIES:
            group = [r for r in rows if r["stochastic"] and r["schedule"] == schedule
                     and r["method"] == method]
            steps = [t["step"] for t in group[0]["trace"]]
            excess = np.array([[t["loss"]-r["fstar"] for t in r["trace"]] for r in group])
            mapping = np.array([[t["projection_mapping"] for t in r["trace"]] for r in group])
            for ax, values in ((axes[0, col], excess), (axes[1, col], mapping)):
                ax.plot(steps, np.maximum(values.mean(axis=0), 1e-16),
                        label=label, color=color, linestyle=style, linewidth=1.65,
                        alpha=.9, zorder=3 if "guarded" in method else 2)
        axes[0, col].set_title("Constant step, 0.02" if col == 0 else
                               "Preset diminishing step", pad=10)
        axes[1, col].set_xlabel("Iteration")
        for ax in axes[:, col]:
            ax.set_yscale("log")
            ax.grid(True, which="major", color="#dedede", linewidth=.65)
            ax.set_xlim(0, data["config"]["steps"])
    axes[0, 0].set_ylabel("Mean objective excess")
    axes[1, 0].set_ylabel("Mean exact-gradient projection residual")
    fig.suptitle("Local spectral-sphere test: stochastic EMA, exact projection", y=.97, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.88, hspace=.20, wspace=.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
