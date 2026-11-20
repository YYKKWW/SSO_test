"""Compact local results and plot verified H20 counterfactuals for publication.

No checkpoints, tensor snapshots, or training data are included. H20 summaries
are independently reproducible with summarize_p2gd_elr_replay.py.
"""
import argparse
import hashlib
import json
from pathlib import Path


def compact_local(source, destination):
    raw = source.read_bytes()
    data = json.loads(raw)
    rows = data["rows"]
    matched = [row for row in rows if row["matched"]]
    result = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "attempted_matches": len(rows),
        "successful_matches": len(matched),
        "maximum_relative_matching_error": max(
            row["relative_match_error"] for row in matched
        ),
        "maximum_radius_error": max(row["radius_error"] for row in matched),
        "protocol": data["protocol"],
        "limitations": data["limitations"],
        "summary": data["summary"],
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")


def plot_replay(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads((root / "h20_summary.json").read_text())
    records = [json.loads(line) for line in
               (root / "h20/repair_loss_replay.jsonl").read_text().splitlines()]
    rows = {row["method"]: row for row in summary["summaries"]}
    methods = [
        ("ambient", "Ambient", "#ad5161"),
        ("top1", "Top-1", "#ad5161"),
        ("cluster_cone", "Cluster cone", "#ad5161"),
        ("finite_momentum", "Finite momentum", "#ad5161"),
        ("anchored_momentum", "Anchored momentum", "#ad5161"),
        ("finite_d0", "Finite D0", "#178a87"),
        ("anchored_d0", "Anchored D0", "#178a87"),
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5),
                             gridspec_kw={"width_ratios": [1.5, 1.25, 1.15]})
    for panel, selected, scale, power in zip(
            axes[:2], (methods[:5], methods[5:]), (1e-3, 1e-5), (-3, -5)):
        for index, (method, label, color) in enumerate(selected):
            values = [next(candidate["heldout_difference_vs_half"]
                           for candidate in record["results"]
                           if candidate["method"] == method)
                      for record in records if record["near_components"]]
            panel.scatter(np.array(values) / scale, index + np.linspace(-.12, .12, len(values)),
                          s=26, color=color, alpha=.55, zorder=3)
            panel.scatter([rows[method]["mean_loss_difference_vs_half"] / scale], [index],
                          marker="D", s=45, color="black", zorder=4)
        panel.set_yticks(range(len(selected)), [item[1] for item in selected])
        panel.invert_yaxis()
        panel.axvline(0, color="#666666", linewidth=.8)
        panel.grid(axis="x", color="#e5e5e5", linewidth=.7)
        panel.set_xlabel(rf"Loss difference from half-D0 ($\times 10^{{{power}}}$)")
        panel.margins(y=.25)
    axes[0].set_title("Momentum family: seven paired steps", fontsize=11)
    axes[1].set_title("D0 repairs: separate horizontal scale", fontsize=11)
    baseline = rows["half_d0"]
    comparison = ["half_d0", "finite_momentum", "finite_d0"]
    y = np.arange(len(comparison))
    for offset, key, label, color in [
        (-.16, "mean_returned_spectral_elr", "Spectral displacement", "#bc7934"),
        (.16, "mean_output_elr", "Local output change", "#366996"),
    ]:
        values = [rows[method][key] / baseline[key] for method in comparison]
        axes[2].barh(y + offset, values, height=.28, color=color, label=label)
        for yy, value in zip(y + offset, values):
            axes[2].text(value + .08, yy, f"{value:.2f}", va="center", fontsize=9)
    axes[2].set_yticks(y, ["Half-D0", "Finite momentum", "Finite D0"])
    axes[2].invert_yaxis()
    axes[2].axvline(1, color="#777777", linewidth=.8, linestyle="--")
    axes[2].set_xlim(0, 9.6)
    axes[2].set_xlabel("Ratio of means to half-D0")
    axes[2].set_title("Equal AUS, unequal movement", fontsize=11)
    axes[2].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Width 256: same-state returned-step replay", fontsize=14, y=.98)
    fig.text(.5, .025, "Dots: correlated steps; black diamonds: means. "
             "Right: 12 near components. These are not independent training runs.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .94), w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"replay_comparison.{extension}", dpi=180)
    plt.close(fig)


def plot_budget(root):
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads((root / "h20_summary.json").read_text())
    rows = {row["name"]: row for row in data["results"]}
    directions = [("finite_d0", "Finite D0"), ("cluster_cone", "Cluster cone"),
                  ("finite_momentum", "Finite momentum")]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0))
    y = np.arange(len(directions))
    for panel, metric, title in zip(axes[:2], ("spectral", "output"),
                                    ("Matched spectral ELR", "Matched output change")):
        for offset, tag, label, color in [
            (-.14, "100", "100% budget", "#366996"),
            (.14, "095", "95% budget", "#bc7934"),
        ]:
            values = [rows[f"{metric}_{tag}_{method}"][
                "mean_difference_vs_matched_control"] * 1e4 for method, _ in directions]
            panel.scatter(values, y + offset, color=color, s=42, label=label, zorder=3)
        panel.set_yticks(y, [label for _, label in directions])
        panel.invert_yaxis()
        panel.axvline(0, color="#777777", linewidth=.8)
        panel.grid(axis="x", color="#e5e5e5", linewidth=.7)
        panel.set_title(title, fontsize=11)
        panel.set_xlabel(r"Loss difference vs matched control ($\times 10^{-4}$)")
        panel.set_ylim(2.5, -.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    base = rows["half_d0"]
    for offset, metric, label, color in [
        (-.14, "spectral", "Spectral matched", "#366996"),
        (.14, "output", "Output matched", "#bc7934"),
    ]:
        values = [rows[f"{metric}_100_{method}"]["mean_independent_output_elr"] /
                  base["mean_independent_output_elr"] for method, _ in directions]
        axes[2].barh(y + offset, values, height=.25, color=color, label=label)
        for yy, value in zip(y + offset, values):
            axes[2].text(value + .02, yy, f"{value:.2f}", fontsize=9, va="center")
    axes[2].set_yticks(y, [label for _, label in directions])
    axes[2].set_ylim(2.5, -.5)
    axes[2].set_xlim(0, 1.25)
    axes[2].axvline(1, color="#777777", linewidth=.8, linestyle="--")
    axes[2].set_title("Independent-input output movement", fontsize=11)
    axes[2].set_xlabel("Ratio of means to half-D0, 100% budgets")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left",
                   bbox_to_anchor=(0, -.24), ncol=2)
    fig.suptitle("Width 256: controlled spectral and output budgets", fontsize=14, y=.98)
    fig.text(.5, .025, "Loss: means over seven correlated steps; lower is better. "
             "Movement: 12 near components. All updates use radial return.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .94), w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"budget_comparison.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--local-source", type=Path)
    parser.add_argument("--budget-root", type=Path)
    args = parser.parse_args()
    if args.local_source:
        compact_local(args.local_source, args.root / "local_summary.json")
    plot_replay(args.root)
    if args.budget_root:
        plot_budget(args.budget_root)


if __name__ == "__main__":
    main()
