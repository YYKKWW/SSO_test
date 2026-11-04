"""Plot matched follow-ups separately by mechanism and LMO step scale."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
STEPS = [250, 500, 750, 1000, 1250, 1500, 1750, 1908]


def load_rows():
    rows = {}
    for name in (
        "cluster_cone_lmo_1b_20261031/final_20261101.json",
        "cluster_mechanism_scales_1b_20261102/final_20261104.json",
    ):
        for row in json.loads((ROOT / "docs/results" / name).read_text())["rows"]:
            assert row["completed"] and row["accounting"]["State"] == "COMPLETED"
            assert [point["step"] for point in row["vals"]] == STEPS
            assert row["config"]["SEED"] == "2345"
            assert row["config"]["EVAL_ITERS"] == "20"
            rows[row["job"]] = row
    return rows


def main():
    rows = load_rows()
    reference = rows["4368382_2"]["vals"]
    families = [
        ("Near-direction mechanisms", [
            ("4368383_3", "Wide cluster c8", "#315bb7", "D"),
            ("4375034_0", "Ambient c8", "#c67a18", "s"),
            ("4375034_1", "Top-1 c8", "#13857f", "^"),
        ]),
        ("Single-LMO step scale (narrow cluster)", [
            ("4375034_2", "LMO c0.5", "#13857f", "^"),
            ("4368382_5", "LMO c1", "#315bb7", "D"),
            ("4375034_3", "LMO c2", "#c67a18", "s"),
        ]),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "cluster-followups-20261104",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    for column, (title, styles) in enumerate(families):
        for row_index in range(2):
            ax = axes[row_index, column]
            indices = range(len(STEPS)) if row_index == 0 else range(4, len(STEPS))
            for job, label, color, marker in styles:
                points = rows[job]["vals"]
                ax.plot(
                    [STEPS[i] for i in indices],
                    [points[i]["loss"] - reference[i]["loss"] for i in indices],
                    color=color, marker=marker, markersize=4.5,
                    linewidth=1.8, label=label,
                )
            ax.axhline(0, color="#606873", linewidth=1, linestyle="--", label="MCSD-TP")
            ax.set_xlabel("Training iteration")
            ax.set_ylabel("Validation loss minus MCSD-TP")
            ax.grid(color="#d8dce1", linewidth=.65, linestyle="--", alpha=.7)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if row_index == 0:
                ax.set(title=title, xlim=(200, 1950), xticks=[250, 750, 1250, 1908])
                ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=.95)
            else:
                ax.set(title="Late-stage detail", xlim=(1210, 1950),
                       xticks=[1250, 1500, 1750, 1908])
    fig.suptitle("Width 256 | 1B tokens | seed 2345 | eval20 | lower is better", fontsize=13)
    fig.subplots_adjust(left=.08, right=.98, bottom=.08, top=.9, hspace=.42, wspace=.32)
    output = ROOT / "docs/experiments/cluster_summary/validation_1b_followups_20261104"
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=180, facecolor="white",
                    metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
