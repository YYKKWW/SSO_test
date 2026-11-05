"""Plot completed same-protocol LMO rank/shape runs and baseline endpoints."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
STEPS = [250, 500, 750, 1000, 1250, 1500, 1750, 1908]
SOURCES = (
    "cluster_cone_lmo_1b_20261031/final_20261101.json",
    "cluster_mechanism_scales_1b_20261102/final_20261104.json",
    "cluster_lmo_shape_1b_20261104/final_20261105.json",
)
STYLES = (
    ("4375034_2", "Unblended LMO c0.5", "#315bb7", "o"),
    ("4389998_0", "Top-1 LMO c0.5", "#c67a18", "s"),
    ("4389998_1", "25% shape blend c0.5", "#13857f", "D"),
)


def load_rows():
    rows = {}
    for source in SOURCES:
        for row in json.loads((ROOT / "docs/results" / source).read_text())["rows"]:
            assert row["completed"] and row["accounting"]["State"] == "COMPLETED"
            assert row["accounting"]["ExitCode"] == "0:0"
            assert [point["step"] for point in row["vals"]] == STEPS
            for key, value in (
                ("WIDTH", "256"), ("SEED", "2345"), ("EVAL_ITERS", "20"),
                ("TRAIN_ITER", "1908"), ("LR", "1.5e-2"),
            ):
                assert row["config"][key] == value, (row["job"], key)
            rows[row["job"]] = row
    control = rows["4368382_2"]["config"]
    for row in rows.values():
        for key in (
            "TRAIN_DATA_PREFIX", "VALID_DATA_PREFIX", "TOKENIZER_MODEL",
            "GLOBAL_BATCH", "SEQ_LENGTH", "NUM_LAYERS", "LR_WARMUP_ITERS",
        ):
            assert row["config"][key] == control[key], (row["job"], key)
    for job, *_ in STYLES:
        for key, value in (
            ("SPEL_PGD_CLUSTER_ITERATIONS", "10"),
            ("SPEL_PGD_CLUSTER_THRESHOLD_REL", "2e-4"),
            ("SPEL_PGD_GAP_THRESHOLD_REL", "5e-5"),
            ("SPEL_PGD_PGD_LR_SCALE", "0.5"),
        ):
            assert rows[job]["config"][key] == value, (job, key)
    return rows


def main():
    rows = load_rows()
    reference = rows["4368382_2"]["vals"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "cluster-lmo-shape-20261105",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, indices, title in (
        (axes[0], range(8), "All recorded validation points"),
        (axes[1], range(4, 8), "Late-stage detail"),
    ):
        for job, label, color, marker in STYLES:
            points = rows[job]["vals"]
            ax.plot(
                [STEPS[i] for i in indices],
                [points[i]["loss"] - reference[i]["loss"] for i in indices],
                color=color, marker=marker, markersize=4.5,
                linewidth=1.7, label=label,
            )
        ax.axhline(0, color="#606873", linewidth=1, linestyle="--", label="MCSD-TP")
        ax.set(title=title, xlabel="Training iteration")
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axes[0].set(ylabel="Validation loss minus MCSD-TP", xlim=(200, 1950),
                xticks=[250, 750, 1250, 1908])
    axes[1].set(xlim=(1210, 1950), xticks=[1250, 1500, 1750, 1908])

    # Display endpoint differences, with absolute losses alongside each point.
    endpoints = (
        ("4389998_1", "25% blend", "#13857f"),
        ("4375034_2", "Unblended", "#315bb7"),
        ("4368382_2", "MCSD-TP", "#606873"),
        ("4389998_0", "Top-1 LMO", "#c67a18"),
        ("4368383_0", "MuonSphere", "#946777"),
        ("4368383_1", "SSO", "#8258a0"),
    )
    ax = axes[2]
    baseline = reference[-1]["loss"]
    for index, (job, label, color) in enumerate(endpoints):
        loss = rows[job]["vals"][-1]["loss"]
        delta = loss - baseline
        ax.hlines(index, 0, delta, color=color, linewidth=2)
        ax.plot(delta, index, "o", color=color, markersize=6)
        ax.annotate(f"{loss:.6f}", (delta, index), xytext=(7, 6),
                    textcoords="offset points", fontsize=9, color=color)
    ax.axvline(0, color="#606873", linewidth=1, linestyle="--")
    ax.set(title="Final validation (step 1908)",
           yticks=range(len(endpoints)), yticklabels=[row[1] for row in endpoints],
           xlabel="Loss minus MCSD-TP", xlim=(-.0015, .0085),
           ylim=(5.6, -.6), xticks=[0, .002, .004, .006, .008])
    for ax in axes:
        ax.grid(color="#d8dce1", linewidth=.65, linestyle="--", alpha=.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .015))
    fig.suptitle("Width 256 | 1B tokens | seed 2345 | eval20 | lower is better", fontsize=13)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.83, wspace=.48)
    output = ROOT / "docs/experiments/cluster_summary/validation_1b_lmo_shape_20261105"
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
