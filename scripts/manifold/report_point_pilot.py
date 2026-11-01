"""Export audited H20 pilot curves; redraw released JSON without Torch or a GPU."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


TRAIN = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+)"
    r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?\| lm loss:\s*([\d.Ee+-]+)")
VAL = re.compile(r"validation loss at iteration\s+(\d+)( on validation set)?"
                 r"\s*\| lm loss value:\s*([\d.Ee+-]+)")
ELAPSED = dict(zip(map(str, range(4368413, 4368419)),
                   ["01:38:03", "01:38:05", "01:38:38", "01:37:33", "01:38:01", "01:38:02"]))
LABELS = {
    "point090_lr005": "Point cap 0.90 / LR 0.005",
    "nesterov095_lr010": "Nesterov 0.95 / LR 0.010",
    "point090_lr0025": "Point cap 0.90 / LR 0.0025",
    "raw_lr010": "No momentum / LR 0.010",
    "point090_lr010": "Point cap 0.90 / LR 0.010",
    "point095_lr005": "Point cap 0.95 / LR 0.005",
}


def export(evidence):
    register = json.loads((evidence / "register.json").read_text())
    runs = []
    for spec in register["runs"]:
        if spec["phase"] != "pilot":
            continue
        folder = evidence / spec["job_id"]
        paths = list(folder.glob("*.out"))
        assert len(paths) == 1
        raw = paths[0].read_bytes()
        text = raw.decode()
        cfg = json.loads((folder / "resolved_config.json").read_text())
        completion = json.loads((folder / "completion.json").read_text())
        assert completion["exit_code"] == 0
        assert cfg["training"]["lr"] == spec["lr"]
        assert cfg["training"]["momentum_beta"] == spec["beta"]
        assert cfg["training"].get("point_extrapolation_cap") == spec["cap"]
        assert cfg["training"]["nesterov"] == spec["nesterov"]
        assert cfg["train_iters"] == 11445 and cfg["warmup_iters"] == 229
        assert cfg["exit_interval"] == 1000
        train = []
        for step, horizon, samples, ms, loss in TRAIN.findall(text):
            assert int(horizon) == 11445 and int(samples) == int(step) * 128
            train.append(dict(step=int(step), loss=float(loss), seconds=float(ms) / 1000))
        val = [dict(step=int(step), loss=float(loss), final=bool(final))
               for step, final, loss in VAL.findall(text)]
        assert [r["step"] for r in train] == [1] + list(range(10, 1001, 10))
        assert [r["step"] for r in val] == list(range(200, 1001, 200))
        assert not any(r["final"] for r in val)
        assert all(math.isfinite(r["loss"]) for r in train + val)
        counters = {}
        for key in ("skipped", "nan"):
            values = [int(v) for v in re.findall(r"number of " + key + r" iterations:\s*(\d+)", text)]
            assert len(values) == len(train) and max(values) == 0
            counters[key] = max(values)
        keep_cfg = {k: cfg[k] for k in ("protocol_id", "source_commit", "model", "numerics",
                    "training", "initialization", "train_iters", "warmup_iters", "exit_interval")}
        hashes = {k: v for k, v in cfg["source_hashes"].items()
                  if "orthogonalized_optimizers/manifold_" in k or k.startswith("scripts/manifold/")}
        runs.append(dict(name=spec["name"], label=LABELS[spec["name"]], job_id=spec["job_id"],
                         lr=spec["lr"], cap=spec["cap"], beta=spec["beta"], nesterov=spec["nesterov"],
                         state="COMPLETED", exit_code="0:0", elapsed=ELAPSED[spec["job_id"]],
                         process_seconds=completion["process_seconds"], val_at_1000=val[-1]["loss"],
                         train_mean_steps_910_1000=statistics.mean(r["loss"] for r in train[-10:]),
                         mean_seconds_steps_910_1000=statistics.mean(r["seconds"] for r in train[-10:]),
                         counters=counters, config=keep_cfg, source_hashes=hashes,
                         stdout_sha256=hashlib.sha256(raw).hexdigest(), train=train, val=val))
    assert len(runs) == 6
    smoke = json.loads((evidence / "4368407/resume_verification.json").read_text())
    assert smoke["exact_match"]
    return dict(checked="2026-11-01", steps=1000, processed_tokens=262144000,
                schedule_tokens=3000000000, validation_model="live Y for point variants",
                exact_resume_verification=smoke, runs=runs)


def plot(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    colors = ["#007F82", "#263D85", "#C76F16", "#858585", "#AD426E", "#607E30"]
    baseline = next(r for r in data["runs"] if r["name"].startswith("nesterov"))
    base = {p["step"]: p["loss"] for p in baseline["val"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    handles = []
    for run, color in zip(data["runs"], colors):
        x = [p["step"] for p in run["val"]]
        y = [p["loss"] for p in run["val"]]
        style = "--" if run["name"] in ("raw_lr010", "point090_lr010") else "-"
        line, = axes[0].plot(x, y, style, color=color, marker="o", ms=4,
                             lw=1.8, label=run["label"])
        handles.append(line)
        if run is not baseline:
            axes[1].plot(x, [v - base[s] for s, v in zip(x, y)], style,
                         color=color, marker="o", ms=4, lw=1.8)
    axes[0].set(title="Validation loss", ylabel="Loss (lower is better)", ylim=(4.0, 6.3))
    axes[1].axhline(0, color=colors[1], lw=1.4)
    axes[1].set(title="Difference from Nesterov control", ylabel="Loss minus Nesterov loss")
    for ax in axes:
        ax.set(xlabel="Training iteration", xticks=[200, 400, 600, 800, 1000])
        ax.grid(alpha=.19)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.005), fontsize=9)
    fig.suptitle("Width 256 | Frobenius sphere | 262M-token pilot | seed 1234", fontsize=12)
    fig.subplots_adjust(bottom=.25, top=.85, wspace=.28)
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("validation." + suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for run, color in zip(data["runs"], colors):
        points = [p for p in run["train"] if p["step"] >= 200]
        ax.plot([p["step"] for p in points], [p["loss"] for p in points],
                color=color, lw=1.5, label=run["label"])
    ax.set(title="Training loss | logged 10-step averages, no extra smoothing",
           xlabel="Training iteration", ylabel="Training loss", xlim=(200, 1000))
    ax.grid(alpha=.19)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.24)
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("training." + suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = export(args.evidence) if args.evidence else json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    plot(data, args.out)
    for run in data["runs"]:
        print(run["job_id"], run["label"], run["val_at_1000"],
              round(run["train_mean_steps_910_1000"], 6),
              round(run["mean_seconds_steps_910_1000"], 3))


if __name__ == "__main__":
    main()
