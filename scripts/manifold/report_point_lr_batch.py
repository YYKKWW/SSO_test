"""Snapshot frozen H20 runs; redraw point and LR-grid results from compact JSON."""

import argparse
import datetime
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


TRAIN = re.compile(r"iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+)"
                   r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
                   r".*?\| lm loss:\s*([\d.Ee+-]+)")
VAL = re.compile(r"validation loss at iteration\s+(\d+)( on validation set)?"
                 r"\s*\| lm loss value:\s*([\d.Ee+-]+)")
METHODS = {"manifold_mcsd": "MCSD", "manifold_muonh": "MuonH",
           "manifold_normuon_sphere": "NorMuon-Sphere"}


def collect(root, extra_registers=()):
    records = []
    names = ("point_pilot_20261031", "point_followup_20261101", "point_lr_batch_20261102", *extra_registers)
    for name in names:
        register = json.loads((root / "results/manifold" / (name + "_register.json")).read_text())
        records.extend(dict(row, batch=name) for row in register["runs"]
                       if row.get("phase", "pilot") == "pilot")
    ids = [row["job_id"] for row in records]
    assert len(set(ids)) == len(ids)
    output = subprocess.check_output(
        ["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o", "JobID,State,ExitCode,Elapsed,Start,End"],
        text=True)
    accounting = {}
    for line in output.splitlines():
        fields = line.split("|")
        if fields[0] in ids:
            accounting[fields[0]] = dict(zip(("job_id", "state", "exit_code", "elapsed", "start", "end"), fields))
    runs = []
    for record in records:
        job = record["job_id"]
        directory = Path(record["run_dir"])
        paths = list((root / "logs").glob(f"*_{job}.out"))
        assert len(paths) <= 1, job
        raw = paths[0].read_bytes() if paths else b""
        text = raw.decode()
        config_raw = (directory / "resolved_config.json").read_bytes()
        cfg = json.loads(config_raw)
        assert cfg["train_iters"] == 11445 and cfg["warmup_iters"] == 229
        assert cfg["model"]["hidden_size"] == 256 and cfg["training"]["seed"] in (1234, 2026)
        train = []
        for step, horizon, samples, ms, loss in TRAIN.findall(text):
            assert int(horizon) == 11445 and int(samples) == int(step) * 128
            train.append(dict(step=int(step), loss=float(loss), seconds=float(ms) / 1000))
        val = [dict(step=int(step), loss=float(loss), final=bool(final))
               for step, final, loss in VAL.findall(text)]
        assert all(math.isfinite(p["loss"]) for p in train + val), job
        assert [p["step"] for p in val] == sorted(set(p["step"] for p in val)), job
        counters = {}
        for key in ("nan", "skipped"):
            values = [int(v) for v in re.findall(r"number of " + key + r" iterations:\s*(\d+)", text)]
            assert len(values) == len(train), job
            counters[key] = max(values) if values else None
        completion_path = directory / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
        target = cfg["exit_interval"] or cfg["train_iters"]
        completed = accounting[job]["state"] == "COMPLETED"
        if completed:
            assert train and val, job
            assert accounting[job]["exit_code"] == "0:0" and completion["exit_code"] == 0, job
            assert val[-1]["step"] == target, job
            assert val[-1]["final"] == (not cfg["exit_interval"]), job
        keep = {key: cfg[key] for key in (
            "source_commit", "protocol_id", "model", "training", "numerics", "initialization",
            "train_iters", "warmup_iters", "target_tokens", "actual_tokens", "exit_interval")}
        keep["dataset_prefix_names"] = {key: Path(cfg[key]).name for key in ("train_prefix", "valid_prefix")}
        runs.append(dict(record, **{key: value for key, value in accounting[job].items() if key != "job_id"},
                         config=keep, stdout_sha256=hashlib.sha256(raw).hexdigest() if paths else None,
                         config_sha256=hashlib.sha256(config_raw).hexdigest(),
                         optimizer_source_hashes={key: value for key, value in cfg["source_hashes"].items()
                                                  if "orthogonalized_optimizers/manifold_" in key},
                         completion=completion, counters=counters,
                         last_train=train[-1] if train else None, last_val=val[-1] if val else None,
                         endpoint_val=val[-1]["loss"] if completed else None,
                         final_3b_val=val[-1]["loss"] if completed and not cfg["exit_interval"] else None,
                         tail_train_mean=statistics.mean(p["loss"] for p in train[-10:]) if train else None,
                         recent_seconds_per_step=statistics.mean(p["seconds"] for p in train[-10:]) if train else None,
                         train=train, val=val))
    return dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                note="Dated snapshot. Running jobs have no final_3b_val. Point runs are early-exit pilots.",
                runs=runs)


def point_series(data):
    jobs = {run["job_id"]: run for run in data["runs"]}
    definitions = [
        ("Nesterov 0.95 / LR 0.010", ["4368414", "4373195"], "#294C85"),
        ("Point 0.90 / LR 0.005", ["4368413", "4373194"], "#929292"),
        ("Point 0.90 / LR 0.0025", ["4368415", "4373196"], "#B36D19"),
        ("Point 0.95 / LR 0.0025", ["4373197", "4375560"], "#A04476"),
        ("Point 0.95 / LR 0.00125", ["4375561"], "#008378"),
    ]
    series = []
    for label, ids, color in definitions:
        merged = {}
        for job in ids:
            for point in jobs[job]["val"]:
                assert point["step"] not in merged
                merged[point["step"]] = point["loss"]
        assert sorted(merged) == list(range(200, 3001, 200))
        series.append((label, merged, color))
    return series


def add_references(data, path):
    selected = {"4329085", "4346665", "4346666"}
    refs = []
    for run in json.loads(path.read_text())["runs"]:
        if run["job_id"] not in selected:
            continue
        assert run["state"] == "COMPLETED" and run["lr"] == .01
        assert run["beta"] == .95 and run["nesterov"] and run["seed"] == 1234
        refs.append({key: run[key] for key in ("job_id", "method", "lr", "beta", "nesterov", "seed",
                     "state", "elapsed", "final_val", "val", "config", "optimizer_source_hashes", "log_sha256")})
    assert len(refs) == 3
    data["references_lr001"] = refs
    data["references_seed2026"] = [
        {key: run[key] for key in ("job_id", "method", "lr", "beta", "nesterov", "seed",
                                  "state", "elapsed", "final_val", "val", "config",
                                  "optimizer_source_hashes", "log_sha256")}
        for run in json.loads(path.read_text())["runs"] if run["job_id"] == "4346667"
    ]


def same_step_differences(points, reference):
    baseline = {point["step"]: point["loss"] for point in reference}
    assert len(baseline) == len(reference), "duplicate reference validation step"
    assert len({point["step"] for point in points}) == len(points), "duplicate validation step"
    return [dict(step=point["step"], loss=point["loss"] - baseline[point["step"]])
            for point in points if point["step"] in baseline]


def plot(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    series = point_series(data)
    base = series[0][1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, values, color in series:
        steps = list(values)
        axes[0].plot(steps, list(values.values()), color=color, lw=1.8, marker="o", ms=3, label=label)
        if values is not base:
            axes[1].plot(steps, [values[s] - base[s] for s in steps], color=color, lw=1.8, marker="o", ms=3)
    axes[0].set(title="Validation loss", ylabel="Loss")
    axes[1].set(title="Difference from Nesterov", ylabel="Loss minus same-step control")
    axes[1].axhline(0, color=series[0][2], lw=1)
    for ax in axes:
        ax.set(xlabel="Training iteration", xticks=[200, 1000, 2000, 3000])
        ax.grid(alpha=.17)
    fig.suptitle("Width 256 | Frobenius sphere | 786M-token pilots | seed 1234", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.26, top=.84, wspace=.27)
    save(fig, "point_validation_3000")

    grid = [run for run in data["runs"] if run["batch"] == "point_lr_batch_20261102"
            and run["config"]["exit_interval"] is None]
    colors = {.01: "#294C85", .012: "#008378", .015: "#B36D19"}
    refs = data["references_lr001"]
    for kind in ("validation", "training"):
        train_values = [p["loss"] for run in grid for p in run["train"] if p["step"] >= 1000]
        limits = (3.43, 4.15) if kind == "validation" else (min(train_values) - .02, max(train_values) + .02)
        fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), sharey=True)
        for ax, (method, name) in zip(axes, METHODS.items()):
            for run in grid:
                if run["method"] != method:
                    continue
                points = run["val"] if kind == "validation" else [p for p in run["train"] if p["step"] >= 1000]
                running = run["state"] != "COMPLETED"
                ax.plot([p["step"] * 262144 / 1e9 for p in points], [p["loss"] for p in points],
                        color=colors[run["lr"]], ls="--" if running else "-", lw=1.5,
                        label=f"LR {run['lr']:.3f}" + (" (running)" if running else ""))
            if kind == "validation":
                ref = next(r for r in refs if r["method"] == method)
                ax.plot([p["step"] * 262144 / 1e9 for p in ref["val"]], [p["loss"] for p in ref["val"]],
                        color=colors[.01], lw=1.5, label="LR 0.010 (reference)")
            ax.set(title=name, xlabel="Processed tokens (billions)", xlim=(.25, 3.03), ylim=limits)
            ax.grid(alpha=.17)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        axes[0].set_ylabel("Validation loss" if kind == "validation" else "Training loss (10-step logged mean)")
        fig.suptitle(f"Width 256 | beta=0.95 Nesterov | 3B schedule | {kind.capitalize()}", fontsize=12)
        footnote = ("Dashed curves stop at the snapshot; running cells have no final result."
                    if any(run["state"] != "COMPLETED" for run in grid)
                    else "All six new LR cells completed the full 3B schedule.")
        fig.text(.5, .015, footnote
                 + (" No LR=0.010 training trace is included." if kind == "training" else ""),
                 ha="center", fontsize=9)
        fig.subplots_adjust(bottom=.17, top=.84, wspace=.12)
        save(fig, "lr_grid_" + kind)

    adaptive = [run for run in data["runs"] if run["batch"] == "norunit_seeds_20261104"
                and run["config"]["training"]["seed"] == 1234 and run["val"]]
    if adaptive:
        control = next(run for run in refs if run["method"] == "manifold_mcsd")
        curves = [(METHODS[run["method"]], run["val"], color, False) for run, color in
                  zip(sorted(refs, key=lambda r: r["method"]), ("#294C85", "#808080", "#008378"))]
        for run in adaptive:
            row = run["config"]["numerics"].get("direction_adaptation") == "normuon"
            curves.append(("MCSD-NorUnit" if row else "MCSD-unitF", run["val"],
                           "#B36D19" if row else "#A04476", run["state"] != "COMPLETED"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        visible_differences = []
        for label, points, color, running in curves:
            suffix = " (running)" if running else ""
            axes[0].plot([p["step"] * 262144 / 1e9 for p in points], [p["loss"] for p in points],
                         label=label + suffix, color=color, ls="--" if running else "-", lw=1.6)
            delta = same_step_differences(points, control["val"])
            visible_differences.extend(p["loss"] for p in delta if p["step"] * 262144 / 1e9 >= .25)
            axes[1].plot([p["step"] * 262144 / 1e9 for p in delta], [p["loss"] for p in delta],
                         color=color, ls="--" if running else "-", lw=1.6)
        axes[0].set(title="Validation loss", ylabel="Loss", ylim=(3.43, 4.15))
        axes[1].set(title="Same-step difference from standard MCSD", ylabel="Loss difference")
        axes[1].set_ylim(min(visible_differences) - .002, max(visible_differences) + .002)
        for ax in axes:
            ax.set(xlabel="Processed tokens (billions)", xlim=(.25, 3.03))
            ax.grid(alpha=.17)
        fig.suptitle("Width 256 | LR 0.010 | beta=0.95 Nesterov | seed 1234", fontsize=12)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=.24, top=.84, wspace=.25)
        save(fig, "normalization_validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--project", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--include-register", action="append", default=[],
                        help="additional register basename without _register.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    data = collect(args.project, args.include_register) if args.project else json.loads(args.data.read_text())
    if args.references:
        add_references(data, args.references)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.plot:
        plot(data, args.out)
    for run in data["runs"]:
        if run["batch"] in ("point_lr_batch_20261102", "norunit_seeds_20261104"):
            print(run["job_id"], run["state"], run["last_train"]["step"] if run["last_train"] else None, run["last_val"],
                  run["elapsed"], run["counters"])


if __name__ == "__main__":
    main()
