"""Verified full-budget B AUS-control readout; raw scalar arrays stay off Git."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics as stats
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import report_band_aus_progress_20261125 as prefix


STOP = 11445
WINDOW = (*range(10000, 11401, 200), STOP)
FIELDS = ["target_chord", "effective_lr", "nominal_lr", "ray_radial_cosine"]


def require_complete(state, receipt, band_progress, replay_progress, stop):
    if (state != "COMPLETED" or receipt["exit_code"] != 0 or receipt["training_exit_code"] != 0
            or not band_progress["completed"] or not replay_progress["completed"]
            or prefix.safe_prefix(band_progress, replay_progress, stop) != stop):
        raise ValueError("not a successful complete-budget replay")


def export(project, output):
    import numpy as np

    directory, = [f.parent for f in (project/"results/manifold").glob("*/job_id.txt")
                   if f.read_text().strip() == str(prefix.JOB)]
    cfg = json.loads((directory/"resolved_config.json").read_text())
    band = cfg["frobenius_band"]
    reference = Path(band["aus_reference_dir"])
    ref_meta = json.loads((reference/"aus_trace.json").read_text())
    if (cfg["train_iters"] != STOP or cfg["exit_interval"] is not None
            or prefix.digest(reference/"resolved_config.json") != band["reference_config_sha256"]
            or prefix.digest(reference/"aus.npy") != band["reference_aus_sha256"]):
        raise ValueError("wrong horizon or changed pinned reference")
    receipt = json.loads((directory/"completion.json").read_text())
    if (directory/"aus_replay_failure.json").exists():
        raise ValueError("a replay failure was recorded")
    bp = json.loads((directory/"band_trace_progress.json").read_text())
    rp = json.loads((directory/"aus_replay_progress.json").read_text())
    state = subprocess.check_output(["sacct", "-X", "-n", "-j", str(prefix.JOB),
        "--format=State,Elapsed,Start,End", "-P"], text=True).strip().splitlines()[0].split("|")
    require_complete(state[0], receipt, bp, rp, STOP)
    meta = json.loads((directory/"aus_trace.json").read_text())
    norm_meta = json.loads((directory/"matrix_norms.json").read_text())
    replay_meta = json.loads((directory/"aus_replay.json").read_text())
    if (meta["steps"] != STOP or meta["matrices"] != ref_meta["matrices"] or len(meta["matrices"]) != 196
            or norm_meta["steps"] != STOP or norm_meta["fields"] != ["before_update", "after_return"]
            or norm_meta["band"] != [.9, 1.1] or norm_meta["weight_storage"] != "FP32 master"
            or norm_meta["layout_sha256"] != prefix.digest(directory/"aus_trace.json")
            or replay_meta["steps"] != STOP or replay_meta["fields"] != FIELDS
            or replay_meta["reference_sha256"] != band["reference_aus_sha256"]
            or replay_meta["tolerance"] != 2e-6):
        raise ValueError("completed trace metadata/layout mismatch")
    for filename, expected in (("aus.npy", meta["sha256"]), ("matrix_norms.npy", norm_meta["sha256"]),
                               ("aus_replay_steps.npy", replay_meta["sha256"])):
        if prefix.digest(directory/filename) != expected:
            raise ValueError("completed trace checksum mismatch")
    arrays = {"aus": np.load(directory/"aus.npy", allow_pickle=False),
              "norms": np.load(directory/"matrix_norms.npy", allow_pickle=False),
              "replay": np.load(directory/"aus_replay_steps.npy", allow_pickle=False)}
    target = np.load(reference/"aus.npy", allow_pickle=False)
    radii = np.array([r["radius"] for r in meta["matrices"]], dtype=np.float64)
    checked = prefix.verify_prefix(arrays["aus"], arrays["norms"], arrays["replay"], target, radii, .9, 1.1)
    if (not math.isclose(checked["max_abs_error"], replay_meta["max_abs_error"], abs_tol=1e-12)
            or not math.isclose(checked["rms_error"], replay_meta["rms_error"], abs_tol=1e-12)):
        raise ValueError("replay receipt errors disagree with full arrays")
    logfile, = list((project/"logs").glob(f"*_{prefix.JOB}.out"))
    stdout = logfile.read_bytes()
    log = stdout.decode()
    values = {int(s): float(v) for s, v in prefix.VAL.findall(log)}
    if set(values) != {*range(200, 11401, 200), STOP}:
        raise ValueError("missing full-budget validation grid")
    health = {}
    for k in ("nan", "skipped"):
        counts = re.findall(rf"number of {k} iterations:\s*(\d+)", log)
        if not counts or max(map(int, counts)):
            raise ValueError("missing or nonzero training health counters")
        health[k] = 0
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"prefix_arrays.npz", **arrays, radii=radii)
    (output/"stdout.out").write_bytes(stdout)
    (output/"matrix_layout.json").write_text(json.dumps(ref_meta, indent=2)+"\n")
    data = dict(observed_at=dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        job=prefix.JOB, slurm_state=state[0], slurm_elapsed=state[1], started_at=state[2], ended_at=state[3],
        planned_steps=STOP, verified_prefix=STOP, validation=values,
        last_logged_step=max(map(int, re.findall(r"iteration\s+(\d+)\s*/", log))),
        training={int(s): float(v) for s, v in prefix.TRAIN.findall(log)}, health=health,
        prefix_checks=checked, band_progress=bp, replay_progress=rp, completion=receipt,
        source_commit=cfg["source_commit"], reference_job=4427413,
        reference_config_sha256=band["reference_config_sha256"], reference_aus_sha256=band["reference_aus_sha256"],
        arrays_sha256=prefix.digest(output/"prefix_arrays.npz"), stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        protocol={k: cfg[k] for k in ("model", "training", "numerics", "initialization", "geometry", "warmup_iters")},
        band_bounds=[.9, 1.1], replay_rule=band["replay_rule"], completed=True,
        raw_array_hashes={k: v for k, v in (("aus", meta["sha256"]), ("norms", norm_meta["sha256"]), ("replay", replay_meta["sha256"]))},
        note="Complete 3B control; all raw-array checksums and completion receipts verified.")
    (output/"snapshot.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), final=values[STOP], checks=checked)))


def compare_curves(reference, treatment):
    if set(reference) != set(treatment) or max(reference) != STOP:
        raise ValueError("expected full matched validation grids")
    delta = {s: treatment[s]-reference[s] for s in sorted(reference)}
    late = [delta[s] for s in WINDOW]
    return dict(endpoint_delta=delta[STOP], late_mean=stats.mean(treatment[s] for s in WINDOW),
                late_mean_delta=stats.mean(late), late_wins=sum(x < 0 for x in late),
                late_points=len(late), late_delta_rms=math.sqrt(stats.mean(x*x for x in late)),
                late_delta_max_abs=max(map(abs, late)), deltas=delta)


def analyze(snapshot, baselines):
    import numpy as np

    data = prefix.analyze(snapshot, baselines)
    if not data.get("completed") or data["slurm_state"] != "COMPLETED" or data["verified_prefix"] != STOP:
        raise ValueError("an intermediate prefix is not a final result")
    ref = {int(s): v for s, v in data["runs"][4427413]["validation"].items()}
    for j, run in data["runs"].items():
        curve = {int(s): v for s, v in run["validation"].items()}
        run["comparison"] = compare_curves(ref, curve)
    old = json.loads(baselines.read_text())["runs"]
    for j in (4427413, 4487882):
        data["runs"][j]["training"] = old[str(j)]["train"]
    data["runs"][prefix.JOB]["training"] = data["training"]
    data["window_steps"] = WINDOW
    data["baseline_scope"] = "Completed historical runs on the full matched validation grid; single seed, not a newly randomized comparison."
    data["replay_minus_original_b_final"] = data["runs"][prefix.JOB]["latest_loss"]-data["runs"][4487882]["latest_loss"]
    data["replay_minus_original_b_late"] = data["runs"][prefix.JOB]["comparison"]["late_mean"]-data["runs"][4487882]["comparison"]["late_mean"]
    train = {int(s): v for s, v in data["training"].items()}
    common = sorted(s for s in train if all(str(s) in data["runs"][j]["training"] for j in (4427413, 4487882)))[-20:]
    data["last_20_common_train_steps"] = common
    data["last_20_train_means"] = {j: stats.mean(float(data["runs"][j]["training"].get(str(s), data["runs"][j]["training"].get(s))) for s in common)
                                   for j in (4427413, 4487882, prefix.JOB)}
    with np.load(snapshot/"prefix_arrays.npz", allow_pickle=False) as archive:
        ratios = archive["norms"][-1, :, 1].astype(float)/archive["radii"]
        data["final_norm_ratio"] = dict(minimum=float(ratios.min()), median=float(stats.median(ratios)),
                                       maximum=float(ratios.max()), upper_fraction=float((abs(ratios-1.1) < 2e-6).mean()))
    return data


def plots(data, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    runs = {int(j): r for j, r in data["runs"].items()}
    def tidy(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e2e5e8", lw=.7)
    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{name}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for j in (4427413, 4346666, 4487882, prefix.JOB):
        r = runs[j]
        curve = {int(s): v for s, v in r["validation"].items()}
        steps = sorted(curve)
        x = [s*128*2048/1e9 for s in steps]
        axes[0].plot(x, [curve[s] for s in steps], color=r["color"], lw=1.5, label=r["label"])
        if j != 4427413:
            ds = {int(s): v for s, v in r["comparison"]["deltas"].items()}
            axes[1].plot(x, [1000*ds[s] for s in steps], color=r["color"], lw=1.5)
    axes[0].set(title="Completed validation trajectories", ylabel="Cross-entropy")
    axes[1].axhline(0, color="#56616e", lw=.9)
    axes[1].axvspan(10000*128*2048/1e9, STOP*128*2048/1e9, color="#edf0f3", zorder=0)
    axes[1].set(title="Difference from MuonH", ylabel="Loss difference (x 0.001)")
    for ax in axes:
        tidy(ax); ax.set_xlabel("Processed tokens (billions)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Completed 3B B AUS replay | width 256, seed 1234", fontsize=12)
    save(fig, "validation_full_3b")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), layout="constrained")
    ids = (4346666, 4487882, prefix.JOB)
    for ax, field, title in ((axes[0], "endpoint_delta", "Final validation"), (axes[1], "late_mean_delta", "Predeclared nine-point late mean")):
        ax.bar(range(3), [1000*runs[j]["comparison"][field] for j in ids], color=[runs[j]["color"] for j in ids], width=.55)
        ax.axhline(0, color="#56616e", lw=.9)
        ax.set(title=title, xticks=range(3), xticklabels=("NorMuon", "Original B", "B + AUS"), ylabel="Difference from MuonH (x 0.001)")
        tidy(ax)
    save(fig, "endpoint_and_late_window")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for j in (4427413, 4346666, 4487882, prefix.JOB):
        r = runs[j]
        curve = {int(s): v for s, v in r["validation"].items()}
        axes[0].plot(WINDOW, [curve[s] for s in WINDOW], color=r["color"], lw=1.4, marker="o", ms=3, label=r["label"])
        if "training" in r:
            train = {int(s): v for s, v in r["training"].items()}
            x = sorted(train)
            means = [stats.mean(train[s] for s in x[max(0, i-19):i+1]) for i in range(len(x))]
            axes[1].plot([s*128*2048/1e9 for s in x], means, color=r["color"], lw=1.4)
    axes[0].set(title="Predeclared late validation window", xlabel="Optimizer step", ylabel="Cross-entropy")
    axes[0].ticklabel_format(useOffset=False, axis="y")
    axes[1].set(title="Train loss, trailing 20 logged points", xlabel="Processed tokens (billions)", ylabel="Mini-batch cross-entropy", ylim=(3.35, 4.3))
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "late_validation_and_training")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    rows = [r for r in data["summaries"] if r["step"] >= 229]
    axes[0].plot([r["step"] for r in rows], [r["actual_to_target_median"] for r in rows], color="#216c56")
    axes[0].axhline(1, color="#56616e", lw=.8, ls="--")
    axes[0].set(title="Returned AUS / MuonH target", ylabel="Matrix median", ylim=(.99999, 1.00001))
    axes[0].ticklabel_format(axis="y", useOffset=False, style="plain")
    axes[1].plot([r["step"] for r in rows], [r["effective_to_nominal_lr_median"] for r in rows], color="#c58027")
    axes[1].set(title="Solved step coefficient", ylabel="Effective LR / nominal LR, median")
    for ax in axes:
        tidy(ax); ax.set_xlabel("Optimizer step")
    fig.suptitle("All 2,243,220 matrix updates verified | FP32 master weights", fontsize=12)
    save(fig, "replay_precision_and_steps")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-project", type=Path)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--results", type=Path)
    p.add_argument("--baselines", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.export_project:
        export(args.export_project, args.output)
        return
    if args.snapshot and not args.baselines:
        p.error("--snapshot needs completed --baselines")
    data = analyze(args.snapshot, args.baselines) if args.snapshot else json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    for name, key in (("validation", "validation"), ("training", "training")):
        with (args.output/f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("job", "step", "processed_tokens", "loss"))
            for j, run in data["runs"].items():
                writer.writerows((j, s, int(s)*128*2048, v) for s, v in run.get(key, {}).items())
    plots(data, args.output)
    print(json.dumps({j:dict(label=r["label"], final=r["latest_loss"], late_mean=r["comparison"]["late_mean"],
                            endpoint_delta=r["comparison"]["endpoint_delta"], late_delta=r["comparison"]["late_mean_delta"],
                            late_wins=r["comparison"]["late_wins"]) for j,r in data["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
