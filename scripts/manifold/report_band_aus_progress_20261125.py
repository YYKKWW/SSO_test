"""Freeze a safely flushed running prefix, then report against completed runs.

Export runs outside the frozen training tree. No running source or array is
modified. Unwritten tails of preallocated files are never treated as results.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics as stats
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo


JOB = 4504355
VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\deE.+-]+)")
TRAIN = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?\blm loss:\s*([\deE.+-]+)")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_prefix(band, replay, planned):
    values = [band["steps_written"], replay["steps_written"]]
    if any(type(x) is not int or not 0 < x <= planned for x in values):
        raise ValueError("invalid published trace progress")
    return min(values)


def verify_prefix(aus, norms, replay, target, radii, lower, upper):
    import numpy as np

    steps, matrices = target.shape
    if (aus.shape != (steps, matrices) or norms.shape != (steps, matrices, 2)
            or replay.shape != (steps, matrices, 4) or len(radii) != matrices
            or any(x.dtype != np.float32 for x in (aus, norms, replay, target))
            or any(not np.isfinite(x).all() for x in (aus, norms, replay, target, radii))
            or (aus < 0).any() or (target < 0).any() or (target >= np.sqrt(2)).any()
            or (norms <= 0).any() or (radii <= 0).any() or (replay[:, :, 1:3] < 0).any()
            or not np.array_equal(replay[:, :, 0], target)):
        raise ValueError("invalid prefix layout, values or reference targets")
    ratios = norms.astype(np.float64) / radii[None, :, None]
    if (ratios < lower-2e-6).any() or (ratios > upper+2e-6).any():
        raise ValueError("FP32 norm outside the annulus")
    error = aus.astype(np.float64) - target.astype(np.float64)
    maximum = float(np.abs(error).max())
    continuity = float(np.max(np.abs(norms[1:, :, 0]-norms[:-1, :, 1])/radii)) if steps > 1 else 0.
    if maximum > 2e-6 or continuity > 2e-6:
        raise ValueError("AUS mismatch or norm discontinuity")
    return dict(steps=steps, matrices=matrices, matrix_decisions=steps*matrices,
                max_abs_error=maximum, rms_error=float(np.sqrt(np.mean(error**2))),
                max_norm_continuity_error=continuity)


def export(project, output):
    import numpy as np

    directory, = [f.parent for f in (project/"results/manifold").glob("*/job_id.txt")
                   if f.read_text().strip() == str(JOB)]
    cfg = json.loads((directory/"resolved_config.json").read_text())
    band = cfg["frobenius_band"]
    reference = Path(band["aus_reference_dir"])
    layout = json.loads((reference/"aus_trace.json").read_text())
    if (digest(reference/"resolved_config.json") != band["reference_config_sha256"]
            or digest(reference/"aus.npy") != band["reference_aus_sha256"]
            or layout["sha256"] != band["reference_aus_sha256"] or len(layout["matrices"]) != 196):
        raise ValueError("pinned reference changed")
    b = json.loads((directory/"band_trace_progress.json").read_text())
    r = json.loads((directory/"aus_replay_progress.json").read_text())
    stop = safe_prefix(b, r, cfg["train_iters"])
    arrays = {"aus": np.load(directory/"aus.npy", mmap_mode="r", allow_pickle=False)[:stop].copy(),
              "norms": np.load(directory/"matrix_norms.npy", mmap_mode="r", allow_pickle=False)[:stop].copy(),
              "replay": np.load(directory/"aus_replay_steps.npy", mmap_mode="r", allow_pickle=False)[:stop].copy()}
    targets = np.load(reference/"aus.npy", mmap_mode="r", allow_pickle=False)[:stop].copy()
    radii = np.array([x["radius"] for x in layout["matrices"]], dtype=np.float64)
    checked = verify_prefix(arrays["aus"], arrays["norms"], arrays["replay"], targets, radii, band["lower"], band["upper"])
    logfile, = list((project/"logs").glob(f"*_{JOB}.out"))
    stdout = logfile.read_bytes()
    log = stdout.decode()
    health = {}
    for k in ("nan", "skipped"):
        counts = re.findall(rf"number of {k} iterations:\s*(\d+)", log)
        if not counts or max(map(int, counts)):
            raise ValueError("missing or nonzero training health counter")
        health[k] = 0
    if (directory/"aus_replay_failure.json").exists():
        raise ValueError("a replay failure was recorded")
    state = subprocess.check_output(["sacct", "-X", "-n", "-j", str(JOB), "--format=State,ElapsedRaw", "-P"], text=True).strip().splitlines()[0].split("|")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"prefix_arrays.npz", **arrays, radii=radii)
    (output/"stdout.out").write_bytes(stdout)
    (output/"matrix_layout.json").write_text(json.dumps(layout, indent=2)+"\n")
    data = dict(observed_at=dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                job=JOB, slurm_state=state[0], elapsed_seconds=int(state[1]),
                planned_steps=cfg["train_iters"], verified_prefix=stop,
                last_logged_step=max(map(int, re.findall(r"iteration\s+(\d+)\s*/", log))),
                validation={int(s): float(v) for s, v in VAL.findall(log) if int(s) <= stop},
                training={int(s): float(v) for s, v in TRAIN.findall(log) if int(s) <= stop},
                health=health, prefix_checks=checked, band_progress=b, replay_progress=r,
                source_commit=cfg["source_commit"], reference_job=4427413,
                reference_config_sha256=band["reference_config_sha256"], reference_aus_sha256=band["reference_aus_sha256"],
                arrays_sha256=digest(output/"prefix_arrays.npz"), stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                protocol={k: cfg[k] for k in ("model", "training", "numerics", "initialization", "geometry", "warmup_iters")},
                replay_rule=band["replay_rule"], band_bounds=[band["lower"], band["upper"]],
                note="Immutable snapshot of the minimum published progress prefix; not a full 3B result.")
    (output/"snapshot.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), observed_at=data["observed_at"], verified_prefix=stop,
                         last_validation=max(data["validation"]), state=data["slurm_state"], checks=checked)))


def analyze(snapshot, baselines):
    import numpy as np

    data = json.loads((snapshot/"snapshot.json").read_text())
    if digest(snapshot/"prefix_arrays.npz") != data["arrays_sha256"]:
        raise ValueError("prefix archive changed")
    with np.load(snapshot/"prefix_arrays.npz", allow_pickle=False) as archive:
        aus, norms, replay, radii = (archive[k] for k in ("aus", "norms", "replay", "radii"))
    verify_prefix(aus, norms, replay, replay[:, :, 0], radii, *data["band_bounds"])
    completed = json.loads(baselines.read_text())
    if (data["planned_steps"] != completed["steps"] or data["reference_job"] != completed["reference_job"]
            or any(v != completed["protocol"].get(k) for k, v in data["protocol"].items())):
        raise ValueError("completed baselines do not share the planned replay protocol")
    full = completed["runs"]
    vals = {int(s): v for s, v in data["validation"].items()}
    last = max(vals)
    runs = {}
    for j in (4427413, 4487882, 4346666):
        row = full[str(j)]
        runs[j] = dict(label=row["label"], color=row["color"],
                       validation={int(s): v for s, v in row["validation"].items() if int(s) in vals})
        if set(runs[j]["validation"]) != set(vals):
            raise ValueError("baseline validation grid differs")
    runs[JOB] = dict(label="B, MuonH AUS replay", color="#216c56", validation=vals)
    for j, row in runs.items():
        row["latest_loss"] = row["validation"][last]
        row["latest_delta_vs_muonh"] = row["latest_loss"]-runs[4427413]["validation"][last]
    ratios = norms[:, :, 1].astype(np.float64)/radii
    valid = replay[:, :, 2] > 1e-12
    scale = np.divide(replay[:, :, 1], replay[:, :, 2], out=np.full(aus.shape, np.nan), where=valid, dtype=np.float64)
    matched = np.divide(aus, replay[:, :, 0], out=np.full(aus.shape, np.nan), where=replay[:, :, 0] > 1e-12, dtype=np.float64)
    summaries = []
    indices = {0, data["verified_prefix"]-1, *range(99, data["verified_prefix"], 100)}
    if data["verified_prefix"] >= 229:
        indices.add(228)
    for i in sorted(indices):
        summaries.append(dict(step=i+1, norm_p10=float(np.quantile(ratios[i], .1)), norm_median=float(np.median(ratios[i])),
                              norm_p90=float(np.quantile(ratios[i], .9)),
                              actual_to_target_median=float(np.nanmedian(matched[i])) if np.isfinite(matched[i]).any() else None,
                              effective_to_nominal_lr_median=float(np.nanmedian(scale[i])) if np.isfinite(scale[i]).any() else None,
                              error_max=float(np.max(abs(aus[i].astype(np.float64)-replay[i,:,0])))))
    data.update(runs=runs, latest_common_validation=last, latest_common_tokens=last*128*2048,
                processed_tokens_verified=data["verified_prefix"]*128*2048, summaries=summaries,
                effective_polar_steps=10, weight_storage="FP32 master",
                latest_delta_replay_minus_original_b=vals[last]-runs[4487882]["latest_loss"],
                baseline_scope="Completed historical runs truncated to identical validation steps; endpoints are not compared against this prefix.")
    return data


def plots(data, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    runs = {int(j): r for j, r in data["runs"].items()}

    def tidy(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e2e5e8", lw=.7)
        ax.set_xlabel("Optimizer step")

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{name}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4), layout="constrained")
    reference = {int(s): v for s, v in runs[4427413]["validation"].items()}
    for j in (4427413, 4346666, 4487882, JOB):
        r = runs[j]
        vals = {int(s): v for s, v in r["validation"].items()}
        x = sorted(vals)
        axes[0].plot(x, [vals[s] for s in x], color=r["color"], lw=1.6, label=r["label"])
        if j != 4427413:
            axes[1].plot(x, [1000*(vals[s]-reference[s]) for s in x], color=r["color"], lw=1.5)
    axes[0].set(title="Validation, matched steps only", ylabel="Validation cross-entropy")
    axes[1].axhline(0, color="#56616e", lw=.9)
    axes[1].set(title="Difference from MuonH", ylabel="Loss difference (x 0.001)")
    for ax in axes: tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Running B AUS control | width 256, seed 1234 | latest validation {data['latest_common_validation']:,}", fontsize=12)
    save(fig, "validation_prefix")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    rows = [r for r in data["summaries"] if r["step"] >= 229]
    axes[0].plot([r["step"] for r in rows], [r["actual_to_target_median"] for r in rows], color="#216c56")
    axes[0].axhline(1, color="#56616e", lw=.8, ls="--")
    axes[0].set(title="Actual AUS / MuonH target", ylabel="Logical-matrix median", ylim=(.99999, 1.00001))
    axes[0].ticklabel_format(axis="y", useOffset=False, style="plain")
    axes[1].plot([r["step"] for r in rows], [r["effective_to_nominal_lr_median"] for r in rows], color="#c58027")
    axes[1].axhline(1, color="#56616e", lw=.8, ls="--")
    axes[1].set(title="Solved constrained-matrix step", ylabel="Effective LR / nominal LR, median")
    for ax in axes: tidy(ax)
    fig.suptitle(f"Verified flushed prefix through {data['verified_prefix']:,} | FP32 returned-iterate chord", fontsize=12)
    save(fig, "replay_precision_and_step")


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
        p.error("--snapshot needs the completed --baselines report")
    data = analyze(args.snapshot, args.baselines) if args.snapshot else json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    with (args.output/"validation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("job", "step", "processed_tokens", "validation_loss"))
        for j, row in data["runs"].items():
            writer.writerows((j, s, int(s)*128*2048, v) for s, v in row["validation"].items())
    plots(data, args.output)
    print(json.dumps(dict(observed_at=data["observed_at"], prefix=data["verified_prefix"], latest_validation=data["latest_common_validation"],
                         rows={j:r["latest_loss"] for j,r in data["runs"].items()}, checks=data["prefix_checks"]), indent=2))


if __name__ == "__main__":
    main()
