"""Verify and summarize the full 3B norm-band A/B; redraw without H20 access."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics as stats
from pathlib import Path


STOP = 11445
WINDOW = (*range(10000, 11401, 200), STOP)
VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\deE.+-]+)")
TRAIN = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?\blm loss:\s*([\deE.+-]+)")
ARMS = {4487881: ("A: band [0.8, 1.2]", "#c58027", 4479799, "18:35:59"),
        4487882: ("B: band [0.9, 1.1], cone", "#a54c70", 4479800, "18:40:47")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effect(reference, treatment):
    if set(reference) != set(treatment) or max(reference) != STOP:
        raise ValueError("expected identical full-budget validation grids")
    delta = {s: treatment[s] - reference[s] for s in sorted(reference)}
    wins = [s for s, d in delta.items() if d < 0]
    return dict(deltas=delta, endpoint_delta=delta[STOP],
                late_mean=stats.mean(treatment[s] for s in WINDOW),
                late_mean_delta=stats.mean(delta[s] for s in WINDOW),
                late_wins=sum(delta[s] < 0 for s in WINDOW), late_points=len(WINDOW),
                last_logged_win=max(wins, default=None))


def check_protocol(candidate, reference, extra=()):
    fields = ("model", "numerics", "initialization", "geometry", "train_iters", "warmup_iters",
              "train_prefix", "valid_prefix", "tokenizer", "gpus")
    if any(candidate.get(k) != reference.get(k) for k in fields):
        raise ValueError("model/data/numerical protocols differ")
    if {k: v for k, v in candidate["training"].items() if k not in extra} != reference["training"]:
        raise ValueError("training recipes differ")


def verify_arrays(directory, steps, reference_layout, lower, upper):
    import numpy as np

    meta = json.loads((directory / "aus_trace.json").read_text())
    norm_meta = json.loads((directory / "matrix_norms.json").read_text())
    progress = json.loads((directory / "band_trace_progress.json").read_text())
    aus = np.load(directory / "aus.npy", mmap_mode="r", allow_pickle=False)
    norms = np.load(directory / "matrix_norms.npy", mmap_mode="r", allow_pickle=False)
    if (meta["steps"] != steps or meta["matrices"] != reference_layout or len(reference_layout) != 196
            or aus.shape != (steps, 196) or norms.shape != (steps, 196, 2)
            or aus.dtype != np.float32 or norms.dtype != np.float32
            or norm_meta["fields"] != ["before_update", "after_return"]
            or norm_meta["steps"] != steps or norm_meta["band"] != [lower, upper]
            or norm_meta["weight_storage"] != "FP32 master"
            or progress["steps_written"] != steps or not progress["completed"]):
        raise ValueError("wrong or incomplete trace layout")
    if (digest(directory / "aus.npy") != meta["sha256"]
            or digest(directory / "matrix_norms.npy") != norm_meta["sha256"]
            or digest(directory / "aus_trace.json") != norm_meta["layout_sha256"]):
        raise ValueError("trace checksum mismatch")
    radii = np.array([r["radius"] for r in reference_layout], dtype=np.float64)
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("invalid radii")
    if (not np.isfinite(aus).all() or (aus < 0).any() or (aus >= math.sqrt(2)).any()
            or not np.isfinite(norms).all() or (norms <= 0).any()):
        raise ValueError("invalid trace measurement")
    ratios = norms.astype(np.float64) / radii[None, :, None]
    if (ratios < lower - 2e-6).any() or (ratios > upper + 2e-6).any():
        raise ValueError("infeasible saved norm")
    continuity = float(np.max(np.abs(norms[1:, :, 0] - norms[:-1, :, 1]) / radii[None, :]))
    if continuity > 2e-6:
        raise ValueError("between-step norm continuity failed")
    return aus, ratios, dict(steps=steps, logical_matrices=196, aus_entries=int(aus.size),
                            norm_entries=int(norms.size), max_relative_continuity_error=continuity,
                            aus_sha256=meta["sha256"], norms_sha256=norm_meta["sha256"])


def read_run(root, job):
    dirs = [f.parent for f in root.rglob("job_id.txt") if f.read_text().strip().split(";")[0] == str(job)]
    logs = list(root.rglob(f"*_{job}.out"))
    if len(dirs) != 1 or len(logs) != 1:
        raise ValueError(f"expected one receipt/log for {job}")
    directory, logfile = dirs[0], logs[0]
    cfg = json.loads((directory / "resolved_config.json").read_text())
    receipt = json.loads((directory / "completion.json").read_text())
    log = logfile.read_text()
    if receipt["exit_code"] != 0 or receipt["training_exit_code"] != 0:
        raise ValueError("unsuccessful training process")
    health = {}
    for key in ("nan", "skipped"):
        counts = re.findall(rf"number of {key} iterations:\s*(\d+)", log)
        if not counts or max(map(int, counts)):
            raise ValueError("missing or nonzero health counters")
        health[key] = 0
    values = {int(s): float(v) for s, v in VAL.findall(log)}
    if set(values) != {*range(200, 11401, 200), STOP}:
        raise ValueError("incomplete full-budget validation grid")
    return directory, cfg, dict(validation=values, train={int(s): float(v) for s, v in TRAIN.findall(log)},
                               process_seconds=receipt["process_seconds"], training_health=health,
                               log_sha256=digest(logfile))


def collect(snapshot, pilot_file, ledger_file):
    import numpy as np

    pilots = json.loads(pilot_file.read_text())["runs"]
    ref_dir, ref_cfg, ref = read_run(snapshot, 4427413)
    ref.update(label="MuonH sphere", color="#56616e", role="historical verified reference")
    ref["effect"] = effect(ref["validation"], ref["validation"])
    ref_meta = json.loads((ref_dir / "aus_trace.json").read_text())
    ref_aus = np.load(ref_dir / "aus.npy", mmap_mode="r", allow_pickle=False)
    if ref_aus.shape != (STOP, 196) or digest(ref_dir / "aus.npy") != ref_meta["sha256"]:
        raise ValueError("invalid reference AUS trace")
    runs, all_rows = {4427413: ref}, []
    for job, (label, color, pilot, elapsed) in ARMS.items():
        directory, cfg, run = read_run(snapshot, job)
        check_protocol(cfg, ref_cfg)
        for name, expected in ref_cfg["source_hashes"].items():
            if (name.startswith("Megatron-LM/megatron/core/models/gpt/")
                    or Path(name).name in ("manifold_polar.py", "manifold_geometry.py", "manifold_layout.py")):
                if cfg["source_hashes"].get(name) != expected:
                    raise ValueError("model/geometry/polar source changed from the reference")
        band = cfg["frobenius_band"]
        if cfg["exit_interval"] or cfg["train_iters"] != STOP or not band["record_every_step"]:
            raise ValueError("not a full-budget, every-step band run")
        prefix = {int(s): v for s, v in pilots[str(pilot)]["validation"].items()}
        prefix_error = max(abs(run["validation"][s] - v) for s, v in prefix.items())
        if prefix_error:
            raise ValueError("observer changed the pilot's validation prefix")
        aus, norms, validation = verify_arrays(directory, STOP, ref_meta["matrices"], band["lower"], band["upper"])
        norm_q = np.quantile(norms[:, :, 1], [0, .1, .5, .9, 1], axis=1)
        aus_q = np.quantile(aus, [.1, .5, .9], axis=1)
        paired = np.divide(aus, ref_aus, out=np.full(aus.shape, np.nan, dtype=np.float64),
                           dtype=np.float64, where=ref_aus > 1e-12)
        ratio_q = np.full((3, STOP), np.nan)
        valid_steps = np.any(np.isfinite(paired), axis=1)
        ratio_q[:, valid_steps] = np.nanquantile(paired[valid_steps], [.1, .5, .9], axis=1)
        occupancy = np.mean(abs(norms[:, :, 1] - band["upper"]) <= 2e-6, axis=1)
        rows = []
        for i in range(STOP):
            row = dict(job=job, step=i+1, **dict(zip(("norm_min", "norm_p10", "norm_median", "norm_p90", "norm_max"), map(float, norm_q[:, i]))),
                       **dict(zip(("aus_p10", "aus_median", "aus_p90"), map(float, aus_q[:, i]))),
                       **dict(zip(("aus_ratio_p10", "aus_ratio_median", "aus_ratio_p90"),
                                  [float(x) if math.isfinite(x) else None for x in ratio_q[:, i]])),
                       upper_bound_fraction=float(occupancy[i]))
            rows.append(row)
        diagnostic = [json.loads(line) for line in (directory / "band_diagnostics.jsonl").read_text().splitlines()]
        angles = []
        for s in sorted({r["step"] for r in diagnostic}):
            sample = [r for r in diagnostic if r["step"] == s]
            if len(sample) != 196 or len({(r["parameter"], r["component"]) for r in sample}) != 196:
                raise ValueError("incomplete/duplicate sampled matrix diagnostics")
            angles.append(dict(step=s, **{k: stats.median(r[k] for r in sample if r[k] is not None)
                for k in ("grad_angle", "momentum_angle", "source_angle", "lmo_input_angle", "polar_angle")}))
        clip = [json.loads(line) for line in (directory / "band_summary.jsonl").read_text().splitlines()]
        if clip[-1]["step"] != 11400 or clip[-1]["component_updates"] != 11400 * 196:
            raise ValueError("unexpected sampled clip-count endpoint")
        groups = {}
        for component in sorted({r["label"] for r in ref_meta["matrices"]}):
            ids = [i for i, r in enumerate(ref_meta["matrices"]) if r["label"] == component]
            groups[component] = dict(norm_median=float(np.median(norms[-1, ids, 1])),
                                     aus_ratio_median=float(np.nanmedian(paired[-1, ids])))
        source_names = {"manifold_band_trace.py", "manifold_frobenius_band.py", "manifold_baselines.py", "manifold_polar.py"}
        run.update(label=label, color=color, band=band, slurm_elapsed=elapsed,
                   effect=effect(ref["validation"], run["validation"]), trace_validation=validation,
                   pilot_prefix_points=len(prefix), pilot_prefix_max_logged_error=prefix_error,
                   trajectory_summary=[r for r in rows if r["step"] in (1, 229, STOP) or r["step"] % 100 == 0],
                   final_groups=groups, angles=angles, clip_counts=clip,
                   source_commit=cfg["source_commit"],
                   source_hashes={k: v for k, v in cfg["source_hashes"].items() if Path(k).name in source_names})
        runs[job] = run
        all_rows.extend(rows)
    ledger = json.loads(ledger_file.read_text())
    baseline, = [r for r in ledger["runs"] if r["job_id"] == "4346666"]
    check_protocol(baseline["config"], ref_cfg, extra=("normuon_beta2", "normuon_eps"))
    curve = {r["step"]: r["loss"] for r in baseline["val"]}
    runs[4346666] = dict(label="NorMuon-Sphere", color="#277da1", validation=curve,
                        effect=effect(ref["validation"], curve), slurm_elapsed=baseline["elapsed"],
                        source_commit=baseline["source_commit"], role="historical matched recipe, different source snapshot")
    protocol = {k: ref_cfg[k] for k in ("model", "training", "numerics", "initialization", "geometry", "train_iters", "warmup_iters")}
    return dict(snapshot_date="2026-11-25", steps=STOP, processed_tokens=STOP*128*2048,
                window_steps=WINDOW, runs=runs, protocol=protocol, reference_job=4427413,
                effective_polar_steps=10, trace_scope="Every-step summaries released; full per-matrix traces retained in H20 and local archive, not embedded in Git.",
                aus_definition="FP32 master returned-iterate unit chord, not nominal LR or radians"), all_rows


def plots(report, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42})
    runs = {int(k): v for k, v in report["runs"].items()}

    def tidy(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=.7, color="#e2e5e8")
        ax.set_xlabel("Processed tokens (billions)")

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.1), layout="constrained")
    for job in (4427413, 4346666, *ARMS):
        r = runs[job]
        vals = {int(s): v for s, v in r["validation"].items()}
        steps = sorted(vals)
        x = [s*128*2048/1e9 for s in steps]
        axes[0].plot(x, [vals[s] for s in steps], lw=1.6, color=r["color"], label=r["label"])
        if job != 4427413:
            ds = {int(s): v for s, v in r["effect"]["deltas"].items()}
            axes[1].plot(x, [1000*ds[s] for s in steps], lw=1.5, color=r["color"])
    axes[0].set(title="Full-budget validation", ylabel="Validation cross-entropy")
    axes[1].axhline(0, color="#56616e", lw=.9)
    axes[1].axvspan(10000*128*2048/1e9, STOP*128*2048/1e9, color="#edf0f3", zorder=0)
    axes[1].set(title="Difference from MuonH", ylabel="Validation loss difference (x 0.001)")
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Full 3B norm-band A/B | width 256, seed 1234, peak LR 0.010", fontsize=12)
    save(fig, "validation_full_3b")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), layout="constrained")
    for job in ARMS:
        r = runs[job]
        rows = r["trajectory_summary"]
        x = [s["step"]*128*2048/1e9 for s in rows]
        axes[0].plot(x, [s["norm_median"] for s in rows], color=r["color"], label=r["label"])
        axes[0].fill_between(x, [s["norm_p10"] for s in rows], [s["norm_p90"] for s in rows], color=r["color"], alpha=.15)
        later = [s for s in rows if s["step"] >= 229]
        axes[1].plot([s["step"]*128*2048/1e9 for s in later], [s["aus_ratio_median"] for s in later], color=r["color"])
    axes[0].axhline(1, color="#56616e", lw=.9, label="Sphere reference")
    axes[0].set(title="Returned norm / reference R", ylabel="Matrix median and P10-P90")
    axes[1].axhline(1, color="#56616e", lw=.9)
    axes[1].set(title="Actual AUS / MuonH AUS", ylabel="Paired logical-matrix median", ylim=(.8, 1.03))
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Every-step measurements, displayed every 100 steps | FP32 master weights", fontsize=12)
    save(fig, "norm_and_aus_full_3b")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), layout="constrained")
    for job in (4427413, *ARMS):
        r = runs[job]
        tr = sorted((int(s), v) for s, v in r["train"].items())
        # A trailing 20-log-point average is about 200 updates, not validation.
        smooth = [(s, stats.mean(v for _, v in tr[max(0,i-19):i+1])) for i, (s, _) in enumerate(tr) if i >= 19]
        axes[0].plot([s*128*2048/1e9 for s, _ in smooth], [v for _, v in smooth], color=r["color"], lw=1.5, label=r["label"])
    axes[0].set(title="Training loss, trailing 20 logged points", ylabel="Smoothed mini-batch cross-entropy")
    for job in ARMS:
        r = runs[job]
        angles = r["angles"]
        axes[1].plot([s["step"]*128*2048/1e9 for s in angles], [s["polar_angle"] for s in angles], color=r["color"], lw=1.5)
    axes[1].set(title="msign(Q) before final cone correction", ylabel="Median angle to iso-norm tangent plane (degrees)")
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "training_and_polar_angles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--results", type=Path)
    parser.add_argument("--pilot-results", type=Path)
    parser.add_argument("--baseline-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot and (not args.pilot_results or not args.baseline_ledger):
        parser.error("collection requires --pilot-results and --baseline-ledger")
    report, rows = collect(args.snapshot, args.pilot_results, args.baseline_ledger) if args.snapshot else (json.loads(args.results.read_text()), None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, field in (("validation", "validation"), ("training", "train")):
        with (args.output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("job", "step", "tokens", "loss"))
            for job, run in report["runs"].items():
                writer.writerows((job, s, int(s)*128*2048, v) for s, v in run.get(field, {}).items())
    if rows is not None:
        with (args.output / "every_step_summaries.csv.gz").open("wb") as f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
                with io.TextIOWrapper(g, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    plots(report, args.output)
    print(json.dumps({j: dict(label=r["label"], endpoint=r["validation"].get(STOP, r["validation"].get(str(STOP))),
                             late_mean=r["effect"]["late_mean"], endpoint_delta=r["effect"]["endpoint_delta"],
                             last_logged_win=r["effect"]["last_logged_win"])
                      for j, r in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
