"""Collect or redraw the completed November 20 Frobenius-annulus pilot.

Collection uses an external archive of logs/receipts, not model checkpoints.
Published JSON contains summaries; the accompanying compressed CSV preserves
all sampled logical-matrix measurements. No CUDA or training imports are used.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import statistics as stats
from pathlib import Path


ARMS = {
    4479797: ("sphere_control", "Sphere / native MuonH", "#56616e", "04:50:47"),
    4479798: ("band10_ambient", "Band [0.9, 1.1]", "#277da1", "04:53:45"),
    4479799: ("band20_ambient", "Band [0.8, 1.2]", "#c58027", "04:51:03"),
    4479800: ("band10_active_cone", "Band [0.9, 1.1], cone", "#a54c70", "04:55:04"),
}
WINDOW = tuple(range(2000, 3001, 200))
SAMPLES = sorted({1, 6, 10, 25, 50, 229, *range(100, 3001, 100)})
METRICS = ("norm_over_R", "trial_norm_over_R", "returned_norm_over_R", "trial_elr",
           "relative_displacement", "aus", "grad_angle", "momentum_angle",
           "source_angle", "lmo_input_angle", "polar_angle")
VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\deE.+-]+)")
VERIFIED_POLAR_SOURCES = {
    "manifold_baselines.py": "f4985fe9a0731127f777c6dd200b73251a64e7730d0b6d52989fdb9162c7a09c",
    "manifold_polar.py": "4e188c52806aaf07a568594f7c08e3fde87080b69fc8459d326941ed15f89ecd",
}


def distribution(values, *, allow_empty=False):
    xs = sorted(x for x in values if x is not None)
    if not xs and allow_empty:
        return dict(count=0, min=None, p10=None, median=None, p90=None, max=None)
    if not xs or not all(math.isfinite(x) for x in xs):
        raise ValueError("expected finite, nonempty measurements")
    deciles = stats.quantiles(xs, n=10, method="inclusive") if len(xs) > 1 else [xs[0]] * 9
    return dict(count=len(xs), min=xs[0], p10=deciles[0], median=stats.median(xs),
                p90=deciles[8], max=xs[-1])


def loss_effect(reference, treatment):
    if set(reference) != set(treatment):
        raise ValueError("validation grids must match")
    delta = {s: treatment[s] - reference[s] for s in sorted(reference)}
    return dict(deltas=delta, endpoint_delta=delta[3000],
                window_mean=stats.mean(treatment[s] for s in WINDOW),
                window_mean_delta=stats.mean(delta[s] for s in WINDOW),
                window_wins=sum(delta[s] < 0 for s in WINDOW), window_points=len(WINDOW))


def verify_rows(rows, lower, upper):
    keys = [(r["step"], r["parameter"], r["component"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate logical-matrix measurement")
    identities = {(r["parameter"], r["component"]) for r in rows}
    if len(identities) != 196 or set(keys) != {(s, p, c) for s in SAMPLES for p, c in identities}:
        raise ValueError("incomplete 196-matrix diagnostic grid")
    for row in rows:
        for name in METRICS:
            value = row[name]
            if value is None and name.endswith("angle"):
                continue
            if value is None or not math.isfinite(value):
                raise ValueError("nonfinite iterate/displacement/angle")
        if not lower - 2e-6 <= row["returned_norm_over_R"] <= upper + 2e-6:
            raise ValueError("FP32 return violates the closed band")
    return identities


def summarize_rows(rows, lower, upper, reference):
    output = []
    for step in SAMPLES:
        sample = [r for r in rows if r["step"] == step]
        record = dict(step=step, **{m: distribution(r[m] for r in sample) for m in METRICS})
        record["at_lower_fraction"] = sum(abs(r["returned_norm_over_R"] - lower) <= 2e-6 for r in sample) / 196
        record["at_upper_fraction"] = sum(abs(r["returned_norm_over_R"] - upper) <= 2e-6 for r in sample) / 196
        for name in ("aus", "trial_elr"):
            ratios = [r[name] / reference[(step, r["parameter"], r["component"])][name]
                      for r in sample if reference[(step, r["parameter"], r["component"])][name] > 0]
            # The first warmup update has zero LR: 0/0 is undefined, not one.
            record[f"paired_{name}_ratio"] = distribution(ratios, allow_empty=True)
        output.append(record)
    final = [r for r in rows if r["step"] == 3000]
    grouped = {c: {m: distribution(r[m] for r in final if r["component"] == c)
                   for m in METRICS} for c in sorted({r["component"] for r in final})}
    return output, grouped


def collect(snapshot, reference_file):
    previous = json.loads(reference_file.read_text())
    old = previous["runs"]["4427413"]
    old_values = {int(s): v for s, v in old["validation"].items() if int(s) <= 3000}
    directories = {f.read_text().strip().split(";")[0]: f.parent for f in snapshot.rglob("job_id.txt")}
    runs, raw, reference, identities = {}, [], None, None
    common = None
    for job, (arm, label, color, elapsed) in ARMS.items():
        directory = directories[str(job)]
        logs = list(snapshot.rglob(f"*_{job}.out"))
        if len(logs) != 1:
            raise ValueError(f"expected exactly one stdout for {job}")
        log = logs[0].read_text()
        values = {int(s): float(v) for s, v in VAL.findall(log)}
        if set(values) != set(range(200, 3001, 200)) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("missing or nonfinite validation endpoint")
        completion = json.loads((directory / "completion.json").read_text())
        if completion["exit_code"] != 0 or completion["training_exit_code"] != 0:
            raise ValueError("training did not complete successfully")
        health = {}
        for name in ("nan", "skipped"):
            counts = re.findall(rf"number of {name} iterations:\s*(\d+)", log)
            if not counts or max(map(int, counts)) != 0:
                raise ValueError(f"missing or nonzero {name} counter")
            health[name] = 0
        cfg = json.loads((directory / "resolved_config.json").read_text())
        hashes = {Path(k).name: v for k, v in cfg["source_hashes"].items()}
        if any(hashes.get(name) != digest for name, digest in VERIFIED_POLAR_SOURCES.items()):
            raise ValueError("effective polar count needs a fresh runtime source audit")
        protocol = {k: cfg[k] for k in ("model", "training", "numerics", "initialization",
                    "geometry", "train_iters", "warmup_iters", "source_commit")}
        if common is not None and protocol != common:
            raise ValueError("pilot arms do not share the locked protocol")
        common = protocol
        band = cfg["frobenius_band"]
        if band["arm"] != arm or cfg["exit_interval"] != 3000:
            raise ValueError("wrong arm or early-exit budget")
        rows = [json.loads(line) for line in (directory / "band_diagnostics.jsonl").read_text().splitlines()]
        actual = verify_rows(rows, band["lower"], band["upper"])
        if identities is not None and actual != identities:
            raise ValueError("logical matrix identities differ across arms")
        identities = actual
        if reference is None:
            if values != old_values:
                raise ValueError("sphere control changed the old MuonH validation prefix")
            reference = {(r["step"], r["parameter"], r["component"]): r for r in rows}
            reference_values = values
        summaries, grouped = summarize_rows(rows, band["lower"], band["upper"], reference)
        clips = [json.loads(line) for line in (directory / "band_summary.jsonl").read_text().splitlines()]
        if [r["step"] for r in clips] != SAMPLES:
            raise ValueError("incomplete clip summary")
        for a, b in zip(clips, clips[1:]):
            if any(b[k] < a[k] for k in ("component_updates", "lower_clip_count", "upper_clip_count")):
                raise ValueError("cumulative clip count decreased")
        for row in clips:
            if row["component_updates"] != 196 * row["step"] or not 0 <= row["lower_clip_count"] + row["upper_clip_count"] <= row["component_updates"]:
                raise ValueError("invalid clip denominator or counts")
        last, start = clips[-1], next(r for r in clips if r["step"] == 2000)
        clipping = {f"{bound}_fraction": last[f"{bound}_clip_count"] / last["component_updates"]
                    for bound in ("lower", "upper")}
        clipping.update({f"last_1000_{bound}_fraction": (last[f"{bound}_clip_count"] - start[f"{bound}_clip_count"]) / (196 * 1000)
                         for bound in ("lower", "upper")})
        source_names = {"manifold_frobenius_band.py", "manifold_baselines.py", "manifold_geometry.py",
                        "manifold_polar.py", "manifold_gradient_probe.py", "run.py",
                        "pretrain_frobenius_band.py", "submit_frobenius_band.py"}
        runs[job] = dict(arm=arm, label=label, color=color, band=band, validation=values,
                         effect=loss_effect(reference_values, values),
                         slurm_state="COMPLETED", slurm_elapsed=elapsed,
                         process_seconds=completion["process_seconds"], training_health=health,
                         diagnostic_rows=len(rows), summaries=summaries, final_component_groups=grouped,
                         clip_counts=clips, clipping=clipping,
                         log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest(),
                         diagnostics_sha256=hashlib.sha256((directory / "band_diagnostics.jsonl").read_bytes()).hexdigest(),
                         source_hashes={k: v for k, v in cfg["source_hashes"].items() if Path(k).name in source_names})
        raw.extend(dict(job=job, **r) for r in rows)
    comparisons = {}
    for job in (4427414, 4427415):
        run = previous["runs"][str(job)]
        curve = {int(s): v for s, v in run["validation"].items() if int(s) <= 3000}
        comparisons[job] = dict(label=run["label"], validation=curve,
                                effect=loss_effect(reference_values, curve),
                                note="Historical prefix, not a newly randomized band arm; replay has different AUS control.")
    report = dict(snapshot_date="2026-11-22", stop_step=3000, processed_tokens=3000 * 128 * 2048,
                  cosine_horizon_steps=11445, logical_matrices=196, sampled_steps=SAMPLES,
                  window_steps=WINDOW, protocol=common, runs=runs, historical_prefixes=comparisons,
                  reference_job=4427413, reference_prefix_points=len(old_values),
                  reference_max_logged_validation_error=0.0,
                  effective_polar=dict(steps=10, polish_steps=0,
                      verification="Frozen _polar calls polar_express_msign without overriding steps=10; hashes verified.",
                      inactive_config_field="baseline_msign_steps=8 does not control this MuonH/band call path"),
                  aus_definition="Unit-normalized chord: norm(W_next/norm(W_next) - W/norm(W)); not the angle in radians.",
                  scope="Single-seed early exit on unchanged 3B schedule, not a completed 3B experiment.")
    return report, raw


def plots(report, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                         "pdf.fonttype": 42})
    runs = list(report["runs"].values())

    def tidy(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e2e5e8", lw=.7)
        ax.set_xlabel("Optimizer step")

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.0), layout="constrained")
    for run in runs:
        vals = {int(s): v for s, v in run["validation"].items()}
        steps = sorted(vals)
        axes[0].plot(steps, [vals[s] for s in steps], color=run["color"], lw=1.7, label=run["label"])
        if run["arm"] != "sphere_control":
            ds = {int(s): v for s, v in run["effect"]["deltas"].items()}
            axes[1].plot(steps, [1000 * ds[s] for s in steps], color=run["color"], marker="o", ms=3, lw=1.4)
    axes[0].set(title="Validation loss", ylabel="Cross-entropy")
    axes[1].axhline(0, color=runs[0]["color"], lw=1)
    axes[1].axvspan(2000, 3000, color="#edf0f3", alpha=.7, zorder=0)
    axes[1].set(title="Difference from native MuonH", ylabel="Validation loss difference (x 0.001)")
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Norm-band pilot | width 256, seed 1234 | 3,000 steps = 0.786B tokens", fontsize=12)
    save(fig, "validation_band")

    fig, axes = plt.subplots(1, 3, figsize=(13.1, 3.9), layout="constrained")
    for run in runs:
        samples = run["summaries"]
        x = [r["step"] for r in samples]
        color = run["color"]
        axes[0].plot(x, [r["returned_norm_over_R"]["median"] for r in samples], color=color, lw=1.6, label=run["label"])
        axes[0].fill_between(x, [r["returned_norm_over_R"]["p10"] for r in samples],
                             [r["returned_norm_over_R"]["p90"] for r in samples], color=color, alpha=.12)
        after_warmup = [r for r in samples if r["step"] >= 229]
        axes[1].plot([r["step"] for r in after_warmup],
                     [r["paired_aus_ratio"]["median"] for r in after_warmup], color=color, lw=1.6)
        if run["arm"] != "sphere_control":
            clips = run["clip_counts"]
            axes[2].plot([r["step"] for r in clips], [100 * r["upper_clip_count"] / r["component_updates"] for r in clips], color=color, lw=1.6)
    axes[0].set(title="Returned matrix norm", ylabel="Norm / reference R")
    axes[1].set(title="Actual angular step, after warmup", ylabel="Median unit-chord AUS / sphere AUS",
                xlim=(229, 3000), ylim=(.80, 1.02))
    axes[2].set(title="Cumulative upper-bound projection", ylabel="Percent of logical-matrix updates", ylim=(0, 100))
    for ax in axes:
        tidy(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Constraint and step-size diagnostics | norm shading: matrix P10-P90", fontsize=12)
    save(fig, "norm_and_aus")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True, layout="constrained")
    for ax, run in zip(axes.flat, runs, strict=True):
        samples = run["summaries"]
        for key, label, color in (("grad_angle", "Gradient G", "#38885e"),
                                   ("momentum_angle", "Stored momentum M", "#a54c70"),
                                   ("source_angle", "Nesterov input S", "#277da1"),
                                   ("polar_angle", "msign(Q), before cone correction", "#c58027")):
            ax.plot([r["step"] for r in samples], [r[key]["median"] for r in samples], color=color, lw=1.5, label=label)
        tidy(ax)
        ax.set(title=run["label"], ylabel="Median angle to tangent plane (degrees)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Measured G, M, S and msign angles | instantaneous iso-norm tangent plane", fontsize=12)
    save(fig, "momentum_and_msign_angles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--results", type=Path, help="Redraw the published JSON only")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot and not args.reference_results:
        parser.error("--snapshot needs the November 19 --reference-results")
    report, rows = collect(args.snapshot, args.reference_results) if args.snapshot else (json.loads(args.results.read_text()), None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("job", "arm", "step", "processed_tokens", "validation_loss", "delta_from_sphere"))
        for job, run in report["runs"].items():
            for step, loss in run["validation"].items():
                writer.writerow((job, run["arm"], step, int(step) * 128 * 2048, loss, run["effect"]["deltas"][step]))
    if rows is not None:
        columns = ("job", "step", "parameter", "component", *METRICS)
        with (args.output / "matrix_diagnostics.csv.gz").open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
                import io
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
    plots(report, args.output)
    print(json.dumps({job: {"label": r["label"], "endpoint": r["validation"].get("3000", r["validation"].get(3000)),
                           "late_mean": r["effect"]["window_mean"],
                           "endpoint_delta": r["effect"]["endpoint_delta"],
                           "clipping": r["clipping"]}
                      for job, r in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
