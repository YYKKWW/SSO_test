"""Rebuild the November 19 readout from archived H20 logs and receipts.

The snapshot is deliberately external: do not commit runtime paths, training
data, or model checkpoints. Plotting needs matplotlib; collection is stdlib-only.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics as stats
from pathlib import Path


JOBS = {
    4427413: "MuonH reference",
    4427414: "MCSD-TP + unit-F, nominal LR",
    4427415: "Nesterov, pre-projection, AUS replay",
    4428781: "Width 384 MuonH",
    4428783: "Width 384 MCSD + unit-F",
    4428784: "Width 384 MCSD-TP + unit-F",
    4453012: "Nesterov, no pre-projection, AUS replay",
    4453013: "Nesterov, pre-projection after step 800",
    4453014: "Tangent EMA, pre-projection, AUS replay",
    4459729: "No momentum, ambient G, AUS replay",
    4459730: "No momentum, projected G, AUS replay",
    4464031: "Held-out audit CUDA gate",
    4464032: "MuonH trajectory with held-out audit",
}
WINDOW = tuple(range(2000, 3001, 200))
TYPES = ("attention.q", "attention.k", "attention.v", "attention.o",
         "mlp.gate", "mlp.up", "mlp.down")
GROUPS = {"Q/K": TYPES[:2], "V/O": TYPES[2:4], "MLP": TYPES[4:]}
VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\deE.+-]+)")


def projection_effect(ambient, projected, steps=WINDOW):
    delta = [projected[s] - ambient[s] for s in steps]
    return {"steps": list(steps), "deltas": delta,
            "endpoint_delta": delta[-1], "window_mean_delta": stats.mean(delta),
            "projected_wins": sum(d < 0 for d in delta), "comparisons": len(delta)}


def audit_pairs(rows):
    pairs = {}
    for row in rows:
        key = (row["step"], row["source"], row["layer"], row["component"])
        pair = pairs.setdefault(key, {})
        if row["preproject"] in pair:
            raise ValueError(f"duplicate audit candidate: {key}")
        if row["stalled"]:
            raise ValueError(f"a stalled ray requires a separate readout: {key}")
        pair[row["preproject"]] = row
    result = []
    for (step, source, layer, component), pair in sorted(pairs.items()):
        if set(pair) != {False, True}:
            raise ValueError("unpaired audit candidate")
        a, p = pair[False], pair[True]
        if a["baseline_losses"] != p["baseline_losses"] or a["target_chord"] != p["target_chord"]:
            raise ValueError("candidates do not share the same baseline and target")
        delta = [x - y for x, y in zip(p["candidate_losses"], a["candidate_losses"], strict=True)]
        if len(delta) != 2 or not all(math.isfinite(x) for x in delta):
            raise ValueError("expected two finite paired losses")
        result.append(dict(step=step, source=source, layer=layer, component=component,
                           batch_deltas=delta, mean_delta=stats.mean(delta),
                           model_chord_relative_difference=p["effective_model_chord"] / a["effective_model_chord"] - 1))
    return result


def collect(snapshot):
    directories = {f.read_text().strip().split(";")[0]: f.parent
                   for f in snapshot.rglob("job_id.txt")}
    runs = {}
    for job, label in JOBS.items():
        directory = directories[str(job)]
        logs = list(snapshot.rglob(f"*_{job}.out"))
        if len(logs) != 1:
            raise ValueError(f"expected one log for {job}")
        log = logs[0].read_text()
        values = {int(s): float(v) for s, v in VAL.findall(log)}
        cfg = json.loads((directory / "resolved_config.json").read_text())
        completion = json.loads((directory / "completion.json").read_text())
        if completion["exit_code"] != 0 or completion["training_exit_code"] != 0:
            raise ValueError(f"failed process: {job}")
        stop = cfg["exit_interval"] or cfg["train_iters"]
        if job != 4464031 and max(values, default=0) != stop:
            raise ValueError(f"incomplete validation endpoint: {job}")
        health = {}
        for key in ("skipped", "nan"):
            counts = re.findall(rf"number of {key} iterations:\s*(\d+)", log)
            if not counts:
                raise ValueError(f"missing training health counters: {job}")
            health[key] = max(map(int, counts))
            if health[key]:
                raise ValueError(f"nonzero {key} counter: {job}")
        provenance = {k: cfg[k] for k in ("model", "training", "numerics", "initialization",
                      "protocol_id", "source_commit", "method", "geometry", "train_iters", "warmup_iters")}
        provenance.update(stop_step=stop, lmo_preprojection=cfg.get("lmo_preprojection"),
                          lmo_preprojection_after=cfg.get("lmo_preprojection_after"),
                          tangent_momentum=cfg.get("tangent_momentum", False),
                          aus_gradient_only=cfg.get("aus_gradient_only", False),
                          train_index_name=Path(cfg["train_prefix"]).name,
                          valid_index_name=Path(cfg["valid_prefix"]).name)
        provenance["optimizer_source_hashes"] = {
            k: v for k, v in cfg["source_hashes"].items()
            if Path(k).name in {"manifold_mcsd.py", "manifold_baselines.py", "manifold_aus.py",
                                "manifold_geometry.py", "manifold_polar.py", "manifold_heldout_audit.py",
                                "pretrain_late_audit.py"}}
        run = dict(label=label, validation=values, config=provenance,
                   process_seconds=completion["process_seconds"], training_health=health,
                   log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest())
        trace = directory / "aus_trace.json"
        if trace.exists():
            meta = json.loads(trace.read_text())
            run["aus"] = {k: v for k, v in meta.items() if k != "matrices"}
            run["aus"]["logical_matrices"] = len(meta["matrices"])
        runs[job] = run
    effects = {
        "no_momentum": projection_effect(runs[4459729]["validation"], runs[4459730]["validation"]),
        "nesterov": projection_effect(runs[4453012]["validation"], runs[4427415]["validation"]),
    }
    effects["interaction_endpoint"] = effects["nesterov"]["endpoint_delta"] - effects["no_momentum"]["endpoint_delta"]
    effects["interaction_window"] = effects["nesterov"]["window_mean_delta"] - effects["no_momentum"]["window_mean_delta"]
    rows = [json.loads(x) for x in (directories["4464032"] / "heldout_direction_audit.jsonl").read_text().splitlines()]
    pairs = audit_pairs(rows)
    expected = {(s, src, layer, t) for s in (3000, 6000, 9000) for src in ("G", "S")
                for layer in (0, 14, 27) for t in TYPES}
    if {(r["step"], r["source"], r["layer"], r["component"]) for r in pairs} != expected:
        raise ValueError("incomplete 3-step, 3-layer held-out audit")
    audit = []
    for step in (3000, 6000, 9000):
        for source in ("G", "S"):
            ps = [p for p in pairs if p["step"] == step and p["source"] == source]
            audit.append(dict(step=step, source=source, mean_delta=stats.mean(p["mean_delta"] for p in ps),
                              median_delta=stats.median(p["mean_delta"] for p in ps),
                              projected_wins=sum(p["mean_delta"] < 0 for p in ps), blocks=len(ps),
                              grouped_means={g: stats.mean(p["mean_delta"] for p in ps if p["component"] in ts)
                                             for g, ts in GROUPS.items()}))
    reference = runs[4427413]["validation"]
    observed = runs[4464032]["validation"]
    prefix_error = max(abs(v - reference[s]) for s, v in observed.items())
    if prefix_error != 0:
        raise ValueError("held-out observer did not reproduce the reference at logged precision")
    angles = []
    for line in (directories["4464032"] / "momentum_angle_probe.jsonl").read_text().splitlines():
        r = json.loads(line)
        angles.append({k: r[k] for k in ("step", "component", "grad_radial_cosine", "momentum_radial_cosine", "source_radial_cosine")})
    angle_summary = []
    for step in sorted({r["step"] for r in angles}):
        rs = [r for r in angles if r["step"] == step]
        if len(rs) != 196:
            raise ValueError("incomplete momentum sample")
        angle_summary.append(dict(step=step, **{
            key: stats.median(math.degrees(math.asin(min(1, abs(r[key + "_radial_cosine"])))) for r in rs)
            for key in ("grad", "momentum", "source")}))
    return dict(snapshot_date="2026-11-19", runs=runs, projection_effects=effects,
                audit_summary=audit, audit_pairs=pairs, audit_records=rows,
                audit_master_max_error=max(abs(r["actual_master_chord"] - r["target_chord"]) for r in rows),
                audit_prefix_max_logged_loss_error=prefix_error, momentum_angles=angle_summary)


def plots(report, output):
    import matplotlib.pyplot as plt

    def tidy(ax):
        ax.grid(axis="y", color="#e1e5e9", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=190)
        plt.close(fig)

    runs = {int(j): r for j, r in report["runs"].items()}
    curves = {j: {int(s): v for s, v in r["validation"].items()} for j, r in runs.items()}
    colors = ("#2e7c99", "#b04a63")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), layout="constrained")
    for ax, ids, title in zip(axes[:2], ((4459729, 4459730), (4453012, 4427415)),
                               ("No momentum", "Nesterov, beta = 0.95"), strict=True):
        for job, color, label in zip(ids, colors, ("Ambient input", "Projected input"), strict=True):
            steps = sorted(s for s in curves[job] if s <= 3000)
            ax.plot(steps, [curves[job][s] for s in steps], color=color, label=label, lw=1.6)
        ax.set(title=title, xlabel="Optimizer step", ylabel="Validation loss")
        ax.legend(frameon=False, fontsize=8)
    for key, label, color in (("no_momentum", "No momentum", colors[0]), ("nesterov", "Nesterov", colors[1])):
        e = report["projection_effects"][key]
        axes[2].plot(e["steps"], [1000*d for d in e["deltas"]], color=color, marker="o", ms=3.5, label=label)
    axes[2].axhline(0, color="#555e67", lw=.8)
    axes[2].set(title="Projected minus ambient input", xlabel="Optimizer step", ylabel="Validation loss difference (x 0.001)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        tidy(ax)
    fig.suptitle("Matched matrix-wise AUS | width 256, seed 1234 | early exit at 3,000", fontsize=12)
    save(fig, "projection_factorial")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True, layout="constrained")
    for ax, source in zip(axes, ("G", "S"), strict=True):
        ps = [p for p in report["audit_pairs"] if p["source"] == source]
        for offset, (group, ts), color in zip((-.16, 0, .16), GROUPS.items(), ("#947349", "#2e7c99", "#478366"), strict=True):
            for x, step in enumerate((3000, 6000, 9000)):
                ds = [p["mean_delta"] * 10000 for p in ps if p["step"] == step and p["component"] in ts]
                jitter = [0 if len(ds) == 1 else (i/(len(ds)-1)-.5)*.07 for i in range(len(ds))]
                ax.scatter([x+offset+j for j in jitter], ds, s=18, color=color, alpha=.7, label=group if x == 0 else None)
        means = [10000*stats.mean(p["mean_delta"] for p in ps if p["step"] == s) for s in (3000, 6000, 9000)]
        ax.plot(range(3), means, color="#242c33", marker="D", ms=4, lw=1.2, label="21-block mean")
        ax.axhline(0, color="#555e67", lw=.8)
        ax.set(title="Current gradient G" if source == "G" else "Nesterov input S",
               xlabel="Pre-update audit step", xticks=range(3), xticklabels=("3,000", "6,000", "9,000"))
        tidy(ax)
    axes[0].set_ylabel("Paired held-out loss difference (x 0.0001)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Pre-projection minus ambient input | same MuonH state | two fixed microbatches", fontsize=12)
    save(fig, "heldout_direction_pairs")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    for ax, ref, ids, labels in ((axes[0], 4427413, (4427414, 4427415), ("MCSD-TP + unit-F", "MCSD-TP + unit-F, AUS replay")),
                                  (axes[1], 4428781, (4428783, 4428784), ("MCSD + unit-F", "MCSD-TP + unit-F"))):
        for job, color, label in zip(ids, colors, labels, strict=True):
            steps = sorted(curves[job])
            ax.plot([s*128*2048/1e9 for s in steps], [1000*(curves[job][s]-curves[ref][s]) for s in steps], color=color, label=label, lw=1.5)
        ax.axhline(0, color="#555e67", lw=.8)
        ax.set(title=f"Width {runs[ref]['config']['model']['hidden_size']}", xlabel="Processed tokens (billions)", ylabel="Validation loss minus MuonH (x 0.001)")
        ax.legend(frameon=False, fontsize=8)
        tidy(ax)
    fig.suptitle("Full 3B trajectories | same seed, nominal peak LR and momentum", fontsize=12)
    save(fig, "full_budget_differences")

    fig, ax = plt.subplots(figsize=(7.6, 3.8), layout="constrained")
    for key, label, color in (("grad", "Current G", "#478366"), ("momentum", "Stored M", colors[1]), ("source", "Nesterov S", colors[0])):
        ax.plot([r["step"] for r in report["momentum_angles"]], [r[key] for r in report["momentum_angles"]], color=color, lw=1.5, label=label)
    ax.set(title="MuonH: angle to the current tangent space", xlabel="Optimizer step", ylabel="Median angle (degrees)")
    ax.legend(frameon=False)
    tidy(ax)
    save(fig, "momentum_angles_through_9000")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--results", type=Path, help="Redraw published data without the external H20 archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.snapshot) if args.snapshot else json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("job", "step", "validation_loss"))
        for job, run in report["runs"].items():
            writer.writerows((job, step, value) for step, value in run["validation"].items())
    with (args.output / "heldout_direction_audit.jsonl").open("w") as stream:
        for row in report["audit_records"]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    plots(report, args.output)
    print(json.dumps({"effects": report["projection_effects"], "audit": report["audit_summary"]}, indent=2))


if __name__ == "__main__":
    main()
