#!/usr/bin/env python3
"""Build an offline experiment index without modifying the source ledgers."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/experiments"
ARCHIVE = DOCS / "experiment_inventory"
LEDGER = DOCS / "frobenius_3b_6b/results.json"
SUPPLEMENT = ARCHIVE / "supplemental.json"
SNAPSHOT = DOCS / "frobenius_3b_6b/latest_batch_20261029.json"
METHODS = ("MCSD", "MCSD-TP", "MuonH", "NorMuon-Sphere")
LRS = (0.005, 0.007, 0.008, 0.010, 0.012, 0.015, 0.020)
TOYS = ("nesterov_small_original.json", "nesterov_small_compact_20261014.json")


def read_json(path):
    return json.loads(path.read_text())


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def momentum(run):
    return f"{run['beta']:g} {'Nesterov' if run['nesterov'] else 'EMA'}"


def budget(run):
    return "100M" if run["tokens"] == 100_000_000 else f"{run['tokens'] / 1e9:g}B"


def label(run):
    return run["label"] + (" + unit-F" if run.get("direction") == "frobenius" else "")


def elapsed_hours(value):
    """Keep raw Slurm values in JSON; display one consistent hours format."""
    if "-" not in value:
        return value
    days, clock = value.split("-", 1)
    hours, minutes, seconds = clock.split(":")
    return f"{24 * int(days) + int(hours):02d}:{minutes}:{seconds}"


def load_inventory():
    current = read_json(LEDGER)
    supplemental = read_json(SUPPLEMENT)
    runs = []
    for original in current["runs"]:
        runs.append({
            **original,
            "protocol": "w256_shape_r1",
            "geometry": "frobenius",
            "radius_protocol": "shape: R=sqrt(D_out), r=1",
            "source": "frobenius_3b_6b/results.json",
        })
    for batch in supplemental["completed_batches"]:
        common = {k: v for k, v in batch.items() if k not in ("runs", "id")}
        runs.extend({**common, **run, "protocol": batch["id"], "direction": "none"} for run in batch["runs"])
    ids = [run["job_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate jobs in completed training inventory")
    for run in runs:
        if run["state"] != "COMPLETED" or not math.isfinite(run["final_val"]):
            raise ValueError(f"Non-final record in completed inventory: {run['job_id']}")
        if not (DOCS / run["source"]).is_file():
            raise FileNotFoundError(run["source"])
    return current, supplemental, sorted(runs, key=lambda run: int(run["job_id"]))


def coverage(runs, tokens, seed, direction="none"):
    selected = [r for r in runs if r["tokens"] == tokens and r["seed"] == seed and r.get("direction") == direction]
    rows = []
    for method in METHODS:
        configs = sorted({(r["beta"], r["nesterov"]) for r in selected if r["label"] == method})
        for beta, nesterov in configs:
            same = [r for r in selected if (r["label"], r["beta"], r["nesterov"]) == (method, beta, nesterov)]
            cells = []
            for lr in LRS:
                matches = [r for r in same if r["lr"] == lr]
                if len(matches) > 1:
                    raise ValueError("Replicates need separate coverage rows")
                cells.append(
                    f"[{matches[0]['final_val']:.6f}](experiment_inventory/COMPLETED_RUNS.md#job-{matches[0]['job_id']})"
                    if matches else "--"
                )
            rows.append([label(same[0]), momentum(same[0]), *cells])
    return table(["Method", "Momentum", *(f"{lr:.3f}" for lr in LRS)], rows)


def completed_markdown(runs):
    sections = [
        "# Completed Training Run Ledger\n",
        "Generated from preserved sources. This lists final-budget results, not best intermediate checkpoints. "
        "All times are Slurm elapsed in hours:minutes:seconds (hours can exceed 24), including startup/evaluation. "
        "A missing recorded exit code is not replaced by an assumed zero.\n",
        "[Experiment overview and coverage](../ALL_EXPERIMENTS.md) | [Machine-readable inventory](index.json)\n",
    ]
    for protocol, title in (
        ("w256_shape_r1", "Width 256, Frobenius Shape Radius r=1"),
        ("w384_init_radius_3b", "Width 384, Initialization-Derived Radii, 3B"),
        ("w384_init_radius_100m", "Width 384, Initialization-Derived Radii, 100M Pilot"),
    ):
        selected = [r for r in runs if r["protocol"] == protocol]
        sections.append(f"## {title}\n\n{len(selected)} completed runs. Do not pool across protocols.\n")
        rows = []
        for r in selected:
            rows.append([
                f'<a id="job-{r["job_id"]}"></a>{r["job_id"]}', r["geometry"], label(r), budget(r),
                f"{r['lr']:.3f}", momentum(r), r["seed"], r["final_step"], f"{r['final_val']:.6f}",
                elapsed_hours(r["elapsed"]), f"[source](../{r['source']})",
            ])
        sections.append(table(["Job", "Geometry", "Method", "Budget", "Peak LR", "Momentum", "Seed", "Final step", "Final val", "Elapsed", "Evidence"], rows))
    return "\n".join(sections)


def toy_markdown():
    parts = [
        "## Local Manifold-Nesterov Tests\n",
        "These are **CPU FP64, 8x8 Rayleigh-quotient problems on the Frobenius sphere**, not LLM or MCSD matrix-sign runs. "
        "Both versions use 12 seeds, 1,500 iterations, beta=0.9, R=1, six LRs, and a target objective gap of 1e-8. "
        "Each version has 3 methods x 6 LRs x 12 seeds = 216 trials. "
        "The revised implementation uses an atan2 log map, one extra matrix per method, and no restart.\n",
    ]
    rows = []
    full = []
    for filename in TOYS:
        data = read_json(ARCHIVE / filename)
        variant = "Original log map" if "original" in filename else "Compact atan2 log map"
        for r in data["best_by_method"]:
            rows.append([variant, r["method"], r["learning_rate"], f"{r['median_reached_iteration']:g}",
                         f"{r['median_final_gap']:.3g}", f"{r['median_elapsed_ms']:.2f}"])
        for r in data["summary"]:
            reached = r["median_reached_iteration"]
            full.append([variant, r["method"], r["learning_rate"], f"{reached:g}" if math.isfinite(reached) else "Not reached",
                         f"{r['median_final_gap']:.3g}", f"{r['median_elapsed_ms']:.2f}"])
    headers = ["Version", "Method", "LR", "Median target-hit iteration", "Median final gap", "Full-run CPU ms"]
    parts.append(table(headers, rows))
    parts.append("Best LR is selected by target-hit iteration, then final gap. CPU time is for the full run, not time-to-target. "
                 "The early target-hit improvement is not an LLM speedup claim. Source JSON retains the original `Infinity` sentinel for unreached targets.\n")
    parts.append("<details>\n<summary>All local LR settings, including unsuccessful settings</summary>\n\n" + table(headers, full) + "\n</details>\n")
    parts.append("[Original summary](experiment_inventory/nesterov_small_original.json) | "
                 "[Compact summary](experiment_inventory/nesterov_small_compact_20261014.json) | "
                 "[Implementation](../../scripts/manifold/compare_manifold_nesterov_small.py)\n")
    return "\n".join(parts)


def overview(current, supplemental, runs, snapshot):
    parts = [
        "# Experiment Inventory\n",
        f"**Organized October 29, 2026.** Latest saved server snapshot: {snapshot['checked_at_hkt']}. "
        "This is an offline inventory, not a fresh queue check.\n",
        "**80 completed training runs in this three-geometry project: 60 current 3B + 4 current 6B + 6 historical 3B + 10 historical 100M pilots.** "
        "Related legacy spectral runs, smokes, cancellations, local toy studies, and plans are listed separately. "
        "No result is removed because it is worse than a later result.\n",
        "[Interpretation and curves](FROBENIUS_3B_6B_SUMMARY.md) | "
        "[All 80 final results, times and job IDs](experiment_inventory/COMPLETED_RUNS.md) | "
        "[Machine-readable ledger](experiment_inventory/index.json) | [Source-document catalog](experiment_inventory/SOURCE_CATALOG.md)\n",
        "## Comparison Boundaries\n",
        table(["Protocol", "Completed", "Keep fixed when comparing", "Evidence"], [
            ["Width 256, shape radius, Frobenius", "60 x 3B; 4 x 6B", "73.418M parameters, R=sqrt(D_out), r=1, sequence 2048", "[Current study](FROBENIUS_3B_6B_SUMMARY.md)"],
            ["Width 384, initialization-derived radii", "6 x 3B", "126.641M parameters; 3 Frobenius + 3 Stiefel", "[Historical main batch](mcsd_three_geometries/MAIN_3B_20260924.md)"],
            ["Width 384, initialization-derived radii", "10 x 100M", "3 Frobenius + 3 Stiefel + 4 spectral; pilot horizon", "[Pilot batch](mcsd_three_geometries/PILOT_20260924.md)"],
        ]),
        "The old width-384 loss near 3.30 and current width-256 loss near 3.46 are **not** matched optimizer results. "
        "Width, model size and radius protocol differ. The corresponding 25-hour versus 18-hour runtimes are not an optimizer-only speed comparison. "
        "Within a protocol, match budget, LR, momentum, seed and variant before attributing a change to the algorithm.\n",
        "## Every Tested Learning Rate\n",
        "All LRs below are **peak matrix learning rates**, not constant per-step rates. "
        "The current protocol uses 2% warmup and cosine decay to 10% of peak; auxiliary AdamW uses a separate LR. "
        "Cells show final validation loss and link to the job record. `--` means **no completed result in this ledger**, not a failed run. "
        "LR=0.015 and LR=0.020 remain visible even when a newer matched comparison uses only 0.010.\n",
        "### 3B, Seed 1234, Standard Direction\n",
        coverage(current["runs"], 3_000_000_000, 1234),
        "No cross-row bold winner is used: momentum differs between rows. "
        "The matched 0.95-Nesterov / 0.010 ordering is NorMuon-Sphere < MuonH < MCSD-TP < MCSD. "
        "0.95 Nesterov has not yet been swept over LR=0.012/0.015.\n",
        "### 3B, Seed 2026, Standard Direction\n",
        coverage(current["runs"], 3_000_000_000, 2026),
        "### 3B, Seed 1234, Extra Unit-F Ablation\n",
        coverage(current["runs"], 3_000_000_000, 1234, "frobenius"),
        "This extra normalization is a separate MCSD variant. `none` in the standard rows does not disable a baseline's native normalization.\n",
        "### All 6B Results\n",
    ]
    six = [r for r in current["runs"] if r["tokens"] == 6_000_000_000]
    parts.append(table(["Method", "Seed", "Peak LR", "Momentum", "Final val", "Elapsed", "Job"], [
        [label(r), r["seed"], f"{r['lr']:.3f}", momentum(r), f"{r['final_val']:.6f}", elapsed_hours(r["elapsed"]), r["job_id"]] for r in six
    ]))
    parts.append("Elapsed times are hours:minutes:seconds, including startup and validation. "
                 "These are fresh 6B trainings, not continuations of the 3B schedule. MCSD and NorMuon-Sphere use different LR/momentum configurations. "
                 "There is no completed 6B MCSD-TP or MuonH result here, nor a 6B test of the new 0.95-Nesterov configuration.\n")
    parts.append("## Historical Three-Geometry Results\n")
    for batch in supplemental["completed_batches"]:
        parts.append(f"### Width 384, {budget(batch)}\n\n[Protocol, audits and original register]({batch['source']}). "
                     "All rows use peak LR=0.010, seed=1234 and initialization-derived frozen radii.\n")
        parts.append(table(["Geometry", "Method", "Momentum", "Final val", "Elapsed", "Job"], [
            [r["geometry"], r["label"], momentum(r), f"{r['final_val']:.6f}", r["elapsed"], r["job_id"]] for r in batch["runs"]
        ]))
    parts.append("The 100M spectral pilots completed, but their sparse post-update norm defects reached approximately 5-16%; "
                 "all four spectral methods were deferred from this historical formal 3B batch. "
                 "Their losses are retained, not presented as exact-constraint results. "
                 "Frobenius and Stiefel sparse audits are likewise not per-update certificates.\n")
    parts.append("## Smokes, Failures and Cancellations\n\nThese are not included in the 80 completed-budget runs. "
                 "Job groups can overlap where a smoke also supplied a recovery reference.\n")
    parts.append(table(["Jobs", "Category", "Recorded outcome", "Source"], [
        [r["jobs"], r["kind"], r["outcome"], f"[record]({r['source']})"] for r in supplemental["non_final_records"]
    ]))
    parts.append("[Momentum formula/regression/resume checks](experiment_inventory/momentum_validation_20261005.json) "
                 "also passed on CPU (6 regression cases, 48 formula/resume cases); these are correctness checks, not training results.\n")
    parts.append(toy_markdown())
    parts.extend([
        "## Planned Is Not Completed\n",
        table(["Item", "Evidence status", "Missing experiment"], [
            ["0.95 Nesterov LR refinement", "Only LR=0.010 completed for all four methods", "[Prepared LR=0.012/0.015 grid](FROBENIUS_NESTEROV_NORUNIT_20261029.md): MCSD/MuonH/NorMuon-Sphere only; no new MCSD-TP run"],
            ["Matched Nesterov seed replication", "MCSD/TP: seeds 1234 and 2026; baselines: 1234 only", "MuonH and NorMuon-Sphere seed 2026, beta=0.95 Nesterov, LR=0.010"],
            ["Shape-radius width transfer", "[Width-384/512 plan](mcsd_three_geometries/WIDTH_TRANSFER_MCSD_NORMUON_20261022.md) was dry-run only; no job IDs", "Run a matched protocol; old width-384 initialization-radius results do not fill these cells"],
            ["6B momentum transfer", "Four old-configuration completed results above", "Freeze equally tuned 3B configurations before a four-method 6B comparison"],
            ["MCSD row adaptation", "Local opt-in implementation/tests only", "[Prepared MCSD-NorUnit run](FROBENIUS_NESTEROV_NORUNIT_20261029.md); no H20 loss evidence; not included in historical MCSD results"],
            ["Manifold point extrapolation", "Small CPU problems only", "No LLM implementation result or acceleration claim"],
        ]),
        "## Related Spectral and Legacy Archives\n",
        "The following six completed runs belong to the **separate legacy spectral-cluster checkout**, not the new three-geometry comparison. "
        "They use sequence length 4096, 5,723 steps, LR=0.015, width 256 and seed 1234; model layout and validation protocol differ. "
        "Do not rank these losses against the Frobenius table. "
        "The [saved snapshot](frobenius_3b_6b/latest_batch_20261029.json) contains their configurations, hashes and curves.\n",
    ])
    external_rows = []
    for r in snapshot["separate_spectral_study"]:
        config = r["config"]
        name = {"muon_ball_dist": "MuonSphere", "spectral_ball_dist": "SSO"}.get(config["OPTIMIZER"], "Spectral-cluster variant")
        external_rows.append([r["job_id"], name, config.get("SPEL_PGD_RADIAL_CLUSTER_MODE", "--"),
                              config["SPEL_PGD_CLUSTER_ITERATIONS"] if name == "Spectral-cluster variant" else "--",
                              config.get("actual_return", "--"), f"{r['final_val']:.6f}", r["elapsed"], r["state"]])
    parts.append(table(["Job", "Method", "Cluster mode", "Block iterations", "Actual return", "Final val", "Elapsed", "Saved status"], external_rows))
    parts.append("The mode names alone are not complete configuration identities; consult the snapshot before combining rows. "
                 "A later seed-transfer array was RUNNING (4346705_2) / PENDING (4346705_3-5) in that snapshot. "
                 "No final-budget results for those four tasks are available in this record; this is not a live status assertion.\n")
    parts.append("The older 1B/12B SSO, MuonBall/MuonSphere, MCSD-PGD, precision, gap and cached-spectrum explorations remain in "
                 "[the inherited README archive](../../README.md#historical-sso-test-experiments), "
                 "[its detailed records](../../README.md#supplementary-results-and-detailed-records), and "
                 "[the old 1B sweep note](width256_sso_mcsd_lr_sweep_1b.md). "
                 "Those historical tables have not been re-audited into this 80-run ledger; their old running/pending labels are dated observations, not today's queue.\n")
    parts.extend([
        "## Curves and Source Preservation\n",
        "- [Original 16-run LR sweep: train and validation curves](mcsd_three_geometries/reports/width256_3b_20261005/curves_and_results.json), "
        "[train figure](mcsd_three_geometries/reports/width256_3b_20261005/figures/all_learning_rates_train.png), "
        "[validation figure](mcsd_three_geometries/reports/width256_3b_20261005/figures/all_learning_rates_validation.png).\n"
        "- [Extended 3B curves/results](mcsd_three_geometries/reports/width256_3b_20261008/curves_and_results.json) and "
        "[12-run momentum/seed follow-up](mcsd_three_geometries/reports/followup_20261020/results.json).\n"
        "- [6B validation curves](frobenius_3b_6b/validation_loss_6b.png), "
        "[matched EMA curves](frobenius_3b_6b/matched_ema095_validation.png), "
        "[matched Nesterov curves](frobenius_3b_6b/nesterov095_matched_20261029.png).\n",
        "The 64-row Frobenius source ledger is preserved unchanged. Historical 16-row results are transcribed with source links; "
        "source hashes are saved in the combined JSON. Small engineering/toy summaries have been copied out of ignored local results so the documentation can stand alone. "
        "Raw checkpoints are not required to read this archive. Not every old raw train log or per-step trace is Git-tracked; "
        "missing raw traces are not synthesized from endpoint losses.\n",
        "To update: add verified final rows to the source ledgers, retain partial/failed/cancelled records separately, "
        "then run `python scripts/manifold/build_experiment_inventory.py` from the repository root. "
        "Use `--check` to detect stale generated documents. Never replace a historical configuration with a later best setting.\n",
    ])
    return "\n".join(parts)


def catalog():
    excluded = {DOCS / "ALL_EXPERIMENTS.md", ARCHIVE / "COMPLETED_RUNS.md", ARCHIVE / "SOURCE_CATALOG.md"}
    rows = []
    for path in sorted(DOCS.rglob("*.md")):
        if path in excluded:
            continue
        title = next((line.lstrip("# ") for line in path.read_text().splitlines() if line.startswith("# ")), path.stem)
        relative = path.relative_to(DOCS)
        rows.append([f"[{title}](../{relative})", f"`{relative}`"])
    return "# Source Document Catalog\n\n[Inventory](../ALL_EXPERIMENTS.md). Dated documents retain their original observation times; " \
        "a submission note is not proof of completion. Use the inventory for current consolidated coverage.\n\n" + table(["Document", "Path"], rows)


def build():
    current, supplemental, runs = load_inventory()
    snapshot = read_json(SNAPSHOT)
    source_paths = [LEDGER, SUPPLEMENT, SNAPSHOT, *(ARCHIVE / name for name in TOYS)]
    source_paths.extend(DOCS / batch["source"] for batch in supplemental["completed_batches"])
    output = {
        "compiled_on": supplemental["compiled_on"],
        "latest_saved_server_check": snapshot["checked_at_hkt"],
        "scope": "Three-geometry project completed training; separate-study jobs are excluded from these counts",
        "counts_by_protocol": dict(Counter(r["protocol"] for r in runs)),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "runs": runs,
        "non_final_records": supplemental["non_final_records"],
        "separate_study_snapshot": str(SNAPSHOT.relative_to(ROOT)),
    }
    return {
        DOCS / "ALL_EXPERIMENTS.md": overview(current, supplemental, runs, snapshot),
        ARCHIVE / "COMPLETED_RUNS.md": completed_markdown(runs),
        ARCHIVE / "SOURCE_CATALOG.md": catalog(),
        ARCHIVE / "index.json": json.dumps(output, indent=2, allow_nan=False) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated files without writing")
    args = parser.parse_args()
    outputs = build()
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(content)
    if stale:
        raise SystemExit("Stale inventory files: " + ", ".join(stale))
    print("Inventory verified" if args.check else "Wrote inventory, complete ledger, source catalog and JSON")


if __name__ == "__main__":
    main()
