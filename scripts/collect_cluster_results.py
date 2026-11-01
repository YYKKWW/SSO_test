"""Collect completed cluster runs without importing the training environment."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def collect(directory):
    log = (directory / "train.log").read_text()
    config = json.loads((directory / "launch_config.json").read_text())
    vals = [dict(step=int(step), loss=float(loss)) for step, loss in re.findall(
        r"validation loss at iteration (\d+)(?: on validation set)? \| lm loss value: ([\d.Ee+-]+)", log)]
    regular = [line for line in log.splitlines() if re.search(r"iteration\s+\d+/\s*\d+ \|", line)]
    line = regular[-1] if regular else ""
    stats = {}
    patterns = {
        "step": r"iteration\s+(\d+)/", "near": r"cumulative pgd branches: (\d+)/",
        "decisions": r"cumulative pgd branches: \d+/(\d+)",
        "bursts": r"cumulative pgd bursts/repeat rate/mean length: (\d+)/",
        "overflow": r"cluster overflow: (\d+)",
        "nan_iterations": r"number of nan iterations:\s+(\d+)",
        "skipped_iterations": r"number of skipped iterations:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            stats[key] = int(match[1])
    q = re.search(r"cluster cumulative q1/q2/q3/q4: (\d+)/(\d+)/(\d+)/(\d+)", line)
    if q:
        stats["q_counts"] = list(map(int, q.groups()))
    if stats.get("decisions"):
        stats["near_rate"] = stats["near"] / stats["decisions"]
    if stats.get("near") and stats.get("bursts"):
        stats["repeat_rate"] = 1 - stats["bursts"] / stats["near"]
        stats["mean_burst_length"] = stats["near"] / stats["bursts"]
    row = dict(job=directory.name, directory=str(directory), config=config, vals=vals,
               final=vals[-1]["loss"] if vals else None, last_regular_statistics=stats,
               last_regular_log=line, log_sha256=hashlib.sha256(log.encode()).hexdigest(),
               source_manifest_sha256=hashlib.sha256((directory / "source.sha256").read_bytes()).hexdigest(),
               completed=(directory / "COMPLETED").exists(),
               training_completed=(directory / "TRAINING_COMPLETED").exists(),
               training_wall_seconds=int((directory / "training_wall_seconds.txt").read_text()))
    for key, name in (("calls", "spectral_call_counts.json"), ("offline_audit", "offline_audit.json")):
        path = directory / "snapshots" / name
        if path.exists():
            row[key] = json.loads(path.read_text())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [collect(path) for path in args.directories]
    accounting = subprocess.check_output([
        "sacct", "-X", "-P", "-j", ",".join(row["job"] for row in rows),
        "--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList"], text=True)
    lines = accounting.splitlines()
    header = lines[0].split("|")
    records = {entry[0]: dict(zip(header, entry)) for line in lines[1:] if (entry := line.split("|"))}
    for row in rows:
        row["accounting"] = records.get(row["job"], {})
        assert row["completed"] and row["training_completed"], row["job"]
        assert row["vals"][-1]["step"] == int(row["config"]["TRAIN_ITER"]), row["job"]
        endpoint = int(row["config"]["TRAIN_ITER"])
        interval = int(row["config"]["EVAL_INTERVAL"])
        expected_steps = sorted(set(range(interval, endpoint + 1, interval)) | {endpoint})
        assert [v["step"] for v in row["vals"]] == expected_steps, row["job"]
        assert row["accounting"].get("State") == "COMPLETED", row["job"]
        assert row["accounting"].get("ExitCode") == "0:0", row["job"]
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                  note="Final endpoint losses; branch statistics end at the last regular log, not necessarily the endpoint. Times include training and validation, not preflight or offline audit.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([dict(job=r["job"], final=r["final"], seconds=r["training_wall_seconds"]) for r in rows], indent=2))


if __name__ == "__main__":
    main()
