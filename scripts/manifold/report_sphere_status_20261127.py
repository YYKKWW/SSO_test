"""Read-only fixed-sphere launch snapshot; never publish a pilot as a 3B endpoint."""

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


VAL = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\deE.+-]+)")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_endpoint(row):
    receipt = row.get("completion") or {}
    return (row["state"] == "COMPLETED" and row["scheduled_steps"] == 11445 and not row["gate"]
            and row["trace_steps"] == 11445 and receipt.get("exit_code") == 0
            and receipt.get("training_exit_code") == 0 and "11445" in row["validation"]
            and row.get("early_exit") is None and row.get("nan_iterations") == 0
            and row.get("skipped_iterations") == 0
            and row.get("replay_max_abs_error") is not None and row["replay_max_abs_error"] <= 2e-6)


def collect(project):
    sys.path.insert(0, str(project/"scripts/manifold"))
    from run import validate_aus_reference

    register = read(project/"results/manifold/sphere_3b_20261126_register.json")
    if register["inflight"] or len(register["runs"]) != 4:
        raise ValueError("incomplete or uncertain submission register")
    source = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    if source != register["source_commit"] or subprocess.check_output(
            ["git", "-C", str(project), "status", "--porcelain"], text=True).strip():
        raise ValueError("frozen source changed")
    ids = [r["job"] for r in register["runs"]]
    accounting = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
        "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End"], text=True)
    states = {r[0]: r[1:] for r in (line.split("|") for line in accounting.splitlines()) if r[0] in ids}
    rows = []
    for entry in register["runs"]:
        directory = Path(entry["directory"])
        path = directory/"resolved_config.json"
        cfg = read(path)
        if digest(path) != entry["config_sha256"] or cfg["source_commit"] != source:
            raise ValueError("submitted configuration changed")
        validate_aus_reference(cfg, Path(cfg["aus"]["reference_dir"]))
        logs = list((project/"logs").glob(f"*_{entry['job']}.out"))
        log = logs[0].read_text() if len(logs) == 1 else ""
        replay = read(directory/"aus_replay.json") or read(directory/"aus_replay_progress.json") or {}
        trace = read(directory/"aus_trace.json")
        if trace and digest(directory/"aus.npy") != trace["sha256"]:
            raise ValueError("completed AUS checksum mismatch")
        state, exit_code, elapsed, start, end, *_ = states.get(entry["job"], ["UNKNOWN", "", "", "", ""])
        gate_matches = re.findall(r"^MCSD_SPHERE_CUDA_GATE PASS (.+)$", log, re.M)
        row = dict(job=entry["job"], arm=entry["arm"], gate=entry["gate"], state=state,
            slurm_exit_code=exit_code, elapsed=elapsed, start=start, end=end,
            dependency=entry["dependency"], run_directory=directory.name,
            config_sha256=entry["config_sha256"], completion=read(directory/"completion.json"),
            scheduled_steps=cfg["train_iters"], early_exit=cfg["exit_interval"],
            last_logged_step=max(map(int, re.findall(r"iteration\s+(\d+)\s*/", log)), default=0),
            validation={s: float(v) for s, v in VAL.findall(log)},
            trace_steps=trace["steps"] if trace else 0,
            trace_sha256=trace["sha256"] if trace else None,
            replay_rule=cfg["aus"].get("replay_rule", "tangent"),
            preprojection=cfg["lmo_preprojection"], method=cfg["method"],
            reference_config_sha256=cfg["aus"]["reference_config_sha256"],
            reference_aus_sha256=cfg["aus"]["reference_aus_sha256"],
            replay_max_abs_error=replay.get("max_abs_error", (trace or {}).get("max_abs_error")),
            replay_rms_error=replay.get("rms_error", (trace or {}).get("rms_error")),
            cuda_gate=json.loads(gate_matches[-1]) if gate_matches else None)
        for key in ("nan", "skipped"):
            counts = list(map(int, re.findall(rf"number of {key} iterations:\s*(\d+)", log)))
            row[key+"_iterations"] = max(counts) if counts else None
        row["full_3b_endpoint_verified"] = full_endpoint(row)
        rows.append(row)
    return dict(observed_at=dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        source_commit=source, base_commit="2db0545a872111be41eed60e349a9eae61eb3bbd",
        maximum_additional_gpus=2, reference_job=4427413, projected_reference_job=4427415,
        full_steps=11445, processed_tokens=3000238080, width=256, seed=1234,
        late_steps=register["late_steps"], checkpoints=False, local_tests_passed=291,
        note="Dated Slurm/config/log snapshot; running or gate values are not final 3B results.", runs=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = collect(args.project.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"observed_at": result["observed_at"], "runs": [
        {k: r[k] for k in ("job", "arm", "gate", "state", "last_logged_step", "trace_steps", "full_3b_endpoint_verified")}
        for r in result["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
