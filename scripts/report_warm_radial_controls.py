"""Collect local, downloaded warm-radial control results without training imports."""
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results/warm_radial_controls_3b_20261117"


def collect(folder, job):
    log = (folder / "train.log").read_text()
    config = json.loads((folder / "launch_config.json").read_text())
    vals = [dict(step=int(step), loss=float(loss)) for step, loss in re.findall(
        r"validation loss at iteration (\d+)(?: on validation set)? \| lm loss value: ([\d.Ee+-]+)", log)]
    expected = list(range(250, 5723, 250)) + [5723]
    assert [v["step"] for v in vals] == expected, (job, vals)
    regular = [line for line in log.splitlines() if re.search(r"iteration\s+\d+/\s*\d+ \|", line)]
    health = {}
    for name in ("nan", "skipped"):
        values = [int(re.search(r"number of " + name + r" iterations:\s+(\d+)", line)[1])
                  for line in regular]
        health["maximum_" + name + "_iterations"] = max(values)
        assert max(values) == 0, (job, name)
    final = re.findall(r"^KERNEL_LMO_FINAL (\{.*\})$", log, re.M)
    assert len(final) == 1
    statistics = json.loads(final[0])
    # The endpoint JSON is authoritative; regular log counters may stop earlier.
    stats_path = folder / "snapshots/pilot_statistics.json"
    if not stats_path.exists():
        stats_path = folder / "pilot_statistics.json"
    assert statistics == json.loads(stats_path.read_text()), job
    c, s = statistics["counts"], statistics["sums"]
    assert c["updates"] == 5722 * 252
    assert c["trial_solves"] == c["updates"]
    assert c["small_ritz_svd"] == c["updates"] + 252
    seconds = int((folder / "training_wall_seconds.txt").read_text())
    return dict(job=job, config=config, vals=vals, final_loss=vals[-1]["loss"],
        training_wall_seconds=seconds,
        training_time=f"{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}",
        regular_reports=len(regular), health=health,
        statistics=statistics,
        near_rate=c["near"]/c["updates"],
        repeat_fraction_of_near=c["near_repeats"]/c["near"],
        near_returned_over_trial=s["near_returned_length"]/s["near_trial_length"],
        near_returned_over_d0_trial=s["near_returned_length"]/s["near_d0_trial_length"],
        log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        audit=json.loads((folder / "warm_spectrum_audit.json").read_text()))


def main():
    rows = []
    for job in ("4453017_0", "4453017_1"):
        folder = BASE / "results" / job
        assert (folder / "COMPLETED").exists() and (folder / "TRAINING_COMPLETED").exists()
        row = collect(folder, job)
        err = (BASE / "logs" / f"warm_controls_{job}.err").read_text()
        row["cuda_tests"] = sum(map(int, re.findall(r"Ran (\d+) tests? in", err)))
        assert row["cuda_tests"] == 44 and "FAILED" not in err
        smoke = BASE / "results" / (job + "_smoke")
        assert (smoke / "TRAINING_COMPLETED").exists()
        row["smoke_completed"] = True
        rows.append(row)
    reference_dir = ROOT / "docs/results/warm_ambient_radial_3b_20261116"
    # Earlier reference has a flattened local result layout.
    rows.append(collect(reference_dir, "4442918_2"))
    base = rows[0]["config"]
    for row in rows:
        row["config_differences_from_control"] = {
            key: [base.get(key), row["config"].get(key)]
            for key in sorted(set(base) | set(row["config"]))
            if base.get(key) != row["config"].get(key)}
        row["loss_difference_from_control"] = row["final_loss"] - rows[0]["final_loss"]
    assert set(rows[1]["config_differences_from_control"]) == {
        "KERNEL_METHOD", "KERNEL_TP_NEAR_RATIO", "KERNEL_SNAPSHOT_DIR"}
    assert set(rows[2]["config_differences_from_control"]) == {
        "KERNEL_METHOD", "KERNEL_PGD_TRIAL_RATIO", "KERNEL_SNAPSHOT_DIR", "PRETRAIN_ENTRYPOINT"}
    report = dict(checked_at="2026-11-18T10:44:58+08:00", rows=rows,
        accounting={
            "4453017_0": dict(state="COMPLETED", exit_code="0:0", elapsed="17:59:11",
                               start="2026-11-17T14:33:15+08:00", end="2026-11-18T08:32:26+08:00"),
            "4453017_1": dict(state="COMPLETED", exit_code="0:0", elapsed="17:59:18",
                               start="2026-11-17T15:42:07+08:00", end="2026-11-18T09:41:25+08:00")},
        server_observation=dict(home_free="18G", account_running=["4459729"],
            account_pending={"4459730": "Dependency"},
            other_job_scope="Separate Frobenius-manifold angle study, not pooled with spectral-sphere results"),
        limitations=["One seed, not a replicated superiority claim.",
            "Counter rates describe each evolving trajectory, not same-state routing agreement.",
            "Eight-state audits are selected samples, not global accuracy bounds.",
            "Main wall times exclude preflight, smoke, and offline audit; GPU/node loads may differ."])
    (BASE / "final_summary_20261118.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(json.dumps({key: row[key] for key in ("job", "final_loss", "training_time", "regular_reports",
            "near_rate", "repeat_fraction_of_near", "near_returned_over_trial", "near_returned_over_d0_trial",
            "config_differences_from_control")}, indent=2))
    print("Late paired validation:")
    for index in range(-5, 0):
        print(rows[0]["vals"][index]["step"], *[r["vals"][index]["loss"] for r in rows])


if __name__ == "__main__":
    main()
