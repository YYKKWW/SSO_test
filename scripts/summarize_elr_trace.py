"""Verify and summarize the six 64-update common-AUS trajectories.

This study has no P2GD branch and must not be merged into the 3B ledger.
Only standard-library dependencies are needed; no tensors are published.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    paths = sorted(root.glob("trajectory_*/metrics/heldout_probe.jsonl"))
    if len(paths) != 6:
        raise ValueError("exactly six trajectories required")
    reference = paths[0].parent
    stream = read(reference/"training_stream.jsonl")
    initial = [r for r in read(reference/"state.jsonl") if r["iteration"] == 253]
    optimizer = read(reference/"optimizer_before.jsonl")
    heldout = read(paths[0])
    ids = [(r["tokens"], r["labels"], r["valid_tokens"]) for r in heldout[0]["before"]]
    baseline = {(r["iteration"], r["component_index"]): r for r in read(reference/"trajectory_components.jsonl")}
    output, evidence = [], {}
    for path in paths:
        metrics, folder = path.parent, path.parent.parent
        rows, components = read(path), read(metrics/"trajectory_components.jsonl")
        assert [r["iteration"] for r in rows] == list(range(253, 317))
        assert read(metrics/"training_stream.jsonl") == stream
        assert [r for r in read(metrics/"state.jsonl") if r["iteration"] == 253] == initial
        assert read(metrics/"optimizer_before.jsonl") == optimizer
        for row in rows:
            for side in ("before", "after"):
                assert [(r["tokens"], r["labels"], r["valid_tokens"]) for r in row[side]] == ids
                weighted = sum(r["loss"]*r["valid_tokens"] for r in row[side])/sum(r["valid_tokens"] for r in row[side])
                assert abs(weighted-row["loss_"+side]) < 1e-10
        grid = {(r["iteration"], r["component_index"]) for r in components}
        assert len(components) == len(grid) == 64*252 and grid == set(baseline)
        for row in components:
            ref = baseline[(row["iteration"], row["component_index"])]
            assert row["base_target"] == ref["aus"]
            assert abs(row["target"]-row["base_target"]*row["target_multiplier"]) < 1e-12
        wall = json.loads((folder/"wall_seconds.json").read_text())
        assert wall["exit_code"] == 0 and (folder/"COMPLETED").exists()
        for name in ("heldout_probe.jsonl", "trajectory_components.jsonl", "training_stream.jsonl",
                     "state.jsonl", "optimizer_before.jsonl", "trajectory_protocol.jsonl", "model_probe.jsonl"):
            p = metrics/name
            evidence[str(p.relative_to(root))] = sha(p)
        values = [r["loss_after"] for r in rows]
        output.append(dict(method=rows[0]["method"], updates=len(rows), components=len(components),
            initial_heldout=rows[0]["loss_before"], final_heldout=values[-1],
            last16_mean=statistics.mean(values[-16:]), wall_seconds=wall["seconds"],
            mean_one_step_heldout_delta=statistics.mean(r["loss_delta"] for r in rows),
            max_master_aus_relative_error=max(r["relative_target_error"] for r in components),
            mean_bf16_over_master_aus=statistics.mean(r["bf16_aus"]/r["aus"] for r in components),
            curve=[dict(iteration=r["iteration"], loss=r["loss_after"]) for r in rows]))
    pairs = []
    for muon, tp in zip(output[:3], output[3:]):
        pairs.append(dict(muon=muon["method"], mcsd_tp=tp["method"],
                          mcsd_minus_muon_final=tp["final_heldout"]-muon["final_heldout"],
                          mcsd_minus_muon_last16=tp["last16_mean"]-muon["last16_mean"]))
    return dict(protocol="width256, seed1234, same step252 checkpoint; updates253-316; 20 fixed held-out microbatches; common returned FP32-master AUS trace",
        paired_training_stream=True, paired_recorded_initial_states=True,
        paired_heldout_batches=True, paired_common_aus_targets=True,
        bf16_displacement_matched=False, p2gd_tested=False, full_3b_run=False,
        methods=output, pairs=pairs, source_sha256=evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    report = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.archive:
        with tarfile.open(args.archive, "w:gz") as archive:
            for name in report["source_sha256"]:
                archive.add(args.root/name, arcname="results/"+name)
            for folder in sorted(args.root.glob("trajectory_*/metrics")):
                for name in ("launch_config.json", "wall_seconds.json", "VERIFIED.json", "COMPLETED"):
                    path = folder.parent/name
                    archive.add(path, arcname="results/"+str(path.relative_to(args.root)))
            for name in ("trajectory_submission.json", "REFERENCE_READY.json",
                         "TRAJECTORY_LANE_0_COMPLETED.json", "TRAJECTORY_LANE_1_COMPLETED.json"):
                archive.add(args.root/name, arcname="results/"+name)
    print(json.dumps([{k:v for k,v in row.items() if k != "curve"} for row in report["methods"]], indent=2))


if __name__ == "__main__":
    main()
