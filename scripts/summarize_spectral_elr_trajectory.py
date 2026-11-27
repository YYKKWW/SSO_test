"""Publish/recompute the six-arm shared-AUS study without model checkpoints.

Collect: python scripts/summarize_spectral_elr_trajectory.py --raw RAW_RESULTS --output OUTPUT_DIR
Replay:  python scripts/summarize_spectral_elr_trajectory.py --evidence OUTPUT_DIR/evidence.json.gz --output /tmp/elr-summary
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

from summarize_spectral_elr_confirmation import require, rows, stats


NAMES = ["muon_reference", "muon_aus0985", "muon_aus095", "mcsd_tp_aus100",
         "mcsd_tp_aus0985", "mcsd_tp_aus095"]
SCALES = [1., .985, .95, 1., .985, .95]
STEPS = list(range(253, 317))
SAMPLED = {0, 1, 5, 8, 120, 121, 125, 128, 240, 241, 245, 248}


def collect(root):
    evidence = dict(schema=1, arms=[], source_sha256={})
    for key in ("REFERENCE_READY", "TRAJECTORY_LANE_0_COMPLETED", "TRAJECTORY_LANE_1_COMPLETED"):
        evidence[key] = json.loads((root/(key+".json")).read_text())
    evidence["audit"] = rows(root/"same_state_return_audit.jsonl")
    evidence["negative"] = {key: rows(root/"reference_identity_gate/metrics"/(key+".jsonl"))
                            for key in ("negative_control", "heldout_probe", "training_stream")}
    for i, name in enumerate(NAMES):
        folder = root/f"trajectory_{i}_{name}"
        require((folder/"COMPLETED").exists(), f"Incomplete: {name}")
        data = {}
        for key in ("components", "heldout_probe", "model_probe", "trajectory_components", "output_components",
                    "theory_geometry", "return_conditioning", "all_component_returns", "training_stream",
                    "optimizer_before", "state", "activation_inputs", "negative_control"):
            path = folder/"metrics"/(key+".jsonl")
            if not path.exists():
                require(key == "negative_control", f"Missing {path.name}")
                continue
            values = rows(path)
            if key in ("state", "activation_inputs"):
                values = [v for v in values if v["iteration"] == 253]
            if key == "output_components":
                values = [dict(iteration=v["iteration"],component_index=v["component_index"],
                    heldout={k:v["heldout"][k] for k in ("master_output_elr","bf16_weight_output_elr")}) for v in values]
            if key == "components":
                values = [{k:v[k] for k in ("iteration","component_index","actual_elr_spectral",
                    "actual_elr_frobenius","aus","weight_stable_rank","displacement_stable_rank")}
                    | {key:{"spectral_normal_cosine":v[key]["spectral_normal_cosine"]}
                       for key in ("gradient","nesterov_input","direction")} for v in values]
            data[key] = values
            evidence["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = json.loads((folder/"launch_config.json").read_text())
        # Do not publish home directories, host configuration or checkpoint contents.
        data["config"] = {k:config[k] for k in ("WIDTH", "NUM_LAYERS", "SEQ_LENGTH", "GLOBAL_BATCH",
            "MICRO_BATCH", "LR", "LR_DECAY_ITERS", "LR_WARMUP_ITERS", "SEED", "SAVE_CHECKPOINT",
            "ELR_TRAJECTORY_ARM", "ELR_FIRST", "ELR_LAST", "ELR_VALID_SKIP_BATCHES")}
        data["time"] = json.loads((folder/"wall_seconds.json").read_text())
        data["verified"] = json.loads((folder/"VERIFIED.json").read_text())
        log = (folder/"train.log").read_text()
        data["log_counters"] = {kind: [int(v) for v in re.findall(
            r"number of "+kind+r" iterations:\s*(\d+)",log)] for kind in ("skipped", "nan")}
        evidence["arms"].append(data)
    return evidence


def indexed(records, components, label):
    result = {(r["iteration"],r["component_index"]):r for r in records}
    require(len(result) == len(records), f"Duplicate {label}")
    require(set(result) == {(s,i) for s in STEPS for i in components}, f"Incomplete {label}")
    return result


def paired_losses(a, b):
    differences = [x["loss_after"]-y["loss_after"] for x,y in zip(a,b)]
    batch_deltas = [x["loss"]-y["loss"] for x,y in zip(a[-1]["after"],b[-1]["after"])]
    return dict(final=differences[-1], last16_mean=mean(differences[-16:]),
                all64_mean=mean(differences), lower_steps=sum(d < 0 for d in differences),
                final_batch_deltas=batch_deltas, lower_final_batches=sum(d < 0 for d in batch_deltas))


def summarize(evidence):
    data = evidence["arms"]
    require(len(data) == 6, "Need six arms")
    ref = data[0]
    target = indexed(ref["trajectory_components"], range(252), "reference targets")
    ref_cs = indexed(ref["components"], SAMPLED, "reference detailed components")
    ref_out = indexed(ref["output_components"], range(252), "reference outputs")
    negative = evidence["negative"]
    for key in ("negative_control", "heldout_probe", "training_stream"):
        strip = lambda rs: [{k:v for k,v in r.items() if k != "method"} for r in rs]
        require(strip(ref[key][:3]) == strip(negative[key]) and len(negative[key]) == 3,
                f"Negative control differs: {key}")
    first, result = {}, []
    for name,scale,arm in zip(NAMES,SCALES,data):
        probes = arm["heldout_probe"]
        require([r["iteration"] for r in probes] == STEPS, "Wrong evaluation sequence")
        require([r["iteration"] for r in arm["model_probe"]] == STEPS, "Wrong training sequence")
        require(arm["training_stream"] == ref["training_stream"], "Full microbatch streams differ")
        require(all(r["microbatches"] == 32 for r in arm["training_stream"]), "Wrong global batch")
        require(all(r["optimizer_update_successful"] for r in arm["model_probe"]), "Skipped update")
        require(arm["time"]["exit_code"] == 0, "Nonzero process exit")
        for counts in arm["log_counters"].values():
            require(bool(counts) and max(counts) == 0, "NaN/skip counters failed")
        for key in ("state", "optimizer_before", "activation_inputs"):
            first.setdefault(key,arm[key])
            require(arm[key] == first[key], f"Initial state mismatch: {key}")
        require(probes[0]["before"] == ref["heldout_probe"][0]["before"], "Initial loss differs")
        for p,rp in zip(probes,ref["heldout_probe"]):
            for phase in ("before","after"):
                batches = p[phase]
                require(len(batches) == 20, "Wrong heldout size")
                require([(b["tokens"],b["labels"],b["valid_tokens"]) for b in batches] ==
                        [(b["tokens"],b["labels"],b["valid_tokens"]) for b in rp[phase]], "Heldout stream differs")
                value = sum(b["loss"]*b["valid_tokens"] for b in batches)/sum(b["valid_tokens"] for b in batches)
                require(math.isfinite(value) and abs(value-p["loss_"+phase]) < 1e-10, "Invalid loss")
        ts = indexed(arm["trajectory_components"],range(252),"actual targets")
        cs = indexed(arm["components"],SAMPLED,"detailed components")
        outs = indexed(arm["output_components"],range(252),"output components")
        theory = indexed(arm["theory_geometry"],SAMPLED,"return geometry")
        conditioning = indexed(arm["return_conditioning"],SAMPLED,"conditioning")
        for key,row in ts.items():
            r = target[key]
            for field in ("parameter","rows","shape","radius"):
                require(row[field] == r[field], f"Layout mismatch: {field}")
            expected = scale*r["aus"]
            require(abs(row["target"]-expected) < 1e-12, "Not the common reference AUS target")
            require(abs(row["aus"]-expected) <= 4e-6+.002*expected, "Returned AUS mismatch")
        returns = arm["all_component_returns"]
        require([r["iteration"] for r in returns] == STEPS and all(r["components"] == 252 for r in returns), "Missing returns")
        require(max(r["max_radius_error"] for r in returns) < 5e-5, "Radius guard failed")
        by_step = []
        for step,p in zip(STEPS,probes):
            keys = [(step,i) for i in range(252)]
            sample = [(step,i) for i in sorted(SAMPLED)]
            row = dict(iteration=step,loss_after=p["loss_after"],
                master_aus_ratio=mean(ts[k]["aus"]/target[k]["aus"] for k in keys),
                bf16_aus_ratio=mean(ts[k]["bf16_aus"]/target[k]["bf16_aus"] for k in keys),
                output_ratio=mean(outs[k]["heldout"]["master_output_elr"]/ref_out[k]["heldout"]["master_output_elr"] for k in keys),
                bf16_output_ratio=mean(outs[k]["heldout"]["bf16_weight_output_elr"]/ref_out[k]["heldout"]["bf16_weight_output_elr"] for k in keys),
                spectral_ratio=mean(cs[k]["actual_elr_spectral"]/ref_cs[k]["actual_elr_spectral"] for k in sample),
                frobenius_ratio=mean(cs[k]["actual_elr_frobenius"]/ref_cs[k]["actual_elr_frobenius"] for k in sample),
                return_residual=mean(theory[k]["first_order_return_residual"] for k in sample),
                gap_median=stats([theory[k]["gap"] for k in sample])["median"],
                perturbation_over_gap_median=stats([conditioning[k]["perturbation_over_gap"] for k in sample
                    if conditioning[k]["perturbation_over_gap"] is not None])["median"])
            for source in ("nesterov_input","direction"):
                row[source+"_angle_degrees"] = mean(math.degrees(math.asin(min(1.,abs(
                    cs[k][source]["spectral_normal_cosine"])))) for k in sample)
            by_step.append(row)
        layerwise = []
        for index in sorted(SAMPLED):
            keys = [(s,index) for s in STEPS]
            layerwise.append(dict(component_index=index,parameter=ts[keys[0]]["parameter"],rows=ts[keys[0]]["rows"],
                momentum_angle_degrees=mean(math.degrees(math.asin(min(1.,abs(cs[k]["nesterov_input"]["spectral_normal_cosine"])))) for k in keys),
                return_residual=stats([theory[k]["first_order_return_residual"] for k in keys]),
                perturbation_over_gap=stats([conditioning[k]["perturbation_over_gap"] for k in keys if conditioning[k]["perturbation_over_gap"] is not None])))
        record = dict(name=name,multiplier=scale,initial_loss=probes[0]["loss_before"],
            final_loss=probes[-1]["loss_after"],last16_loss=mean(p["loss_after"] for p in probes[-16:]),
            wall_seconds=arm["time"]["seconds"],component_returns=len(ts),
            max_radius_error=max(r["max_radius_error"] for r in returns),
            target_relative_error=stats([abs(r["aus"]-r["target"])/r["target"] for r in ts.values()]),
            master_bf16_aus_ratio=stats([r["bf16_aus"]/r["aus"] for r in ts.values()]),
            by_step=by_step,layerwise=layerwise,
            relative_gap=stats([r["gap"] for r in theory.values()]),
            perturbation_over_gap=stats([r["perturbation_over_gap"] for r in conditioning.values()
                                        if r["perturbation_over_gap"] is not None]),
            return_residual=stats([r["first_order_return_residual"] for r in theory.values()]),
            reference_comparison=paired_losses(probes,ref["heldout_probe"]))
        record["mean_ratios_to_reference_trajectory"] = {key:mean(s[key] for s in by_step) for key in
            ("master_aus_ratio","bf16_aus_ratio","spectral_ratio","frobenius_ratio","output_ratio","bf16_output_ratio")}
        if name.startswith("mcsd_tp"):
            counterpart = data[SCALES.index(scale)]
            record["same_multiplier_muon_comparison"] = paired_losses(probes,counterpart["heldout_probe"])
        result.append(record)
    audit_groups = defaultdict(list)
    audit = evidence["audit"]
    audit_keys = {(r["iteration"],r["component_index"],r["method"],r["policy"],r["fraction"]) for r in audit}
    require(len(audit_keys) == len(audit) == 576, "Missing/duplicate audit rows")
    for row in audit:
        audit_groups[(row["method"],row["policy"],row["fraction"])].append(row)
    audit_summary = []
    for (method,policy,fraction),rs in sorted(audit_groups.items()):
        require(len(rs) == 24, "Audit group does not cover 24 snapshots")
        audit_summary.append(dict(method=method,policy=policy,fraction=fraction,
            metrics={key:stats([r[key] for r in rs]) for key in ("return_residual","aus","actual_elr_spectral",
                "actual_elr_frobenius","heldout_local_output_elr","gradient_work","predicted_gradient_work")},
            nonnegative_gradient_work=sum(r["gradient_work"] >= 0 for r in rs)))
    return dict(date="2026-11-27",jobs=["4514694","4514695_0","4514695_1"],
        width=256,seed=1234,checkpoint=252,updates_per_arm=64,tokens_per_arm=33554432,
        evaluation_microbatches=20,evaluation_tokens=sum(b["valid_tokens"] for b in ref["heldout_probe"][0]["before"]),
        gates="6 CUDA tests; full training-stream/initial-state pairing; 3-step bitwise identity; all component targets and radii passed",
        source_sha256=evidence["source_sha256"],reference_gate=evidence["REFERENCE_READY"],
        arms=result,audit=audit_summary,
        limitations=["One seed and early checkpoint; repeated batches/components are not independent seeds",
            "Shared FP32-master AUS, not exact BF16 AUS or equal spectral/output movement",
            "Output ratios use each trajectory's own evolving frozen pre-update inputs, not common activations after divergence",
            "252 components for AUS/output; only 12 preselected components for detailed spectral geometry",
            "Common exact radial direction interventions, not native optimizer implementations",
            "Matrix audit gradient work is not measured full-model loss; only Muon/MCSD-TP have new training branches",
            "Timing includes calibration, exact SVD and repeated diagnostics; no production-speed claim",
            "Old own-state/coefficient-attenuated study has different targets and evaluation batches"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw",type=Path)
    inputs.add_argument("--evidence",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.raw:
        evidence = collect(args.raw)
    else:
        with gzip.open(args.evidence,"rt") as stream: evidence = json.load(stream)
    summary = summarize(evidence)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.raw:
        content = json.dumps(evidence,separators=(",",":"),allow_nan=False).encode()
        (args.output/"evidence.json.gz").write_bytes(gzip.compress(content,mtime=0))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for arm in summary["arms"]:
        print(arm["name"],f"final={arm['final_loss']:.9f}",f"tail16={arm['last16_loss']:.9f}",
              f"seconds={arm['wall_seconds']:.1f}",arm["mean_ratios_to_reference_trajectory"])
    for arm in summary["arms"][3:]:
        print(arm["name"],arm["same_multiplier_muon_comparison"])


if __name__ == "__main__":
    main()
