"""Compare two TRUSTED Megatron torch checkpoints from a split/continuous run.

Run with the matching Megatron directory on PYTHONPATH. Checkpoint unpickling
can execute code: never pass a checkpoint from an untrusted source.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def differences(left, right, path="root") -> list[str]:
    """Find exact state mismatches, including nested optimizer and RNG state."""
    if type(left) is not type(right):
        return [path + ": type"]
    if isinstance(left, torch.Tensor):
        return [] if left.dtype == right.dtype and torch.equal(left, right) else [path + ": tensor"]
    if isinstance(left, np.ndarray):
        return [] if np.array_equal(left, right) else [path + ": array"]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ": keys"]
        return [entry for key in left for entry in differences(left[key], right[key], f"{path}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [path + ": length"]
        return [entry for i, (a, b) in enumerate(zip(left, right)) for entry in differences(a, b, f"{path}[{i}]")]
    return [] if left == right else [path + ": value"]


def inspect_state(state: dict) -> dict:
    components, master_tensors = [], 0
    for wrapper in state["optimizer"]:
        for group in wrapper["fp32_from_fp16_params"]:
            for tensor in group:
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("non-FP32 or nonfinite master weight")
                master_tensors += 1
        optimizer = wrapper["optimizer"]
        for group in optimizer["param_groups"]:
            if "geometry" in group and (group["wd_mult"] != 0 or group["weight_decay"] != 0):
                raise ValueError("weight decay on a constrained matrix")
        for item in optimizer["state"].values():
            for component in item.get("components", []):
                if not all(math.isfinite(component[key]) and component[key] > 0 for key in ("radius", "scale")):
                    raise ValueError("invalid radius/scale")
                if component["momentum"].dtype != torch.float32 or not torch.isfinite(component["momentum"]).all():
                    raise ValueError("non-FP32 or nonfinite component momentum")
                if component["update_count"] != state["iteration"]:
                    raise ValueError("component update count does not match training iteration")
                components.append(component)
    if not components:
        raise ValueError("no manifold component state")
    return {
        "iteration": state["iteration"], "logical_components": len(components),
        "fp32_master_tensors": master_tensors,
        "model_tensor_dtypes": sorted({str(x.dtype) for x in state["model"].values() if isinstance(x, torch.Tensor)}),
        "constrained_weight_decay": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("resumed", type=Path)
    p.add_argument("continuous", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    left = torch.load(args.resumed, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(args.continuous, map_location="cpu", weights_only=False, mmap=True)
    keys = ("iteration", "model", "optimizer", "opt_param_scheduler", "rng_state")
    mismatch = [entry for key in keys for entry in differences(left[key], right[key], key)]
    report = {
        "resumed": inspect_state(left), "continuous": inspect_state(right),
        "checked": list(keys), "exact_match": not mismatch,
        "mismatch_count": len(mismatch), "mismatches": mismatch[:20],
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
    raise SystemExit(1 if mismatch else 0)


if __name__ == "__main__":
    main()
