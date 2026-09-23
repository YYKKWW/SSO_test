"""Execute one frozen config with the existing Megatron pretrain_gpt entry."""

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def training_args(cfg: dict) -> list[str]:
    m, t, n = cfg["model"], cfg["training"], cfg["numerics"]
    args = []

    def option(name, value):
        args.extend(["--" + name, str(value)])

    for key, value in m.items():
        option(key.replace("_", "-"), value)
    args += [
        "--group-query-attention", "--bf16", "--swiglu", "--untie-embeddings-and-output-weights",
        "--qk-layernorm", "--disable-bias-linear", "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion", "--no-masked-softmax-fusion", "--no-rope-fusion",
        "--no-create-attention-mask-in-dataloader", "--spectral-mup-init", "--use-cpu-initialization",
        "--no-mmap-bin-files", "--log-throughput",
    ]
    values = {
        "max-position-embeddings": m["seq_length"], "attention-dropout": 0, "hidden-dropout": 0,
        "position-embedding-type": "rope", "rotary-base": 1000000,
        "normalization": "RMSNorm", "norm-epsilon": 1e-6, "transformer-impl": "local",
        "init-method-std": 0.02, "split-qkv-init-mode": "head",
        "tokenizer-type": "HuggingFaceTokenizer", "tokenizer-model": cfg["tokenizer"],
        "data-cache-path": str(Path(cfg["project"]) / "data_cache/manifold_dense"),
        "num-dataset-builder-threads": 8, "num-workers": 2, "distributed-timeout-minutes": 60,
        "seed": t["seed"], "optimizer": cfg["method"], "manifold-geometry": cfg["geometry"],
        "manifold-momentum-beta": t["momentum_beta"], "manifold-aux-lr": t["aux_lr"],
        "manifold-lmo-mode": n["lmo_mode"], "manifold-stiefel-return-mode": n["stiefel_return_mode"],
        "manifold-spectral-solver": n["spectral_solver"], "manifold-power-steps": n["power_steps"],
        "manifold-topk-rank": n["topk_rank"], "manifold-spectral-audit-interval": n["audit_interval"],
        "manifold-baseline-msign-steps": n["baseline_msign_steps"],
        "manifold-baseline-solver-tolerance": n["baseline_solver_tolerance"],
        "manifold-baseline-solver-iterations": n["baseline_solver_iterations"],
        "lr": t["lr"], "min-lr": t["lr"] * t["min_lr_ratio"],
        "lr-warmup-iters": cfg["warmup_iters"], "lr-decay-style": "cosine",
        "lr-decay-iters": cfg["train_iters"], "train-iters": cfg["train_iters"],
        "adam-beta1": 0.9, "adam-beta2": 0.95, "adam-eps": 1e-8,
        "clip-grad": 1.0, "weight-decay": 0.1,
        "tensor-model-parallel-size": 1, "pipeline-model-parallel-size": 1,
        "micro-batch-size": t["micro_batch"], "global-batch-size": t["global_batch"],
        "eval-interval": t["eval_interval"], "eval-iters": t["eval_iters"],
        "log-interval": 1 if cfg["stage"] == "smoke" else 10,
        "tensorboard-dir": str(Path(cfg["run_dir"]) / "tensorboard"),
        "ckpt-format": cfg["checkpoint_format"],
    }
    for key, value in values.items():
        option(key, value)
    args += ["--train-data-path", "1", cfg["train_prefix"], "--valid-data-path", "1", cfg["valid_prefix"]]
    if t["nesterov"]:
        args.append("--manifold-baseline-nesterov")
    if t["save_interval"]:
        option("save", Path(cfg["run_dir"]) / "checkpoints")
        option("save-interval", t["save_interval"])
    if cfg["resume"]:
        option("load", cfg["resume"])
    if cfg["exit_interval"]:
        option("exit-interval", cfg["exit_interval"])
    return args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    project, run_dir = Path(cfg["project"]), Path(cfg["run_dir"])
    mismatches = [name for name, digest in cfg["source_hashes"].items()
                  if not (project / name).is_file() or hashlib.sha256((project / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError("source changed after submission: " + ", ".join(mismatches[:10]))
    import torch

    modules = {}
    for name in (
        "megatron.core.models.gpt.gpt_model",
        "megatron.training",
        "emerging_optimizers.orthogonalized_optimizers.manifold_mcsd",
        "emerging_optimizers.orthogonalized_optimizers.manifold_baselines",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to((project / "Megatron-LM").resolve()):
            raise RuntimeError(f"loaded wrong source: {name} from {path}")
        modules[name] = str(path)
    runtime = {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "source_commit": cfg["source_commit"],
        "modules": modules, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    os.environ["MCSD_MANIFOLD_MANIFEST_PATH"] = str(run_dir / "parameter_manifest.json")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(cfg["gpus"]), "pretrain_gpt.py", *training_args(cfg)]
    (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    print(f"Run: {run_dir}\nActual tokens: {cfg['actual_tokens']}", flush=True)
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=project / "Megatron-LM", check=False)
    (run_dir / "completion.json").write_text(json.dumps({
        "exit_code": result.returncode, "process_seconds": time.monotonic() - start,
        "note": "Includes startup, indexing, evaluation and checkpoint time; not optimizer-only time.",
    }, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
