"""Dry-run-first Megatron launcher for MCSD/TP and geometry-matched baselines."""

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import uuid
from pathlib import Path

METHODS = {
    "manifold_mcsd": None, "manifold_mcsd_tp": None,
    "manifold_muonh": "frobenius", "manifold_imuon": "stiefel",
    "manifold_sso": "spectral", "manifold_muonsphere": "spectral",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--project", default=str(root))
    p.add_argument("--config", default=str(root / "configs/manifold/dense_lm.json"))
    p.add_argument("--geometry", required=True, choices=("frobenius", "spectral", "stiefel"))
    p.add_argument("--method", required=True, choices=tuple(METHODS))
    p.add_argument("--stage", default="smoke", choices=("smoke", "pilot", "tune", "main"))
    p.add_argument("--steps", type=int, help="smoke steps; full stages use --tokens")
    p.add_argument("--tokens", type=int)
    p.add_argument("--width", type=int, choices=(256, 384, 512))
    p.add_argument("--gpus", type=int, default=1, choices=(1, 2, 4, 8))
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--momentum-beta", type=float)
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--lmo-mode", choices=("exact", "ns"))
    p.add_argument("--stiefel-return-mode", choices=("exact", "ns"))
    p.add_argument("--spectral-solver", choices=("exact", "pi_topk"))
    p.add_argument("--audit-interval", type=int)
    p.add_argument("--save-interval", type=int)
    p.add_argument("--eval-iters", type=int)
    p.add_argument("--resume", type=Path, help="restore optimizer, RNG, master weights and scheduler")
    p.add_argument("--verify-against-run", type=Path, help="single-GPU smoke: compare the resumed final checkpoint to a continuous run")
    p.add_argument("--exit-interval", type=int, help="exit early without shortening the LR horizon")
    home = Path.home()
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(home / "projects/SSO_test/data/olmo_mix_1124_3b")))
    p.add_argument("--train-prefix", default=os.environ.get("TRAIN_DATA_PREFIX"))
    p.add_argument("--valid-prefix", default=os.environ.get("VALID_DATA_PREFIX"))
    p.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_MODEL", str(home / "projects/SSO_test/data/olmo_mix_1124_1b/tokenizer/OLMo-2-1124-7B")))
    p.add_argument("--env-dir", default=os.environ.get("ENV_DIR", str(home / "envs/sso_h20")))
    p.add_argument("--time-limit", default="3-00:00:00")
    p.add_argument("--partition", default="c_math_gpu")
    p.add_argument("--qos", default="gpu")
    p.add_argument("--dependency", help="e.g. afterok:123")
    p.add_argument("--submit", action="store_true")
    return p


def resolve_config(args: argparse.Namespace) -> dict:
    project = Path(args.project).expanduser().resolve()
    cfg = json.loads(Path(args.config).expanduser().read_text())
    if args.method not in METHODS or METHODS[args.method] not in (None, args.geometry):
        raise ValueError("method does not implement the requested constraint")
    model, train, numerics = cfg["model"], cfg["training"], cfg["numerics"]
    if args.width:
        model.update(hidden_size=args.width, ffn_hidden_size=3*args.width,
                     num_attention_heads=args.width//64, num_query_groups=args.width//128)
    if args.lr is not None:
        train["lr"] = args.lr
    if args.seed is not None:
        train["seed"] = args.seed
    if train["lr"] <= 0 or not math.isfinite(train["lr"]):
        raise ValueError("learning rate must be positive and finite")
    if train["global_batch"] % (args.gpus * train["micro_batch"]):
        raise ValueError("global batch must be divisible by DP * micro batch")
    batch_tokens = train["global_batch"] * model["seq_length"]
    if args.stage == "smoke":
        steps = 100 if args.steps is None else args.steps
        if args.tokens is not None or not 1 <= steps <= 100:
            raise ValueError("smoke requires 1--100 steps and no token override")
        target_tokens = steps * batch_tokens
        train.update(eval_interval=max(1, steps), eval_iters=2, save_interval=0)
    else:
        if args.steps is not None:
            raise ValueError("full stages use --tokens, not --steps")
        target_tokens = args.tokens if args.tokens is not None else cfg["budgets"][args.stage]
        if target_tokens <= 0:
            raise ValueError("token budget must be positive")
        steps = math.ceil(target_tokens / batch_tokens)
    for key in ("lmo_mode", "stiefel_return_mode", "spectral_solver", "audit_interval"):
        if getattr(args, key) is not None:
            numerics[key] = getattr(args, key)
    if numerics["audit_interval"] < 1:
        raise ValueError("audit interval must be positive")
    for key in ("save_interval", "eval_iters"):
        if getattr(args, key) is not None:
            train[key] = getattr(args, key)
    if train["save_interval"] < 0 or train["eval_iters"] < 1:
        raise ValueError("invalid checkpoint or evaluation settings")
    native = args.method in ("manifold_muonh", "manifold_sso", "manifold_muonsphere")
    train["momentum_beta"] = args.momentum_beta if args.momentum_beta is not None else (0.9 if native else 0.95)
    train["nesterov"] = args.nesterov if args.nesterov is not None else native
    if not 0 <= train["momentum_beta"] < 1:
        raise ValueError("momentum beta must be in [0, 1)")
    if train["nesterov"] and not native:
        raise ValueError("MCSD/TP and iMuon use EMA, not Nesterov")
    if args.method in ("manifold_sso", "manifold_muonsphere") and numerics["lmo_mode"] != "ns":
        raise ValueError("native SSO/MuonSphere kernels require NS")
    if args.exit_interval is not None and args.exit_interval < 1:
        raise ValueError("exit interval must be positive")
    if args.verify_against_run and (
        not args.resume or args.stage != "smoke" or args.gpus != 1
        or not train["save_interval"] or args.exit_interval
    ):
        raise ValueError("resume verification requires single-GPU smoke, checkpoints and no early exit")
    data_root = Path(args.data_root).expanduser().resolve()
    cfg.update(
        project=str(project), stage=args.stage, geometry=args.geometry, method=args.method,
        gpus=args.gpus, target_tokens=target_tokens, actual_tokens=steps*batch_tokens,
        train_iters=steps, warmup_iters=0 if steps == 1 else max(1, round(steps*train["warmup_fraction"])),
        train_prefix=str(Path(args.train_prefix).expanduser().resolve()) if args.train_prefix else str(data_root / "indexed/train/olmo_mix_1124_3b_train_text_document"),
        valid_prefix=str(Path(args.valid_prefix).expanduser().resolve()) if args.valid_prefix else str(data_root / "indexed/valid/olmo_mix_1124_3b_valid_text_document"),
        tokenizer=str(Path(args.tokenizer).expanduser().resolve()),
        env_dir=str(Path(args.env_dir).expanduser().resolve()),
        resume=str(args.resume.expanduser().resolve()) if args.resume else None,
        verify_against_run=str(args.verify_against_run.expanduser().resolve()) if args.verify_against_run else None,
        dependency=args.dependency,
        exit_interval=args.exit_interval, checkpoint_format="torch",
    )
    return cfg


def git_output(project: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(project), *args], text=True).strip()


def validate_resume(cfg: dict) -> None:
    """Do not silently mix geometry, data or LR horizons during continuation."""
    if not cfg["resume"]:
        return
    original_path = Path(cfg["resume"]).parent / "resolved_config.json"
    if not original_path.is_file():
        raise ValueError("resume needs its original resolved_config.json beside checkpoints")
    original = json.loads(original_path.read_text())
    keys = ("protocol_id", "method", "geometry", "model", "numerics", "train_iters",
            "warmup_iters", "train_prefix", "valid_prefix", "tokenizer", "checkpoint_format")
    changed = [key for key in keys if cfg[key] != original[key]]
    for key in ("global_batch", "micro_batch", "seed", "lr", "min_lr_ratio", "aux_lr", "momentum_beta", "nesterov"):
        if cfg["training"][key] != original["training"][key]:
            changed.append("training." + key)
    if changed:
        raise ValueError("resume protocol mismatch: " + ", ".join(changed))


def validate_checkpoint_dependency(cfg: dict) -> None:
    """Allow a queued parent only when afterok names that exact producer job."""
    if not cfg["resume"] or (Path(cfg["resume"]) / "latest_checkpointed_iteration.txt").is_file():
        return
    producer = Path(cfg["resume"]).parent / "job_id.txt"
    dependencies = (cfg.get("dependency") or "").split(":")
    if not producer.is_file() or dependencies[0] != "afterok" or producer.read_text().strip() not in dependencies[1:]:
        raise ValueError("missing checkpoint: use afterok dependency on its exact producer job")


def validate_submission(cfg: dict) -> None:
    project = Path(cfg["project"])
    if git_output(project, "status", "--porcelain"):
        raise ValueError("commit the experiment worktree before submission")
    required = [project / "Megatron-LM/megatron/core/models/gpt/gpt_model.py", Path(cfg["tokenizer"]), Path(cfg["env_dir"]) / "bin/python"]
    required += [Path(cfg[key] + suffix) for key in ("train_prefix", "valid_prefix") for suffix in (".bin", ".idx")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing runtime inputs: " + ", ".join(missing))
    if (project / "Megatron-LM/megatron/core/models").is_symlink():
        raise ValueError("model source must be the tracked snapshot, not a mutable symlink")
    validate_resume(cfg)
    validate_checkpoint_dependency(cfg)
    if cfg["verify_against_run"]:
        comparison = dict(cfg, resume=str(Path(cfg["verify_against_run"]) / "checkpoints"))
        validate_resume(comparison)
        validate_checkpoint_dependency(comparison)


def main() -> None:
    args = parser().parse_args()
    cfg = resolve_config(args)
    print(json.dumps(cfg, indent=2, sort_keys=True))
    if not args.submit:
        print("Dry run only; add --submit on the target H20 host.")
        return
    validate_submission(cfg)
    project = Path(cfg["project"])
    cfg["source_commit"] = git_output(project, "rev-parse", "HEAD")
    source_files = git_output(project, "ls-files", "Megatron-LM", "scripts/manifold", "slurm/manifold_dense_h20.sbatch", "configs/manifold").splitlines()
    cfg["source_hashes"] = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in source_files if name.endswith((".py", ".json", ".sbatch"))}
    run_id = f"{args.stage}_{args.geometry}_{args.method}_w{cfg['model']['hidden_size']}_{uuid.uuid4().hex[:10]}"
    run_dir = project / "results/manifold" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (project / "logs").mkdir(exist_ok=True)
    cfg["run_dir"] = str(run_dir)
    path = run_dir / "resolved_config.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    command = ["sbatch", "--parsable", f"--job-name={run_id[:80]}", f"--partition={args.partition}", f"--qos={args.qos}", f"--gres=gpu:nvidia_h20:{args.gpus}", f"--time={args.time_limit}", f"--chdir={project}", f"--export=ALL,MANIFOLD_RUN_CONFIG={path}"]
    if args.dependency:
        command.append(f"--dependency={args.dependency}")
    command.append(str(project / "slurm/manifold_dense_h20.sbatch"))
    print(shlex.join(command))
    job_id = subprocess.check_output(command, cwd=project, text=True).strip()
    (run_dir / "job_id.txt").write_text(job_id + "\n")
    print(f"Submitted {job_id}: {run_dir}")


if __name__ == "__main__":
    main()
