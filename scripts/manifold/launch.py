"""Dry-run-first launcher for the three-geometry H20 smoke protocol."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def build_submission(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    project = Path(args.project).expanduser().resolve()
    script = project / "slurm" / "manifold_dense_h20.sbatch"
    if not script.is_file():
        raise FileNotFoundError(script)
    if args.steps < 1 or args.steps > 100:
        raise ValueError("first-phase smoke runs must have 1-100 optimizer steps")
    if args.gpus < 1 or args.gpus > 8 or 128 % (args.gpus * 4):
        raise ValueError("GPU count must be 1, 2, 4, or 8 for batch 128/micro 4")
    source_commit = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("commit the experiment worktree before submitting a smoke run")
    settings = {
        "PROJECT_DIR": str(project),
        "SOURCE_COMMIT": source_commit,
        "GEOMETRY": args.geometry,
        "OPTIMIZER": args.method,
        "LMO_MODE": args.lmo_mode,
        "STIEFEL_RETURN_MODE": args.stiefel_return_mode,
        "SPECTRAL_SOLVER": args.spectral_solver,
        "TRAIN_ITERS": str(args.steps),
        "GPUS_PER_NODE": str(args.gpus),
        "LR": str(args.lr),
        "MIN_LR": str(args.lr * 0.1),
        "SEED": str(args.seed),
        "WARMUP_ITERS": str(0 if args.steps == 1 else max(1, round(args.steps * 0.02))),
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in settings.items())
    command = [
        "sbatch",
        f"--gres=gpu:nvidia_h20:{args.gpus}",
        f"--export={export}",
        str(script),
    ]
    return command, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--geometry", required=True, choices=("frobenius", "spectral", "stiefel"))
    parser.add_argument("--method", required=True, choices=("manifold_mcsd", "manifold_mcsd_tp"))
    parser.add_argument("--lmo-mode", default="ns", choices=("exact", "ns"))
    parser.add_argument("--stiefel-return-mode", default="ns", choices=("exact", "ns"))
    parser.add_argument("--spectral-solver", default="pi_topk", choices=("exact", "pi_topk"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--submit", action="store_true", help="submit rather than print")
    args = parser.parse_args()
    command, settings = build_submission(args)
    print(json.dumps(settings, indent=2, sort_keys=True))
    print(shlex.join(command))
    if args.submit:
        os.makedirs(Path(args.project).expanduser() / "logs", exist_ok=True)
        subprocess.run(command, check=True, cwd=Path(args.project).expanduser().resolve())


if __name__ == "__main__":
    main()
