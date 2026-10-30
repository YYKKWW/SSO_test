"""Compare three momentum choices on a small Frobenius-sphere problem.

The problem is a matrix-shaped Rayleigh quotient

    min  -0.5 <W, A W>    subject to ||W||_F = R,

where ``A`` is diagonal after vectorizing ``W``.  Its optimum is known, so
the experiment can report an objective gap instead of only a training loss.

The three methods intentionally share the same exponential-map step:

* ``ema``: ambient exponential moving average, projected at the current point;
* ``muon_nesterov``: the gradient/EMA blend used by Muon-style optimizers;
* ``manifold_nesterov``: extrapolate on the sphere and evaluate the next
  gradient at the extrapolated point.

This is a mechanics experiment, not a convergence claim for MCSD.  It keeps
the objective and geometry small enough to run deterministically on a CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch

Method = Literal["ema", "muon_nesterov", "manifold_nesterov"]
METHODS: tuple[Method, ...] = ("ema", "muon_nesterov", "manifold_nesterov")


@dataclass(frozen=True)
class Record:
    iteration: int
    objective: float
    objective_gap: float
    gradient_norm: float
    top_alignment: float
    constraint_defect: float


@dataclass(frozen=True)
class RunResult:
    method: Method
    learning_rate: float
    seed: int
    elapsed_seconds: float
    reached_iteration: int | None
    records: tuple[Record, ...]
    optimizer_state_bytes: int


def _check_sphere_point(x: torch.Tensor, radius: float) -> None:
    if x.ndim != 2 or x.dtype != torch.float64:
        raise ValueError("the small benchmark expects one FP64 matrix")
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if not torch.isfinite(x).all():
        raise ValueError("sphere point is not finite")


def sphere_project(x: torch.Tensor, z: torch.Tensor, radius: float) -> torch.Tensor:
    """Orthogonally project ``z`` onto the Frobenius-sphere tangent at ``x``."""
    _check_sphere_point(x, radius)
    if z.shape != x.shape or z.dtype != x.dtype:
        raise ValueError("tangent source must match the sphere point")
    return z - x * (torch.sum(x * z) / radius**2)


def sphere_exp(x: torch.Tensor, tangent: torch.Tensor, radius: float) -> torch.Tensor:
    """Exact exponential map of the radius-``radius`` Frobenius sphere."""
    _check_sphere_point(x, radius)
    tangent = sphere_project(x, tangent, radius)
    norm = torch.linalg.vector_norm(tangent)
    if norm.item() <= 32 * torch.finfo(x.dtype).eps * radius:
        trial = x + tangent
        return radius * trial / torch.linalg.vector_norm(trial)
    angle = norm / radius
    return torch.cos(angle) * x + radius * torch.sin(angle) * tangent / norm


def sphere_log(x: torch.Tensor, target: torch.Tensor, radius: float) -> torch.Tensor:
    """Shortest sphere logarithm from ``x`` to a non-antipodal ``target``."""
    _check_sphere_point(x, radius)
    _check_sphere_point(target, radius)
    # atan2 retains tiny angles that acos(dot) rounds to zero near coincidence.
    unit_x = x / torch.linalg.vector_norm(x)
    unit_target = target / torch.linalg.vector_norm(target)
    cosine = torch.clamp(torch.sum(unit_x * unit_target), -1.0, 1.0)
    tangent = unit_target - cosine * unit_x
    tangent = tangent - torch.sum(unit_x * tangent) * unit_x
    tangent_norm = torch.linalg.vector_norm(tangent)
    angle = torch.atan2(tangent_norm, cosine)
    if math.pi - angle.item() <= 1e-10:
        raise ValueError("sphere logarithm is ambiguous at the antipode")
    if angle.item() <= 32 * torch.finfo(x.dtype).eps:
        return torch.zeros_like(x)
    return radius * angle * tangent / tangent_norm


def _spectrum(size: int) -> torch.Tensor:
    """A reproducible spectrum with one separated leading eigenvalue."""
    if size < 3:
        raise ValueError("matrix must contain at least three entries")
    values = torch.linspace(1.0, 8.0, size, dtype=torch.float64)
    values[-1] = 10.0
    return values


def objective_and_gradient(
    x: torch.Tensor, eigenvalues: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.reshape(-1)
    objective = -0.5 * torch.sum(eigenvalues * flat.square())
    gradient = -(eigenvalues * flat).reshape_as(x)
    return objective, gradient


def _record(
    iteration: int,
    x: torch.Tensor,
    eigenvalues: torch.Tensor,
    radius: float,
) -> Record:
    objective, gradient = objective_and_gradient(x, eigenvalues)
    riemannian_gradient = sphere_project(x, gradient, radius)
    optimum = -0.5 * eigenvalues[-1].item() * radius**2
    alignment = abs(x.reshape(-1)[-1].item()) / radius
    return Record(
        iteration=iteration,
        objective=objective.item(),
        objective_gap=max(0.0, objective.item() - optimum),
        gradient_norm=torch.linalg.vector_norm(riemannian_gradient).item(),
        top_alignment=alignment,
        constraint_defect=abs(torch.linalg.vector_norm(x).item() / radius - 1.0),
    )


def _fista_factor(t: float) -> tuple[float, float]:
    next_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
    return (t - 1.0) / next_t, next_t


class SpherePointExtrapolation:
    """One accepted-point buffer in addition to the live model parameter Y.

    Compute gradients at ``parameter`` before calling ``step``. It contains
    Y_k between calls; ``accepted`` contains X_k. This replaces an EMA buffer
    and does not combine point extrapolation with a second momentum buffer.
    Temporary vectors still exist during the update. This CPU reference is
    not a peak-memory-optimized or checkpoint-integrated LLM optimizer.
    """

    @torch.no_grad()
    def __init__(self, parameter: torch.Tensor, radius: float) -> None:
        _check_sphere_point(parameter, radius)
        self.parameter = parameter
        self.accepted = parameter.detach().clone()
        self.radius = radius
        self.t = 1.0

    @property
    def state_nbytes(self) -> int:
        return self.accepted.numel() * self.accepted.element_size()

    @torch.no_grad()
    def step(self, gradient: torch.Tensor, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        direction = -sphere_project(self.parameter, gradient, self.radius)
        next_x = sphere_exp(self.parameter, learning_rate * direction, self.radius)
        del direction
        gamma, next_t = _fista_factor(self.t)
        backward = sphere_log(next_x, self.accepted, self.radius)
        next_y = sphere_exp(next_x, -gamma * backward, self.radius)
        self.parameter.copy_(next_y)
        self.accepted.copy_(next_x)
        self.t = next_t


@torch.no_grad()
def run_method(
    method: Method,
    *,
    shape: tuple[int, int],
    radius: float,
    learning_rate: float,
    beta: float,
    iterations: int,
    seed: int,
    target_gap: float,
) -> RunResult:
    """Run one deterministic trial and retain the accepted-point trajectory."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if learning_rate <= 0 or not 0 <= beta < 1 or iterations < 1:
        raise ValueError("invalid benchmark hyperparameters")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    x.mul_(radius / torch.linalg.vector_norm(x))
    extrapolation = SpherePointExtrapolation(x, radius) if method == "manifold_nesterov" else None
    momentum = torch.zeros_like(x) if extrapolation is None else None
    state_bytes = (extrapolation.state_nbytes if extrapolation is not None
                   else momentum.numel() * momentum.element_size())
    eigenvalues = _spectrum(x.numel())
    records = [_record(0, x, eigenvalues, radius)]
    reached = 0 if records[-1].objective_gap <= target_gap else None
    start = time.perf_counter()

    for iteration in range(1, iterations + 1):
        if extrapolation is not None:
            _, gradient = objective_and_gradient(extrapolation.parameter, eigenvalues)
            extrapolation.step(gradient, learning_rate)
            accepted_x = extrapolation.accepted
        else:
            _, gradient = objective_and_gradient(x, eigenvalues)
            momentum.mul_(beta).add_(gradient, alpha=1.0 - beta)
            source = (
                gradient.lerp(momentum, beta)
                if method == "muon_nesterov"
                else momentum
            )
            direction = -sphere_project(x, source, radius)
            x.copy_(sphere_exp(x, learning_rate * direction, radius))
            accepted_x = x

        records.append(_record(iteration, accepted_x, eigenvalues, radius))
        if reached is None and records[-1].objective_gap <= target_gap:
            reached = iteration

    return RunResult(
        method=method,
        learning_rate=learning_rate,
        seed=seed,
        elapsed_seconds=time.perf_counter() - start,
        reached_iteration=reached,
        records=tuple(records),
        optimizer_state_bytes=state_bytes,
    )


def _median_or_infinity(values: Iterable[int | None]) -> float:
    materialized = list(values)
    if any(value is None for value in materialized):
        return float("inf")
    return statistics.median(float(value) for value in materialized if value is not None)


def summarize(runs: Iterable[RunResult]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, float], list[RunResult]] = {}
    for run in runs:
        grouped.setdefault((run.method, run.learning_rate), []).append(run)
    rows = []
    for (method, learning_rate), group in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "learning_rate": learning_rate,
                "seeds": len(group),
                "median_final_gap": statistics.median(
                    run.records[-1].objective_gap for run in group
                ),
                "median_final_gradient_norm": statistics.median(
                    run.records[-1].gradient_norm for run in group
                ),
                "median_top_alignment": statistics.median(
                    run.records[-1].top_alignment for run in group
                ),
                "median_reached_iteration": _median_or_infinity(
                    run.reached_iteration for run in group
                ),
                "maximum_constraint_defect": max(
                    record.constraint_defect for run in group for record in run.records
                ),
                "median_elapsed_ms": 1000.0
                * statistics.median(run.elapsed_seconds for run in group),
                "optimizer_state_bytes": max(run.optimizer_state_bytes for run in group),
            }
        )
    return rows


def best_rows(summary: Iterable[dict[str, float | int | str]]) -> list[dict]:
    selected = []
    for method in METHODS:
        candidates = [row for row in summary if row["method"] == method]

        def selection_key(row: dict) -> tuple[bool, float, float]:
            hit = float(row["median_reached_iteration"])
            final_gap = float(row["median_final_gap"])
            return math.isinf(hit), hit if math.isfinite(hit) else final_gap, final_gap

        selected.append(min(candidates, key=selection_key))
    return selected


def _write_outputs(
    output_dir: Path,
    runs: list[RunResult],
    summary: list[dict],
    selected: list[dict],
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "summary": summary, "best_by_method": selected}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_lr = {row["method"]: row["learning_rate"] for row in selected}
    with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "learning_rate", "seed", *Record.__dataclass_fields__],
        )
        writer.writeheader()
        for run in runs:
            if run.seed != 0 or run.learning_rate != selected_lr[run.method]:
                continue
            for record in run.records:
                writer.writerow(
                    {
                        "method": run.method,
                        "learning_rate": run.learning_rate,
                        "seed": run.seed,
                        **asdict(record),
                    }
                )
    lines = [
        "# Small Frobenius-sphere momentum comparison",
        "",
        "All methods minimize the same matrix Rayleigh quotient on `||W||_F = R`.",
        "A row is selected first by the median iteration that reaches the configured",
        "target gap, then by final objective gap if no candidate reaches the target.",
        "",
        "- EMA uses `M <- beta M + (1-beta) G` and steps along `-Proj_X(M)`.",
        "- Muon Nesterov steps along `-Proj_X((1-beta) G + beta M)`.",
        "- Manifold Nesterov computes `X_next = Exp_Y(-eta grad f(Y))`, then",
        "  `Y_next = Exp_Xnext(-gamma Log_Xnext(X))` with the FISTA factor `gamma`.",
        "",
        "Each method keeps one extra matrix: EMA/Muon store M; point extrapolation",
        "stores X while the live parameter stores Y. No EMA is allocated for that method.",
        "Temporary vectors, model/gradient storage and recorded diagnostics are excluded",
        "from the persistent-state count. Peak device memory has not been benchmarked.",
        "The timings include CPU updates and per-iteration diagnostics, not LLM wall times.",
        "Sphere Log uses atan2 to preserve small-angle accuracy. Restart is disabled.",
        "",
        "| Method | Best LR | Median final gap | Median grad norm | Median hit step | Max defect | Median ms | State bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        hit = row["median_reached_iteration"]
        hit_text = "not reached" if math.isinf(float(hit)) else f"{float(hit):.0f}"
        lines.append(
            f"| {row['method']} | {float(row['learning_rate']):.4g} | "
            f"{float(row['median_final_gap']):.3e} | "
            f"{float(row['median_final_gradient_norm']):.3e} | {hit_text} | "
            f"{float(row['maximum_constraint_defect']):.3e} | "
            f"{float(row['median_elapsed_ms']):.2f} | {row['optimizer_state_bytes']} |"
        )
    lines += [
        "",
        "For P constrained FP32 entries, the live model costs 4P bytes and the",
        "one extra state buffer costs another 4P bytes, for either EMA or point",
        "extrapolation alone. Combining MCSD's EMA with point extrapolation",
        "requires both M and X: 8P extra state bytes. This comparison does not",
        "include such a combined variant, model activations or auxiliary AdamW states.",
        "A zero-history implementation cannot generally reproduce this two-point recurrence.",
        "",
        "The earlier acos-based Log lost small angles. Its late accuracy plateau",
        "is not sufficient evidence that a restart is necessary. The current test",
        "uses no restart; coarse learning-rate instability should still be checked.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--shape", type=int, nargs=2, default=(8, 8), metavar=("M", "N"))
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(0.04, 0.08, 0.12, 0.16, 0.24, 0.32),
    )
    parser.add_argument("--target-gap", type=float, default=1e-8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/manifold/nesterov_small"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "iterations": args.iterations,
        "seeds": args.seeds,
        "shape": list(args.shape),
        "radius": args.radius,
        "beta": args.beta,
        "learning_rates": args.learning_rates,
        "target_gap": args.target_gap,
        "dtype": "float64",
        "device": "cpu",
        "state_layout": "one extra matrix per method; live extrapolated model parameter",
        "log_map": "atan2",
        "restart": False,
    }
    runs = [
        run_method(
            method,
            shape=tuple(args.shape),
            radius=args.radius,
            learning_rate=learning_rate,
            beta=args.beta,
            iterations=args.iterations,
            seed=seed,
            target_gap=args.target_gap,
        )
        for method in METHODS
        for learning_rate in args.learning_rates
        for seed in range(args.seeds)
    ]
    summary = summarize(runs)
    selected = best_rows(summary)
    _write_outputs(args.output_dir, runs, summary, selected, config)
    print((args.output_dir / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
