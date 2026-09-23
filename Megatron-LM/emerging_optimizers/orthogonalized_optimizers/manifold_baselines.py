"""Geometry-matched baselines for the Dense LM experiment.

These adapters share MCSD's logical matrix layout, initialization, fixed radii,
FP32 state and auxiliary-parameter routing. They are NOT untouched reproductions
of the baselines' complete training recipes. SSO and MuonSphere call the existing
vendored update functions, including their pre-step hard radial return.

MuonH: Hyperball Eq. (1), https://arxiv.org/abs/2606.16899.
iMuon: Stiefel product-ball direction, https://arxiv.org/abs/2605.09238,
with the current-sample ambient EMA adaptation specified in our paper.
"""

import logging
import math

import torch

from .manifold_geometry import (
    GeometryParameters,
    constraint_defect,
    partial_polar,
    return_to_manifold,
)
from .manifold_mcsd import ManifoldMCSD
from .manifold_polar import polar_express_msign

logger = logging.getLogger(__name__)
BASELINE_GEOMETRIES = {
    "muonh": "frobenius",
    "imuon": "stiefel",
    "sso": "spectral",
    "muonsphere": "spectral",
}


def _polar(matrix: torch.Tensor, mode: str) -> torch.Tensor:
    return partial_polar(matrix) if mode == "exact" else polar_express_msign(matrix)


@torch.no_grad()
def imuon_direction(w: torch.Tensor, momentum: torch.Tensor, radius: float, mode: str) -> torch.Tensor:
    """Product-ball direction on scaled (column or row) Stiefel.

    W/radius = Q, K = skew(Q.T M), H = (I - Q Q.T) M.
    Structural zero blocks must not amplify floating-point cancellation.
    The separate spectral unit balls need not give a unit ambient direction.
    """
    transpose = w.shape[0] < w.shape[1]
    q = (w.T if transpose else w) / radius
    m = momentum.T if transpose else momentum
    qt_m = q.T @ m
    k = 0.5 * (qt_m - qt_m.T)
    h = m - q @ qt_m if q.shape[0] > q.shape[1] else torch.zeros_like(m)
    cutoff = 8 * torch.finfo(m.dtype).eps * torch.linalg.vector_norm(m)

    def block_sign(block):
        if torch.linalg.vector_norm(block) <= cutoff:
            return torch.zeros_like(block)
        return _polar(block, mode)

    direction = -q @ block_sign(k) - block_sign(h)
    return direction.T if transpose else direction


@torch.no_grad()
def baseline_update(
    w: torch.Tensor,
    momentum: torch.Tensor,
    values: GeometryParameters,
    *,
    baseline: str,
    lr: float,
    lmo_mode: str = "ns",
    stiefel_return_mode: str = "ns",
    power_steps: int = 10,
    msign_steps: int = 8,
    solver_tolerance: float = 2e-4,
    solver_iterations: int = 20,
) -> torch.Tensor:
    """One update; spectral kernels modify the local W copy before the trial."""
    if BASELINE_GEOMETRIES.get(baseline) != values.name:
        raise ValueError(f"{baseline} is not a {values.name} baseline")
    if baseline == "muonh":
        sign = _polar(momentum, lmo_mode)
        norm = torch.linalg.vector_norm(sign)
        direction = sign / norm.clamp_min(torch.finfo(sign.dtype).tiny)
        trial = w - lr * values.radius * direction
        return return_to_manifold(trial, values, mode="projection")[0]
    if baseline == "imuon":
        direction = imuon_direction(w, momentum, values.radius, lmo_mode)
        trial = w + lr * values.scale * direction
        return return_to_manifold(
            trial, values, mode="projection", polar_mode=stiefel_return_mode
        )[0]
    if lmo_mode != "ns":
        raise ValueError("native spectral baseline kernels use BF16 NS, not exact LMO")
    if baseline == "sso":
        from .spectral_ball_utils import compute_spectral_ball_update

        direction, _, _ = compute_spectral_ball_update(
            w, momentum, values.radius, power_steps, msign_steps,
            "bisection", solver_tolerance, solver_iterations, retract_mode="hard",
        )
    else:
        from .muon_ball import compute_muon_ball_update

        direction, _, _ = compute_muon_ball_update(
            w, momentum, values.radius, power_steps, msign_steps, retract_mode="hard"
        )
    # Preserve the upstream pre-return/update order; no silent post-projection.
    return w - lr * values.scale * direction.float()


class ManifoldBaseline(ManifoldMCSD):
    """Baseline adapter sharing only component/state infrastructure with MCSD."""

    def __init__(
        self, params, *, baseline: str, use_nesterov: bool = True,
        msign_steps: int = 8, solver_tolerance: float = 2e-4,
        solver_iterations: int = 20, **kwargs,
    ) -> None:
        if BASELINE_GEOMETRIES.get(baseline) != kwargs.get("geometry"):
            raise ValueError("baseline and manifold geometry do not match")
        if baseline == "imuon" and use_nesterov:
            raise ValueError("iMuon adaptation uses ambient EMA without Nesterov")
        if msign_steps < 1 or solver_iterations < 1 or solver_tolerance <= 0:
            raise ValueError("invalid baseline numerical solver settings")
        kwargs.pop("method", None)
        super().__init__(params, method="mcsd", **kwargs)
        for group in self.param_groups:
            group.update(
                baseline=baseline, use_nesterov=use_nesterov,
                msign_steps=msign_steps, solver_tolerance=solver_tolerance,
                solver_iterations=solver_iterations,
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        decisions, audits, max_step = 0, 0, 0
        max_defect = 0.0
        for group in self.param_groups:
            if group.get("weight_decay", 0) != 0 or group.get("wd_mult", 0) != 0:
                raise ValueError("constrained matrices must have zero weight decay")
            self._init_group(group)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dtype != torch.float32 or param.ndim != 2:
                    raise ValueError("baseline adapter requires FP32 master matrices")
                for index, component in enumerate(self.state[param]["components"]):
                    rows = self._row_indices[param][index]
                    w = param.data.index_select(0, rows)
                    g = param.grad.index_select(0, rows).float()
                    values = GeometryParameters(group["geometry"], component["radius"], component["scale"])
                    if not math.isfinite(values.radius) or not math.isfinite(values.scale):
                        raise ValueError("missing initialized radii; load optimizer checkpoint")
                    beta = group["momentum_beta"]
                    m = component["momentum"]
                    m.lerp_(g, 1 - beta)
                    source = g.lerp(m, beta) if group["use_nesterov"] else m
                    updated = baseline_update(
                        w, source, values, baseline=group["baseline"], lr=group["lr"],
                        lmo_mode=group["lmo_mode"], stiefel_return_mode=group["stiefel_return_mode"],
                        power_steps=group["power_steps"], msign_steps=group["msign_steps"],
                        solver_tolerance=group["solver_tolerance"], solver_iterations=group["solver_iterations"],
                    )
                    param.data.index_copy_(0, rows, updated)
                    component["update_count"] += 1
                    max_step = max(max_step, component["update_count"])
                    decisions += 1
                    if (component["update_count"] + decisions) % group["spectral_audit_interval"] == 0:
                        max_defect = max(max_defect, constraint_defect(updated, values))
                        audits += 1
        self.last_step_stats = {
            "step": max_step, "components": decisions, "constraint_audits": audits,
            "maximum_constraint_defect": max_defect if audits else float("nan"),
        }
        if audits and max_step % 50 == 0:
            logger.info("manifold baseline stats: %s", self.last_step_stats)
        return loss
