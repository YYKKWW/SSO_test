"""Reference geometry for the three constrained matrix experiments.

The functions operate on one logical FP32/FP64 matrix.  Fused QKV and FC1
storage is handled separately by ``manifold_layout``.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from .manifold_polar import polar_express_msign

GeometryName = Literal["frobenius", "spectral", "stiefel"]
ReturnName = Literal["projection", "radial"]
PolarMode = Literal["exact", "ns"]


class GeometryFailure(RuntimeError):
    """A named numerical or geometric boundary of a reference update."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class GeometryParameters:
    """Fixed radius and step scale obtained from the unconstrained initial matrix."""

    name: GeometryName
    radius: float
    scale: float


@dataclass(frozen=True)
class ReturnInfo:
    """Diagnostics of a completed return, computed from its trial SVD when used."""

    trial_sigma_min: float | None = None
    returned_relative_gap: float | None = None
    returned_top_normal: torch.Tensor | None = None
    top_gap_below_warning: bool = False
    return_approximate: bool = False


def _check_matrix(x: torch.Tensor) -> None:
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("expected a floating-point matrix")
    if not torch.isfinite(x).all():
        raise GeometryFailure("nonfinite_matrix")


def _svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(x, full_matrices=False)
    except torch.linalg.LinAlgError as exc:
        raise GeometryFailure("svd_failure", str(exc)) from exc


def _rank_tolerance(x: torch.Tensor, sigma_max: torch.Tensor) -> float:
    return float(torch.finfo(x.dtype).eps * max(x.shape) * sigma_max.item())


@torch.no_grad()
def partial_polar(x: torch.Tensor) -> torch.Tensor:
    """Exact spectral-ball LMO sign: zero singular modes contribute zero."""
    _check_matrix(x)
    if not torch.any(x):
        return torch.zeros_like(x)
    u, sigma, vh = _svd(x)
    active = sigma > _rank_tolerance(x, sigma[0])
    return (u * active.to(x.dtype)) @ vh


@torch.no_grad()
def full_polar_msign(x: torch.Tensor) -> torch.Tensor:
    """Full rectangular matrix sign U V^T used for a Stiefel return.

    This reference computes the polar factor by SVD.  It is not the
    rank-truncated/partial sign used by the spectral-ball LMO.
    """
    _check_matrix(x)
    u, sigma, vh = _svd(x)
    if sigma[-1].item() <= _rank_tolerance(x, sigma[0]):
        raise GeometryFailure("rank_deficient_trial", f"sigma_min={sigma[-1].item():.8g}")
    return u @ vh


@torch.no_grad()
def initialize_matrix(
    raw: torch.Tensor, name: GeometryName, *, polar_mode: PolarMode = "exact"
) -> tuple[torch.Tensor, GeometryParameters]:
    """Map one common raw initialization to a geometry and fix its scale."""
    _check_matrix(raw)
    if raw.dtype not in (torch.float32, torch.float64):
        raise ValueError("initialization must precede BF16 conversion")
    fro = torch.linalg.vector_norm(raw).item()
    if fro == 0.0:
        raise GeometryFailure("zero_initial_matrix")
    scale = fro / min(raw.shape) ** 0.5
    if name == "frobenius":
        return raw.clone(), GeometryParameters(name, fro, scale)
    if name == "spectral":
        sigma = torch.linalg.svdvals(raw)
        radius = sigma[0].item()
        return raw.clone(), GeometryParameters(name, radius, scale)
    if name == "stiefel":
        polar = (
            full_polar_msign(raw)
            if polar_mode == "exact"
            else polar_express_msign(raw, polish_steps=3)
        )
        return scale * polar, GeometryParameters(name, scale, scale)
    raise ValueError(f"unknown geometry: {name}")


@torch.no_grad()
def relative_spectral_gap(x: torch.Tensor) -> float:
    """Return the accurate relative top singular-value gap of a matrix."""
    sigma = torch.linalg.svdvals(x)
    if sigma.numel() < 2 or sigma[0].item() == 0.0:
        raise GeometryFailure("undefined_spectral_gap")
    return (1.0 - sigma[1] / sigma[0]).item()


@torch.no_grad()
def spectral_tangent_normal(x: torch.Tensor) -> torch.Tensor:
    """Chosen rank-one top normal, exact for a simple leading singular value."""
    u, _, vh = _svd(x)
    return torch.outer(u[:, 0], vh[0, :])


@torch.no_grad()
def tangent_project(
    x: torch.Tensor,
    z: torch.Tensor,
    params: GeometryParameters,
) -> torch.Tensor:
    """Frobenius-orthogonal tangent projection at a feasible matrix."""
    _check_matrix(x)
    if x.shape != z.shape or z.dtype != x.dtype:
        raise ValueError("tangent vector must match matrix shape and dtype")
    if params.name == "frobenius":
        return z - x * (torch.sum(x * z) / params.radius**2)
    if params.name == "stiefel":
        transposed = x.shape[0] < x.shape[1]
        w, v = (x.T, z.T) if transposed else (x, z)
        product = w.T @ v
        sym = (product + product.T) * 0.5
        projected = v - (w @ sym) / params.radius**2
        return projected.T if transposed else projected
    if params.name == "spectral":
        # At a repeated top singular value this selects one numerical normal.
        # The run continues, but the smooth-stratum tangent interpretation does not.
        normal = spectral_tangent_normal(x)
        return z - torch.sum(normal * z) * normal
    raise ValueError(f"unknown geometry: {params.name}")


@torch.no_grad()
def return_to_manifold(
    trial: torch.Tensor,
    params: GeometryParameters,
    *,
    mode: ReturnName,
    gap_stop: float = 1e-4,
    polar_mode: PolarMode = "exact",
) -> tuple[torch.Tensor, ReturnInfo]:
    """Apply the specified exact metric projection or spectral radial return."""
    _check_matrix(trial)
    radius = params.radius
    if params.name == "frobenius":
        norm = torch.linalg.vector_norm(trial).item()
        if norm == 0.0:
            raise GeometryFailure("zero_trial")
        return trial * (radius / norm), ReturnInfo()
    if params.name == "stiefel" and polar_mode == "ns":
        returned = radius * polar_express_msign(trial, polish_steps=3)
        return returned, ReturnInfo(return_approximate=True)
    u, sigma, vh = _svd(trial)
    if params.name == "stiefel":
        minimum = sigma[-1].item()
        if minimum <= _rank_tolerance(trial, sigma[0]):
            raise GeometryFailure("rank_deficient_trial", f"sigma_min={minimum:.8g}")
        # U @ Vh is the full rectangular matrix sign (polar factor).
        return radius * (u @ vh), ReturnInfo(trial_sigma_min=minimum)
    if params.name == "spectral":
        if mode == "projection":
            adjusted = torch.clamp(sigma, max=radius)
            adjusted[0] = radius
            returned = (u * adjusted) @ vh
            gap = (1.0 - adjusted[1] / adjusted[0]).item() if sigma.numel() > 1 else 1.0
        elif mode == "radial":
            if sigma[0].item() == 0.0:
                raise GeometryFailure("zero_trial")
            returned = trial * (radius / sigma[0])
            gap = (1.0 - sigma[1] / sigma[0]).item() if sigma.numel() > 1 else 1.0
        else:
            raise ValueError(f"unknown spectral return: {mode}")
        return returned, ReturnInfo(
            returned_relative_gap=gap,
            returned_top_normal=torch.outer(u[:, 0], vh[0, :]),
            top_gap_below_warning=gap <= gap_stop,
        )
    raise ValueError(f"unknown geometry: {params.name}")


@torch.no_grad()
def constraint_defect(x: torch.Tensor, params: GeometryParameters) -> float:
    """Dimensionless constraint residual for the selected geometry."""
    if params.name == "frobenius":
        return abs(torch.linalg.vector_norm(x).item() / params.radius - 1.0)
    if params.name == "spectral":
        return abs(torch.linalg.matrix_norm(x, ord=2).item() / params.radius - 1.0)
    transposed = x.shape[0] < x.shape[1]
    w = x.T if transposed else x
    gram = (w.T @ w) / params.radius**2
    identity = torch.eye(w.shape[1], device=x.device, dtype=x.dtype)
    return (torch.linalg.vector_norm(gram - identity) / w.shape[1] ** 0.5).item()
