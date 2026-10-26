# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from the SpEL optimizer implementation.  This variant adds a
# gap-triggered PGD fallback while keeping the safe-region SpEL path identical
# to the original SpEL implementation by default.

"""SpEL--PGD optimizer with a gap-triggered PGD fallback.

This file implements a practical MCSD/SpEL--PGD variant for matrix weights
constrained to a spectral sphere.  It is intended to live next to ``spel.py``
in the same optimizer package.

Key design choice
-----------------
The default mode is the engineering version recommended for LLM pretraining:
the safe branch is exactly the original SpEL update, while the unsafe branch
uses an exact SVD projection for the PGD fallback.  This keeps the baseline
SpEL behavior intact when the singular gap is safe.

For ablations, ``projection_mode`` can also force both branches through the
same post-step projection/retraction.  The old experimental implementation is

    Z = W - effective_lr * D
    Y = Retr_spel_style(Z)

which corresponds to ``projection_mode='shared_retraction'``.  In all modes,
the branch rule only controls the direction:

* SpEL branch: tangent-projected ``msign`` direction.
* PGD branch: momentum/Nesterov-momentum direction.

No line search is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch

from emerging_optimizers import utils

from .adaptive_spectral_svd import (
    adaptive_block_krylov_sphere_return,
    adaptive_warm_fixed_subspace_radial_return,
    accurate_svd_fp32,
    accurate_svdvals_fp32,
    block_krylov_topk_sphere_projection,
    hierarchical_warm_radial_return,
)
from .spel import SpEL, _as_column_unit_vector, project_to_tangent_plane
from .spectral_ball_utils import (
    apply_retract,
    compute_target_radius,
    get_spectral_ball_scale_factor,
    msign,
    normalize_direction_spectral_fp32,
    power_iteration,
)
from .spectral_p2gd import project_spectral_sphere_cone


BranchMode = Literal["auto", "spel", "pgd"]
GapEstimatorMode = Literal[
    "deflated_power",
    "deflated_fp32_gap_only",
    "block2_fp32",
    "block2_fp32_gap_only",
    "exact_svd_gap_only",
]
PGDDirectionNormalization = Literal["none", "fro", "spectral", "exact_spectral"]
ProjectionMode = Literal[
    "fallback_exact",
    "fallback_retraction",
    "fallback_retraction_exact_norm",
    "fallback_topk",
    "local_top1_fallback_exact",
    "shared_exact",
    "shared_retraction",
    "shared_topk",
    "tp_radial_pgd_exact",
    "tp_radial_pgd_topk",
    "tp_radial_pgd_krylov_topk",
    "tp_radial_pgd_adaptive_krylov",
    "tp_radial_pgd_adaptive_block2",
    "tp_radial_pgd_multirate",
    "tp_radial_pgd_hierarchical",
    "tp_radial_pi",
]

HIERARCHICAL_REJECTION_REASONS = (
    "rank1_prior_bounds",
    "rank1_top_identity",
    "rank1_warning_gap",
    "rank1_angle",
    "rank1_residual",
    "rank1_radial",
    "block4_residual",
    "block4_warning_gap",
    "block4_residual_gap",
    "block8_residual",
    "block8_warning_gap",
    "block8_residual_gap",
)
MainPowerDtype = Literal["bf16", "fp32", "fp32_highest", "fp64"]

__all__ = [
    "SpELPGDSameProjection",
    "SpELPGD",
    "SpELPGDUpdateInfo",
    "SpectralPointCache",
    "apply_spel_style_spectral_projection_",
    "compute_spel_pgd_same_projection_update",
    "estimate_top2_singular_values_block2_fp32",
    "estimate_second_singular_value",
    "estimate_second_singular_value_fp32",
    "power_iteration_fp32",
    "power_iteration_fp32_highest",
    "power_iteration_fp64",
    "power_iteration_warm_start",
    "project_to_spectral_sphere_local_top1_fp32",
    "project_to_spectral_sphere_exact",
    "project_to_spectral_sphere_topk",
    "retract_to_spectral_sphere_pi_norm",
    "retract_to_spectral_sphere_exact_norm",
]


@dataclass(frozen=True)
class SpectralPointCache:
    """Top spectral data for a returned FP32 master-matrix component."""

    sigma1: float
    sigma2: float
    rel_gap: float
    u1: torch.Tensor
    v1: torch.Tensor
    leading_simple: bool
    u_basis: Optional[torch.Tensor] = None
    v_basis: Optional[torch.Tensor] = None
    gap_lower: float = float("nan")
    gap_upper: float = float("nan")
    is_exact: bool = True
    sigma1_lower: float = float("nan")
    sigma1_upper: float = float("nan")
    sigma2_lower: float = float("nan")
    sigma2_upper: float = float("nan")
    bounds_certified: bool = False
    top_u: Optional[torch.Tensor] = None
    top_v: Optional[torch.Tensor] = None
    s_basis: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SpELPGDUpdateInfo:
    """Diagnostics for one matrix/component update."""

    branch: str
    rel_gap: float
    sigma1: float
    gap_sigma1: float
    sigma2: float
    pre_retract_bias: float
    trial_sigma: float
    post_retract_bias: float
    projection_mode: str
    projection_rank: int
    gap_probed: bool
    next_v: Optional[torch.Tensor] = None
    gap_audited: bool = False
    exact_sigma1: float = float("nan")
    exact_sigma2: float = float("nan")
    exact_rel_gap: float = float("nan")
    delta1_rel: float = float("nan")
    delta2_rel: float = float("nan")
    gap_abs_error: float = float("nan")
    main_sigma1_rel_error: float = float("nan")
    main_triplet_residual_rel: float = float("nan")
    false_safe: bool = False
    false_pgd: bool = False
    pgd_trial_rel_gap: float = float("nan")
    pgd_post_rel_gap: float = float("nan")
    pgd_projection_clipped_count: int = 0
    spectral_cache_hit: bool = False
    next_spectral_cache: Optional[SpectralPointCache] = None
    approximate_spectral_iterations: int = 0
    approximate_spectral_residual_rel: float = float("nan")
    approximate_gap_interval_width: float = float("nan")
    approximate_exact_fallback: bool = False
    approximate_subspace_dimension: int = 0
    approximate_mode: str = "none"
    approximate_rejection_reasons: Tuple[str, ...] = ()
    p2gd_top_multiplicity: int = 0
    cluster_dimension: int = 0
    cluster_cap_overflow: bool = False
    cluster_step_multiplier: float = 1.0
    radial_change_ratio: float = float("nan")
    cluster_flatten_distance: float = 0.0


@dataclass(frozen=True)
class _PostProjectionDiagnostics:
    """Exact spectral diagnostics reused from a PGD full-SVD projection."""

    trial_rel_gap: float = float("nan")
    post_rel_gap: float = float("nan")
    clipped_count: int = 0
    next_spectral_cache: Optional[SpectralPointCache] = None
    approximate_iterations: int = 0
    approximate_residual_rel: float = float("nan")
    approximate_gap_interval_width: float = float("nan")
    approximate_exact_fallback: bool = False
    approximate_subspace_dimension: int = 0
    approximate_mode: str = "none"
    approximate_rejection_reasons: Tuple[str, ...] = ()


def _spectral_point_cache_from_decomposition(
    U: torch.Tensor,
    S_returned: torch.Tensor,
    Vh: torch.Tensor,
    *,
    gap_lower: Optional[float] = None,
    gap_upper: Optional[float] = None,
    is_exact: bool = True,
    sigma1_lower: Optional[float] = None,
    sigma1_upper: Optional[float] = None,
    sigma2_lower: Optional[float] = None,
    sigma2_upper: Optional[float] = None,
    bounds_certified: Optional[bool] = None,
    eps: float = 1e-7,
) -> SpectralPointCache:
    """Keep top-two data after a spectral return changes only singular values."""
    if S_returned.numel() == 0:
        raise ValueError("cannot cache spectral data for an empty matrix")

    sigma1 = float(S_returned[0].item())
    sigma2 = float(S_returned[1].item()) if S_returned.numel() > 1 else 0.0
    scale = max(abs(sigma1), eps)
    rel_gap = max(0.0, 1.0 - sigma2 / scale)
    if gap_lower is None:
        gap_lower = rel_gap
    if gap_upper is None:
        gap_upper = rel_gap
    gap_lower = max(0.0, min(float(gap_lower), 1.0))
    gap_upper = max(gap_lower, min(float(gap_upper), 1.0))
    if bounds_certified is None:
        bounds_certified = is_exact
    if is_exact:
        sigma1_lower = sigma1_upper = sigma1
        sigma2_lower = sigma2_upper = sigma2
    elif bounds_certified:
        if any(
            value is None
            for value in (
                sigma1_lower,
                sigma1_upper,
                sigma2_lower,
                sigma2_upper,
            )
        ):
            raise ValueError("certified approximate cache requires spectral bounds")
    else:
        sigma1_lower = sigma1_upper = float("nan")
        sigma2_lower = sigma2_upper = float("nan")
    leading_simple = (
        float(sigma1_lower) - float(sigma2_upper)
        > eps * max(1.0, abs(sigma1))
        if bounds_certified
        else (sigma1 - sigma2) > eps * max(1.0, abs(sigma1))
    )
    # Retain the complete algebraic plateau CREATED by the spectral return.
    # This is independent of the branch gap threshold and any near-root tolerance.
    top_q = int((S_returned == S_returned[0]).sum().item()) if is_exact else 0
    return SpectralPointCache(
        sigma1=sigma1,
        sigma2=sigma2,
        rel_gap=rel_gap,
        u1=U[:, :1].detach().to(torch.float32).clone(),
        v1=Vh[:1, :].transpose(-2, -1).detach().to(torch.float32).clone(),
        leading_simple=leading_simple,
        u_basis=U[:, : min(8, U.shape[1])].detach().to(torch.float32).clone(),
        v_basis=(
            Vh[: min(8, Vh.shape[0]), :]
            .transpose(-2, -1)
            .detach()
            .to(torch.float32)
            .clone()
        ),
        gap_lower=gap_lower,
        gap_upper=gap_upper,
        is_exact=is_exact,
        sigma1_lower=float(sigma1_lower),
        sigma1_upper=float(sigma1_upper),
        sigma2_lower=float(sigma2_lower),
        sigma2_upper=float(sigma2_upper),
        bounds_certified=bool(bounds_certified),
        top_u=U[:, :top_q].detach().float().clone() if top_q else None,
        top_v=Vh[:top_q].T.detach().float().clone() if top_q else None,
        s_basis=S_returned[: min(8, U.shape[1])].detach().float().clone(),
    )


def _clone_spectral_point_cache(cache: SpectralPointCache) -> SpectralPointCache:
    """Detach persistent cache tensors from temporary update objects."""
    return SpectralPointCache(
        sigma1=cache.sigma1,
        sigma2=cache.sigma2,
        rel_gap=cache.rel_gap,
        u1=cache.u1.detach().clone(),
        v1=cache.v1.detach().clone(),
        top_u=cache.top_u.detach().clone() if cache.top_u is not None else None,
        top_v=cache.top_v.detach().clone() if cache.top_v is not None else None,
        s_basis=cache.s_basis.detach().clone() if cache.s_basis is not None else None,
        leading_simple=cache.leading_simple,
        u_basis=(
            cache.u_basis.detach().clone()
            if cache.u_basis is not None
            else None
        ),
        v_basis=(
            cache.v_basis.detach().clone()
            if cache.v_basis is not None
            else None
        ),
        gap_lower=cache.gap_lower,
        gap_upper=cache.gap_upper,
        is_exact=cache.is_exact,
        sigma1_lower=cache.sigma1_lower,
        sigma1_upper=cache.sigma1_upper,
        sigma2_lower=cache.sigma2_lower,
        sigma2_upper=cache.sigma2_upper,
        bounds_certified=cache.bounds_certified,
    )


def _spectral_point_cache_is_compatible(
    cache: SpectralPointCache,
    matrix: torch.Tensor,
    target_radius: float,
    *,
    eps: float = 1e-5,
) -> bool:
    """Check metadata and feasibility before reusing post-return spectral data."""
    if matrix.ndim != 2:
        return False
    if cache.u1.shape != (matrix.shape[0], 1):
        return False
    if cache.v1.shape != (matrix.shape[1], 1):
        return False
    if cache.u1.device != matrix.device or cache.v1.device != matrix.device:
        return False
    if not all(
        math.isfinite(value)
        for value in (
            cache.sigma1,
            cache.sigma2,
            cache.rel_gap,
            cache.gap_lower,
            cache.gap_upper,
        )
    ):
        return False
    if not (0.0 <= cache.gap_lower <= cache.gap_upper <= 1.0):
        return False
    if cache.bounds_certified:
        if not all(
            math.isfinite(value)
            for value in (
                cache.sigma1_lower,
                cache.sigma1_upper,
                cache.sigma2_lower,
                cache.sigma2_upper,
            )
        ):
            return False
        if not (
            0.0 <= cache.sigma1_lower <= cache.sigma1_upper
            and 0.0 <= cache.sigma2_lower <= cache.sigma2_upper
        ):
            return False
    radius_scale = max(1.0, abs(target_radius))
    return abs(cache.sigma1 - target_radius) <= eps * radius_scale


@torch.no_grad()
def _float_value(x: Any, *, name: str) -> float:
    """Convert a Python scalar or scalar tensor to float."""
    if x is None:
        raise ValueError(f"{name} must not be None")
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape={tuple(x.shape)}")
        return float(x.detach().item())
    return float(x)


@torch.no_grad()
def _canonicalize_uv_for_matrix(
    W: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top singular vectors as column vectors compatible with ``W``.

    This mirrors the robustness in ``project_to_tangent_plane``: if a utility
    returns ``u`` and ``v`` in swapped order, dimensions are used to fix it.
    """
    W_fp32 = W.to(torch.float32)
    u_col = _as_column_unit_vector(u, eps=eps)
    v_col = _as_column_unit_vector(v, eps=eps)

    if u_col.shape[-2] != W_fp32.shape[-2] or v_col.shape[-2] != W_fp32.shape[-1]:
        if u_col.shape[-2] == W_fp32.shape[-1] and v_col.shape[-2] == W_fp32.shape[-2]:
            u_col, v_col = v_col, u_col
        else:
            raise ValueError(
                "Top singular vector dimensions are incompatible with W: "
                f"W.shape={tuple(W_fp32.shape)}, "
                f"u.shape={tuple(u.shape)}, v.shape={tuple(v.shape)}"
            )
    return u_col, v_col


@torch.no_grad()
def estimate_second_singular_value(
    W: torch.Tensor,
    sigma1: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    steps: int = 3,
    eps: float = 1e-8,
) -> float:
    """Estimate sigma_2(W) by power iteration on the rank-one residual.

    The estimate is cheap and is used only to decide whether the top singular
    direction is safe enough for the SpEL tangent-plane branch.
    """
    if steps < 1:
        return 0.0

    W_fp32 = W.to(torch.float32)
    u_col, v_col = _canonicalize_uv_for_matrix(W_fp32, u, v, eps=eps)
    residual = W_fp32 - sigma1.to(torch.float32) * torch.matmul(
        u_col, v_col.transpose(-2, -1)
    )
    sigma2, _, _ = power_iteration(residual, steps=steps)
    return float(sigma2.item())


@torch.no_grad()
def _prepare_right_warm_start(
    W: torch.Tensor,
    v_init: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    """Return a compatible normalized right-vector initial state."""
    default_v = torch.ones_like(W[..., :1, :].transpose(-2, -1), dtype=dtype)
    if v_init is None:
        return default_v

    candidate = v_init.detach().to(device=W.device, dtype=dtype)
    if candidate.shape != default_v.shape:
        return default_v

    return torch.nn.functional.normalize(candidate, dim=-2, eps=eps)


@torch.no_grad()
def power_iteration_warm_start(
    W: torch.Tensor,
    *,
    steps: int,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original BF16 leading singular triplet with an optional warm-started v."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if W.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    W_bf16 = W.to(torch.bfloat16)
    v = _prepare_right_warm_start(W_bf16, v_init, dtype=W_bf16.dtype, eps=eps)
    for _ in range(steps):
        v = torch.nn.functional.normalize(
            W_bf16.transpose(-2, -1) @ (W_bf16 @ v),
            dim=-2,
            eps=eps,
        )
    u = torch.nn.functional.normalize(W_bf16 @ v, dim=-2, eps=eps)
    sigma = (u.transpose(-2, -1) @ W_bf16 @ v).squeeze(-1).squeeze(-1)
    return sigma, u, v


@torch.no_grad()
def power_iteration_fp32(
    W: torch.Tensor,
    *,
    steps: int,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading singular triplet via the original PI structure, but in FP32."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if W.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    W_fp32 = W.to(torch.float32)
    v = _prepare_right_warm_start(W_fp32, v_init, dtype=W_fp32.dtype, eps=eps)
    for _ in range(steps):
        v = torch.nn.functional.normalize(
            W_fp32.transpose(-2, -1) @ (W_fp32 @ v),
            dim=-2,
            eps=eps,
        )
    u = torch.nn.functional.normalize(W_fp32 @ v, dim=-2, eps=eps)
    sigma = (u.transpose(-2, -1) @ W_fp32 @ v).squeeze(-1).squeeze(-1)
    return sigma, u, v


@torch.no_grad()
def power_iteration_fp32_highest(
    W: torch.Tensor,
    *,
    steps: int,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 PI with TF32 disabled through PyTorch's highest precision mode."""
    with utils.fp32_matmul_precision("highest"):
        return power_iteration_fp32(
            W,
            steps=steps,
            v_init=v_init,
            eps=eps,
        )


@torch.no_grad()
def power_iteration_fp64(
    W: torch.Tensor,
    *,
    steps: int,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading singular triplet of the stored matrix using FP64 PI."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if W.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    W_fp64 = W.to(torch.float64)
    v = _prepare_right_warm_start(W_fp64, v_init, dtype=W_fp64.dtype, eps=eps)
    for _ in range(steps):
        v = torch.nn.functional.normalize(
            W_fp64.transpose(-2, -1) @ (W_fp64 @ v),
            dim=-2,
            eps=eps,
        )
    u = torch.nn.functional.normalize(W_fp64 @ v, dim=-2, eps=eps)
    sigma = (u.transpose(-2, -1) @ W_fp64 @ v).squeeze(-1).squeeze(-1)
    return sigma, u, v


@torch.no_grad()
def _main_power_iteration(
    W: torch.Tensor,
    *,
    steps: int,
    v_init: Optional[torch.Tensor],
    main_power_dtype: MainPowerDtype,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the SpEL branch top triplet in the selected main-path dtype."""
    if main_power_dtype == "bf16":
        if v_init is None:
            return power_iteration(W, steps=steps)
        return power_iteration_warm_start(W, steps=steps, v_init=v_init, eps=eps)
    if main_power_dtype == "fp32":
        return power_iteration_fp32(W, steps=steps, v_init=v_init, eps=eps)
    if main_power_dtype == "fp32_highest":
        return power_iteration_fp32_highest(
            W,
            steps=steps,
            v_init=v_init,
            eps=eps,
        )
    if main_power_dtype == "fp64":
        sigma, u, v = power_iteration_fp64(
            W,
            steps=steps,
            v_init=v_init,
        )
        return sigma, u.to(torch.float32), v.to(torch.float32)
    raise ValueError(
        "main_power_dtype must be one of: 'bf16', 'fp32', "
        "'fp32_highest', 'fp64'"
    )


@torch.no_grad()
def estimate_second_singular_value_fp32(
    W: torch.Tensor,
    sigma1: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-8,
) -> float:
    """Estimate sigma_2 by FP32 rank-one deflation and FP32 power iteration."""
    if steps < 1:
        return 0.0

    W_fp32 = W.to(torch.float32)
    u_col, v_col = _canonicalize_uv_for_matrix(W_fp32, u, v, eps=eps)
    residual = W_fp32 - sigma1.to(torch.float32) * torch.matmul(
        u_col, v_col.transpose(-2, -1)
    )
    sigma2, _, _ = power_iteration_fp32(residual, steps=steps, eps=eps)
    return float(sigma2.item())


@torch.no_grad()
def _deterministic_right_block(
    n: int,
    block_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a deterministic orthonormal right block for subspace iteration."""
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    idx = torch.arange(1, n + 1, device=device, dtype=dtype)
    cols = []
    if v_init is not None:
        candidate = v_init.detach().to(device=device, dtype=dtype)
        if candidate.shape == (n, 1):
            cols.append(candidate.squeeze(-1))

    cols.append(torch.sin(0.73 * idx) + torch.cos(0.37 * idx))
    cols.append(torch.sin(1.37 * idx) + torch.cos(0.19 * idx))
    cols.append(torch.sin(2.11 * idx) + torch.cos(0.53 * idx))

    V0 = torch.stack(cols, dim=-1)
    Q, _ = torch.linalg.qr(V0, mode="reduced")
    return Q[:, :block_size]


@torch.no_grad()
def estimate_top2_singular_values_block2_fp32(
    W: torch.Tensor,
    *,
    steps: int = 10,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Estimate ``sigma1,u1,v1,sigma2`` with FP32 block subspace iteration.

    Unlike the cheaper deflated estimator, this computes the top two Ritz
    singular values from the same two-dimensional subspace.  The returned
    ``u1`` and ``v1`` are the first Ritz singular vectors and can be used by the
    SpEL branch exactly as the ordinary power-iteration vectors are used.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if W.ndim != 2:
        raise ValueError("block2_fp32 gap estimator expects one 2D matrix")

    W_fp32 = W.to(torch.float32)
    m, n = W_fp32.shape
    block_size = min(2, m, n)
    if block_size < 1:
        zero = W_fp32.new_tensor(0.0)
        u = W_fp32.new_zeros((m, 1))
        v = W_fp32.new_zeros((n, 1))
        return zero, u, v, 0.0

    V = _deterministic_right_block(
        n,
        block_size,
        device=W_fp32.device,
        dtype=W_fp32.dtype,
        v_init=v_init,
        eps=eps,
    )

    for _ in range(steps):
        U, _ = torch.linalg.qr(W_fp32 @ V, mode="reduced")
        V, _ = torch.linalg.qr(W_fp32.transpose(-2, -1) @ U, mode="reduced")

    B = U.transpose(-2, -1) @ W_fp32 @ V
    U_small, S, Vh_small = torch.linalg.svd(B, full_matrices=False)
    u1 = U @ U_small[:, :1]
    v1 = V @ Vh_small.transpose(-2, -1)[:, :1]

    sigma1 = S[0].clamp_min(eps)
    sigma2 = float(S[1].item()) if S.numel() > 1 else 0.0
    return sigma1, u1, v1, sigma2


@torch.no_grad()
def apply_spel_style_spectral_projection_(
    Z: torch.Tensor,
    target_radius: float,
    *,
    power_iteration_steps: int,
    retract_mode: str = "hard",
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Project/retract ``Z`` with the same operator used by the original SpEL.

    This is intentionally *not* an exact Euclidean SVD projection.  It uses
    ``power_iteration`` followed by ``apply_retract`` so the PGD projection and
    the SpEL projection/retraction are the same engineering operator.

    Args:
        Z: Trial matrix. It is modified in-place and also returned.
        target_radius: Spectral radius/sphere target.
        power_iteration_steps: Number of PI steps used to estimate sigma_1(Z).
        retract_mode: Same value accepted by ``apply_retract``.
        retract_alpha: Same value accepted by ``apply_retract``.
        current_lr: Current param-group lr; used only by dynamic retraction.

    Returns:
        ``(Z, retract_bias, sigma_before_retract)`` where ``Z`` has been
        retracted in-place.
    """
    sigma, _, _ = power_iteration(Z, steps=power_iteration_steps)
    sigma_value = float(sigma.item())
    retract_bias = apply_retract(
        Z,
        sigma_value,
        target_radius,
        mode=retract_mode,
        alpha=retract_alpha,
        current_lr=current_lr,
    )
    return Z, float(retract_bias), sigma_value


@torch.no_grad()
def project_to_spectral_sphere_local_top1_fp32(
    Z: torch.Tensor,
    target_radius: float,
    *,
    power_iteration_steps: int,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """Apply the smooth-stratum local Frobenius projection.

    When the leading singular value is simple and ``sigma2(Z) < target_radius``,
    the local projection is

        Z + (target_radius - sigma1(Z)) * u1(Z) * v1(Z)^T.

    The leading triplet is estimated in FP32 and may be initialized by the
    current-point right singular vector.
    """
    if power_iteration_steps < 1:
        raise ValueError("power_iteration_steps must be at least 1")

    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)
    sigma, u, v = power_iteration_fp32(
        Z_fp32,
        steps=power_iteration_steps,
        v_init=v_init,
        eps=eps,
    )
    sigma_value = float(sigma.item())
    u_col, v_col = _canonicalize_uv_for_matrix(Z_fp32, u, v, eps=eps)
    correction = (target_radius - sigma.to(torch.float32)) * torch.matmul(
        u_col, v_col.transpose(-2, -1)
    )
    return (Z_fp32 + correction).to(dtype=dtype), sigma_value


@torch.no_grad()
def project_to_spectral_sphere_exact(
    Z: torch.Tensor,
    target_radius: float,
) -> Tuple[torch.Tensor, float]:
    """Euclidean projection onto ``{W: ||W||_2 = target_radius}``.

    The projection is exact but uses a full SVD, so it is intended for the
    low-frequency PGD fallback or for small-model ablations.
    """
    Y, trial_sigma, _ = _project_to_spectral_sphere_exact_with_diagnostics(
        Z,
        target_radius,
    )
    return Y, trial_sigma


@torch.no_grad()
def _project_to_spectral_sphere_exact_with_diagnostics(
    Z: torch.Tensor,
    target_radius: float,
    *,
    build_spectral_cache: bool = False,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float, _PostProjectionDiagnostics]:
    """Exact projection plus gap statistics from the already computed SVD."""
    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)
    U, S, Vh = accurate_svd_fp32(Z_fp32)
    S_proj = S.clone()
    trial_sigma = float(S_proj[0].item()) if S_proj.numel() else 0.0

    if S_proj.numel() == 0:
        return Z_fp32.to(dtype=dtype), trial_sigma, _PostProjectionDiagnostics()

    trial_sigma2 = float(S[1].item()) if S.numel() > 1 else 0.0
    trial_rel_gap = max(0.0, 1.0 - trial_sigma2 / max(trial_sigma, eps))
    clipped_count = int(torch.count_nonzero(S > target_radius).item())

    if S_proj[0] >= target_radius:
        S_proj.clamp_(max=target_radius)
        S_proj[0] = target_radius
    else:
        S_proj[0] = target_radius

    Y = (U * S_proj.unsqueeze(-2)) @ Vh
    post_sigma1 = float(S_proj[0].item())
    post_sigma2 = float(S_proj[1].item()) if S_proj.numel() > 1 else 0.0
    post_rel_gap = max(0.0, 1.0 - post_sigma2 / max(post_sigma1, eps))
    next_spectral_cache = (
        _spectral_point_cache_from_decomposition(U, S_proj, Vh)
        if build_spectral_cache
        else None
    )
    diagnostics = _PostProjectionDiagnostics(
        trial_rel_gap=trial_rel_gap,
        post_rel_gap=post_rel_gap,
        clipped_count=clipped_count,
        next_spectral_cache=next_spectral_cache,
    )
    return Y.to(dtype=dtype), trial_sigma, diagnostics


@torch.no_grad()
def retract_to_spectral_sphere_exact_norm(
    Z: torch.Tensor,
    target_radius: float,
    *,
    singular_values: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float]:
    """Apply the exact radial return ``R * Z / ||Z||_2`` in FP32.

    Only singular values are requested because the radial map does not need
    singular vectors. The zero matrix has no radial return, so use the exact
    Frobenius projection as a deterministic fallback in that degenerate case.
    """
    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)
    if singular_values is None:
        singular_values = accurate_svdvals_fp32(Z_fp32)
    else:
        singular_values = singular_values.to(device=Z_fp32.device, dtype=torch.float32)
    trial_sigma = float(singular_values[0].item()) if singular_values.numel() else 0.0

    if trial_sigma <= eps:
        return project_to_spectral_sphere_exact(Z_fp32, target_radius)

    scale = target_radius / singular_values[0].clamp_min(eps)
    return (Z_fp32 * scale).to(dtype=dtype), trial_sigma


@torch.no_grad()
def _retract_to_spectral_sphere_exact_with_cache(
    Z: torch.Tensor,
    target_radius: float,
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float, SpectralPointCache]:
    """Radially return ``Z`` and cache the exact spectrum of the result."""
    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)
    U, S, Vh = accurate_svd_fp32(Z_fp32)
    trial_sigma = float(S[0].item()) if S.numel() else 0.0
    if trial_sigma <= eps:
        projected, trial_sigma, diagnostics = (
            _project_to_spectral_sphere_exact_with_diagnostics(
                Z_fp32,
                target_radius,
                build_spectral_cache=True,
            )
        )
        if diagnostics.next_spectral_cache is None:
            raise RuntimeError("zero-matrix projection did not produce a cache")
        return projected.to(dtype=dtype), trial_sigma, diagnostics.next_spectral_cache

    scale = target_radius / S[0].clamp_min(eps)
    returned_singular_values = S * scale
    cache = _spectral_point_cache_from_decomposition(
        U,
        returned_singular_values,
        Vh,
    )
    return (Z_fp32 * scale).to(dtype=dtype), trial_sigma, cache


@torch.no_grad()
def retract_to_spectral_sphere_pi_norm(
    Z: torch.Tensor,
    target_radius: float,
    *,
    power_iteration_steps: int,
    main_power_dtype: MainPowerDtype,
    v_init: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Apply ``R * Z / sigma1_hat`` using the selected PI implementation.

    This is an approximate radial return used to measure whether high-accuracy
    PI can replace the full-SVD norm in MCSD-TP. The returned vectors can seed
    the next PI call. A full SVD is used only for the degenerate zero estimate.
    """
    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)
    sigma, u, v = _main_power_iteration(
        Z_fp32,
        steps=power_iteration_steps,
        v_init=v_init,
        main_power_dtype=main_power_dtype,
        eps=eps,
    )
    trial_sigma = float(sigma.item())
    if trial_sigma <= eps:
        projected, exact_sigma = project_to_spectral_sphere_exact(
            Z_fp32,
            target_radius,
        )
        return projected.to(dtype=dtype), exact_sigma, u, v

    sigma_fp32 = sigma.to(device=Z_fp32.device, dtype=torch.float32)
    scale = target_radius / sigma_fp32.clamp_min(eps)
    return (Z_fp32 * scale).to(dtype=dtype), trial_sigma, u, v


@torch.no_grad()
def project_to_spectral_sphere_topk(
    Z: torch.Tensor,
    target_radius: float,
    *,
    rank: int,
    power_iteration_steps: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """Approximate spectral-sphere projection with top-k deflation.

    This avoids a full SVD.  If the top singular value is below the target,
    the top singular direction is lifted to ``target_radius``.  Otherwise, the
    first ``rank`` singular values estimated by repeated power iteration and
    deflation are clipped to ``target_radius``.  Singular values below the
    target stop the correction because the remaining deflated spectrum should
    be no larger than the current estimate.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if power_iteration_steps < 1:
        raise ValueError("power_iteration_steps must be at least 1")

    dtype = Z.dtype
    Y = Z.to(torch.float32)
    residual = Y.clone()
    trial_sigma = 0.0

    for idx in range(rank):
        sigma, u, v = power_iteration(residual, steps=power_iteration_steps)
        sigma_value = float(sigma.item())
        if idx == 0:
            trial_sigma = sigma_value
        if sigma_value <= eps:
            break

        u_col, v_col = _canonicalize_uv_for_matrix(residual, u, v, eps=eps)
        outer = torch.matmul(u_col, v_col.transpose(-2, -1))

        if idx == 0 and sigma_value < target_radius:
            Y = Y + (target_radius - sigma_value) * outer
            break

        if sigma_value <= target_radius:
            break

        Y = Y - (sigma_value - target_radius) * outer
        residual = residual - sigma_value * outer

    return Y.to(dtype=dtype), trial_sigma


@torch.no_grad()
def _pgd_direction(
    M: torch.Tensor,
    *,
    normalization: PGDDirectionNormalization = "none",
    power_iteration_steps: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the PGD fallback direction from momentum/Nesterov momentum."""
    D = M.to(torch.float32)
    if normalization == "none":
        return D
    if normalization == "fro":
        return D / torch.linalg.norm(D, dim=(-2, -1), keepdim=True).clamp_min(eps)
    if normalization == "spectral":
        sigma, _, _ = power_iteration(D, steps=power_iteration_steps)
        denom = sigma.to(torch.float32).clamp_min(eps)
        while denom.ndim < D.ndim:
            denom = denom.unsqueeze(-1)
        return D / denom
    if normalization == "exact_spectral":
        singular_values = accurate_svdvals_fp32(D)
        sigma = (
            singular_values[0]
            if singular_values.numel()
            else torch.zeros((), device=D.device, dtype=torch.float32)
        )
        return D / sigma.clamp_min(eps)
    raise ValueError(f"Invalid PGD direction normalization: {normalization}")


@torch.no_grad()
def _select_branch(
    *,
    branch_mode: BranchMode,
    use_pgd_fallback: bool,
    rel_gap: float,
    gap_threshold_rel: float,
) -> str:
    """Select SpEL or PGD branch without any line search."""
    if branch_mode == "spel":
        return "spel"
    if branch_mode == "pgd":
        return "pgd"
    if branch_mode != "auto":
        raise ValueError("branch_mode must be one of: 'auto', 'spel', 'pgd'")
    if use_pgd_fallback and rel_gap < gap_threshold_rel:
        return "pgd"
    return "spel"


def _matrix_spectral_norm_upper_bound(matrix: torch.Tensor) -> float:
    """Cheap certified upper bound: min(Frobenius, sqrt(one * infinity))."""
    value = matrix.to(torch.float32)
    frobenius = torch.linalg.vector_norm(value)
    absolute = value.abs()
    one_norm = absolute.sum(dim=0).max()
    infinity_norm = absolute.sum(dim=1).max()
    induced_bound = torch.sqrt(one_norm * infinity_norm)
    return float(torch.minimum(frobenius, induced_bound).item())


@torch.no_grad()
def _post_project_trial(
    Y: torch.Tensor,
    target_radius: float,
    *,
    branch: str,
    projection_mode: ProjectionMode,
    projection_rank: int,
    power_iteration_steps: int,
    main_power_dtype: MainPowerDtype,
    v_init: Optional[torch.Tensor],
    retract_mode: str,
    retract_alpha: float,
    current_lr: Optional[float],
    build_spectral_cache: bool = False,
    gap_threshold_rel: float = 3e-5,
    u_init: Optional[torch.Tensor] = None,
    krylov_initial_iterations: int = 8,
    krylov_max_iterations: int = 40,
    krylov_safe_block_size: int = 2,
    krylov_residual_tolerance: float = 2e-6,
    krylov_gap_interval_fraction: float = 0.2,
    multirate_exact_safe_return: bool = False,
    prior_spectral_cache: Optional[SpectralPointCache] = None,
    perturbation_norm_upper: float = 0.0,
    gap_probe_safe_multiplier: float = 10.0,
    hierarchical_rank1_iterations: int = 3,
    hierarchical_residual_gap_fraction: float = 0.1,
) -> Tuple[
    torch.Tensor,
    float,
    float,
    Optional[torch.Tensor],
    _PostProjectionDiagnostics,
]:
    """Apply the selected post-step projection/retraction to a trial point."""
    cache_projection_modes = (
        "tp_radial_pgd_exact",
        "tp_radial_pgd_adaptive_krylov",
        "tp_radial_pgd_adaptive_block2",
        "tp_radial_pgd_multirate",
        "tp_radial_pgd_hierarchical",
    )
    if build_spectral_cache and projection_mode not in cache_projection_modes:
        raise ValueError(
            "post-return spectral caching requires an exact or adaptive "
            "theory-facing projection mode"
        )

    if projection_mode == "tp_radial_pgd_adaptive_krylov":
        returned, U, returned_s, Vh, return_info = (
            adaptive_block_krylov_sphere_return(
                Y,
                target_radius,
                branch=branch,
                projection_rank=max(2, projection_rank),
                initial_iterations=krylov_initial_iterations,
                max_iterations=krylov_max_iterations,
                residual_tolerance=krylov_residual_tolerance,
                gap_threshold_rel=gap_threshold_rel,
                gap_interval_fraction=krylov_gap_interval_fraction,
                initial_u=u_init,
                initial_v=v_init,
            )
        )
        next_cache = _spectral_point_cache_from_decomposition(
            U,
            returned_s,
            Vh,
            gap_lower=return_info.gap_lower,
            gap_upper=return_info.gap_upper,
            is_exact=return_info.used_exact_fallback,
        )
        trial_gap = (
            0.5 * (return_info.gap_lower + return_info.gap_upper)
        )
        diagnostics = _PostProjectionDiagnostics(
            trial_rel_gap=trial_gap if branch == "pgd" else float("nan"),
            post_rel_gap=(next_cache.rel_gap if branch == "pgd" else float("nan")),
            clipped_count=return_info.clipped_count,
            next_spectral_cache=next_cache,
            approximate_iterations=return_info.iterations,
            approximate_residual_rel=return_info.max_residual_relative,
            approximate_gap_interval_width=(
                return_info.gap_upper - return_info.gap_lower
            ),
            approximate_exact_fallback=return_info.used_exact_fallback,
            approximate_subspace_dimension=return_info.subspace_dimension,
            approximate_mode=return_info.mode,
            approximate_rejection_reasons=return_info.rejection_reasons,
        )
        return returned, 0.0, return_info.trial_sigma1, None, diagnostics

    if projection_mode == "tp_radial_pgd_adaptive_block2":
        if branch == "spel":
            returned, U, returned_s, Vh, return_info = (
                adaptive_warm_fixed_subspace_radial_return(
                    Y,
                    target_radius,
                    gap_threshold_rel=gap_threshold_rel,
                    block_size=krylov_safe_block_size,
                    initial_u=u_init,
                    initial_v=v_init,
                    initial_iterations=krylov_initial_iterations,
                    max_iterations=krylov_max_iterations,
                    residual_tolerance=krylov_residual_tolerance,
                    gap_interval_fraction=krylov_gap_interval_fraction,
                )
            )
        else:
            returned, U, returned_s, Vh, return_info = (
                adaptive_block_krylov_sphere_return(
                    Y,
                    target_radius,
                    branch="pgd",
                    projection_rank=max(2, projection_rank),
                    initial_iterations=krylov_initial_iterations,
                    max_iterations=krylov_max_iterations,
                    residual_tolerance=krylov_residual_tolerance,
                    gap_threshold_rel=gap_threshold_rel,
                    gap_interval_fraction=krylov_gap_interval_fraction,
                    initial_u=u_init,
                    initial_v=v_init,
                )
            )
        next_cache = _spectral_point_cache_from_decomposition(
            U,
            returned_s,
            Vh,
            gap_lower=return_info.gap_lower,
            gap_upper=return_info.gap_upper,
            is_exact=return_info.used_exact_fallback,
        )
        diagnostics = _PostProjectionDiagnostics(
            trial_rel_gap=(
                0.5 * (return_info.gap_lower + return_info.gap_upper)
                if branch == "pgd"
                else float("nan")
            ),
            post_rel_gap=(next_cache.rel_gap if branch == "pgd" else float("nan")),
            clipped_count=return_info.clipped_count,
            next_spectral_cache=next_cache,
            approximate_iterations=return_info.iterations,
            approximate_residual_rel=return_info.max_residual_relative,
            approximate_gap_interval_width=(
                return_info.gap_upper - return_info.gap_lower
            ),
            approximate_exact_fallback=return_info.used_exact_fallback,
            approximate_subspace_dimension=return_info.subspace_dimension,
            approximate_mode=return_info.mode,
        )
        return returned, 0.0, return_info.trial_sigma1, None, diagnostics

    if projection_mode == "tp_radial_pgd_hierarchical":
        if not build_spectral_cache:
            raise ValueError(
                "tp_radial_pgd_hierarchical requires post-return spectral caching"
            )
        if branch == "pgd":
            projected, trial_sigma, diagnostics = (
                _project_to_spectral_sphere_exact_with_diagnostics(
                    Y,
                    target_radius,
                    build_spectral_cache=True,
                )
            )
            return projected, 0.0, trial_sigma, None, diagnostics
        if multirate_exact_safe_return:
            projected, trial_sigma, next_cache = (
                _retract_to_spectral_sphere_exact_with_cache(
                    Y,
                    target_radius,
                )
            )
            diagnostics = _PostProjectionDiagnostics(
                next_spectral_cache=next_cache,
                approximate_mode="warning_exact",
            )
            return projected, 0.0, trial_sigma, None, diagnostics

        prior = prior_spectral_cache
        returned, U, returned_s, Vh, return_info = hierarchical_warm_radial_return(
            Y,
            target_radius,
            warning_gap_rel=gap_threshold_rel * gap_probe_safe_multiplier,
            prior_sigma1_lower=(prior.sigma1_lower if prior is not None else float("nan")),
            prior_sigma1_upper=(prior.sigma1_upper if prior is not None else float("nan")),
            prior_sigma2_lower=(prior.sigma2_lower if prior is not None else float("nan")),
            prior_sigma2_upper=(prior.sigma2_upper if prior is not None else float("nan")),
            prior_bounds_certified=(
                prior.bounds_certified if prior is not None else False
            ),
            perturbation_norm_upper=perturbation_norm_upper,
            initial_u=u_init,
            initial_v=v_init,
            rank1_iterations=hierarchical_rank1_iterations,
            residual_gap_fraction=hierarchical_residual_gap_fraction,
            max_block_size=krylov_safe_block_size,
            block_initial_iterations=krylov_initial_iterations,
            block_max_iterations=krylov_max_iterations,
            residual_tolerance=krylov_residual_tolerance,
        )
        next_cache = _spectral_point_cache_from_decomposition(
            U,
            returned_s,
            Vh,
            gap_lower=return_info.gap_lower,
            gap_upper=return_info.gap_upper,
            is_exact=return_info.used_exact_fallback,
            sigma1_lower=return_info.sigma1_lower,
            sigma1_upper=return_info.sigma1_upper,
            sigma2_lower=return_info.sigma2_lower,
            sigma2_upper=return_info.sigma2_upper,
            bounds_certified=return_info.bounds_certified,
        )
        diagnostics = _PostProjectionDiagnostics(
            next_spectral_cache=next_cache,
            approximate_iterations=return_info.iterations,
            approximate_residual_rel=return_info.max_residual_relative,
            approximate_gap_interval_width=(
                return_info.gap_upper - return_info.gap_lower
            ),
            approximate_exact_fallback=return_info.used_exact_fallback,
            approximate_subspace_dimension=return_info.subspace_dimension,
            approximate_mode=return_info.mode,
            approximate_rejection_reasons=return_info.rejection_reasons,
        )
        return returned, 0.0, return_info.trial_sigma1, None, diagnostics

    if projection_mode == "tp_radial_pgd_multirate":
        if not build_spectral_cache:
            raise ValueError(
                "tp_radial_pgd_multirate requires post-return spectral caching"
            )
        if branch == "pgd":
            projected, trial_sigma, diagnostics = (
                _project_to_spectral_sphere_exact_with_diagnostics(
                    Y,
                    target_radius,
                    build_spectral_cache=True,
                )
            )
            return projected, 0.0, trial_sigma, None, diagnostics
        if multirate_exact_safe_return:
            projected, trial_sigma, next_cache = (
                _retract_to_spectral_sphere_exact_with_cache(
                    Y,
                    target_radius,
                )
            )
            diagnostics = _PostProjectionDiagnostics(
                next_spectral_cache=next_cache,
            )
            return projected, 0.0, trial_sigma, None, diagnostics

        returned, U, returned_s, Vh, return_info = (
            adaptive_warm_fixed_subspace_radial_return(
                Y,
                target_radius,
                gap_threshold_rel=gap_threshold_rel,
                block_size=krylov_safe_block_size,
                initial_u=u_init,
                initial_v=v_init,
                initial_iterations=krylov_initial_iterations,
                max_iterations=krylov_max_iterations,
                residual_tolerance=krylov_residual_tolerance,
                gap_interval_fraction=krylov_gap_interval_fraction,
                require_gap_certificate=False,
            )
        )
        next_cache = _spectral_point_cache_from_decomposition(
            U,
            returned_s,
            Vh,
            gap_lower=return_info.gap_lower,
            gap_upper=return_info.gap_upper,
            is_exact=return_info.used_exact_fallback,
        )
        diagnostics = _PostProjectionDiagnostics(
            next_spectral_cache=next_cache,
            approximate_iterations=return_info.iterations,
            approximate_residual_rel=return_info.max_residual_relative,
            approximate_gap_interval_width=(
                return_info.gap_upper - return_info.gap_lower
            ),
            approximate_exact_fallback=return_info.used_exact_fallback,
            approximate_subspace_dimension=return_info.subspace_dimension,
            approximate_mode=return_info.mode,
        )
        return returned, 0.0, return_info.trial_sigma1, None, diagnostics

    if projection_mode == "tp_radial_pi":
        if branch != "spel":
            raise ValueError("tp_radial_pi supports only the MCSD-TP branch")
        projected, trial_sigma, _, trial_v = retract_to_spectral_sphere_pi_norm(
            Y,
            target_radius,
            power_iteration_steps=power_iteration_steps,
            main_power_dtype=main_power_dtype,
            v_init=v_init,
        )
        return (
            projected,
            0.0,
            trial_sigma,
            trial_v.detach(),
            _PostProjectionDiagnostics(),
        )

    if projection_mode in (
        "tp_radial_pgd_exact",
        "tp_radial_pgd_topk",
        "tp_radial_pgd_krylov_topk",
    ):
        if branch == "spel":
            if build_spectral_cache:
                projected, trial_sigma, next_cache = (
                    _retract_to_spectral_sphere_exact_with_cache(
                        Y,
                        target_radius,
                    )
                )
                return (
                    projected,
                    0.0,
                    trial_sigma,
                    None,
                    _PostProjectionDiagnostics(next_spectral_cache=next_cache),
                )
            projected, trial_sigma = retract_to_spectral_sphere_exact_norm(
                Y,
                target_radius,
            )
        elif projection_mode == "tp_radial_pgd_topk":
            projected, trial_sigma = project_to_spectral_sphere_topk(
                Y,
                target_radius,
                rank=projection_rank,
                power_iteration_steps=power_iteration_steps,
            )
        elif projection_mode == "tp_radial_pgd_krylov_topk":
            projected, projection_info = block_krylov_topk_sphere_projection(
                Y,
                target_radius,
                rank=projection_rank,
            )
            trial_sigma = projection_info.trial_sigma1
        else:
            (
                projected,
                trial_sigma,
                projection_diagnostics,
            ) = _project_to_spectral_sphere_exact_with_diagnostics(
                Y,
                target_radius,
                build_spectral_cache=build_spectral_cache,
            )
            return projected, 0.0, trial_sigma, None, projection_diagnostics
        return projected, 0.0, trial_sigma, None, _PostProjectionDiagnostics()

    if projection_mode == "local_top1_fallback_exact":
        if branch == "spel":
            projected, trial_sigma = project_to_spectral_sphere_local_top1_fp32(
                Y,
                target_radius,
                power_iteration_steps=power_iteration_steps,
                v_init=v_init,
            )
        else:
            projected, trial_sigma, projection_diagnostics = (
                _project_to_spectral_sphere_exact_with_diagnostics(
                    Y,
                    target_radius,
                )
            )
            return projected, 0.0, trial_sigma, None, projection_diagnostics
        return projected, 0.0, trial_sigma, None, _PostProjectionDiagnostics()

    if projection_mode in ("fallback_exact", "shared_exact"):
        projected, trial_sigma, projection_diagnostics = (
            _project_to_spectral_sphere_exact_with_diagnostics(
                Y,
                target_radius,
            )
        )
        return projected, 0.0, trial_sigma, None, projection_diagnostics

    if projection_mode == "fallback_retraction_exact_norm":
        projected, trial_sigma = retract_to_spectral_sphere_exact_norm(
            Y,
            target_radius,
        )
        return projected, 0.0, trial_sigma, None, _PostProjectionDiagnostics()

    if projection_mode in ("fallback_topk", "shared_topk"):
        projected, trial_sigma = project_to_spectral_sphere_topk(
            Y,
            target_radius,
            rank=projection_rank,
            power_iteration_steps=power_iteration_steps,
        )
        return projected, 0.0, trial_sigma, None, _PostProjectionDiagnostics()

    if projection_mode in ("fallback_retraction", "shared_retraction"):
        projected, retract_bias, trial_sigma = apply_spel_style_spectral_projection_(
            Y,
            target_radius,
            power_iteration_steps=power_iteration_steps,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
        )
        return (
            projected,
            retract_bias,
            trial_sigma,
            None,
            _PostProjectionDiagnostics(),
        )

    raise ValueError(
        "projection_mode must be one of: fallback_exact, fallback_retraction, "
        "fallback_retraction_exact_norm, fallback_topk, "
        "local_top1_fallback_exact, shared_exact, "
        "shared_retraction, shared_topk, tp_radial_pgd_exact, "
        "tp_radial_pgd_topk, tp_radial_pgd_krylov_topk, "
        "tp_radial_pgd_adaptive_krylov, tp_radial_pgd_adaptive_block2, "
        "tp_radial_pgd_multirate, tp_radial_pi"
    )


@torch.no_grad()
def compute_spel_pgd_same_projection_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    *,
    effective_lr: float,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
    retract_mode: str = "hard",
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_pgd_fallback: bool = True,
    branch_mode: BranchMode = "auto",
    gap_threshold_rel: float = 1e-3,
    gap_probe_safe_multiplier: float = 10.0,
    sigma2_power_iteration_steps: int = 5,
    gap_estimator_mode: GapEstimatorMode = "deflated_power",
    pgd_direction_normalization: PGDDirectionNormalization = "spectral",
    fallback_direction: str = "pgd",
    pgd_lr_scale: float = 0.5,
    projection_mode: ProjectionMode = "fallback_exact",
    projection_rank: int = 1,
    tangent_project_after_msign: bool = False,
    normalize_msign_spectral: bool = False,
    msign_spectral_norm_steps: int = 2,
    main_power_dtype: MainPowerDtype = "bf16",
    warm_start_v: Optional[torch.Tensor] = None,
    audit_exact_gap: bool = False,
    force_exact_current_spectrum: bool = False,
    post_return_spectral_cache: bool = False,
    current_spectral_cache: Optional[SpectralPointCache] = None,
    krylov_initial_iterations: int = 8,
    krylov_max_iterations: int = 40,
    krylov_safe_block_size: int = 2,
    krylov_residual_tolerance: float = 2e-6,
    krylov_gap_interval_fraction: float = 0.2,
    hierarchical_rank1_iterations: int = 3,
    hierarchical_residual_gap_fraction: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, SpELPGDUpdateInfo]:
    """Compute a SpEL--PGD update.

    The function follows the same external optimizer contract as ``SpEL``:
    it returns an update direction ``u``.  The caller should still multiply it
    by the spectral-ball scale factor before returning to the base optimizer.

    In ``fallback_*`` modes, the SpEL branch returns the original SpEL direction
    and only the PGD fallback branch encodes a projected target ``Y`` as
    ``u = (W_retracted - Y) / (current_lr * scale_factor)``.  In ``shared_*``
    modes, both branches encode a post-projected target this way.
    ``tp_radial_pgd_exact`` instead applies an exact-norm radial retraction to
    the tangent SpEL branch and an exact Frobenius projection to the ambient
    PGD branch. ``tp_radial_pgd_topk`` keeps the same tangent/radial branch but
    replaces only the ambient PGD projection with sequential PI/deflation.
    ``tp_radial_pgd_krylov_topk`` instead uses a guarded FP32 block-Krylov
    top-k correction and falls back to full SVD when its residual or clipping
    boundary test is ambiguous. ``tp_radial_pi`` is the matched MCSD-TP
    calibration mode: it uses the configured PI for both the tangent normal
    and the post-step radial norm, without an SVD unless auditing is enabled.
    ``tp_radial_pgd_multirate`` uses a fixed warm subspace for MCSD radial
    returns, exact full-SVD projection for PGD, and lets the optimizer schedule
    exact current-point gap refreshes independently.

    No line search, loss reevaluation, Armijo, or Wolfe step is performed.
    """
    from .spectral_ball_utils import (
        _tp_gather_along_dim,
        _tp_split_along_dim,
        _tp_world_and_rank,
    )

    if power_iteration_steps < 1:
        raise ValueError("power_iteration_steps must be at least 1")
    if msign_steps < 1:
        raise ValueError("msign_steps must be at least 1")
    if gap_threshold_rel < 0.0:
        raise ValueError("gap_threshold_rel must be non-negative")
    if gap_probe_safe_multiplier < 1.0:
        raise ValueError("gap_probe_safe_multiplier must be at least 1")
    if main_power_dtype not in ("bf16", "fp32", "fp32_highest", "fp64"):
        raise ValueError(
            "main_power_dtype must be one of: 'bf16', 'fp32', "
            "'fp32_highest', 'fp64'"
        )
    if gap_estimator_mode not in (
        "deflated_power",
        "deflated_fp32_gap_only",
        "block2_fp32",
        "block2_fp32_gap_only",
        "exact_svd_gap_only",
    ):
        raise ValueError(
            "gap_estimator_mode must be one of: 'deflated_power', "
            "'deflated_fp32_gap_only', "
            "'block2_fp32', 'block2_fp32_gap_only', 'exact_svd_gap_only'"
        )
    if pgd_lr_scale <= 0.0:
        raise ValueError("pgd_lr_scale must be positive")
    if projection_mode not in (
        "fallback_exact",
        "fallback_retraction",
        "fallback_retraction_exact_norm",
        "fallback_topk",
        "local_top1_fallback_exact",
        "shared_exact",
        "shared_retraction",
        "shared_topk",
        "tp_radial_pgd_exact",
        "tp_radial_pgd_topk",
        "tp_radial_pgd_krylov_topk",
        "tp_radial_pgd_adaptive_krylov",
        "tp_radial_pgd_adaptive_block2",
        "tp_radial_pgd_multirate",
        "tp_radial_pgd_hierarchical",
        "tp_radial_pi",
    ):
        raise ValueError(
            "projection_mode must be one of: fallback_exact, fallback_retraction, "
            "fallback_retraction_exact_norm, fallback_topk, "
            "local_top1_fallback_exact, shared_exact, "
            "shared_retraction, shared_topk, tp_radial_pgd_exact, "
            "tp_radial_pgd_topk, tp_radial_pgd_krylov_topk, "
            "tp_radial_pgd_adaptive_krylov, tp_radial_pgd_adaptive_block2, "
            "tp_radial_pgd_multirate, tp_radial_pgd_hierarchical, tp_radial_pi"
        )
    if projection_rank < 1:
        raise ValueError("projection_rank must be at least 1")
    if not (1 <= krylov_initial_iterations <= krylov_max_iterations):
        raise ValueError("invalid adaptive Krylov iteration range")
    if not (2 <= krylov_safe_block_size <= 8):
        raise ValueError("krylov_safe_block_size must be between 2 and 8")
    if krylov_residual_tolerance <= 0.0:
        raise ValueError("krylov_residual_tolerance must be positive")
    if krylov_gap_interval_fraction <= 0.0:
        raise ValueError("krylov_gap_interval_fraction must be positive")
    if hierarchical_rank1_iterations < 1:
        raise ValueError("hierarchical_rank1_iterations must be at least 1")
    if not (0.0 < hierarchical_residual_gap_fraction < 1.0):
        raise ValueError("hierarchical_residual_gap_fraction must be in (0, 1)")
    if post_return_spectral_cache and projection_mode not in (
        "tp_radial_pgd_exact",
        "tp_radial_pgd_adaptive_krylov",
        "tp_radial_pgd_adaptive_block2",
        "tp_radial_pgd_multirate",
        "tp_radial_pgd_hierarchical",
    ):
        raise ValueError(
            "post_return_spectral_cache requires an exact or adaptive "
            "theory-facing projection mode"
        )
    if projection_mode in (
        "tp_radial_pgd_multirate",
        "tp_radial_pgd_hierarchical",
    ) and not post_return_spectral_cache:
        raise ValueError(
            f"{projection_mode} requires post_return_spectral_cache=True"
        )
    if (
        projection_mode
        in (
            "tp_radial_pgd_exact",
            "tp_radial_pgd_topk",
            "tp_radial_pgd_krylov_topk",
            "tp_radial_pgd_adaptive_krylov",
            "tp_radial_pgd_adaptive_block2",
            "tp_radial_pgd_multirate",
            "tp_radial_pgd_hierarchical",
            "tp_radial_pi",
        )
        and branch_mode != "pgd"
        and not tangent_project_after_msign
    ):
        raise ValueError(
            f"{projection_mode} requires tangent_project_after_msign=True "
            "whenever the SpEL/MCSD branch can be selected"
        )
    if projection_mode == "tp_radial_pi" and branch_mode != "spel":
        raise ValueError("tp_radial_pi requires branch_mode='spel'")

    effective_lr_value = float(effective_lr)

    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if tp_enabled:
        W_work = _tp_gather_along_dim(W, tp_group, partition_dim)
        M_work = _tp_gather_along_dim(M, tp_group, partition_dim)
    else:
        W_work = W
        M_work = M

    should_estimate_gap = branch_mode == "auto" and use_pgd_fallback
    if fallback_direction not in ("pgd", "p2gd"):
        raise ValueError("fallback_direction must be pgd or p2gd")
    if fallback_direction == "p2gd" and (
        projection_mode != "tp_radial_pgd_exact"
        or not post_return_spectral_cache
        or pgd_direction_normalization != "none"
        or not tangent_project_after_msign
    ):
        raise ValueError("P2GD requires cached exact returns, TP, and raw momentum")

    theory_projection_mode = projection_mode in (
        "tp_radial_pgd_exact",
        "tp_radial_pgd_topk",
        "tp_radial_pgd_krylov_topk",
        "tp_radial_pgd_adaptive_krylov",
        "tp_radial_pgd_adaptive_block2",
        "tp_radial_pgd_multirate",
        "tp_radial_pgd_hierarchical",
    )
    exact_singular_values_cache: Optional[torch.Tensor] = None
    exact_left_vectors_cache: Optional[torch.Tensor] = None
    exact_right_vectors_h_cache: Optional[torch.Tensor] = None
    current_leading_simple = True
    spectral_cache_hit = bool(
        post_return_spectral_cache
        and current_spectral_cache is not None
        and not force_exact_current_spectrum
        and _spectral_point_cache_is_compatible(
            current_spectral_cache,
            W_work,
            target_radius,
        )
    )
    if spectral_cache_hit:
        assert current_spectral_cache is not None
        current_leading_simple = current_spectral_cache.leading_simple
        exact_singular_values_cache = torch.tensor(
            [current_spectral_cache.sigma1, current_spectral_cache.sigma2],
            device=W_work.device,
            dtype=torch.float32,
        )
        exact_left_vectors_cache = current_spectral_cache.u1.to(
            device=W_work.device,
            dtype=torch.float32,
        )
        exact_right_vectors_h_cache = current_spectral_cache.v1.to(
            device=W_work.device,
            dtype=torch.float32,
        ).transpose(-2, -1)
    elif theory_projection_mode:
        # A theory-facing tangent step needs the leading singular vectors, not
        # only an exact spectral norm. Reuse one full decomposition for the
        # tangent normal, pre-step radial return, and exact gap decision.
        (
            exact_left_vectors_cache,
            exact_singular_values_cache,
            exact_right_vectors_h_cache,
        ) = accurate_svd_fp32(W_work)
        exact_sigma1_for_simple = float(exact_singular_values_cache[0].item())
        exact_sigma2_for_simple = (
            float(exact_singular_values_cache[1].item())
            if exact_singular_values_cache.numel() > 1
            else 0.0
        )
        current_leading_simple = (
            exact_sigma1_for_simple - exact_sigma2_for_simple
        ) > eps * max(1.0, abs(exact_sigma1_for_simple))
    elif audit_exact_gap or (
        should_estimate_gap and gap_estimator_mode == "exact_svd_gap_only"
    ):
        exact_singular_values_cache = accurate_svdvals_fp32(W_work)

    if theory_projection_mode:
        if (
            exact_singular_values_cache is None
            or exact_left_vectors_cache is None
            or exact_right_vectors_h_cache is None
            or exact_singular_values_cache.numel() == 0
        ):
            raise ValueError("theory projection modes require a nonempty matrix")
        sigma1 = exact_singular_values_cache[0]
        u = exact_left_vectors_cache[:, :1]
        v = exact_right_vectors_h_cache[:1, :].transpose(-2, -1)
        sigma1_value = float(sigma1.item())
        gap_sigma1_value = sigma1_value
    elif should_estimate_gap and gap_estimator_mode == "block2_fp32":
        # The top Ritz triplet is still the SpEL tangent object; sigma2 is only
        # used for deciding whether that object is sufficiently well separated.
        block_steps = max(power_iteration_steps, sigma2_power_iteration_steps)
        sigma1, u, v, sigma2_value = estimate_top2_singular_values_block2_fp32(
            W_work,
            steps=block_steps,
            v_init=warm_start_v,
            eps=eps,
        )
        sigma1_value = float(sigma1.item())
        gap_sigma1_value = sigma1_value
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    else:
        # Estimate top singular data on the incoming current matrix.  This mirrors
        # original SpEL in bf16 mode and supplies the default deflated gap-test data.
        sigma1, u, v = _main_power_iteration(
            W_work,
            steps=power_iteration_steps,
            v_init=warm_start_v,
            main_power_dtype=main_power_dtype,
            eps=eps,
        )
        sigma1_value = float(sigma1.item())
        gap_sigma1_value = sigma1_value

    if should_estimate_gap and post_return_spectral_cache:
        if exact_singular_values_cache is None:
            raise RuntimeError("post-return spectral cache requires exact top-two data")
        gap_sigma1_value = (
            float(exact_singular_values_cache[0].item())
            if exact_singular_values_cache.numel()
            else 0.0
        )
        sigma2_value = (
            float(exact_singular_values_cache[1].item())
            if exact_singular_values_cache.numel() > 1
            else 0.0
        )
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    elif should_estimate_gap and gap_estimator_mode == "deflated_power":
        sigma2_value = estimate_second_singular_value(
            W_work,
            sigma1,
            u,
            v,
            steps=sigma2_power_iteration_steps,
            eps=eps,
        )
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    elif should_estimate_gap and gap_estimator_mode == "deflated_fp32_gap_only":
        # Preserve the original SpEL top-vector path above, but recompute the
        # original deflated gap rule in FP32 for a cleaner region test.
        gap_sigma1, gap_u, gap_v = power_iteration_fp32(
            W_work,
            steps=power_iteration_steps,
            v_init=warm_start_v,
            eps=eps,
        )
        gap_sigma1_value = float(gap_sigma1.item())
        sigma2_value = estimate_second_singular_value_fp32(
            W_work,
            gap_sigma1,
            gap_u,
            gap_v,
            steps=sigma2_power_iteration_steps,
            eps=eps,
        )
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    elif should_estimate_gap and gap_estimator_mode == "block2_fp32_gap_only":
        # Preserve the original SpEL top-vector path above, but use a separate
        # FP32 block-2 estimate only for the region/gap decision.
        block_steps = max(power_iteration_steps, sigma2_power_iteration_steps)
        gap_sigma1, _, _, sigma2_value = estimate_top2_singular_values_block2_fp32(
            W_work,
            steps=block_steps,
            v_init=warm_start_v,
            eps=eps,
        )
        gap_sigma1_value = float(gap_sigma1.item())
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    elif should_estimate_gap and gap_estimator_mode == "exact_svd_gap_only":
        if exact_singular_values_cache is None:
            exact_singular_values_cache = accurate_svdvals_fp32(W_work)
        gap_sigma1_value = (
            float(exact_singular_values_cache[0].item())
            if exact_singular_values_cache.numel()
            else 0.0
        )
        sigma2_value = (
            float(exact_singular_values_cache[1].item())
            if exact_singular_values_cache.numel() > 1
            else 0.0
        )
        rel_gap = max(0.0, 1.0 - sigma2_value / max(gap_sigma1_value, eps))
    elif not should_estimate_gap:
        sigma2_value = 0.0
        gap_sigma1_value = sigma1_value
        rel_gap = float("inf")

    exact_sigma1 = float("nan")
    exact_sigma2 = float("nan")
    exact_rel_gap = float("nan")
    delta1_rel = float("nan")
    delta2_rel = float("nan")
    gap_abs_error = float("nan")
    main_sigma1_rel_error = float("nan")
    main_triplet_residual_rel = float("nan")
    if audit_exact_gap:
        W_audit = W_work.to(torch.float32)
        cache_is_approximate = bool(
            spectral_cache_hit
            and current_spectral_cache is not None
            and not current_spectral_cache.is_exact
        )
        exact_singular_values = (
            None if cache_is_approximate else exact_singular_values_cache
        )
        if exact_singular_values is None:
            exact_singular_values = accurate_svdvals_fp32(W_audit)
        if exact_singular_values.numel():
            exact_sigma1 = float(exact_singular_values[0].item())
            exact_sigma2 = (
                float(exact_singular_values[1].item())
                if exact_singular_values.numel() > 1
                else 0.0
            )
            exact_scale = max(exact_sigma1, eps)
            exact_rel_gap = max(0.0, 1.0 - exact_sigma2 / exact_scale)
            main_sigma1_rel_error = abs(sigma1_value - exact_sigma1) / exact_scale

            u_col, v_col = _canonicalize_uv_for_matrix(W_audit, u, v, eps=eps)
            sigma1_fp32 = torch.as_tensor(
                sigma1_value,
                device=W_audit.device,
                dtype=torch.float32,
            )
            left_residual = torch.linalg.vector_norm(
                W_audit @ v_col - sigma1_fp32 * u_col
            )
            right_residual = torch.linalg.vector_norm(
                W_audit.transpose(-2, -1) @ u_col - sigma1_fp32 * v_col
            )
            main_triplet_residual_rel = float(
                torch.maximum(left_residual, right_residual).item() / exact_scale
            )

            if should_estimate_gap:
                delta1_rel = abs(gap_sigma1_value - exact_sigma1) / exact_scale
                delta2_rel = abs(sigma2_value - exact_sigma2) / exact_scale
                gap_abs_error = abs(rel_gap - exact_rel_gap)

    # The theory-facing modes use an exact-norm radial correction at the
    # current point. Positive radial scaling preserves both the exact singular
    # vectors used above and the relative singular-value gap.
    current_return_cache: Optional[SpectralPointCache] = None
    if theory_projection_mode:
        W_retracted, _ = retract_to_spectral_sphere_exact_norm(
            W_work,
            target_radius,
            singular_values=exact_singular_values_cache,
        )
        W_work.copy_(W_retracted)
        pre_retract_bias = 0.0
        if post_return_spectral_cache:
            if spectral_cache_hit:
                assert current_spectral_cache is not None
                current_return_cache = _clone_spectral_point_cache(
                    current_spectral_cache
                )
            else:
                assert exact_singular_values_cache is not None
                assert exact_left_vectors_cache is not None
                assert exact_right_vectors_h_cache is not None
                radial_scale = target_radius / exact_singular_values_cache[0].clamp_min(
                    eps
                )
                current_return_cache = _spectral_point_cache_from_decomposition(
                    exact_left_vectors_cache,
                    exact_singular_values_cache * radial_scale,
                    exact_right_vectors_h_cache,
                )
    else:
        # Preserve the original SpEL engineering retraction for existing modes.
        pre_retract_bias = apply_retract(
            W_work,
            sigma1_value,
            target_radius,
            mode=retract_mode,
            alpha=retract_alpha,
            current_lr=current_lr,
        )

    W_base = W_work.to(torch.float32)

    # If lr is zero, keep only the in-place current retraction and return zero.
    if effective_lr_value <= 0.0:
        update_unscaled = torch.zeros_like(W_base)
        info = SpELPGDUpdateInfo(
            branch="zero_lr",
            rel_gap=rel_gap,
            sigma1=sigma1_value,
            gap_sigma1=gap_sigma1_value,
            sigma2=sigma2_value,
            pre_retract_bias=float(pre_retract_bias),
            trial_sigma=sigma1_value,
            post_retract_bias=0.0,
            projection_mode=projection_mode,
            projection_rank=projection_rank,
            gap_probed=should_estimate_gap,
            next_v=v.detach(),
            gap_audited=audit_exact_gap,
            exact_sigma1=exact_sigma1,
            exact_sigma2=exact_sigma2,
            exact_rel_gap=exact_rel_gap,
            delta1_rel=delta1_rel,
            delta2_rel=delta2_rel,
            gap_abs_error=gap_abs_error,
            main_sigma1_rel_error=main_sigma1_rel_error,
            main_triplet_residual_rel=main_triplet_residual_rel,
            spectral_cache_hit=spectral_cache_hit,
            next_spectral_cache=current_return_cache,
        )
    else:
        if (
            post_return_spectral_cache
            and branch_mode == "auto"
            and not current_leading_simple
        ):
            # A repeated leading singular value has no unique tangent normal.
            # Force the ambient branch even when the configured threshold is 0.
            branch = "pgd"
        else:
            branch = _select_branch(
                branch_mode=branch_mode,
                use_pgd_fallback=use_pgd_fallback,
                rel_gap=rel_gap,
                gap_threshold_rel=gap_threshold_rel,
            )
        projection_diagnostics = _PostProjectionDiagnostics()
        p2gd_top_multiplicity = 0

        if branch == "spel":
            M_projected = project_to_tangent_plane(M_work, u, v, eps=eps)
            M_projected = M_projected / torch.linalg.norm(
                M_projected, dim=(-2, -1), keepdim=True
            ).clamp_min(eps)
            D = msign(M_projected, steps=msign_steps)
            if tangent_project_after_msign:
                D = project_to_tangent_plane(D, u, v, eps=eps)
            if normalize_msign_spectral:
                D = normalize_direction_spectral_fp32(
                    D,
                    steps=msign_spectral_norm_steps,
                    eps=eps,
                )
        else:
            D = _pgd_direction(
                M_work,
                normalization=pgd_direction_normalization,
                power_iteration_steps=power_iteration_steps,
                eps=eps,
            )
            if fallback_direction == "p2gd":
                if (
                    current_return_cache is None
                    or not current_return_cache.is_exact
                    or current_return_cache.top_u is None
                    or current_return_cache.top_v is None
                ):
                    raise ValueError("P2GD requires a complete current top-subspace cache")
                p2gd_top_multiplicity = current_return_cache.top_u.shape[1]
                # D is a positive descent-gradient convention; the trial below
                # subtracts D. The cone is not symmetric, so project -D first.
                D = -project_spectral_sphere_cone(
                    -D, current_return_cache.top_u, current_return_cache.top_v
                )

        if branch == "spel" and projection_mode.startswith("fallback_"):
            # Default engineering path: preserve original SpEL exactly in the
            # safe region.  The current W has already been pre-retracted, and
            # the outer optimizer will apply W <- W - lr * scale * D.
            update_unscaled = D.to(torch.float32)
            post_retract_bias = 0.0
            trial_sigma = sigma1_value
        else:
            trial_lr = effective_lr_value * (pgd_lr_scale if branch == "pgd" else 1.0)
            Y = W_base - trial_lr * D.to(torch.float32)
            perturbation_norm_upper = trial_lr * _matrix_spectral_norm_upper_bound(D)
            (
                Y,
                post_retract_bias,
                trial_sigma,
                trial_v,
                projection_diagnostics,
            ) = _post_project_trial(
                Y,
                target_radius,
                branch=branch,
                projection_mode=projection_mode,
                projection_rank=projection_rank,
                power_iteration_steps=power_iteration_steps,
                main_power_dtype=main_power_dtype,
                v_init=(
                    current_spectral_cache.v_basis
                    if (
                        projection_mode in (
                            "tp_radial_pgd_adaptive_krylov",
                            "tp_radial_pgd_adaptive_block2",
                            "tp_radial_pgd_multirate",
                            "tp_radial_pgd_hierarchical",
                        )
                        and spectral_cache_hit
                        and current_spectral_cache is not None
                        and current_spectral_cache.v_basis is not None
                    )
                    else v
                ),
                retract_mode=retract_mode,
                retract_alpha=retract_alpha,
                current_lr=current_lr,
                build_spectral_cache=post_return_spectral_cache,
                gap_threshold_rel=gap_threshold_rel,
                u_init=(
                    current_spectral_cache.u_basis
                    if (
                        projection_mode in (
                            "tp_radial_pgd_adaptive_krylov",
                            "tp_radial_pgd_adaptive_block2",
                            "tp_radial_pgd_multirate",
                            "tp_radial_pgd_hierarchical",
                        )
                        and spectral_cache_hit
                        and current_spectral_cache is not None
                        and current_spectral_cache.u_basis is not None
                    )
                    else u
                ),
                krylov_initial_iterations=krylov_initial_iterations,
                krylov_max_iterations=krylov_max_iterations,
                krylov_safe_block_size=krylov_safe_block_size,
                krylov_residual_tolerance=krylov_residual_tolerance,
                krylov_gap_interval_fraction=krylov_gap_interval_fraction,
                multirate_exact_safe_return=(
                    projection_mode
                    in (
                        "tp_radial_pgd_multirate",
                        "tp_radial_pgd_hierarchical",
                    )
                    and should_estimate_gap
                    and rel_gap
                    <= gap_threshold_rel * gap_probe_safe_multiplier
                ),
                prior_spectral_cache=current_return_cache,
                perturbation_norm_upper=perturbation_norm_upper,
                gap_probe_safe_multiplier=gap_probe_safe_multiplier,
                hierarchical_rank1_iterations=hierarchical_rank1_iterations,
                hierarchical_residual_gap_fraction=(
                    hierarchical_residual_gap_fraction
                ),
            )
            if trial_v is not None:
                v = trial_v
            if projection_diagnostics.next_spectral_cache is not None:
                v = projection_diagnostics.next_spectral_cache.v1
            update_unscaled = (W_base - Y.to(torch.float32)) / effective_lr_value

        info = SpELPGDUpdateInfo(
            branch=branch,
            rel_gap=rel_gap,
            sigma1=sigma1_value,
            gap_sigma1=gap_sigma1_value,
            sigma2=sigma2_value,
            pre_retract_bias=float(pre_retract_bias),
            trial_sigma=float(trial_sigma),
            post_retract_bias=float(post_retract_bias),
            projection_mode=projection_mode,
            projection_rank=projection_rank,
            gap_probed=should_estimate_gap,
            next_v=v.detach(),
            gap_audited=audit_exact_gap,
            exact_sigma1=exact_sigma1,
            exact_sigma2=exact_sigma2,
            exact_rel_gap=exact_rel_gap,
            delta1_rel=delta1_rel,
            delta2_rel=delta2_rel,
            gap_abs_error=gap_abs_error,
            main_sigma1_rel_error=main_sigma1_rel_error,
            main_triplet_residual_rel=main_triplet_residual_rel,
            false_safe=(
                audit_exact_gap
                and branch_mode == "auto"
                and branch == "spel"
                and exact_rel_gap < gap_threshold_rel
            ),
            false_pgd=(
                audit_exact_gap
                and branch_mode == "auto"
                and branch == "pgd"
                and exact_rel_gap >= gap_threshold_rel
            ),
            pgd_trial_rel_gap=projection_diagnostics.trial_rel_gap,
            pgd_post_rel_gap=projection_diagnostics.post_rel_gap,
            pgd_projection_clipped_count=projection_diagnostics.clipped_count,
            p2gd_top_multiplicity=p2gd_top_multiplicity,
            spectral_cache_hit=spectral_cache_hit,
            next_spectral_cache=projection_diagnostics.next_spectral_cache,
            approximate_spectral_iterations=(
                projection_diagnostics.approximate_iterations
            ),
            approximate_spectral_residual_rel=(
                projection_diagnostics.approximate_residual_rel
            ),
            approximate_gap_interval_width=(
                projection_diagnostics.approximate_gap_interval_width
            ),
            approximate_exact_fallback=(
                projection_diagnostics.approximate_exact_fallback
            ),
            approximate_subspace_dimension=(
                projection_diagnostics.approximate_subspace_dimension
            ),
            approximate_mode=projection_diagnostics.approximate_mode,
            approximate_rejection_reasons=(
                projection_diagnostics.approximate_rejection_reasons
            ),
        )

    if tp_enabled:
        # Original SpEL copies the gathered/retracted current matrix back to
        # the local shard before returning the direction.  Preserve that
        # contract so the outer optimizer starts from W_base.
        W_local = _tp_split_along_dim(W_work, tp_group, partition_dim)
        W.copy_(W_local)
        update_local = _tp_split_along_dim(update_unscaled, tp_group, partition_dim)
        return update_local, info

    return update_unscaled, info


class SpELPGDSameProjection(SpEL):
    """Drop-in SpEL subclass with PGD fallback and shared SpEL-style projection.

    Constructor usage is the same as ``SpEL`` plus the following keyword-only
    arguments:

    Args:
        use_pgd_fallback: If True and ``branch_mode='auto'``, use PGD when the
            relative singular-value gap is below ``gap_threshold_rel``.
        branch_mode: ``'auto'`` uses the gap rule, ``'spel'`` always uses the
            tangent-projected SpEL direction, and ``'pgd'`` always uses the PGD
            momentum direction.
        gap_threshold_rel: Threshold for ``1 - sigma2 / sigma1``.
        sigma2_power_iteration_steps: Power iterations on the rank-one residual.
        gap_estimator_mode: ``'deflated_power'`` preserves the original cheap
            sigma2 estimator. ``'deflated_fp32_gap_only'`` keeps the original
            SpEL top-vector path and recomputes the deflated gap rule in FP32.
            ``'block2_fp32'`` estimates ``sigma1,u1,v1,sigma2`` from one FP32
            two-dimensional subspace and uses those vectors for the SpEL
            branch. ``'block2_fp32_gap_only'`` keeps the original SpEL top-vector
            path and uses a separate FP32 block-2 estimate only for branch
            selection. ``'exact_svd_gap_only'`` uses the FP32 full spectrum;
            theory projection modes also reuse the same full SVD's leading
            vectors for the exact tangent normal.
        pgd_direction_normalization: ``'none'`` for standard momentum PGD;
            ``'fro'`` for Frobenius-normalized momentum fallback; ``'spectral'``
            for spectral-norm-normalized PGD fallback.
        pgd_lr_scale: Extra step-size multiplier used only when the PGD fallback
            branch is selected.
        projection_mode: ``'fallback_exact'`` preserves original SpEL in the
            safe branch and uses exact SVD projection only for PGD fallback.
            ``'fallback_retraction'`` does the same but uses cheap SpEL-style
            retraction for PGD. ``'fallback_retraction_exact_norm'`` instead
            applies the FP32 radial map using an exact spectral norm from
            ``torch.linalg.svdvals``. ``'local_top1_fallback_exact'`` applies
            the FP32 local rank-one projection in the safe branch and exact SVD
            in the PGD branch. ``'shared_exact'`` and ``'shared_retraction'``
            project both branches after the trial step and are intended for
            ablations. ``'tp_radial_pgd_exact'`` applies the exact-norm FP32
            radial retraction after the tangent SpEL/MCSD trial and the exact
            FP32 SVD projection after the ambient PGD trial.
            ``'tp_radial_pgd_topk'`` changes only that PGD projection to the
            sequential PI/deflation approximation. The guarded
            ``'tp_radial_pgd_krylov_topk'`` uses FP32 block Krylov and an exact
            fallback for ambiguous projections. ``'tp_radial_pi'`` is an
            MCSD-TP-only calibration mode that replaces both the current-point
            leading triplet and post-step radial norm with the selected PI.
            ``'tp_radial_pgd_multirate'`` uses a fixed warm FP32 subspace for
            MCSD radial returns, exact full-SVD projection for PGD, and exact
            current-gap refreshes only when its temporal guard requests one.
        projection_rank: Rank used by the top-k approximate projection modes.
        main_power_dtype: Dtype for the ordinary SpEL top-vector power
            iteration. ``'bf16'`` preserves the original path, ``'fp32'`` uses
            the enclosing optimizer's matmul policy, ``'fp32_highest'``
            disables TF32 for PI, and ``'fp64'`` is a diagnostic oracle.
            ``gap_estimator_mode='block2_fp32'`` uses block-2 vectors for the
            main path regardless of this setting. Exact theory projection modes
            use full-SVD vectors and ignore this option.
        warm_start_uv: If True, cache the previous right singular vector per
            matrix/component and use it to initialize the next power iteration.
        gap_probe_interval: Run the automatic sigma2/gap estimator once every
            this many optimizer steps while the last measured gap is safe.
            Non-probe steps use the SpEL branch.
        gap_probe_safe_multiplier: Treat a measured relative gap larger than
            this multiple of ``gap_threshold_rel`` as safe for reduced probing.
        gap_audit_interval: If positive, use FP32 SVD every this many optimizer
            steps to audit the estimated top-two singular values and branch
            decision. Zero disables the diagnostic.
        post_return_spectral_cache: Reuse exact top-two singular data and the
            leading vectors produced by the previous radial/project return.
            Adaptive modes also cache their approximate top subspaces and gap
            intervals.
    """

    def __init__(
        self,
        *args: Any,
        use_pgd_fallback: bool = True,
        branch_mode: BranchMode = "auto",
        gap_threshold_rel: float = 1e-3,
        sigma2_power_iteration_steps: int = 5,
        gap_estimator_mode: GapEstimatorMode = "deflated_power",
        pgd_direction_normalization: PGDDirectionNormalization = "spectral",
        fallback_direction: str = "pgd",
        pgd_lr_scale: float = 0.5,
        projection_mode: ProjectionMode = "fallback_exact",
        projection_rank: int = 1,
        tangent_project_after_msign: bool = False,
        normalize_msign_spectral: bool = False,
        msign_spectral_norm_steps: int = 2,
        main_power_dtype: MainPowerDtype = "bf16",
        warm_start_uv: bool = False,
        gap_probe_interval: int = 1,
        gap_probe_safe_multiplier: float = 10.0,
        gap_audit_interval: int = 0,
        post_return_spectral_cache: bool = False,
        krylov_initial_iterations: int = 8,
        krylov_max_iterations: int = 40,
        krylov_safe_block_size: int = 2,
        krylov_residual_tolerance: float = 2e-6,
        krylov_gap_interval_fraction: float = 0.2,
        hierarchical_rank1_iterations: int = 3,
        hierarchical_residual_gap_fraction: float = 0.1,
        radial_cluster_mode: str = "none",
        cluster_threshold_rel: float = 5e-5,
        cluster_spectrum_mode: str = "exact",
        cluster_iterations: int = 10,
        cluster_direction_mode: str = "matched_fro",
        **kwargs: Any,
    ) -> None:
        if radial_cluster_mode not in ("none", "control", "fixed2", "fixed4", "adaptive4"):
            raise ValueError("invalid radial_cluster_mode")
        if cluster_spectrum_mode not in ("exact", "warm_block8") or cluster_iterations < 1:
            raise ValueError("invalid cluster spectrum mode or iteration count")
        if cluster_direction_mode not in ("matched_fro", "momentum_fro"):
            raise ValueError("invalid cluster direction mode")
        if cluster_direction_mode != "matched_fro" and radial_cluster_mode == "none":
            raise ValueError("normalized cluster momentum requires the cluster-radial override")
        if cluster_spectrum_mode != "exact" and radial_cluster_mode == "none":
            raise ValueError("warm block-8 requires the cluster-radial override")
        if radial_cluster_mode != "none" and (
            projection_mode != "tp_radial_pgd_exact"
            or not post_return_spectral_cache
            or not tangent_project_after_msign
            or pgd_direction_normalization != "none"
            or normalize_msign_spectral
            or gap_audit_interval != 0
            or cluster_threshold_rel < gap_threshold_rel
        ):
            raise ValueError("cluster-radial requires cached radial geometry, TP, raw momentum, "
                             "no msign normalization/audit, and cluster threshold >= routing gap")
        self.radial_cluster_mode = radial_cluster_mode
        self.cluster_threshold_rel = float(cluster_threshold_rel)
        self.cluster_spectrum_mode = cluster_spectrum_mode
        self.cluster_iterations = int(cluster_iterations)
        self.cluster_direction_mode = cluster_direction_mode
        if fallback_direction not in ("pgd", "p2gd"):
            raise ValueError("fallback_direction must be pgd or p2gd")
        if fallback_direction == "p2gd" and (
            projection_mode != "tp_radial_pgd_exact"
            or not post_return_spectral_cache
            or pgd_direction_normalization != "none"
            or not tangent_project_after_msign
        ):
            raise ValueError("P2GD requires cached exact returns, TP, and raw momentum")
        if branch_mode not in ("auto", "spel", "pgd"):
            raise ValueError("branch_mode must be one of: 'auto', 'spel', 'pgd'")
        if sigma2_power_iteration_steps < 1:
            raise ValueError("sigma2_power_iteration_steps must be at least 1")
        if main_power_dtype not in ("bf16", "fp32", "fp32_highest", "fp64"):
            raise ValueError(
                "main_power_dtype must be one of: 'bf16', 'fp32', "
                "'fp32_highest', 'fp64'"
            )
        if gap_estimator_mode not in (
            "deflated_power",
            "deflated_fp32_gap_only",
            "block2_fp32",
            "block2_fp32_gap_only",
            "exact_svd_gap_only",
        ):
            raise ValueError(
                "gap_estimator_mode must be one of: 'deflated_power', "
                "'deflated_fp32_gap_only', "
                "'block2_fp32', 'block2_fp32_gap_only', 'exact_svd_gap_only'"
            )
        if pgd_direction_normalization not in (
            "none",
            "fro",
            "spectral",
            "exact_spectral",
        ):
            raise ValueError(
                "pgd_direction_normalization must be one of: "
                "'none', 'fro', 'spectral', 'exact_spectral'"
            )
        if gap_threshold_rel < 0.0:
            raise ValueError("gap_threshold_rel must be non-negative")
        if pgd_lr_scale <= 0.0:
            raise ValueError("pgd_lr_scale must be positive")
        if projection_mode not in (
            "fallback_exact",
            "fallback_retraction",
            "fallback_retraction_exact_norm",
            "fallback_topk",
            "local_top1_fallback_exact",
            "shared_exact",
            "shared_retraction",
            "shared_topk",
            "tp_radial_pgd_exact",
            "tp_radial_pgd_topk",
            "tp_radial_pgd_krylov_topk",
            "tp_radial_pgd_adaptive_krylov",
            "tp_radial_pgd_adaptive_block2",
            "tp_radial_pgd_multirate",
            "tp_radial_pgd_hierarchical",
            "tp_radial_pi",
        ):
            raise ValueError(
                "projection_mode must be one of: fallback_exact, "
                "fallback_retraction, fallback_retraction_exact_norm, "
                "fallback_topk, local_top1_fallback_exact, shared_exact, "
                "shared_retraction, shared_topk, tp_radial_pgd_exact, "
                "tp_radial_pgd_topk, tp_radial_pgd_krylov_topk, "
                "tp_radial_pgd_adaptive_krylov, tp_radial_pgd_adaptive_block2, "
                "tp_radial_pgd_multirate, tp_radial_pgd_hierarchical, "
                "tp_radial_pi"
            )
        if projection_rank < 1:
            raise ValueError("projection_rank must be at least 1")
        if (
            projection_mode
            in (
                "tp_radial_pgd_exact",
                "tp_radial_pgd_topk",
                "tp_radial_pgd_krylov_topk",
                "tp_radial_pgd_adaptive_krylov",
                "tp_radial_pgd_adaptive_block2",
                "tp_radial_pgd_multirate",
                "tp_radial_pgd_hierarchical",
                "tp_radial_pi",
            )
            and branch_mode != "pgd"
            and not tangent_project_after_msign
        ):
            raise ValueError(
                f"{projection_mode} requires tangent_project_after_msign=True "
                "whenever the SpEL/MCSD branch can be selected"
            )
        if projection_mode == "tp_radial_pi" and branch_mode != "spel":
            raise ValueError("tp_radial_pi requires branch_mode='spel'")
        if gap_probe_interval < 1:
            raise ValueError("gap_probe_interval must be at least 1")
        if gap_probe_safe_multiplier < 1.0:
            raise ValueError("gap_probe_safe_multiplier must be at least 1")
        if gap_audit_interval < 0:
            raise ValueError("gap_audit_interval must be non-negative")
        if not (1 <= krylov_initial_iterations <= krylov_max_iterations):
            raise ValueError("invalid adaptive Krylov iteration range")
        if not (2 <= krylov_safe_block_size <= 8):
            raise ValueError("krylov_safe_block_size must be between 2 and 8")
        if krylov_residual_tolerance <= 0.0:
            raise ValueError("krylov_residual_tolerance must be positive")
        if krylov_gap_interval_fraction <= 0.0:
            raise ValueError("krylov_gap_interval_fraction must be positive")
        if hierarchical_rank1_iterations < 1:
            raise ValueError("hierarchical_rank1_iterations must be at least 1")
        if not (0.0 < hierarchical_residual_gap_fraction < 1.0):
            raise ValueError("hierarchical_residual_gap_fraction must be in (0, 1)")
        if (
            projection_mode == "tp_radial_pgd_hierarchical"
            and krylov_safe_block_size not in (4, 8)
        ):
            raise ValueError(
                "tp_radial_pgd_hierarchical requires krylov_safe_block_size 4 or 8"
            )
        if post_return_spectral_cache and projection_mode not in (
            "tp_radial_pgd_exact",
            "tp_radial_pgd_adaptive_krylov",
            "tp_radial_pgd_adaptive_block2",
            "tp_radial_pgd_multirate",
            "tp_radial_pgd_hierarchical",
        ):
            raise ValueError(
                "post_return_spectral_cache requires an exact or adaptive "
                "theory-facing projection mode"
            )
        if (
            projection_mode
            in ("tp_radial_pgd_multirate", "tp_radial_pgd_hierarchical")
            and not post_return_spectral_cache
        ):
            raise ValueError(
                f"{projection_mode} requires "
                "post_return_spectral_cache=True"
            )

        super().__init__(
            *args,
            tangent_project_after_msign=tangent_project_after_msign,
            normalize_msign_spectral=normalize_msign_spectral,
            msign_spectral_norm_steps=msign_spectral_norm_steps,
            **kwargs,
        )

        self.use_pgd_fallback = use_pgd_fallback
        self.branch_mode = branch_mode
        self.gap_threshold_rel = gap_threshold_rel
        self.sigma2_power_iteration_steps = sigma2_power_iteration_steps
        self.gap_estimator_mode = gap_estimator_mode
        self.pgd_direction_normalization = pgd_direction_normalization
        self.fallback_direction = fallback_direction
        self.pgd_lr_scale = pgd_lr_scale
        self.projection_mode = projection_mode
        self.projection_rank = projection_rank
        self.main_power_dtype = main_power_dtype
        self.warm_start_uv = bool(warm_start_uv)
        self.gap_probe_interval = int(gap_probe_interval)
        self.gap_probe_safe_multiplier = float(gap_probe_safe_multiplier)
        self.gap_audit_interval = int(gap_audit_interval)
        self.post_return_spectral_cache = bool(post_return_spectral_cache)
        self.krylov_initial_iterations = int(krylov_initial_iterations)
        self.krylov_max_iterations = int(krylov_max_iterations)
        self.krylov_safe_block_size = int(krylov_safe_block_size)
        self.krylov_residual_tolerance = float(krylov_residual_tolerance)
        self.krylov_gap_interval_fraction = float(krylov_gap_interval_fraction)
        self.hierarchical_rank1_iterations = int(hierarchical_rank1_iterations)
        self.hierarchical_residual_gap_fraction = float(
            hierarchical_residual_gap_fraction
        )
        self._optimizer_step = 0

        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.post_projection_count = 0
        self.gap_probe_count = 0
        self.gap_audit_count = 0
        self.gap_audit_estimate_count = 0
        self.gap_audit_exact_pgd_count = 0
        self.gap_audit_false_safe_count = 0
        self.gap_audit_false_pgd_count = 0
        self.gap_audit_delta1_sum = 0.0
        self.gap_audit_delta2_sum = 0.0
        self.gap_audit_gap_error_sum = 0.0
        self.gap_audit_exact_gap_sum = 0.0
        self.gap_audit_estimated_gap_sum = 0.0
        self.gap_audit_main_sigma1_error_sum = 0.0
        self.gap_audit_main_triplet_residual_sum = 0.0
        self.pgd_burst_start_count = 0
        self.pgd_repeat_count = 0
        self.pgd_projection_diagnostic_count = 0
        self.pgd_trial_gap_sum = 0.0
        self.pgd_post_gap_sum = 0.0
        self.pgd_zero_post_gap_count = 0
        self.pgd_multi_clip_count = 0
        self.pgd_clipped_singular_values_sum = 0
        self.spectral_cache_hit_count = 0
        self.spectral_cache_bootstrap_count = 0
        self.approximate_spectral_count = 0
        self.approximate_spectral_fallback_count = 0
        self.approximate_spectral_iterations_sum = 0
        self.approximate_spectral_residual_sum = 0.0
        self.approximate_gap_interval_width_sum = 0.0
        self.approximate_rank1_count = 0
        self.approximate_block4_count = 0
        self.approximate_block8_count = 0
        self.approximate_rejection_counts = {
            reason: 0 for reason in HIERARCHICAL_REJECTION_REASONS
        }
        self.cumulative_spel_branch_count = 0
        self.cumulative_pgd_branch_count = 0
        self.cumulative_zero_lr_branch_count = 0
        self.cumulative_p2gd_count = 0
        self.cumulative_p2gd_multiple_count = 0
        self.p2gd_max_top_multiplicity = 0
        self.cluster_rank_counts = {rank: 0 for rank in range(1, 5)}
        self.cluster_overflow_count = 0
        self.cluster_multiplier_sum = 0.0
        self.cluster_radial_ratio_max = 0.0
        self.cluster_flatten_distance_max = 0.0
        self.cumulative_post_projection_count = 0
        self.cumulative_gap_probe_count = 0
        self.cumulative_gap_audit_count = 0
        self.cumulative_gap_audit_estimate_count = 0
        self.cumulative_gap_audit_exact_pgd_count = 0
        self.cumulative_gap_audit_false_safe_count = 0
        self.cumulative_gap_audit_false_pgd_count = 0
        self.cumulative_gap_audit_delta1_sum = 0.0
        self.cumulative_gap_audit_delta2_sum = 0.0
        self.cumulative_gap_audit_gap_error_sum = 0.0
        self.cumulative_gap_audit_exact_gap_sum = 0.0
        self.cumulative_gap_audit_estimated_gap_sum = 0.0
        self.cumulative_gap_audit_main_sigma1_error_sum = 0.0
        self.cumulative_gap_audit_main_triplet_residual_sum = 0.0
        self.cumulative_pgd_burst_start_count = 0
        self.cumulative_pgd_repeat_count = 0
        self.cumulative_pgd_projection_diagnostic_count = 0
        self.cumulative_pgd_trial_gap_sum = 0.0
        self.cumulative_pgd_post_gap_sum = 0.0
        self.cumulative_pgd_zero_post_gap_count = 0
        self.cumulative_pgd_multi_clip_count = 0
        self.cumulative_pgd_clipped_singular_values_sum = 0
        self.cumulative_spectral_cache_hit_count = 0
        self.cumulative_spectral_cache_bootstrap_count = 0
        self.cumulative_approximate_spectral_count = 0
        self.cumulative_approximate_spectral_fallback_count = 0
        self.cumulative_approximate_spectral_iterations_sum = 0
        self.cumulative_approximate_spectral_residual_sum = 0.0
        self.cumulative_approximate_gap_interval_width_sum = 0.0
        self.cumulative_approximate_rank1_count = 0
        self.cumulative_approximate_block4_count = 0
        self.cumulative_approximate_block8_count = 0
        self.cumulative_approximate_rejection_counts = {
            reason: 0 for reason in HIERARCHICAL_REJECTION_REASONS
        }
        self.branch_info_dict: Dict[str, Dict[str, bool | float | str]] = {}
        self.post_retract_bias_dict: Dict[str, float] = {}
        self.trial_spectral_norm_dict: Dict[str, float] = {}
        self._warm_start_v_cache: Dict[str, torch.Tensor] = {}
        self._spectral_point_cache: Dict[str, SpectralPointCache] = {}
        self._next_gap_probe_step_cache: Dict[str, int] = {}
        self._pgd_run_length_cache: Dict[str, int] = {}
        self._active_param_cache_key: Optional[str] = None

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform one optimizer step and clear per-step diagnostics."""
        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.post_projection_count = 0
        self.gap_probe_count = 0
        self.gap_audit_count = 0
        self.gap_audit_estimate_count = 0
        self.gap_audit_exact_pgd_count = 0
        self.gap_audit_false_safe_count = 0
        self.gap_audit_false_pgd_count = 0
        self.gap_audit_delta1_sum = 0.0
        self.gap_audit_delta2_sum = 0.0
        self.gap_audit_gap_error_sum = 0.0
        self.gap_audit_exact_gap_sum = 0.0
        self.gap_audit_estimated_gap_sum = 0.0
        self.gap_audit_main_sigma1_error_sum = 0.0
        self.gap_audit_main_triplet_residual_sum = 0.0
        self.pgd_burst_start_count = 0
        self.pgd_repeat_count = 0
        self.pgd_projection_diagnostic_count = 0
        self.pgd_trial_gap_sum = 0.0
        self.pgd_post_gap_sum = 0.0
        self.pgd_zero_post_gap_count = 0
        self.pgd_multi_clip_count = 0
        self.pgd_clipped_singular_values_sum = 0
        self.spectral_cache_hit_count = 0
        self.spectral_cache_bootstrap_count = 0
        self.approximate_spectral_count = 0
        self.approximate_spectral_fallback_count = 0
        self.approximate_spectral_iterations_sum = 0
        self.approximate_spectral_residual_sum = 0.0
        self.approximate_gap_interval_width_sum = 0.0
        self.approximate_rank1_count = 0
        self.approximate_block4_count = 0
        self.approximate_block8_count = 0
        self.approximate_rejection_counts = {
            reason: 0 for reason in HIERARCHICAL_REJECTION_REASONS
        }
        self.branch_info_dict.clear()
        self.post_retract_bias_dict.clear()
        self.trial_spectral_norm_dict.clear()
        result = super().step(closure)
        self._optimizer_step += 1
        return result

    def clear_post_return_spectral_cache(self) -> None:
        """Invalidate spectral data after an external parameter-state change."""
        self._spectral_point_cache.clear()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state and bootstrap spectral caches on the next step."""
        self.clear_post_return_spectral_cache()
        super().load_state_dict(state_dict)

    def _record_update_info(
        self,
        info: SpELPGDUpdateInfo,
        *,
        param_name: Optional[str],
        component_label: Optional[str],
        cache_key: Optional[str] = None,
    ) -> None:
        key_parts = []
        if param_name:
            key_parts.append(str(param_name))
        if component_label:
            key_parts.append(str(component_label))
        key = cache_key or (".".join(key_parts) if key_parts else "matrix")
        if info.cluster_dimension:
            self.cluster_rank_counts[info.cluster_dimension] += 1
            self.cluster_overflow_count += int(info.cluster_cap_overflow)
            self.cluster_multiplier_sum += info.cluster_step_multiplier
            self.cluster_flatten_distance_max = max(
                self.cluster_flatten_distance_max, info.cluster_flatten_distance
            )
        if math.isfinite(info.radial_change_ratio):
            self.cluster_radial_ratio_max = max(
                self.cluster_radial_ratio_max, info.radial_change_ratio
            )
        if info.p2gd_top_multiplicity:
            self.cumulative_p2gd_count += 1
            self.cumulative_p2gd_multiple_count += int(info.p2gd_top_multiplicity > 1)
            self.p2gd_max_top_multiplicity = max(
                self.p2gd_max_top_multiplicity, info.p2gd_top_multiplicity
            )

        if info.branch == "spel":
            self.spel_branch_count += 1
            self.cumulative_spel_branch_count += 1
        elif info.branch == "pgd":
            self.pgd_branch_count += 1
            self.cumulative_pgd_branch_count += 1
        elif info.branch == "zero_lr":
            self.zero_lr_branch_count += 1
            self.cumulative_zero_lr_branch_count += 1

        if info.next_spectral_cache is not None:
            if info.spectral_cache_hit:
                self.spectral_cache_hit_count += 1
                self.cumulative_spectral_cache_hit_count += 1
            else:
                self.spectral_cache_bootstrap_count += 1
                self.cumulative_spectral_cache_bootstrap_count += 1

        if info.approximate_spectral_iterations > 0:
            self.approximate_spectral_count += 1
            self.cumulative_approximate_spectral_count += 1
            self.approximate_spectral_iterations_sum += (
                info.approximate_spectral_iterations
            )
            self.cumulative_approximate_spectral_iterations_sum += (
                info.approximate_spectral_iterations
            )
            if math.isfinite(info.approximate_spectral_residual_rel):
                self.approximate_spectral_residual_sum += (
                    info.approximate_spectral_residual_rel
                )
                self.cumulative_approximate_spectral_residual_sum += (
                    info.approximate_spectral_residual_rel
                )
            if math.isfinite(info.approximate_gap_interval_width):
                self.approximate_gap_interval_width_sum += (
                    info.approximate_gap_interval_width
                )
                self.cumulative_approximate_gap_interval_width_sum += (
                    info.approximate_gap_interval_width
                )
            if info.approximate_exact_fallback:
                self.approximate_spectral_fallback_count += 1
                self.cumulative_approximate_spectral_fallback_count += 1
            if info.approximate_mode == "hierarchical_rank1":
                self.approximate_rank1_count += 1
                self.cumulative_approximate_rank1_count += 1
            elif info.approximate_mode == "hierarchical_block4":
                self.approximate_block4_count += 1
                self.cumulative_approximate_block4_count += 1
            elif info.approximate_mode == "hierarchical_block8":
                self.approximate_block8_count += 1
                self.cumulative_approximate_block8_count += 1
            for reason in info.approximate_rejection_reasons:
                if reason not in self.approximate_rejection_counts:
                    continue
                self.approximate_rejection_counts[reason] += 1
                self.cumulative_approximate_rejection_counts[reason] += 1

        if info.branch == "pgd":
            previous_run = self._pgd_run_length_cache.get(key, 0)
            current_run = previous_run + 1 if previous_run else 1
            self._pgd_run_length_cache[key] = current_run
            if previous_run:
                self.pgd_repeat_count += 1
                self.cumulative_pgd_repeat_count += 1
            else:
                self.pgd_burst_start_count += 1
                self.cumulative_pgd_burst_start_count += 1
        else:
            self._pgd_run_length_cache[key] = 0
        if info.branch == "pgd" and math.isfinite(info.pgd_post_rel_gap):
            self.pgd_projection_diagnostic_count += 1
            self.cumulative_pgd_projection_diagnostic_count += 1
            self.pgd_trial_gap_sum += info.pgd_trial_rel_gap
            self.cumulative_pgd_trial_gap_sum += info.pgd_trial_rel_gap
            self.pgd_post_gap_sum += info.pgd_post_rel_gap
            self.cumulative_pgd_post_gap_sum += info.pgd_post_rel_gap
            clipped_count = int(info.pgd_projection_clipped_count)
            self.pgd_clipped_singular_values_sum += clipped_count
            self.cumulative_pgd_clipped_singular_values_sum += clipped_count
            if info.pgd_post_rel_gap <= 1e-7:
                self.pgd_zero_post_gap_count += 1
                self.cumulative_pgd_zero_post_gap_count += 1
            if clipped_count >= 2:
                self.pgd_multi_clip_count += 1
                self.cumulative_pgd_multi_clip_count += 1
        if (
            info.branch == "pgd"
            or info.projection_mode.startswith("shared_")
            or (
                info.branch == "spel"
                and info.projection_mode
                in (
                    "cluster_radial",
                    "local_top1_fallback_exact",
                    "tp_radial_pgd_exact",
                    "tp_radial_pgd_topk",
                    "tp_radial_pgd_krylov_topk",
                    "tp_radial_pgd_adaptive_krylov",
                    "tp_radial_pgd_adaptive_block2",
                    "tp_radial_pgd_multirate",
                    "tp_radial_pgd_hierarchical",
                    "tp_radial_pi",
                )
            )
        ):
            self.post_projection_count += 1
            self.cumulative_post_projection_count += 1
        if info.gap_probed:
            self.gap_probe_count += 1
            self.cumulative_gap_probe_count += 1
        if info.gap_audited:
            self.gap_audit_count += 1
            self.cumulative_gap_audit_count += 1
            if info.exact_rel_gap < self.gap_threshold_rel:
                self.gap_audit_exact_pgd_count += 1
                self.cumulative_gap_audit_exact_pgd_count += 1
            if info.false_safe:
                self.gap_audit_false_safe_count += 1
                self.cumulative_gap_audit_false_safe_count += 1
            if info.false_pgd:
                self.gap_audit_false_pgd_count += 1
                self.cumulative_gap_audit_false_pgd_count += 1

            self.gap_audit_exact_gap_sum += info.exact_rel_gap
            self.cumulative_gap_audit_exact_gap_sum += info.exact_rel_gap
            self.gap_audit_main_sigma1_error_sum += info.main_sigma1_rel_error
            self.cumulative_gap_audit_main_sigma1_error_sum += (
                info.main_sigma1_rel_error
            )
            self.gap_audit_main_triplet_residual_sum += (
                info.main_triplet_residual_rel
            )
            self.cumulative_gap_audit_main_triplet_residual_sum += (
                info.main_triplet_residual_rel
            )

            if info.gap_probed and math.isfinite(info.gap_abs_error):
                self.gap_audit_estimate_count += 1
                self.cumulative_gap_audit_estimate_count += 1
                self.gap_audit_delta1_sum += info.delta1_rel
                self.gap_audit_delta2_sum += info.delta2_rel
                self.gap_audit_gap_error_sum += info.gap_abs_error
                self.gap_audit_estimated_gap_sum += info.rel_gap
                self.cumulative_gap_audit_delta1_sum += info.delta1_rel
                self.cumulative_gap_audit_delta2_sum += info.delta2_rel
                self.cumulative_gap_audit_gap_error_sum += info.gap_abs_error
                self.cumulative_gap_audit_estimated_gap_sum += info.rel_gap

        if param_name is None and component_label is None:
            return

        self.branch_info_dict[key] = {
            "branch": info.branch,
            "rel_gap": info.rel_gap,
            "sigma1": info.sigma1,
            "gap_sigma1": info.gap_sigma1,
            "sigma2": info.sigma2,
            "pre_retract_bias": info.pre_retract_bias,
            "trial_sigma": info.trial_sigma,
            "post_retract_bias": info.post_retract_bias,
            "projection_mode": info.projection_mode,
            "projection_rank": info.projection_rank,
            "gap_probed": info.gap_probed,
            "gap_audited": info.gap_audited,
            "exact_sigma1": info.exact_sigma1,
            "exact_sigma2": info.exact_sigma2,
            "exact_rel_gap": info.exact_rel_gap,
            "delta1_rel": info.delta1_rel,
            "delta2_rel": info.delta2_rel,
            "gap_abs_error": info.gap_abs_error,
            "main_sigma1_rel_error": info.main_sigma1_rel_error,
            "main_triplet_residual_rel": info.main_triplet_residual_rel,
            "false_safe": info.false_safe,
            "false_pgd": info.false_pgd,
            "pgd_trial_rel_gap": info.pgd_trial_rel_gap,
            "pgd_post_rel_gap": info.pgd_post_rel_gap,
            "pgd_projection_clipped_count": info.pgd_projection_clipped_count,
            "spectral_cache_hit": info.spectral_cache_hit,
            "approximate_spectral_iterations": info.approximate_spectral_iterations,
            "approximate_spectral_residual_rel": (
                info.approximate_spectral_residual_rel
            ),
            "approximate_gap_interval_width": (
                info.approximate_gap_interval_width
            ),
            "approximate_exact_fallback": info.approximate_exact_fallback,
            "approximate_subspace_dimension": info.approximate_subspace_dimension,
            "approximate_mode": info.approximate_mode,
            "approximate_rejection_reasons": ",".join(
                info.approximate_rejection_reasons
            ),
            "next_cached_rel_gap": (
                info.next_spectral_cache.rel_gap
                if info.next_spectral_cache is not None
                else float("nan")
            ),
        }

        if self.retract_mode == "dynamic":
            if info.pre_retract_bias != 0.0:
                self.retract_bias_dict[key] = info.pre_retract_bias
                self.spectral_norm_dict[key] = info.sigma1
            if info.post_retract_bias != 0.0:
                self.post_retract_bias_dict[key] = info.post_retract_bias
                self.trial_spectral_norm_dict[key] = info.trial_sigma

    def _warm_start_key(
        self,
        W: torch.Tensor,
        *,
        param_name: Optional[str],
        component_label: Optional[str],
    ) -> str:
        """Stable cache key for one matrix/component warm-start vector."""
        key_parts = [
            str(param_name)
            if param_name
            else self._active_param_cache_key or f"anonymous_param:{id(W)}"
        ]
        if component_label:
            key_parts.append(str(component_label))
        return ".".join(key_parts)

    def _should_probe_gap(self, cache_key: str) -> bool:
        """Return whether this matrix should run the gap estimator now."""
        if self.branch_mode != "auto" or not self.use_pgd_fallback:
            return False
        if self.gap_probe_interval == 1:
            return True
        return self._optimizer_step >= self._next_gap_probe_step_cache.get(cache_key, 0)

    def _should_audit_gap(self) -> bool:
        """Return whether this optimizer step should run the exact SVD audit."""
        return (
            self.gap_audit_interval > 0
            and self._optimizer_step % self.gap_audit_interval == 0
        )

    def _multirate_needs_exact_refresh(
        self,
        cache_key: str,
        cache: Optional[SpectralPointCache],
    ) -> bool:
        """Guard an approximate cache with periodic and interval-based refresh."""
        if cache is None:
            return True
        if cache.is_exact:
            return False
        if self._should_probe_gap(cache_key):
            return True

        safe_gap = self.gap_probe_safe_multiplier * self.gap_threshold_rel
        return cache.gap_lower <= safe_gap

    def _update_gap_probe_schedule(
        self,
        cache_key: str,
        info: SpELPGDUpdateInfo,
    ) -> None:
        """Reduce probing only after a comfortably separated measured gap."""
        if not info.gap_probed:
            return
        safe_gap = self.gap_probe_safe_multiplier * self.gap_threshold_rel
        if info.rel_gap > safe_gap:
            next_probe_step = self._optimizer_step + self.gap_probe_interval
        else:
            next_probe_step = self._optimizer_step + 1
        self._next_gap_probe_step_cache[cache_key] = next_probe_step

    def _compute_component_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        tp_group: Any,
        partition_dim: Optional[int],
        current_lr: Optional[float] = None,
        param_name: Optional[str] = None,
        component_label: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute a scaled update for one 2D component.

        The returned tensor is ready for the base optimizer.  It already
        includes the spectral-ball scale factor, matching the original SpEL
        contract.
        """
        if current_lr is None:
            raise RuntimeError(
                "SpELPGDSameProjection requires current_lr from the param group. "
                "The base optimizer should pass kwargs['lr'] into orthogonalize()."
            )

        current_lr_value = _float_value(current_lr, name="current_lr")
        target_radius = compute_target_radius(shape=W.shape, radius_mode=self.radius_mode)
        scale_factor = get_spectral_ball_scale_factor(
            W.shape[0], W.shape[1], mode=self.scale_mode
        )
        effective_lr = current_lr_value * float(scale_factor)
        warm_start_key = self._warm_start_key(
            W,
            param_name=param_name,
            component_label=component_label,
        )
        warm_start_v = (
            self._warm_start_v_cache.get(warm_start_key)
            if self.warm_start_uv
            else None
        )
        current_spectral_cache = (
            self._spectral_point_cache.get(warm_start_key)
            if self.post_return_spectral_cache
            else None
        )
        force_exact_current_spectrum = False
        if self.projection_mode in (
            "tp_radial_pgd_multirate",
            "tp_radial_pgd_hierarchical",
        ):
            force_exact_current_spectrum = self._multirate_needs_exact_refresh(
                warm_start_key,
                current_spectral_cache,
            )
            should_probe_gap = (
                current_spectral_cache is None
                or current_spectral_cache.is_exact
                or force_exact_current_spectrum
            )
        else:
            should_probe_gap = (
                True
                if self.post_return_spectral_cache
                else self._should_probe_gap(warm_start_key)
            )

        update_fn = compute_spel_pgd_same_projection_update
        cluster_kwargs = {}
        if self.radial_cluster_mode != "none":
            from .spectral_cluster_radial import compute_cluster_radial_update

            update_fn = compute_cluster_radial_update
            cluster_kwargs = dict(
                cluster_mode=self.radial_cluster_mode,
                cluster_threshold_rel=self.cluster_threshold_rel,
                cluster_spectrum_mode=self.cluster_spectrum_mode,
                cluster_iterations=self.cluster_iterations,
                cluster_direction_mode=self.cluster_direction_mode,
                optimizer_step=self._optimizer_step,
                param_name=param_name,
                component_label=component_label,
            )
        update_unscaled, info = update_fn(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            effective_lr=effective_lr,
            tp_group=tp_group,
            partition_dim=partition_dim,
            retract_mode=self.retract_mode,
            retract_alpha=self.retract_alpha,
            current_lr=current_lr_value,
            use_pgd_fallback=should_probe_gap,
            branch_mode=self.branch_mode,
            gap_threshold_rel=self.gap_threshold_rel,
            gap_probe_safe_multiplier=self.gap_probe_safe_multiplier,
            sigma2_power_iteration_steps=self.sigma2_power_iteration_steps,
            gap_estimator_mode=self.gap_estimator_mode,
            pgd_direction_normalization=self.pgd_direction_normalization,
            fallback_direction=self.fallback_direction,
            pgd_lr_scale=self.pgd_lr_scale,
            projection_mode=self.projection_mode,
            projection_rank=self.projection_rank,
            tangent_project_after_msign=self.tangent_project_after_msign,
            normalize_msign_spectral=self.normalize_msign_spectral,
            msign_spectral_norm_steps=self.msign_spectral_norm_steps,
            main_power_dtype=self.main_power_dtype,
            warm_start_v=warm_start_v,
            audit_exact_gap=self._should_audit_gap(),
            force_exact_current_spectrum=force_exact_current_spectrum,
            post_return_spectral_cache=self.post_return_spectral_cache,
            current_spectral_cache=current_spectral_cache,
            krylov_initial_iterations=self.krylov_initial_iterations,
            krylov_max_iterations=self.krylov_max_iterations,
            krylov_safe_block_size=self.krylov_safe_block_size,
            krylov_residual_tolerance=self.krylov_residual_tolerance,
            krylov_gap_interval_fraction=self.krylov_gap_interval_fraction,
            hierarchical_rank1_iterations=self.hierarchical_rank1_iterations,
            hierarchical_residual_gap_fraction=(
                self.hierarchical_residual_gap_fraction
            ),
            **cluster_kwargs,
        )
        self._update_gap_probe_schedule(warm_start_key, info)

        if self.warm_start_uv and info.next_v is not None:
            self._warm_start_v_cache[warm_start_key] = info.next_v.detach().clone()

        if self.post_return_spectral_cache:
            if info.next_spectral_cache is None:
                self._spectral_point_cache.pop(warm_start_key, None)
            else:
                self._spectral_point_cache[warm_start_key] = (
                    _clone_spectral_point_cache(info.next_spectral_cache)
                )

        self._record_update_info(
            info,
            param_name=param_name,
            component_label=component_label,
            cache_key=warm_start_key,
        )

        return update_unscaled * scale_factor

    def _uses_split_path(self, p: torch.Tensor) -> bool:
        """Return True when original SpEL.orthogonalize will use split logic."""
        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            return True

        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            return True

        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):
            num_local_experts = getattr(p, "num_local_experts", None)
            param_name = getattr(p, "param_name", "")
            if num_local_experts is not None and num_local_experts > 1:
                if "weight1" in param_name or "weight2" in param_name:
                    return True

        return False

    def _resolve_tp_context(
        self, p: torch.Tensor
    ) -> Tuple[Any, Optional[int]]:
        """Resolve tensor-parallel context exactly as in SpEL."""
        tp_group = None
        partition_dim = None

        if self.pg_collection is not None:
            try:
                tp_group = (
                    self.pg_collection.expt_tp
                    if getattr(p, "expert_tp", False)
                    else self.pg_collection.tp
                )
            except Exception:
                tp_group = None

        if hasattr(p, "partition_dim"):
            partition_dim = getattr(p, "partition_dim")
            if partition_dim == -1:
                partition_dim = None

        return tp_group, partition_dim

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return the scaled SpEL--PGD update direction.

        Split QKV/FC1/MoE paths are delegated to the original SpEL splitting
        logic; because ``_compute_component_update`` is overridden here, every
        split component receives the SpEL--PGD same-projection update.
        """
        current_lr = kwargs.get("lr", None)
        if current_lr is None:
            raise RuntimeError(
                "SpELPGDSameProjection requires kwargs['lr'] so it can encode "
                "the post-projection target as an outer optimizer update."
            )

        param_name = getattr(p, "param_name", None)
        self._active_param_cache_key = (
            str(param_name) if param_name else f"anonymous_param:{id(p)}"
        )
        try:
            if self._uses_split_path(p):
                return super().orthogonalize(p, grad, **kwargs)

            tp_group, partition_dim = self._resolve_tp_context(p)
            return self._compute_component_update(
                p.data,
                grad,
                tp_group,
                partition_dim,
                current_lr=current_lr,
                param_name=param_name,
                component_label="matrix" if param_name else None,
            )
        finally:
            self._active_param_cache_key = None

    def get_branch_stats(self) -> Dict[str, int | float]:
        """Return per-step branch counts and fallback rate."""
        total = self.spel_branch_count + self.pgd_branch_count + self.zero_lr_branch_count
        cumulative_total = (
            self.cumulative_spel_branch_count
            + self.cumulative_pgd_branch_count
            + self.cumulative_zero_lr_branch_count
        )
        return {
            "spel_branch_count": self.spel_branch_count,
            "pgd_branch_count": self.pgd_branch_count,
            "cumulative_p2gd_count": self.cumulative_p2gd_count,
            "cumulative_p2gd_multiple_count": self.cumulative_p2gd_multiple_count,
            "p2gd_max_top_multiplicity": self.p2gd_max_top_multiplicity,
            **{f"cluster_rank{q}_count": n for q, n in self.cluster_rank_counts.items()},
            "cluster_overflow_count": self.cluster_overflow_count,
            "cluster_multiplier_sum": self.cluster_multiplier_sum,
            "cluster_radial_enabled": int(self.radial_cluster_mode != "none"),
            "cluster_radial_ratio_max": self.cluster_radial_ratio_max,
            "cluster_flatten_distance_max": self.cluster_flatten_distance_max,
            "zero_lr_branch_count": self.zero_lr_branch_count,
            "post_projection_count": self.post_projection_count,
            "gap_probe_count": self.gap_probe_count,
            "gap_audit_count": self.gap_audit_count,
            "gap_audit_estimate_count": self.gap_audit_estimate_count,
            "gap_audit_exact_pgd_count": self.gap_audit_exact_pgd_count,
            "gap_audit_false_safe_count": self.gap_audit_false_safe_count,
            "gap_audit_false_pgd_count": self.gap_audit_false_pgd_count,
            "gap_audit_delta1_sum": self.gap_audit_delta1_sum,
            "gap_audit_delta2_sum": self.gap_audit_delta2_sum,
            "gap_audit_gap_error_sum": self.gap_audit_gap_error_sum,
            "gap_audit_exact_gap_sum": self.gap_audit_exact_gap_sum,
            "gap_audit_estimated_gap_sum": self.gap_audit_estimated_gap_sum,
            "gap_audit_main_sigma1_error_sum": self.gap_audit_main_sigma1_error_sum,
            "gap_audit_main_triplet_residual_sum": (
                self.gap_audit_main_triplet_residual_sum
            ),
            "pgd_burst_start_count": self.pgd_burst_start_count,
            "pgd_repeat_count": self.pgd_repeat_count,
            "pgd_projection_diagnostic_count": (
                self.pgd_projection_diagnostic_count
            ),
            "pgd_trial_gap_sum": self.pgd_trial_gap_sum,
            "pgd_post_gap_sum": self.pgd_post_gap_sum,
            "pgd_zero_post_gap_count": self.pgd_zero_post_gap_count,
            "pgd_multi_clip_count": self.pgd_multi_clip_count,
            "pgd_clipped_singular_values_sum": (
                self.pgd_clipped_singular_values_sum
            ),
            "spectral_cache_hit_count": self.spectral_cache_hit_count,
            "spectral_cache_bootstrap_count": self.spectral_cache_bootstrap_count,
            "approximate_spectral_count": self.approximate_spectral_count,
            "approximate_spectral_fallback_count": (
                self.approximate_spectral_fallback_count
            ),
            "approximate_spectral_iterations_sum": (
                self.approximate_spectral_iterations_sum
            ),
            "approximate_spectral_residual_sum": (
                self.approximate_spectral_residual_sum
            ),
            "approximate_gap_interval_width_sum": (
                self.approximate_gap_interval_width_sum
            ),
            "approximate_rank1_count": self.approximate_rank1_count,
            "approximate_block4_count": self.approximate_block4_count,
            "approximate_block8_count": self.approximate_block8_count,
            **{
                f"approximate_reject_{reason}_count": count
                for reason, count in self.approximate_rejection_counts.items()
            },
            "total_matrix_updates": total,
            "pgd_fallback_rate": (self.pgd_branch_count / total) if total else 0.0,
            "post_projection_rate": (self.post_projection_count / total) if total else 0.0,
            "cumulative_spel_branch_count": self.cumulative_spel_branch_count,
            "cumulative_pgd_branch_count": self.cumulative_pgd_branch_count,
            "cumulative_zero_lr_branch_count": self.cumulative_zero_lr_branch_count,
            "cumulative_post_projection_count": self.cumulative_post_projection_count,
            "cumulative_gap_probe_count": self.cumulative_gap_probe_count,
            "cumulative_gap_audit_count": self.cumulative_gap_audit_count,
            "cumulative_gap_audit_estimate_count": (
                self.cumulative_gap_audit_estimate_count
            ),
            "cumulative_gap_audit_exact_pgd_count": (
                self.cumulative_gap_audit_exact_pgd_count
            ),
            "cumulative_gap_audit_false_safe_count": (
                self.cumulative_gap_audit_false_safe_count
            ),
            "cumulative_gap_audit_false_pgd_count": (
                self.cumulative_gap_audit_false_pgd_count
            ),
            "cumulative_gap_audit_delta1_sum": self.cumulative_gap_audit_delta1_sum,
            "cumulative_gap_audit_delta2_sum": self.cumulative_gap_audit_delta2_sum,
            "cumulative_gap_audit_gap_error_sum": (
                self.cumulative_gap_audit_gap_error_sum
            ),
            "cumulative_gap_audit_exact_gap_sum": (
                self.cumulative_gap_audit_exact_gap_sum
            ),
            "cumulative_gap_audit_estimated_gap_sum": (
                self.cumulative_gap_audit_estimated_gap_sum
            ),
            "cumulative_gap_audit_main_sigma1_error_sum": (
                self.cumulative_gap_audit_main_sigma1_error_sum
            ),
            "cumulative_gap_audit_main_triplet_residual_sum": (
                self.cumulative_gap_audit_main_triplet_residual_sum
            ),
            "cumulative_pgd_burst_start_count": (
                self.cumulative_pgd_burst_start_count
            ),
            "cumulative_pgd_repeat_count": self.cumulative_pgd_repeat_count,
            "cumulative_pgd_projection_diagnostic_count": (
                self.cumulative_pgd_projection_diagnostic_count
            ),
            "cumulative_pgd_trial_gap_sum": self.cumulative_pgd_trial_gap_sum,
            "cumulative_pgd_post_gap_sum": self.cumulative_pgd_post_gap_sum,
            "cumulative_pgd_zero_post_gap_count": (
                self.cumulative_pgd_zero_post_gap_count
            ),
            "cumulative_pgd_multi_clip_count": self.cumulative_pgd_multi_clip_count,
            "cumulative_pgd_clipped_singular_values_sum": (
                self.cumulative_pgd_clipped_singular_values_sum
            ),
            "cumulative_spectral_cache_hit_count": (
                self.cumulative_spectral_cache_hit_count
            ),
            "cumulative_spectral_cache_bootstrap_count": (
                self.cumulative_spectral_cache_bootstrap_count
            ),
            "cumulative_approximate_spectral_count": (
                self.cumulative_approximate_spectral_count
            ),
            "cumulative_approximate_spectral_fallback_count": (
                self.cumulative_approximate_spectral_fallback_count
            ),
            "cumulative_approximate_spectral_iterations_sum": (
                self.cumulative_approximate_spectral_iterations_sum
            ),
            "cumulative_approximate_spectral_residual_sum": (
                self.cumulative_approximate_spectral_residual_sum
            ),
            "cumulative_approximate_gap_interval_width_sum": (
                self.cumulative_approximate_gap_interval_width_sum
            ),
            "cumulative_approximate_rank1_count": (
                self.cumulative_approximate_rank1_count
            ),
            "cumulative_approximate_block4_count": (
                self.cumulative_approximate_block4_count
            ),
            "cumulative_approximate_block8_count": (
                self.cumulative_approximate_block8_count
            ),
            **{
                f"cumulative_approximate_reject_{reason}_count": count
                for reason, count in self.cumulative_approximate_rejection_counts.items()
            },
            "cumulative_total_matrix_updates": cumulative_total,
            "cumulative_pgd_fallback_rate": (
                self.cumulative_pgd_branch_count / cumulative_total
            )
            if cumulative_total
            else 0.0,
            "cumulative_post_projection_rate": (
                self.cumulative_post_projection_count / cumulative_total
            )
            if cumulative_total
            else 0.0,
        }

    def get_branch_info_dict(self) -> Optional[Dict[str, Dict[str, bool | float | str]]]:
        """Return detailed per-component branch diagnostics for the last step."""
        return self.branch_info_dict if self.branch_info_dict else None

    def get_post_retract_bias_dict(self) -> Optional[Dict[str, float]]:
        """Return dynamic post-projection/retraction bias diagnostics."""
        return self.post_retract_bias_dict if self.post_retract_bias_dict else None

    def get_trial_spectral_norm_dict(self) -> Optional[Dict[str, float]]:
        """Return spectral norms of trial points before post-retraction."""
        return self.trial_spectral_norm_dict if self.trial_spectral_norm_dict else None


# Short alias for configs that prefer optimizer names without suffixes.
SpELPGD = SpELPGDSameProjection
