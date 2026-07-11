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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch

from .spel import SpEL, _as_column_unit_vector, project_to_tangent_plane
from .spectral_ball_utils import (
    apply_retract,
    compute_target_radius,
    get_spectral_ball_scale_factor,
    msign,
    power_iteration,
)


BranchMode = Literal["auto", "spel", "pgd"]
GapEstimatorMode = Literal[
    "deflated_power",
    "deflated_fp32_gap_only",
    "block2_fp32",
    "block2_fp32_gap_only",
]
PGDDirectionNormalization = Literal["none", "fro", "spectral"]
ProjectionMode = Literal[
    "fallback_exact",
    "fallback_retraction",
    "fallback_topk",
    "shared_exact",
    "shared_retraction",
    "shared_topk",
]
MainPowerDtype = Literal["bf16", "fp32"]

__all__ = [
    "SpELPGDSameProjection",
    "SpELPGD",
    "SpELPGDUpdateInfo",
    "apply_spel_style_spectral_projection_",
    "compute_spel_pgd_same_projection_update",
    "estimate_top2_singular_values_block2_fp32",
    "estimate_second_singular_value",
    "estimate_second_singular_value_fp32",
    "power_iteration_fp32",
    "power_iteration_warm_start",
    "project_to_spectral_sphere_exact",
    "project_to_spectral_sphere_topk",
]


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
    raise ValueError("main_power_dtype must be one of: 'bf16', 'fp32'")


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
def project_to_spectral_sphere_exact(
    Z: torch.Tensor,
    target_radius: float,
) -> Tuple[torch.Tensor, float]:
    """Euclidean projection onto ``{W: ||W||_2 = target_radius}``.

    The projection is exact but uses a full SVD, so it is intended for the
    low-frequency PGD fallback or for small-model ablations.
    """
    dtype = Z.dtype
    Z_fp32 = Z.to(torch.float32)

    U, S, Vh = torch.linalg.svd(Z_fp32, full_matrices=False)
    S_proj = S.clone()
    trial_sigma = float(S_proj[0].item()) if S_proj.numel() else 0.0

    if S_proj.numel() == 0:
        return Z_fp32.to(dtype=dtype), trial_sigma

    if S_proj[0] >= target_radius:
        S_proj.clamp_(max=target_radius)
        S_proj[0] = target_radius
    else:
        S_proj[0] = target_radius

    Y = (U * S_proj.unsqueeze(-2)) @ Vh
    return Y.to(dtype=dtype), trial_sigma


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


@torch.no_grad()
def _post_project_trial(
    Y: torch.Tensor,
    target_radius: float,
    *,
    projection_mode: ProjectionMode,
    projection_rank: int,
    power_iteration_steps: int,
    retract_mode: str,
    retract_alpha: float,
    current_lr: Optional[float],
) -> Tuple[torch.Tensor, float, float]:
    """Apply the selected post-step projection/retraction to a trial point."""
    if projection_mode in ("fallback_exact", "shared_exact"):
        projected, trial_sigma = project_to_spectral_sphere_exact(Y, target_radius)
        return projected, 0.0, trial_sigma

    if projection_mode in ("fallback_topk", "shared_topk"):
        projected, trial_sigma = project_to_spectral_sphere_topk(
            Y,
            target_radius,
            rank=projection_rank,
            power_iteration_steps=power_iteration_steps,
        )
        return projected, 0.0, trial_sigma

    if projection_mode in ("fallback_retraction", "shared_retraction"):
        return apply_spel_style_spectral_projection_(
            Y,
            target_radius,
            power_iteration_steps=power_iteration_steps,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
        )

    raise ValueError(
        "projection_mode must be one of: fallback_exact, fallback_retraction, "
        "fallback_topk, shared_exact, shared_retraction, shared_topk"
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
    sigma2_power_iteration_steps: int = 5,
    gap_estimator_mode: GapEstimatorMode = "deflated_power",
    pgd_direction_normalization: PGDDirectionNormalization = "spectral",
    pgd_lr_scale: float = 0.5,
    projection_mode: ProjectionMode = "fallback_exact",
    projection_rank: int = 1,
    tangent_project_after_msign: bool = False,
    main_power_dtype: MainPowerDtype = "bf16",
    warm_start_v: Optional[torch.Tensor] = None,
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
    if main_power_dtype not in ("bf16", "fp32"):
        raise ValueError("main_power_dtype must be one of: 'bf16', 'fp32'")
    if gap_estimator_mode not in (
        "deflated_power",
        "deflated_fp32_gap_only",
        "block2_fp32",
        "block2_fp32_gap_only",
    ):
        raise ValueError(
            "gap_estimator_mode must be one of: 'deflated_power', "
            "'deflated_fp32_gap_only', "
            "'block2_fp32', 'block2_fp32_gap_only'"
        )
    if pgd_lr_scale <= 0.0:
        raise ValueError("pgd_lr_scale must be positive")
    if projection_mode not in (
        "fallback_exact",
        "fallback_retraction",
        "fallback_topk",
        "shared_exact",
        "shared_retraction",
        "shared_topk",
    ):
        raise ValueError(
            "projection_mode must be one of: fallback_exact, fallback_retraction, "
            "fallback_topk, shared_exact, shared_retraction, shared_topk"
        )
    if projection_rank < 1:
        raise ValueError("projection_rank must be at least 1")

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
    if should_estimate_gap and gap_estimator_mode == "block2_fp32":
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

    if should_estimate_gap and gap_estimator_mode == "deflated_power":
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
    elif not should_estimate_gap:
        sigma2_value = 0.0
        gap_sigma1_value = sigma1_value
        rel_gap = float("inf")

    # First make the current point feasible using the exact same pre-step
    # retraction used by the original SpEL implementation.
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
        )
    else:
        branch = _select_branch(
            branch_mode=branch_mode,
            use_pgd_fallback=use_pgd_fallback,
            rel_gap=rel_gap,
            gap_threshold_rel=gap_threshold_rel,
        )

        if branch == "spel":
            M_projected = project_to_tangent_plane(M_work, u, v, eps=eps)
            M_projected = M_projected / torch.linalg.norm(
                M_projected, dim=(-2, -1), keepdim=True
            ).clamp_min(eps)
            D = msign(M_projected, steps=msign_steps)
            if tangent_project_after_msign:
                D = project_to_tangent_plane(D, u, v, eps=eps)
        else:
            D = _pgd_direction(
                M_work,
                normalization=pgd_direction_normalization,
                power_iteration_steps=power_iteration_steps,
                eps=eps,
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
            Y, post_retract_bias, trial_sigma = _post_project_trial(
                Y,
                target_radius,
                projection_mode=projection_mode,
                projection_rank=projection_rank,
                power_iteration_steps=power_iteration_steps,
                retract_mode=retract_mode,
                retract_alpha=retract_alpha,
                current_lr=current_lr,
            )
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
            selection.
        pgd_direction_normalization: ``'none'`` for standard momentum PGD;
            ``'fro'`` for Frobenius-normalized momentum fallback; ``'spectral'``
            for spectral-norm-normalized PGD fallback.
        pgd_lr_scale: Extra step-size multiplier used only when the PGD fallback
            branch is selected.
        projection_mode: ``'fallback_exact'`` preserves original SpEL in the
            safe branch and uses exact SVD projection only for PGD fallback.
            ``'fallback_retraction'`` does the same but uses cheap SpEL-style
            retraction for PGD.  ``'shared_exact'`` and ``'shared_retraction'``
            project both branches after the trial step and are intended for
            ablations.
        projection_rank: Rank used by the top-k approximate projection modes.
        main_power_dtype: Dtype for the ordinary SpEL top-vector power
            iteration. ``'bf16'`` preserves the original path; ``'fp32'`` is a
            high-precision ablation. ``gap_estimator_mode='block2_fp32'`` uses
            the block2 vectors for the main path regardless of this setting.
        warm_start_uv: If True, cache the previous right singular vector per
            matrix/component and use it to initialize the next power iteration.
        gap_probe_interval: Run the automatic sigma2/gap estimator once every
            this many optimizer steps while the last measured gap is safe.
            Non-probe steps use the SpEL branch.
        gap_probe_safe_multiplier: Treat a measured relative gap larger than
            this multiple of ``gap_threshold_rel`` as safe for reduced probing.
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
        pgd_lr_scale: float = 0.5,
        projection_mode: ProjectionMode = "fallback_exact",
        projection_rank: int = 1,
        tangent_project_after_msign: bool = False,
        main_power_dtype: MainPowerDtype = "bf16",
        warm_start_uv: bool = False,
        gap_probe_interval: int = 1,
        gap_probe_safe_multiplier: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if branch_mode not in ("auto", "spel", "pgd"):
            raise ValueError("branch_mode must be one of: 'auto', 'spel', 'pgd'")
        if sigma2_power_iteration_steps < 1:
            raise ValueError("sigma2_power_iteration_steps must be at least 1")
        if main_power_dtype not in ("bf16", "fp32"):
            raise ValueError("main_power_dtype must be one of: 'bf16', 'fp32'")
        if gap_estimator_mode not in (
            "deflated_power",
            "deflated_fp32_gap_only",
            "block2_fp32",
            "block2_fp32_gap_only",
        ):
            raise ValueError(
                "gap_estimator_mode must be one of: 'deflated_power', "
                "'deflated_fp32_gap_only', "
                "'block2_fp32', 'block2_fp32_gap_only'"
            )
        if pgd_direction_normalization not in ("none", "fro", "spectral"):
            raise ValueError(
                "pgd_direction_normalization must be one of: 'none', 'fro', 'spectral'"
            )
        if gap_threshold_rel < 0.0:
            raise ValueError("gap_threshold_rel must be non-negative")
        if pgd_lr_scale <= 0.0:
            raise ValueError("pgd_lr_scale must be positive")
        if projection_mode not in (
            "fallback_exact",
            "fallback_retraction",
            "fallback_topk",
            "shared_exact",
            "shared_retraction",
            "shared_topk",
        ):
            raise ValueError(
                "projection_mode must be one of: fallback_exact, fallback_retraction, "
                "fallback_topk, shared_exact, shared_retraction, shared_topk"
            )
        if projection_rank < 1:
            raise ValueError("projection_rank must be at least 1")
        if gap_probe_interval < 1:
            raise ValueError("gap_probe_interval must be at least 1")
        if gap_probe_safe_multiplier < 1.0:
            raise ValueError("gap_probe_safe_multiplier must be at least 1")

        super().__init__(
            *args,
            tangent_project_after_msign=tangent_project_after_msign,
            **kwargs,
        )

        self.use_pgd_fallback = use_pgd_fallback
        self.branch_mode = branch_mode
        self.gap_threshold_rel = gap_threshold_rel
        self.sigma2_power_iteration_steps = sigma2_power_iteration_steps
        self.gap_estimator_mode = gap_estimator_mode
        self.pgd_direction_normalization = pgd_direction_normalization
        self.pgd_lr_scale = pgd_lr_scale
        self.projection_mode = projection_mode
        self.projection_rank = projection_rank
        self.main_power_dtype = main_power_dtype
        self.warm_start_uv = bool(warm_start_uv)
        self.gap_probe_interval = int(gap_probe_interval)
        self.gap_probe_safe_multiplier = float(gap_probe_safe_multiplier)
        self._optimizer_step = 0

        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.post_projection_count = 0
        self.gap_probe_count = 0
        self.cumulative_spel_branch_count = 0
        self.cumulative_pgd_branch_count = 0
        self.cumulative_zero_lr_branch_count = 0
        self.cumulative_post_projection_count = 0
        self.cumulative_gap_probe_count = 0
        self.branch_info_dict: Dict[str, Dict[str, bool | float | str]] = {}
        self.post_retract_bias_dict: Dict[str, float] = {}
        self.trial_spectral_norm_dict: Dict[str, float] = {}
        self._warm_start_v_cache: Dict[str, torch.Tensor] = {}
        self._next_gap_probe_step_cache: Dict[str, int] = {}

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform one optimizer step and clear per-step diagnostics."""
        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.post_projection_count = 0
        self.gap_probe_count = 0
        self.branch_info_dict.clear()
        self.post_retract_bias_dict.clear()
        self.trial_spectral_norm_dict.clear()
        result = super().step(closure)
        self._optimizer_step += 1
        return result

    def _record_update_info(
        self,
        info: SpELPGDUpdateInfo,
        *,
        param_name: Optional[str],
        component_label: Optional[str],
    ) -> None:
        if info.branch == "spel":
            self.spel_branch_count += 1
            self.cumulative_spel_branch_count += 1
        elif info.branch == "pgd":
            self.pgd_branch_count += 1
            self.cumulative_pgd_branch_count += 1
        elif info.branch == "zero_lr":
            self.zero_lr_branch_count += 1
            self.cumulative_zero_lr_branch_count += 1
        if info.branch == "pgd" or info.projection_mode.startswith("shared_"):
            self.post_projection_count += 1
            self.cumulative_post_projection_count += 1
        if info.gap_probed:
            self.gap_probe_count += 1
            self.cumulative_gap_probe_count += 1

        if param_name is None and component_label is None:
            return

        key_parts = []
        if param_name:
            key_parts.append(str(param_name))
        if component_label:
            key_parts.append(str(component_label))
        key = ".".join(key_parts) if key_parts else "matrix"

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
        key_parts = []
        if param_name:
            key_parts.append(str(param_name))
        if component_label:
            key_parts.append(str(component_label))
        if key_parts:
            return ".".join(key_parts)
        return f"anonymous:{id(W)}"

    def _should_probe_gap(self, cache_key: str) -> bool:
        """Return whether this matrix should run the gap estimator now."""
        if self.branch_mode != "auto" or not self.use_pgd_fallback:
            return False
        if self.gap_probe_interval == 1:
            return True
        return self._optimizer_step >= self._next_gap_probe_step_cache.get(cache_key, 0)

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
        should_probe_gap = self._should_probe_gap(warm_start_key)

        update_unscaled, info = compute_spel_pgd_same_projection_update(
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
            sigma2_power_iteration_steps=self.sigma2_power_iteration_steps,
            gap_estimator_mode=self.gap_estimator_mode,
            pgd_direction_normalization=self.pgd_direction_normalization,
            pgd_lr_scale=self.pgd_lr_scale,
            projection_mode=self.projection_mode,
            projection_rank=self.projection_rank,
            tangent_project_after_msign=self.tangent_project_after_msign,
            main_power_dtype=self.main_power_dtype,
            warm_start_v=warm_start_v,
        )
        self._update_gap_probe_schedule(warm_start_key, info)

        if self.warm_start_uv and info.next_v is not None:
            self._warm_start_v_cache[warm_start_key] = info.next_v.detach().clone()

        self._record_update_info(
            info,
            param_name=param_name,
            component_label=component_label,
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

        if self._uses_split_path(p):
            return super().orthogonalize(p, grad, **kwargs)

        tp_group, partition_dim = self._resolve_tp_context(p)
        param_name = getattr(p, "param_name", None)
        return self._compute_component_update(
            p.data,
            grad,
            tp_group,
            partition_dim,
            current_lr=current_lr,
            param_name=param_name,
            component_label="matrix" if param_name else None,
        )

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
            "zero_lr_branch_count": self.zero_lr_branch_count,
            "post_projection_count": self.post_projection_count,
            "gap_probe_count": self.gap_probe_count,
            "total_matrix_updates": total,
            "pgd_fallback_rate": (self.pgd_branch_count / total) if total else 0.0,
            "post_projection_rate": (self.post_projection_count / total) if total else 0.0,
            "cumulative_spel_branch_count": self.cumulative_spel_branch_count,
            "cumulative_pgd_branch_count": self.cumulative_pgd_branch_count,
            "cumulative_zero_lr_branch_count": self.cumulative_zero_lr_branch_count,
            "cumulative_post_projection_count": self.cumulative_post_projection_count,
            "cumulative_gap_probe_count": self.cumulative_gap_probe_count,
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
