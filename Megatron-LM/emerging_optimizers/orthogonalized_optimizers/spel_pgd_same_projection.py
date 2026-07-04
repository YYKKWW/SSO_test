# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from the SpEL optimizer implementation.  This variant adds a
# SpEL--PGD branch while keeping the PGD projection/retraction identical to
# the SpEL-style spectral-sphere retraction used by the original code.

"""SpEL--PGD optimizer with the same projection/retraction as SpEL.

This file implements a practical MCSD/SpEL--PGD variant for matrix weights
constrained to a spectral sphere.  It is intended to live next to ``spel.py``
in the same optimizer package.

Key design choice
-----------------
The PGD branch does **not** use the expensive exact SVD projection.  Instead,
both the SpEL branch and the PGD branch form a trial point

    Z = W - effective_lr * D

and then apply the same SpEL-style spectral-sphere retraction:

    sigma_Z = power_iteration(Z)
    apply_retract(Z, sigma_Z, target_radius, ...)

Thus the two branches differ only in the direction ``D``:

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
PGDDirectionNormalization = Literal["none", "fro"]

__all__ = [
    "SpELPGDSameProjection",
    "SpELPGD",
    "SpELPGDUpdateInfo",
    "apply_spel_style_spectral_projection_",
    "compute_spel_pgd_same_projection_update",
    "estimate_second_singular_value",
]


@dataclass(frozen=True)
class SpELPGDUpdateInfo:
    """Diagnostics for one matrix/component update."""

    branch: str
    rel_gap: float
    sigma1: float
    sigma2: float
    pre_retract_bias: float
    trial_sigma: float
    post_retract_bias: float


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
def _pgd_direction(
    M: torch.Tensor,
    *,
    normalization: PGDDirectionNormalization = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the PGD fallback direction from momentum/Nesterov momentum."""
    D = M.to(torch.float32)
    if normalization == "none":
        return D
    if normalization == "fro":
        return D / torch.linalg.norm(D, dim=(-2, -1), keepdim=True).clamp_min(eps)
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
    gap_threshold_rel: float = 5e-3,
    sigma2_power_iteration_steps: int = 3,
    pgd_direction_normalization: PGDDirectionNormalization = "none",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, SpELPGDUpdateInfo]:
    """Compute a SpEL--PGD update using SpEL-style projection for both branches.

    The function follows the same external optimizer contract as ``SpEL``:
    it returns an update direction ``u``.  The caller should still multiply it
    by the spectral-ball scale factor before returning to the base optimizer.

    If the caller returns ``u * scale_factor`` and the base optimizer applies
    ``W <- W - current_lr * returned_update``, then the actual new iterate is

        Y = Retr_spel_style(W_retracted - current_lr * scale_factor * D).

    This is achieved by returning

        u = (W_retracted - Y) / (current_lr * scale_factor).

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

    effective_lr_value = float(effective_lr)

    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if tp_enabled:
        W_work = _tp_gather_along_dim(W, tp_group, partition_dim)
        M_work = _tp_gather_along_dim(M, tp_group, partition_dim)
    else:
        W_work = W
        M_work = M

    # Estimate top singular data on the incoming current matrix.  This mirrors
    # original SpEL and also supplies the gap-test data.
    sigma1, u, v = power_iteration(W_work, steps=power_iteration_steps)
    sigma1_value = float(sigma1.item())

    if branch_mode == "auto" and use_pgd_fallback:
        sigma2_value = estimate_second_singular_value(
            W_work,
            sigma1,
            u,
            v,
            steps=sigma2_power_iteration_steps,
            eps=eps,
        )
        rel_gap = max(0.0, 1.0 - sigma2_value / max(sigma1_value, eps))
    else:
        sigma2_value = 0.0
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
            sigma2=sigma2_value,
            pre_retract_bias=float(pre_retract_bias),
            trial_sigma=sigma1_value,
            post_retract_bias=0.0,
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
            D = project_to_tangent_plane(D, u, v, eps=eps)
        else:
            D = _pgd_direction(
                M_work,
                normalization=pgd_direction_normalization,
                eps=eps,
            )

        # Shared projection/retraction for SpEL and PGD branches.
        # This is the part that makes PGD projection match original SpEL's
        # spectral-sphere retraction instead of exact SVD projection.
        Y = W_base - effective_lr_value * D.to(torch.float32)
        Y, post_retract_bias, trial_sigma = apply_spel_style_spectral_projection_(
            Y,
            target_radius,
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
            sigma2=sigma2_value,
            pre_retract_bias=float(pre_retract_bias),
            trial_sigma=float(trial_sigma),
            post_retract_bias=float(post_retract_bias),
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
            momentum direction.  All modes still use the same post-step
            SpEL-style projection/retraction.
        gap_threshold_rel: Threshold for ``1 - sigma2 / sigma1``.
        sigma2_power_iteration_steps: Power iterations on the rank-one residual.
        pgd_direction_normalization: ``'none'`` for standard momentum PGD;
            ``'fro'`` for Frobenius-normalized momentum fallback.
    """

    def __init__(
        self,
        *args: Any,
        use_pgd_fallback: bool = True,
        branch_mode: BranchMode = "auto",
        gap_threshold_rel: float = 5e-3,
        sigma2_power_iteration_steps: int = 3,
        pgd_direction_normalization: PGDDirectionNormalization = "none",
        **kwargs: Any,
    ) -> None:
        if branch_mode not in ("auto", "spel", "pgd"):
            raise ValueError("branch_mode must be one of: 'auto', 'spel', 'pgd'")
        if sigma2_power_iteration_steps < 1:
            raise ValueError("sigma2_power_iteration_steps must be at least 1")
        if pgd_direction_normalization not in ("none", "fro"):
            raise ValueError("pgd_direction_normalization must be one of: 'none', 'fro'")
        if gap_threshold_rel < 0.0:
            raise ValueError("gap_threshold_rel must be non-negative")

        super().__init__(*args, **kwargs)

        self.use_pgd_fallback = use_pgd_fallback
        self.branch_mode = branch_mode
        self.gap_threshold_rel = gap_threshold_rel
        self.sigma2_power_iteration_steps = sigma2_power_iteration_steps
        self.pgd_direction_normalization = pgd_direction_normalization

        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.branch_info_dict: Dict[str, Dict[str, float | str]] = {}
        self.post_retract_bias_dict: Dict[str, float] = {}
        self.trial_spectral_norm_dict: Dict[str, float] = {}

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform one optimizer step and clear per-step diagnostics."""
        self.spel_branch_count = 0
        self.pgd_branch_count = 0
        self.zero_lr_branch_count = 0
        self.branch_info_dict.clear()
        self.post_retract_bias_dict.clear()
        self.trial_spectral_norm_dict.clear()
        return super().step(closure)

    def _record_update_info(
        self,
        info: SpELPGDUpdateInfo,
        *,
        param_name: Optional[str],
        component_label: Optional[str],
    ) -> None:
        if info.branch == "spel":
            self.spel_branch_count += 1
        elif info.branch == "pgd":
            self.pgd_branch_count += 1
        elif info.branch == "zero_lr":
            self.zero_lr_branch_count += 1

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
            "sigma2": info.sigma2,
            "pre_retract_bias": info.pre_retract_bias,
            "trial_sigma": info.trial_sigma,
            "post_retract_bias": info.post_retract_bias,
        }

        if self.retract_mode == "dynamic":
            if info.pre_retract_bias != 0.0:
                self.retract_bias_dict[key] = info.pre_retract_bias
                self.spectral_norm_dict[key] = info.sigma1
            if info.post_retract_bias != 0.0:
                self.post_retract_bias_dict[key] = info.post_retract_bias
                self.trial_spectral_norm_dict[key] = info.trial_sigma

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
            use_pgd_fallback=self.use_pgd_fallback,
            branch_mode=self.branch_mode,
            gap_threshold_rel=self.gap_threshold_rel,
            sigma2_power_iteration_steps=self.sigma2_power_iteration_steps,
            pgd_direction_normalization=self.pgd_direction_normalization,
        )

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
        return {
            "spel_branch_count": self.spel_branch_count,
            "pgd_branch_count": self.pgd_branch_count,
            "zero_lr_branch_count": self.zero_lr_branch_count,
            "total_matrix_updates": total,
            "pgd_fallback_rate": (self.pgd_branch_count / total) if total else 0.0,
        }

    def get_branch_info_dict(self) -> Optional[Dict[str, Dict[str, float | str]]]:
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
