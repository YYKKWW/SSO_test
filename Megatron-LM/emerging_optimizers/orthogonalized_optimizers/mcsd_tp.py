# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone MCSD-TP optimizer.

This module contains only the smooth-stratum MCSD tangent-plane path.  It
deliberately does not estimate a spectral gap and has no PGD, exact-SVD,
top-k, Krylov, certificate, or fallback branch.  The method assumes that a
usable leading singular pair exists, as in the original SSO engineering path.

For one matrix component, an optimizer step is

1. use the cached leading pair of the current returned point (or bootstrap it
   with power iteration on the first step);
2. project momentum onto ``{H: <u_1 v_1^T, H>_F = 0}``;
3. apply matrix sign and project the result onto the same tangent plane;
4. form the trial point and radially return it with a power-iteration estimate
   of its spectral norm; and
5. cache the trial leading pair, which radial scaling leaves unchanged.

After bootstrap this needs one leading-triplet power iteration per matrix and
optimizer step.  Optional gap logic belongs in ``spel_pgd_same_projection``.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from emerging_optimizers import utils

from .spectral_ball_utils import (
    _tp_gather_along_dim,
    _tp_split_along_dim,
    _tp_world_and_rank,
    compute_target_radius,
    get_spectral_ball_scale_factor,
    msign,
)
from .spel import SpEL

PowerDtype = Literal["bf16", "fp32", "fp32_highest"]

__all__ = [
    "MCSDTP",
    "MCSDTPConfig",
    "MCSDTPSpectralState",
    "MCSDTPStepInfo",
    "leading_singular_triplet",
    "mcsd_tp_step",
    "project_to_spectral_tangent",
]


@dataclass(frozen=True)
class MCSDTPConfig:
    """Numerical settings for the standalone MCSD-TP matrix step."""

    power_iteration_steps: int = 10
    power_dtype: PowerDtype = "bf16"
    warm_start: bool = True
    msign_steps: int = 8
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.power_iteration_steps < 1:
            raise ValueError("power_iteration_steps must be at least 1")
        if self.power_dtype not in ("bf16", "fp32", "fp32_highest"):
            raise ValueError("power_dtype must be bf16, fp32, or fp32_highest")
        if self.msign_steps < 1:
            raise ValueError("msign_steps must be at least 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass
class MCSDTPSpectralState:
    """Leading singular vectors cached at a returned feasible point."""

    u1: torch.Tensor
    v1: torch.Tensor


@dataclass(frozen=True)
class MCSDTPStepInfo:
    """Small diagnostics that do not affect the optimizer decision."""

    bootstrapped: bool
    power_iteration_calls: int
    incoming_sigma1_estimate: float
    trial_sigma1_estimate: float


def _unit_column(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    x_fp32 = x.detach().to(torch.float32)
    if x_fp32.ndim == 1:
        x_fp32 = x_fp32.unsqueeze(-1)
    if x_fp32.ndim != 2 or x_fp32.shape[-1] != 1:
        raise ValueError(f"expected one column vector, got shape {tuple(x.shape)}")
    return x_fp32 / torch.linalg.vector_norm(x_fp32).clamp_min(eps)


def _initial_right_vector(
    W: torch.Tensor, v_init: torch.Tensor | None, *, dtype: torch.dtype, eps: float
) -> torch.Tensor:
    if v_init is None or tuple(v_init.shape) != (W.shape[1], 1):
        v = torch.ones((W.shape[1], 1), device=W.device, dtype=dtype)
    else:
        v = v_init.detach().to(device=W.device, dtype=dtype)
    return v / torch.linalg.vector_norm(v).clamp_min(eps)


@torch.no_grad()
def leading_singular_triplet(
    W: torch.Tensor,
    *,
    steps: int,
    power_dtype: PowerDtype,
    v_init: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the leading singular triplet using bilateral power iteration."""
    if W.ndim != 2:
        raise ValueError(f"MCSD-TP expects a 2D matrix, got shape {tuple(W.shape)}")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    work_dtype = torch.bfloat16 if power_dtype == "bf16" else torch.float32
    W_work = W.detach().to(work_dtype)
    v = _initial_right_vector(W_work, v_init, dtype=work_dtype, eps=eps)

    precision = "highest" if power_dtype == "fp32_highest" else "medium"
    with utils.fp32_matmul_precision(precision):
        for _ in range(steps):
            v = W_work.transpose(-2, -1) @ (W_work @ v)
            v = v / torch.linalg.vector_norm(v).clamp_min(eps)
        u = W_work @ v
        u = u / torch.linalg.vector_norm(u).clamp_min(eps)
        sigma = (u.transpose(-2, -1) @ W_work @ v).squeeze()

    return sigma.to(torch.float32), u.to(torch.float32), v.to(torch.float32)


@torch.no_grad()
def project_to_spectral_tangent(
    matrix: torch.Tensor, u1: torch.Tensor, v1: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Frobenius-project a matrix onto the assumed spectral tangent plane."""
    matrix_fp32 = matrix.to(torch.float32)
    u = _unit_column(u1, eps=eps)
    v = _unit_column(v1, eps=eps)
    if u.shape[0] != matrix_fp32.shape[0] or v.shape[0] != matrix_fp32.shape[1]:
        raise ValueError(
            "leading-vector shapes do not match the matrix: "
            f"matrix={tuple(matrix.shape)}, u={tuple(u.shape)}, v={tuple(v.shape)}"
        )
    coefficient = u.transpose(-2, -1) @ matrix_fp32 @ v
    return matrix_fp32 - coefficient * (u @ v.transpose(-2, -1))


def _state_is_compatible(state: MCSDTPSpectralState | None, W: torch.Tensor) -> bool:
    return bool(
        state is not None
        and tuple(state.u1.shape) == (W.shape[0], 1)
        and tuple(state.v1.shape) == (W.shape[1], 1)
    )


@torch.no_grad()
def mcsd_tp_step(
    W: torch.Tensor,
    momentum: torch.Tensor,
    *,
    target_radius: float,
    effective_lr: float,
    config: MCSDTPConfig,
    state: MCSDTPSpectralState | None = None,
) -> tuple[torch.Tensor, MCSDTPSpectralState, MCSDTPStepInfo]:
    """Return the complete next MCSD-TP matrix and its reusable spectral state.

    No spectral-gap decision is made.  A compatible state certifies only that
    the vectors came from this implementation's previous radial return; it is
    not a mathematical simplicity certificate.
    """
    if W.ndim != 2 or momentum.ndim != 2 or W.shape != momentum.shape:
        raise ValueError("W and momentum must be 2D tensors with equal shapes")
    if target_radius <= 0.0:
        raise ValueError("target_radius must be positive")
    if effective_lr < 0.0:
        raise ValueError("effective_lr must be non-negative")

    W_fp32 = W.to(torch.float32)
    momentum_fp32 = momentum.to(torch.float32)
    bootstrapped = not _state_is_compatible(state, W_fp32)

    if bootstrapped:
        sigma, u1, v1 = leading_singular_triplet(
            W_fp32,
            steps=config.power_iteration_steps,
            power_dtype=config.power_dtype,
            eps=config.eps,
        )
        sigma_value = float(sigma.item())
        if sigma_value <= config.eps:
            raise ValueError("cannot radially return a matrix with zero estimated norm")
        W_base = W_fp32 * (target_radius / sigma.clamp_min(config.eps))
        incoming_sigma = sigma_value
        power_iteration_calls = 1
    else:
        assert state is not None
        u1 = _unit_column(state.u1, eps=config.eps)
        v1 = _unit_column(state.v1, eps=config.eps)
        W_base = W_fp32
        incoming_sigma = float(target_radius)
        power_iteration_calls = 0

    tangent_momentum = project_to_spectral_tangent(momentum_fp32, u1, v1, eps=config.eps)
    tangent_momentum = tangent_momentum / torch.linalg.matrix_norm(
        tangent_momentum, ord="fro"
    ).clamp_min(config.eps)

    direction = msign(tangent_momentum, steps=config.msign_steps).to(torch.float32)
    direction = project_to_spectral_tangent(direction, u1, v1, eps=config.eps)

    trial = W_base - float(effective_lr) * direction
    trial_sigma, trial_u, trial_v = leading_singular_triplet(
        trial,
        steps=config.power_iteration_steps,
        power_dtype=config.power_dtype,
        v_init=v1 if config.warm_start else None,
        eps=config.eps,
    )
    power_iteration_calls += 1
    trial_sigma_value = float(trial_sigma.item())
    if trial_sigma_value <= config.eps:
        raise ValueError("cannot radially return a trial matrix with zero estimated norm")

    next_W = trial * (target_radius / trial_sigma.clamp_min(config.eps))
    next_state = MCSDTPSpectralState(u1=trial_u.detach().clone(), v1=trial_v.detach().clone())
    info = MCSDTPStepInfo(
        bootstrapped=bootstrapped,
        power_iteration_calls=power_iteration_calls,
        incoming_sigma1_estimate=incoming_sigma,
        trial_sigma1_estimate=trial_sigma_value,
    )
    return next_W.to(dtype=W.dtype), next_state, info


class MCSDTP(SpEL):
    """Megatron adapter for the standalone, gap-free MCSD-TP matrix step."""

    def __init__(
        self,
        *args: Any,
        main_power_dtype: PowerDtype = "bf16",
        warm_start_uv: bool = True,
        **kwargs: Any,
    ) -> None:
        if main_power_dtype not in ("bf16", "fp32", "fp32_highest"):
            raise ValueError("main_power_dtype must be bf16, fp32, or fp32_highest")

        # These options belong to legacy SpEL ablations.  MCSD-TP always uses
        # tangent projection after msign and a PI radial return of its trial.
        kwargs.pop("projection_mode", None)
        kwargs.pop("projection_rank", None)
        kwargs.pop("tangent_project_after_msign", None)
        kwargs.pop("normalize_msign_spectral", None)
        kwargs.pop("msign_spectral_norm_steps", None)
        super().__init__(
            *args,
            projection_mode="retraction",
            projection_rank=1,
            tangent_project_after_msign=True,
            normalize_msign_spectral=False,
            **kwargs,
        )
        self.main_power_dtype = main_power_dtype
        self.warm_start_uv = bool(warm_start_uv)
        self._mcsd_tp_spectral_state: dict[str, MCSDTPSpectralState] = {}
        self.bootstrap_power_iteration_count = 0
        self.trial_power_iteration_count = 0

    @staticmethod
    def _cache_key(W: torch.Tensor, *, param_name: str | None, component_label: str | None) -> str:
        if param_name:
            return f"{param_name}:{component_label or 'matrix'}"
        storage = W.untyped_storage() if hasattr(W, "untyped_storage") else W.storage()
        return (
            f"storage={storage.data_ptr()}:offset={W.storage_offset()}:"
            f"shape={tuple(W.shape)}:component={component_label or 'matrix'}"
        )

    def clear_spectral_state(self) -> None:
        """Discard cached leading vectors; the next update will bootstrap PI."""
        self._mcsd_tp_spectral_state.clear()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer tensors and rebuild non-persistent spectral caches.

        The cached singular vectors are deliberately absent from checkpoints.
        They are numerical accelerators rather than optimizer state, and a
        freshly loaded parameter may not be bitwise identical to the point
        that produced an old in-memory cache.
        """
        super().load_state_dict(state_dict)
        self.clear_spectral_state()

    def _compute_component_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        tp_group: Any,
        partition_dim: int | None,
        current_lr: float | None = None,
        param_name: str | None = None,
        component_label: str | None = None,
    ) -> torch.Tensor:
        """Compute the encoded update for one matrix or fused component."""
        if current_lr is None:
            raise RuntimeError("MCSDTP requires the current parameter-group learning rate")
        lr = float(current_lr)

        world_size, _ = _tp_world_and_rank(tp_group)
        tp_enabled = tp_group is not None and partition_dim is not None and world_size > 1
        if tp_enabled:
            W_work = _tp_gather_along_dim(W, tp_group, partition_dim)
            M_work = _tp_gather_along_dim(M, tp_group, partition_dim)
        else:
            W_work = W
            M_work = M

        target_radius = compute_target_radius(shape=W_work.shape, radius_mode=self.radius_mode)
        scale_factor = get_spectral_ball_scale_factor(
            W_work.shape[0], W_work.shape[1], mode=self.scale_mode
        )
        key = self._cache_key(W, param_name=param_name, component_label=component_label)
        config = MCSDTPConfig(
            power_iteration_steps=self.power_iteration_steps,
            power_dtype=self.main_power_dtype,
            warm_start=self.warm_start_uv,
            msign_steps=self.msign_steps,
        )
        next_W, next_state, info = mcsd_tp_step(
            W_work,
            M_work,
            target_radius=target_radius,
            effective_lr=lr * scale_factor,
            config=config,
            state=self._mcsd_tp_spectral_state.get(key),
        )
        self._mcsd_tp_spectral_state[key] = next_state
        self.bootstrap_power_iteration_count += int(info.bootstrapped)
        self.trial_power_iteration_count += 1

        if lr == 0.0:
            W_work.copy_(next_W)
            update = torch.zeros_like(W_work)
        else:
            update = (W_work.to(torch.float32) - next_W.to(torch.float32)) / lr
            update = update.to(dtype=W_work.dtype)

        if tp_enabled:
            if lr == 0.0:
                W.copy_(_tp_split_along_dim(W_work, tp_group, partition_dim))
            return _tp_split_along_dim(update, tp_group, partition_dim)
        return update
