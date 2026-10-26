# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Accurate FP32 SVD helpers and guarded block-Krylov spectral returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass(frozen=True)
class BlockKrylovSVDResult:
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    residuals_absolute: torch.Tensor
    iterations: int
    subspace_dimension: int


@dataclass(frozen=True)
class TopKProjectionInfo:
    mode: str
    iterations: int
    subspace_dimension: int
    max_residual_relative: float
    used_exact_fallback: bool
    trial_sigma1: float


@dataclass(frozen=True)
class AdaptiveSpectralReturnInfo:
    """Accuracy and fallback diagnostics for one adaptive spectral return."""

    mode: str
    iterations: int
    subspace_dimension: int
    max_residual_relative: float
    gap_lower: float
    gap_upper: float
    used_exact_fallback: bool
    trial_sigma1: float
    clipped_count: int
    sigma1_lower: float = float("nan")
    sigma1_upper: float = float("nan")
    sigma2_lower: float = float("nan")
    sigma2_upper: float = float("nan")
    bounds_certified: bool = False
    rejection_reasons: Tuple[str, ...] = ()


def accurate_svd_fp32(
    W: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use QR-based cuSOLVER ``gesvd`` on CUDA and LAPACK on CPU."""
    W32 = W.to(torch.float32)
    if W32.is_cuda:
        return torch.linalg.svd(W32, full_matrices=False, driver="gesvd")
    return torch.linalg.svd(W32, full_matrices=False)


def accurate_svdvals_fp32(W: torch.Tensor) -> torch.Tensor:
    """Return FP32 singular values from the accurate driver on this device."""
    W32 = W.to(torch.float32)
    if W32.is_cuda:
        return torch.linalg.svdvals(W32, driver="gesvd")
    return torch.linalg.svdvals(W32)


def _reorthogonalized_block(
    candidate: torch.Tensor,
    basis: torch.Tensor,
    *,
    passes: int,
    eps: float,
) -> torch.Tensor:
    block = candidate
    for _ in range(passes):
        block = block - basis @ (basis.transpose(-2, -1) @ block)
    Q, R = torch.linalg.qr(block, mode="reduced")
    diagonal = torch.abs(torch.diagonal(R))
    if diagonal.numel() == 0:
        return Q[:, :0]
    tolerance = eps * max(block.shape) * max(float(diagonal.max().item()), eps)
    rank = int(torch.count_nonzero(diagonal > tolerance).item())
    if rank == 0:
        return Q[:, :0]
    Q = Q[:, :rank]
    for _ in range(passes):
        Q = Q - basis @ (basis.transpose(-2, -1) @ Q)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    return Q


def _projected_svd_from_basis(
    W: torch.Tensor,
    basis: torch.Tensor,
    *,
    basis_side: str,
    wanted: int,
) -> BlockKrylovSVDResult:
    if basis_side == "right":
        U_small, S, Vh_small = accurate_svd_fp32(W @ basis)
        take = min(wanted, S.numel())
        U = U_small[:, :take]
        V = basis @ Vh_small[:take, :].transpose(-2, -1)
        Vh = V.transpose(-2, -1)
    else:
        U_small, S, Vh = accurate_svd_fp32(basis.transpose(-2, -1) @ W)
        take = min(wanted, S.numel())
        U = basis @ U_small[:, :take]
        Vh = Vh[:take, :]

    S = S[:take]
    V = Vh.transpose(-2, -1)
    left = W @ V - U * S.unsqueeze(0)
    right = W.transpose(-2, -1) @ U - V * S.unsqueeze(0)
    residuals = torch.sqrt(
        (left * left).sum(dim=0) + (right * right).sum(dim=0)
    ) / (2.0**0.5)
    return BlockKrylovSVDResult(
        U=U,
        S=S,
        Vh=Vh,
        residuals_absolute=residuals,
        iterations=0,
        subspace_dimension=basis.shape[1],
    )


@torch.no_grad()
def block_krylov_svd_fp32(
    W: torch.Tensor,
    *,
    wanted: int,
    block_size: int,
    initial_iterations: int = 8,
    max_iterations: int = 40,
    seed: int = 4321,
    reorthogonalization_passes: int = 2,
    stop_rule: Optional[Callable[[BlockKrylovSVDResult], bool]] = None,
    initial_basis: Optional[torch.Tensor] = None,
    check_interval: int = 1,
    eps: float = 1e-8,
) -> BlockKrylovSVDResult:
    """Build a fully reorthogonalized block Krylov space on the smaller side."""
    if W.ndim != 2:
        raise ValueError("block Krylov SVD expects one 2D matrix")
    if wanted < 1 or block_size < wanted:
        raise ValueError("block_size must be at least wanted")
    if not (1 <= initial_iterations <= max_iterations):
        raise ValueError("invalid block Krylov iteration range")
    if check_interval < 1:
        raise ValueError("check_interval must be at least 1")

    W32 = W.to(torch.float32)
    m, n = W32.shape
    if min(m, n) < wanted:
        raise ValueError("wanted exceeds the matrix minimum dimension")
    basis_side = "right" if n <= m else "left"
    side_dimension = n if basis_side == "right" else m
    generator = torch.Generator(device=W32.device)
    generator.manual_seed(seed)
    start_width = min(block_size, side_dimension)
    random_width = start_width
    initial = None
    if initial_basis is not None:
        initial = initial_basis.to(device=W32.device, dtype=torch.float32)
        if initial.ndim == 1:
            initial = initial.unsqueeze(1)
        if initial.ndim != 2 or initial.shape[0] != side_dimension:
            raise ValueError("initial_basis has the wrong Krylov-side shape")
        initial = initial[:, :start_width]
        random_width = max(0, start_width - initial.shape[1])

    random = torch.randn(
        (side_dimension, random_width),
        device=W32.device,
        dtype=torch.float32,
        generator=generator,
    )
    start = random if initial is None else torch.cat((initial, random), dim=1)
    basis, _ = torch.linalg.qr(start, mode="reduced")
    last_block = basis
    latest = None

    def normal_operator(X: torch.Tensor) -> torch.Tensor:
        if basis_side == "right":
            return W32.transpose(-2, -1) @ (W32 @ X)
        return W32 @ (W32.transpose(-2, -1) @ X)

    for iteration in range(1, max_iterations + 1):
        saturated = basis.shape[1] >= side_dimension
        if iteration > 1 and not saturated:
            new_block = _reorthogonalized_block(
                normal_operator(last_block),
                basis,
                passes=reorthogonalization_passes,
                eps=eps,
            )
            new_block = new_block[:, : side_dimension - basis.shape[1]]
            if new_block.shape[1] == 0:
                saturated = True
            else:
                basis = torch.cat((basis, new_block), dim=1)
                last_block = new_block
                saturated = basis.shape[1] >= side_dimension

        should_check = (
            iteration >= initial_iterations
            and (iteration - initial_iterations) % check_interval == 0
        )
        if should_check or saturated or iteration == max_iterations:
            result = _projected_svd_from_basis(
                W32, basis, basis_side=basis_side, wanted=wanted
            )
            latest = BlockKrylovSVDResult(
                U=result.U,
                S=result.S,
                Vh=result.Vh,
                residuals_absolute=result.residuals_absolute,
                iterations=iteration,
                subspace_dimension=basis.shape[1],
            )
            if stop_rule is not None and stop_rule(latest):
                return latest
            if saturated:
                return latest

    if latest is None:
        raise RuntimeError("block Krylov iteration produced no Ritz data")
    return latest


def _relative_gap_interval(
    result: BlockKrylovSVDResult,
    *,
    eps: float,
) -> Tuple[float, float]:
    """Bound the top relative gap using Ritz values and residual norms."""
    if result.S.numel() < 2:
        return 1.0, 1.0
    sigma1 = float(result.S[0].item())
    sigma2 = float(result.S[1].item())
    residual1 = float(result.residuals_absolute[0].item())
    residual2 = float(result.residuals_absolute[1].item())
    sigma1_lower = max(sigma1 - residual1, eps)
    sigma1_upper = max(sigma1 + residual1, eps)
    sigma2_lower = max(sigma2 - residual2, 0.0)
    sigma2_upper = max(sigma2 + residual2, 0.0)
    lower = max(0.0, 1.0 - sigma2_upper / sigma1_lower)
    upper = max(0.0, 1.0 - sigma2_lower / sigma1_upper)
    return min(lower, upper), max(lower, upper)


@torch.no_grad()
def adaptive_warm_fixed_subspace_radial_return(
    Z: torch.Tensor,
    target_radius: float,
    *,
    gap_threshold_rel: float,
    block_size: int = 2,
    initial_u: Optional[torch.Tensor] = None,
    initial_v: Optional[torch.Tensor] = None,
    initial_iterations: int = 5,
    max_iterations: int = 20,
    check_interval: int = 5,
    residual_tolerance: float = 1e-4,
    gap_interval_fraction: float = 0.2,
    require_gap_certificate: bool = True,
    seed: int = 4321,
    exact_fallback: bool = True,
    eps: float = 1e-8,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AdaptiveSpectralReturnInfo,
]:
    """Radial return from a fixed-width warm subspace iteration.

    Unlike expanding block Krylov, this keeps the working subspace at a fixed
    dimension. Extra vectors help track clustered leading singular spaces.
    By default, acceptance also requires the leading-two gap interval to
    certify the branch decision.  ``require_gap_certificate=False`` decouples
    the radial-norm approximation from branch selection: acceptance then uses
    only the Ritz residual, while the returned interval remains available for
    diagnostics and an independently scheduled exact gap refresh.
    """
    if not (1 <= initial_iterations <= max_iterations):
        raise ValueError("invalid fixed-subspace iteration range")
    if check_interval < 1:
        raise ValueError("check_interval must be at least 1")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    Z32 = Z.to(torch.float32)
    m, n = Z32.shape
    wanted = min(2, m, n)
    working_width = min(block_size, m, n)
    if wanted < 2:
        U, S, Vh = accurate_svd_fp32(Z32)
        returned = Z32 * (target_radius / S[0].clamp_min(eps))
        returned_s = S * (target_radius / S[0].clamp_min(eps))
        info = AdaptiveSpectralReturnInfo(
            "full_svd_fallback", 0, wanted, 0.0, 1.0, 1.0, True,
            float(S[0].item()), 0,
            float(returned_s[0].item()), float(returned_s[0].item()),
            (float(returned_s[1].item()) if returned_s.numel() > 1 else 0.0),
            (float(returned_s[1].item()) if returned_s.numel() > 1 else 0.0),
            True,
        )
        return returned, U, returned_s, Vh, info

    basis_side = "right" if n <= m else "left"
    side_dimension = n if basis_side == "right" else m
    initial_basis = initial_v if basis_side == "right" else initial_u
    generator = torch.Generator(device=Z32.device)
    generator.manual_seed(seed)
    if initial_basis is not None:
        basis = initial_basis.detach().to(device=Z32.device, dtype=torch.float32)
        if basis.ndim == 1:
            basis = basis.unsqueeze(1)
        if basis.shape[0] != side_dimension:
            raise ValueError("warm fixed-subspace basis has the wrong shape")
        basis = basis[:, :working_width]
    else:
        basis = Z32.new_empty((side_dimension, 0))
    basis = basis[:, :working_width]
    if basis.shape[1] < working_width:
        random = torch.randn(
            (side_dimension, working_width - basis.shape[1]),
            device=Z32.device,
            dtype=torch.float32,
            generator=generator,
        )
        basis = torch.cat((basis, random), dim=1)
    basis, _ = torch.linalg.qr(basis, mode="reduced")

    def normal_operator(X: torch.Tensor) -> torch.Tensor:
        if basis_side == "right":
            return Z32.transpose(-2, -1) @ (Z32 @ X)
        return Z32 @ (Z32.transpose(-2, -1) @ X)

    gap_width_target = max(
        gap_interval_fraction * max(gap_threshold_rel, 1e-5),
        2e-6,
    )
    result = None
    converged = False
    gap_interval = (0.0, 1.0)
    for iteration in range(1, max_iterations + 1):
        basis, _ = torch.linalg.qr(normal_operator(basis), mode="reduced")
        should_check = (
            iteration >= initial_iterations
            and (iteration - initial_iterations) % check_interval == 0
        )
        if should_check or iteration == max_iterations:
            candidate = _projected_svd_from_basis(
                Z32,
                basis,
                basis_side=basis_side,
                wanted=working_width,
            )
            result = BlockKrylovSVDResult(
                candidate.U,
                candidate.S,
                candidate.Vh,
                candidate.residuals_absolute,
                iteration,
                working_width,
            )
            scale = max(float(result.S[0].item()), eps)
            residual = float(result.residuals_absolute[:2].max().item()) / scale
            gap_interval = _relative_gap_interval(result, eps=eps)
            interval_width = gap_interval[1] - gap_interval[0]
            decision_certified = not (
                gap_interval[0] <= gap_threshold_rel <= gap_interval[1]
            )
            gap_certified = (
                interval_width
                <= max(gap_width_target, 2.0 * residual_tolerance)
                and decision_certified
            )
            converged = residual <= residual_tolerance and (
                gap_certified or not require_gap_certificate
            )
            if converged:
                break

    if result is None:
        raise RuntimeError("fixed-subspace iteration produced no Ritz data")
    scale = max(float(result.S[0].item()), eps)
    max_residual = float(result.residuals_absolute[:2].max().item()) / scale
    if exact_fallback and not converged:
        U, S, Vh = accurate_svd_fp32(Z32)
        radial_scale = target_radius / S[0].clamp_min(eps)
        returned = Z32 * radial_scale
        returned_s = S * radial_scale
        exact_gap = max(0.0, 1.0 - float(S[1].item()) / max(float(S[0].item()), eps))
        info = AdaptiveSpectralReturnInfo(
            "full_svd_fallback", result.iterations, working_width, max_residual,
            exact_gap, exact_gap, True, float(S[0].item()), 0,
            float(returned_s[0].item()), float(returned_s[0].item()),
            (float(returned_s[1].item()) if returned_s.numel() > 1 else 0.0),
            (float(returned_s[1].item()) if returned_s.numel() > 1 else 0.0),
            True,
        )
        return returned, U, returned_s, Vh, info

    radial_scale = target_radius / result.S[0].clamp_min(eps)
    returned = Z32 * radial_scale
    returned_s = result.S * radial_scale
    residuals = result.residuals_absolute * radial_scale
    sigma1_lower = max(
        0.0,
        float((returned_s[0] - residuals[0]).item()),
    )
    sigma1_upper = float((returned_s[0] + residuals[0]).item())
    sigma2_lower = max(
        0.0,
        float((returned_s[1] - residuals[1]).item()),
    )
    sigma2_upper = float((returned_s[1] + residuals[1]).item())
    info = AdaptiveSpectralReturnInfo(
        f"adaptive_warm_block{working_width}", result.iterations,
        working_width, max_residual,
        gap_interval[0], gap_interval[1], False, float(result.S[0].item()), 0,
        sigma1_lower, sigma1_upper, sigma2_lower, sigma2_upper, converged,
    )
    return returned, result.U, returned_s, result.Vh, info


def _warm_rank1_ritz(
    Z: torch.Tensor,
    *,
    initial_u: Optional[torch.Tensor],
    initial_v: Optional[torch.Tensor],
    iterations: int,
    seed: int,
    eps: float,
) -> BlockKrylovSVDResult:
    """Track one warm leading singular triplet with a fixed rank-one space."""
    Z32 = Z.to(torch.float32)
    m, n = Z32.shape
    basis_side = "right" if n <= m else "left"
    side_dimension = n if basis_side == "right" else m
    initial = initial_v if basis_side == "right" else initial_u
    if initial is not None:
        basis = initial.detach().to(device=Z32.device, dtype=torch.float32)
        if basis.ndim == 1:
            basis = basis.unsqueeze(1)
        if basis.shape[0] != side_dimension:
            raise ValueError("warm rank-one basis has the wrong shape")
        basis = basis[:, :1]
    else:
        generator = torch.Generator(device=Z32.device)
        generator.manual_seed(seed)
        basis = torch.randn(
            (side_dimension, 1),
            device=Z32.device,
            dtype=torch.float32,
            generator=generator,
        )
    basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(eps)

    for _ in range(iterations):
        if basis_side == "right":
            candidate = Z32.transpose(-2, -1) @ (Z32 @ basis)
        else:
            candidate = Z32 @ (Z32.transpose(-2, -1) @ basis)
        basis = candidate / torch.linalg.vector_norm(
            candidate, dim=0, keepdim=True
        ).clamp_min(eps)

    result = _projected_svd_from_basis(
        Z32,
        basis,
        basis_side=basis_side,
        wanted=1,
    )
    return BlockKrylovSVDResult(
        U=result.U,
        S=result.S,
        Vh=result.Vh,
        residuals_absolute=result.residuals_absolute,
        iterations=iterations,
        subspace_dimension=1,
    )


@torch.no_grad()
def hierarchical_warm_radial_return(
    Z: torch.Tensor,
    target_radius: float,
    *,
    warning_gap_rel: float,
    prior_sigma1_lower: float,
    prior_sigma1_upper: float,
    prior_sigma2_lower: float,
    prior_sigma2_upper: float,
    prior_bounds_certified: bool,
    perturbation_norm_upper: float,
    initial_u: Optional[torch.Tensor] = None,
    initial_v: Optional[torch.Tensor] = None,
    rank1_iterations: int = 3,
    residual_gap_fraction: float = 0.1,
    max_block_size: int = 8,
    block_initial_iterations: int = 3,
    block_max_iterations: int = 10,
    residual_tolerance: float = 5e-4,
    seed: int = 4321,
    eps: float = 1e-8,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AdaptiveSpectralReturnInfo,
]:
    """Escalate rank-1 -> block-4/8 -> exact SVD for a radial return.

    Rank one is accepted only when certified bounds from the previous returned
    point, Weyl's perturbation bound, and the current Ritz residual jointly
    identify the leading singular triplet. Fixed block stages refresh the
    leading-two interval. Any unresolved or warning-band case uses full SVD.
    """
    if rank1_iterations < 1:
        raise ValueError("rank1_iterations must be at least 1")
    if not (0.0 < residual_gap_fraction < 1.0):
        raise ValueError("residual_gap_fraction must be in (0, 1)")
    if max_block_size not in (4, 8):
        raise ValueError("max_block_size must be 4 or 8")
    if perturbation_norm_upper < 0.0:
        raise ValueError("perturbation_norm_upper must be non-negative")

    Z32 = Z.to(torch.float32)
    total_iterations = 0
    rejection_reasons = []
    rank1 = _warm_rank1_ritz(
        Z32,
        initial_u=initial_u,
        initial_v=initial_v,
        iterations=rank1_iterations,
        seed=seed,
        eps=eps,
    )
    total_iterations += rank1.iterations
    sigma_hat = float(rank1.S[0].item())
    residual_abs = float(rank1.residuals_absolute[0].item())
    residual_rel = residual_abs / max(sigma_hat, eps)

    bounds_finite = all(
        math.isfinite(value)
        for value in (
            prior_sigma1_lower,
            prior_sigma1_upper,
            prior_sigma2_lower,
            prior_sigma2_upper,
        )
    )
    if prior_bounds_certified and bounds_finite:
        delta = float(perturbation_norm_upper)
        numerical_slack = (
            32.0
            * torch.finfo(torch.float32).eps
            * max(prior_sigma1_upper + delta, 1.0)
        )
        sigma1_lower = max(prior_sigma1_lower - delta - numerical_slack, eps)
        sigma1_upper = max(
            prior_sigma1_upper + delta + numerical_slack,
            sigma1_lower,
        )
        sigma2_lower = max(prior_sigma2_lower - delta - numerical_slack, 0.0)
        sigma2_upper = max(
            prior_sigma2_upper + delta + numerical_slack,
            sigma2_lower,
        )

        # The residual localizes the tracked Ritz value. Intersecting it with
        # Weyl's interval tightens the radial-norm certificate.
        tracked_lower = max(sigma1_lower, sigma_hat - residual_abs)
        tracked_upper = min(sigma1_upper, sigma_hat + residual_abs)
        separated = tracked_lower > sigma2_upper
        identifies_top = (
            tracked_lower <= tracked_upper
            and sigma_hat + residual_abs >= sigma1_lower
            and sigma_hat - residual_abs > sigma2_upper
        )
        gap_lower = (
            max(0.0, 1.0 - sigma2_upper / tracked_lower)
            if separated
            else 0.0
        )
        gap_upper = max(
            gap_lower,
            1.0 - sigma2_lower / max(tracked_upper, eps),
        )
        angle_ratio = residual_abs / max(tracked_lower - sigma2_upper, eps)
        radial_scale = target_radius / max(sigma_hat, eps)
        returned_sigma1_lower = radial_scale * tracked_lower
        returned_sigma1_upper = radial_scale * tracked_upper
        radial_error = max(
            abs(returned_sigma1_lower - target_radius),
            abs(returned_sigma1_upper - target_radius),
        ) / max(target_radius, eps)
        rank1_checks = (
            ("rank1_top_identity", identifies_top),
            ("rank1_warning_gap", gap_lower > warning_gap_rel),
            ("rank1_angle", angle_ratio <= residual_gap_fraction),
            ("rank1_residual", residual_rel <= residual_tolerance),
            ("rank1_radial", radial_error <= residual_tolerance),
        )
        rejection_reasons.extend(
            reason for reason, passed in rank1_checks if not passed
        )
        if all(passed for _, passed in rank1_checks):
            returned = Z32 * radial_scale
            sigma2_mid = 0.5 * (sigma2_lower + sigma2_upper) * radial_scale
            returned_s = torch.stack(
                (Z32.new_tensor(target_radius), Z32.new_tensor(sigma2_mid))
            )
            info = AdaptiveSpectralReturnInfo(
                mode="hierarchical_rank1",
                iterations=total_iterations,
                subspace_dimension=1,
                max_residual_relative=residual_rel,
                gap_lower=gap_lower,
                gap_upper=min(1.0, gap_upper),
                used_exact_fallback=False,
                trial_sigma1=sigma_hat,
                clipped_count=0,
                sigma1_lower=returned_sigma1_lower,
                sigma1_upper=returned_sigma1_upper,
                sigma2_lower=radial_scale * sigma2_lower,
                sigma2_upper=radial_scale * sigma2_upper,
                bounds_certified=True,
                rejection_reasons=tuple(rejection_reasons),
            )
            return returned, rank1.U, returned_s, rank1.Vh, info
    else:
        rejection_reasons.append("rank1_prior_bounds")

    stage_u = initial_u if initial_u is not None else rank1.U
    stage_v = initial_v if initial_v is not None else rank1.Vh.transpose(-2, -1)
    block_sizes = [4] if max_block_size == 4 else [4, 8]
    latest_info = None
    for block_size in block_sizes:
        returned, U, returned_s, Vh, block_info = (
            adaptive_warm_fixed_subspace_radial_return(
                Z32,
                target_radius,
                gap_threshold_rel=warning_gap_rel,
                block_size=block_size,
                initial_u=stage_u,
                initial_v=stage_v,
                initial_iterations=block_initial_iterations,
                max_iterations=block_max_iterations,
                check_interval=1,
                residual_tolerance=residual_tolerance,
                require_gap_certificate=False,
                seed=seed + block_size,
                exact_fallback=False,
                eps=eps,
            )
        )
        total_iterations += block_info.iterations
        latest_info = block_info
        dynamic_ratio = block_info.max_residual_relative / max(
            block_info.gap_lower,
            eps,
        )
        block_checks = (
            (f"block{block_size}_residual", block_info.bounds_certified),
            (f"block{block_size}_warning_gap", block_info.gap_lower > warning_gap_rel),
            (f"block{block_size}_residual_gap", dynamic_ratio <= residual_gap_fraction),
        )
        rejection_reasons.extend(
            reason for reason, passed in block_checks if not passed
        )
        if all(passed for _, passed in block_checks):
            info = AdaptiveSpectralReturnInfo(
                mode=f"hierarchical_block{block_info.subspace_dimension}",
                iterations=total_iterations,
                subspace_dimension=block_info.subspace_dimension,
                max_residual_relative=block_info.max_residual_relative,
                gap_lower=block_info.gap_lower,
                gap_upper=block_info.gap_upper,
                used_exact_fallback=False,
                trial_sigma1=block_info.trial_sigma1,
                clipped_count=0,
                sigma1_lower=block_info.sigma1_lower,
                sigma1_upper=block_info.sigma1_upper,
                sigma2_lower=block_info.sigma2_lower,
                sigma2_upper=block_info.sigma2_upper,
                bounds_certified=True,
                rejection_reasons=tuple(rejection_reasons),
            )
            return returned, U, returned_s, Vh, info
        stage_u = U
        stage_v = Vh.transpose(-2, -1)

    U, S, Vh = accurate_svd_fp32(Z32)
    radial_scale = target_radius / S[0].clamp_min(eps)
    returned = Z32 * radial_scale
    returned_s = S * radial_scale
    exact_gap = (
        max(0.0, 1.0 - float(S[1].item()) / max(float(S[0].item()), eps))
        if S.numel() > 1
        else 1.0
    )
    fallback_residual = (
        latest_info.max_residual_relative if latest_info is not None else residual_rel
    )
    sigma2 = float(returned_s[1].item()) if returned_s.numel() > 1 else 0.0
    info = AdaptiveSpectralReturnInfo(
        mode="full_svd_fallback",
        iterations=total_iterations,
        subspace_dimension=max_block_size,
        max_residual_relative=fallback_residual,
        gap_lower=exact_gap,
        gap_upper=exact_gap,
        used_exact_fallback=True,
        trial_sigma1=float(S[0].item()),
        clipped_count=0,
        sigma1_lower=float(returned_s[0].item()),
        sigma1_upper=float(returned_s[0].item()),
        sigma2_lower=sigma2,
        sigma2_upper=sigma2,
        bounds_certified=True,
        rejection_reasons=tuple(rejection_reasons),
    )
    return returned, U, returned_s, Vh, info


@torch.no_grad()
def adaptive_warm_block2_radial_return(
    Z: torch.Tensor,
    target_radius: float,
    **kwargs,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AdaptiveSpectralReturnInfo,
]:
    """Compatibility wrapper for the original fixed warm block-2 return."""
    return adaptive_warm_fixed_subspace_radial_return(
        Z,
        target_radius,
        block_size=2,
        **kwargs,
    )


@torch.no_grad()
def adaptive_block_krylov_sphere_return(
    Z: torch.Tensor,
    target_radius: float,
    *,
    branch: str,
    projection_rank: int = 8,
    oversampling: int = 2,
    initial_iterations: int = 8,
    max_iterations: int = 40,
    residual_tolerance: float = 2e-6,
    gap_threshold_rel: float = 3e-5,
    gap_interval_fraction: float = 0.2,
    boundary_margin_rel: float = 2e-5,
    initial_u: Optional[torch.Tensor] = None,
    initial_v: Optional[torch.Tensor] = None,
    seed: int = 4321,
    exact_fallback: bool = True,
    eps: float = 1e-8,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AdaptiveSpectralReturnInfo,
]:
    """Apply a radial or projected return from one adaptive decomposition.

    The tangent branch computes only the leading two triplets.  The ambient
    PGD branch computes ``projection_rank`` triplets plus one sentinel pair so
    all certified active singular values can be clipped and the omitted tail
    can be bounded.  A full FP32 SVD is used only when the residual-based gap
    interval or PGD clipping boundary remains ambiguous.
    Returned ``U, S, Vh`` describe the returned point and can seed the next
    optimizer step without another decomposition.
    """
    if branch not in ("spel", "pgd"):
        raise ValueError("branch must be 'spel' or 'pgd'")
    if target_radius <= 0.0:
        raise ValueError("target_radius must be positive")
    if projection_rank < 2:
        raise ValueError("projection_rank must be at least 2")

    Z32 = Z.to(torch.float32)
    minimum_dimension = min(Z32.shape)
    # A rank-k projected return needs the (k + 1)-st Ritz pair as a sentinel:
    # without it, sigma_k < R does not certify that no omitted singular value
    # also needs clipping.  The sentinel is never clipped in an accepted
    # approximate return; it only certifies the truncation boundary.
    wanted = min(
        2 if branch == "spel" else projection_rank + 1,
        minimum_dimension,
    )
    block_size = min(wanted + oversampling, minimum_dimension)
    basis_side = "right" if Z32.shape[1] <= Z32.shape[0] else "left"
    initial_basis = initial_v if basis_side == "right" else initial_u
    converged = False
    latest_gap = (0.0, 1.0)

    gap_width_target = max(
        gap_interval_fraction * max(gap_threshold_rel, 1e-5),
        2.0 * residual_tolerance,
    )

    def stop_rule(result: BlockKrylovSVDResult) -> bool:
        nonlocal converged, latest_gap
        scale = max(float(result.S[0].item()), eps)
        max_residual = float(result.residuals_absolute.max().item()) / scale
        latest_gap = _relative_gap_interval(result, eps=eps)
        gap_width = latest_gap[1] - latest_gap[0]
        converged = (
            max_residual <= residual_tolerance and gap_width <= gap_width_target
        )
        return converged

    result = block_krylov_svd_fp32(
        Z32,
        wanted=wanted,
        block_size=block_size,
        initial_iterations=initial_iterations,
        max_iterations=max_iterations,
        seed=seed,
        reorthogonalization_passes=2,
        stop_rule=stop_rule,
        initial_basis=initial_basis,
        check_interval=4,
        eps=eps,
    )
    latest_gap = _relative_gap_interval(result, eps=eps)
    scale = max(float(result.S[0].item()), eps)
    max_residual = float(result.residuals_absolute.max().item()) / scale
    gap_ambiguous = (
        latest_gap[0] <= gap_threshold_rel <= latest_gap[1]
        or latest_gap[1] - latest_gap[0] > gap_width_target
    )
    boundary_ambiguous = False
    if branch == "pgd" and minimum_dimension > projection_rank:
        if result.S.numel() <= projection_rank:
            boundary_ambiguous = True
        else:
            sentinel_upper = float(
                result.S[projection_rank].item()
                + result.residuals_absolute[projection_rank].item()
            )
            boundary_ambiguous = sentinel_upper >= target_radius * (
                1.0 - boundary_margin_rel
            )

    if exact_fallback and (not converged or gap_ambiguous or boundary_ambiguous):
        U, S, Vh = accurate_svd_fp32(Z32)
        trial_sigma1 = float(S[0].item())
        returned_s = S.clone()
        if branch == "spel":
            radial_scale = target_radius / S[0].clamp_min(eps)
            returned = Z32 * radial_scale
            returned_s = S * radial_scale
            clipped_count = 0
        else:
            clipped_count = int(torch.count_nonzero(S > target_radius).item())
            returned_s.clamp_(max=target_radius)
            returned_s[0] = target_radius
            returned = (U * returned_s.unsqueeze(0)) @ Vh
        exact_gap = (
            max(0.0, 1.0 - float(S[1].item()) / max(trial_sigma1, eps))
            if S.numel() > 1
            else 1.0
        )
        info = AdaptiveSpectralReturnInfo(
            mode="full_svd_fallback",
            iterations=result.iterations,
            subspace_dimension=result.subspace_dimension,
            max_residual_relative=max_residual,
            gap_lower=exact_gap,
            gap_upper=exact_gap,
            used_exact_fallback=True,
            trial_sigma1=trial_sigma1,
            clipped_count=clipped_count,
        )
        return returned, U, returned_s, Vh, info

    returned_s = result.S.clone()
    if branch == "spel":
        radial_scale = target_radius / result.S[0].clamp_min(eps)
        returned = Z32 * radial_scale
        returned_s.mul_(radial_scale)
        clipped_count = 0
    else:
        clipped_count = int(
            torch.count_nonzero(result.S > target_radius).item()
        )
        returned_s.clamp_(max=target_radius)
        returned_s[0] = target_radius
        correction = result.U @ (
            (returned_s - result.S).unsqueeze(-1) * result.Vh
        )
        returned = Z32 + correction

    info = AdaptiveSpectralReturnInfo(
        mode="adaptive_block_krylov",
        iterations=result.iterations,
        subspace_dimension=result.subspace_dimension,
        max_residual_relative=max_residual,
        gap_lower=latest_gap[0],
        gap_upper=latest_gap[1],
        used_exact_fallback=False,
        trial_sigma1=float(result.S[0].item()),
        clipped_count=clipped_count,
    )
    return returned, result.U, returned_s, result.Vh, info


@torch.no_grad()
def full_svd_sphere_projection(
    Z: torch.Tensor,
    target_radius: float,
) -> Tuple[torch.Tensor, float]:
    Z32 = Z.to(torch.float32)
    U, S, Vh = accurate_svd_fp32(Z32)
    if S.numel() == 0:
        return Z32, 0.0
    trial_sigma1 = float(S[0].item())
    projected = S.clone()
    if projected[0] >= target_radius:
        projected.clamp_(max=target_radius)
    projected[0] = target_radius
    return (U * projected.unsqueeze(0)) @ Vh, trial_sigma1


@torch.no_grad()
def block_krylov_topk_sphere_projection(
    Z: torch.Tensor,
    target_radius: float,
    *,
    rank: int = 8,
    oversampling: int = 2,
    initial_iterations: int = 8,
    max_iterations: int = 40,
    residual_tolerance: float = 1e-4,
    boundary_margin_rel: float = 5e-4,
    seed: int = 4321,
    exact_fallback: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, TopKProjectionInfo]:
    """Top-k sphere projection with residual and clipping-boundary guards."""
    Z32 = Z.to(torch.float32)
    if min(Z32.shape) < rank:
        projected, sigma1 = full_svd_sphere_projection(Z32, target_radius)
        return projected, TopKProjectionInfo(
            "full_svd_fallback", 0, min(Z32.shape), 0.0, True, sigma1
        )

    converged = False

    def active_residual_relative(result: BlockKrylovSVDResult) -> float:
        active = result.S >= target_radius * (1.0 - boundary_margin_rel)
        if not bool(torch.any(active)):
            active = torch.zeros_like(result.S, dtype=torch.bool)
            active[0] = True
        scale = max(float(result.S[0].item()), eps)
        return float(result.residuals_absolute[active].max().item()) / scale

    def stop_rule(result: BlockKrylovSVDResult) -> bool:
        nonlocal converged
        converged = active_residual_relative(result) <= residual_tolerance
        return converged

    result = block_krylov_svd_fp32(
        Z32,
        wanted=rank,
        block_size=rank + oversampling,
        initial_iterations=initial_iterations,
        max_iterations=max_iterations,
        seed=seed,
        reorthogonalization_passes=2,
        stop_rule=stop_rule,
        eps=eps,
    )
    max_residual = active_residual_relative(result)
    boundary_ambiguous = float(result.S[-1].item()) >= target_radius * (
        1.0 - boundary_margin_rel
    )
    if exact_fallback and (not converged or boundary_ambiguous):
        projected, sigma1 = full_svd_sphere_projection(Z32, target_radius)
        return projected, TopKProjectionInfo(
            "full_svd_fallback",
            result.iterations,
            result.subspace_dimension,
            max_residual,
            True,
            sigma1,
        )

    target = result.S.clone()
    if target[0] >= target_radius:
        target.clamp_(max=target_radius)
    target[0] = target_radius
    correction = result.U @ ((target - result.S).unsqueeze(-1) * result.Vh)
    projected = Z32 + correction
    return projected, TopKProjectionInfo(
        "block_krylov_topk",
        result.iterations,
        result.subspace_dimension,
        max_residual,
        False,
        float(result.S[0].item()),
    )
