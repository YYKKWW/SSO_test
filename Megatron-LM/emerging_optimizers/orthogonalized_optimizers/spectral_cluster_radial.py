"""Experimental near-cluster directions with a unified radial return.

This is not exact P2GD at a positive gap. Both branches share one trial
spectral solve (exact SVD by default, or fixed warm block-8 Ritz), and cache
the returned spectrum. There is no projection fallback or online accuracy
gate. The cluster direction either matches the MCSD Frobenius length (legacy)
or projects Frobenius-normalized momentum without further normalization.
Neither step rule carries a stochastic convergence guarantee here.
"""

import torch


def compute_cluster_radial_update(
    W, M, target_radius, power_iteration_steps, msign_steps, *,
    effective_lr, cluster_mode, cluster_threshold_rel, gap_threshold_rel,
    pgd_lr_scale, current_spectral_cache=None, tp_group=None,
    partition_dim=None, cluster_spectrum_mode="exact",
    cluster_iterations=10, cluster_direction_mode="matched_fro",
    cluster_near_projection="cone", **unused,
):
    # Imported here because the host optimizer dispatches to this module.
    from . import spel_pgd_same_projection as host
    from .spectral_ball_utils import _tp_world_and_rank

    if cluster_mode not in ("control", "fixed2", "fixed4", "adaptive4"):
        raise ValueError("invalid radial cluster mode")
    if cluster_threshold_rel < gap_threshold_rel or pgd_lr_scale <= 0:
        raise ValueError("cluster threshold must cover the routing threshold; scale must be positive")
    if _tp_world_and_rank(tp_group)[0] != 1:
        raise ValueError("the cluster-radial pilot supports TP=1 only")
    if W.ndim != 2 or min(W.shape) < 2:
        raise ValueError("cluster-radial requires a nontrivial matrix")
    if cluster_spectrum_mode not in ("exact", "warm_block8") or cluster_iterations < 1:
        raise ValueError("invalid cluster spectral solver")
    if cluster_direction_mode not in ("matched_fro", "momentum_fro"):
        raise ValueError("invalid cluster direction mode")
    if cluster_near_projection not in ("cone", "ambient", "tangent1"):
        raise ValueError("invalid near-branch projection")
    if cluster_near_projection != "cone" and cluster_direction_mode != "momentum_fro":
        raise ValueError("mechanism controls require Frobenius-normalized momentum")

    approximate = cluster_spectrum_mode == "warm_block8"

    def decompose(matrix, previous=None):
        if approximate:
            from .warm_block_ritz import warm_block8_ritz
            return warm_block8_ritz(
                matrix, iterations=cluster_iterations,
                initial_v=previous.v_basis if previous is not None else None,
            )
        return host.accurate_svd_fp32(matrix)

    def returned_cache(u, s, vh):
        options = dict(is_exact=False, bounds_certified=False,
                       gap_lower=0., gap_upper=1.) if approximate else {}
        return host._spectral_point_cache_from_decomposition(u, s, vh, **options)

    approximation_info = dict(
        approximate_spectral_iterations=cluster_iterations,
        approximate_subspace_dimension=min(8, *W.shape),
        approximate_mode="warm_block8_ritz",
    ) if approximate else {}

    eps = 1e-7
    cache = current_spectral_cache
    hit = bool(
        cache is not None
        and (approximate or cache.is_exact)
        and cache.s_basis is not None
        and (not approximate or cache.v_basis is not None
             and cache.v_basis.shape == (W.shape[1], min(8, *W.shape)))
        and host._spectral_point_cache_is_compatible(cache, W, target_radius)
    )
    if not hit:
        u, s, vh = decompose(W)
        if float(s[0]) <= eps:
            raise ValueError("cannot initialize a spectral sphere from a zero matrix")
        W.copy_(W.float() * (target_radius / s[0]))
        cache = returned_cache(
            u, s * (target_radius / s[0]), vh
        )
    # A valid cache is already at the returned point; do not re-decompose it.
    x = W.float()
    gap = cache.rel_gap
    fallback = cluster_mode != "control" and gap <= gap_threshold_rel
    rank = active = 0
    multiplier = 1.0
    flatten_distance = radial_ratio = 0.0
    branch = "pgd" if fallback else "spel"

    if effective_lr <= 0:
        return torch.zeros_like(x), host.SpELPGDUpdateInfo(
            branch="zero_lr", rel_gap=gap, sigma1=cache.sigma1,
            gap_sigma1=cache.sigma1, sigma2=cache.sigma2, pre_retract_bias=0.,
            trial_sigma=cache.sigma1, post_retract_bias=0.,
            projection_mode="cluster_radial", projection_rank=0, gap_probed=True,
            spectral_cache_hit=hit, next_spectral_cache=cache,
            **(approximation_info if not hit else {}),
        )

    # Normalized-momentum cluster steps do not need an MCSD length reference.
    if not fallback or cluster_direction_mode == "matched_fro":
        projected = host.project_to_tangent_plane(M.float(), cache.u1, cache.v1, eps=eps)
        projected = projected / torch.linalg.vector_norm(projected).clamp_min(eps)
        reference = host.msign(projected, steps=msign_steps).float()
        reference = host.project_to_tangent_plane(reference, cache.u1, cache.v1, eps=eps)
        direction = -reference
    if fallback:
        relative = (cache.s_basis[0] - cache.s_basis) / cache.s_basis[0]
        active = int((relative <= cluster_threshold_rel).sum())
        cap = 2 if cluster_mode == "fixed2" else 4
        rank = min(cap, len(cache.s_basis), active if cluster_mode == "adaptive4" else cap)
        # Routing uses cached scalar values; accommodate FP32 boundary rounding.
        rank = max(1, rank)
        momentum = M.float()
        if cluster_direction_mode == "momentum_fro":
            momentum = momentum / torch.linalg.vector_norm(momentum).clamp_min(1e-12)
        if cluster_near_projection == "ambient":
            direction = -momentum
            rank = 0
        elif cluster_near_projection == "tangent1":
            direction = host.project_to_tangent_plane(-momentum, cache.u1, cache.v1, eps=eps)
            rank = 1
        else:
            direction = host.project_spectral_sphere_cone(
                -momentum, cache.u_basis[:, :rank], cache.v_basis[:, :rank]
            )
        if cluster_direction_mode == "momentum_fro":
            multiplier = float(pgd_lr_scale)
        else:
            length = torch.linalg.vector_norm(direction)
            multiplier = float(pgd_lr_scale * torch.linalg.vector_norm(reference) /
                               length.clamp_min(1e-30))
        flatten_distance = float(
            torch.linalg.vector_norm(target_radius - cache.s_basis[:rank]) / target_radius
        )

    alpha = float(effective_lr) * multiplier
    y = x + alpha * direction
    u, s, vh = decompose(y, cache)
    if float(s[0]) <= eps:
        raise ValueError("zero trial has no defined radial return")
    factor = target_radius / s[0]
    returned = y * factor
    next_cache = returned_cache(u, s * factor, vh)
    radial_ratio = float(
        torch.linalg.vector_norm(returned - y) /
        (alpha * torch.linalg.vector_norm(direction)).clamp_min(1e-30)
    )
    info = host.SpELPGDUpdateInfo(
        branch=branch, rel_gap=gap, sigma1=cache.sigma1,
        gap_sigma1=cache.sigma1, sigma2=cache.sigma2, pre_retract_bias=0.,
        trial_sigma=float(s[0]), post_retract_bias=float("nan") if approximate else 0.,
        projection_mode="cluster_radial", projection_rank=rank, gap_probed=True,
        next_v=next_cache.v1, spectral_cache_hit=hit, next_spectral_cache=next_cache,
        pgd_trial_rel_gap=next_cache.rel_gap if fallback else float("nan"),
        pgd_post_rel_gap=next_cache.rel_gap if fallback else float("nan"),
        cluster_dimension=rank,
        cluster_cap_overflow=bool(cluster_near_projection == "cone" and active > rank),
        cluster_step_multiplier=multiplier, radial_change_ratio=radial_ratio,
        cluster_flatten_distance=flatten_distance,
        **approximation_info,
    )
    return (x - returned) / float(effective_lr), info
