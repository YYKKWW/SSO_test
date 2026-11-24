"""One selected BF16-matched policy; no counterfactual loss or rollback."""
import torch

from p2gd_budget_geometry import norm2, radial
from p2gd_elr_geometry import repair
from p2gd_late_geometry import angle, forward_match

POLICIES = ('half_d0', 'ambient', 'top1', 'cluster_cone', 'half_output_of_cluster')


@torch.no_grad()
def selected_return(policy, x, momentum, d0, u, s, v, radius, alpha, h, before,
                    dtype=torch.bfloat16):
    if policy not in POLICIES or dtype != torch.bfloat16:
        raise ValueError('Unknown policy or non-BF16 forward weights')
    baseline = radial(x + alpha * .5 * d0, radius)
    base_delta = baseline.to(dtype).float() - before
    spectral_target = float(norm2(base_delta) / norm2(before))
    direction = .5 * d0
    reference = dict(effective_alpha=alpha, target=0., measured=0., tolerance=0.,
                     relative_error=0., evaluations=0, matching_satisfied=True,
                     matching_enforced=False, metric='reference', norm_svd_calls=0)
    if policy == 'half_d0' or spectral_target == 0:
        point, info = baseline, dict(reference, zero_budget=spectral_target == 0)
    else:
        name = 'cluster_cone' if policy == 'half_output_of_cluster' else policy
        fn = lambda coefficient: repair(name, d0, momentum, u, s, v, radius, coefficient)
        direction, point, info = forward_match(x, fn, radius, alpha, spectral_target,
                                               'spectral', h, before, dtype)
        if policy == 'half_output_of_cluster':
            cluster_info = info
            target = float((h @ (point.to(dtype).float() - before).T).norm() /
                           (h @ before.T).norm())
            if target == 0:
                raise ValueError('Zero output target: do not silently accept a nonzero step')
            direction, point, info = forward_match(x, lambda coefficient: .5*d0,
                radius, alpha, target, 'output', h, before, dtype)
            info['reference_cluster_match'] = cluster_info
            info['norm_svd_calls'] += cluster_info['norm_svd_calls']
            info['evaluations'] += cluster_info['evaluations']
    delta = point - x
    rounded = point.to(dtype).float() - before
    z = -.5 * d0.norm() * momentum / momentum.norm().clamp_min(1e-30)
    a = u.T @ z @ v
    a = .5 * (a + a.T)
    block = u.T @ direction @ v
    block = .5 * (block + block.T)
    corrected = policy in ('ambient', 'top1', 'cluster_cone') and spectral_target > 0
    correction = block - a
    cs = torch.linalg.svdvals(correction)
    rank_tol = max(float(z.norm())*1e-7, float(cs.max())*1e-5, 1e-12)
    info.update(policy=policy, spectral_target=spectral_target,
        bf16_spectral_elr=float(norm2(rounded) / norm2(before)),
        bf16_output_elr=float((h @ rounded.T).norm() / (h @ before.T).norm()),
        master_spectral_elr=float(norm2(delta) / norm2(x)),
        alpha_ratio=info['effective_alpha']/alpha,
        correction_rank=int((cs > rank_tol).sum()) if corrected else None,
        correction_fraction=float(correction.norm()/z.norm().clamp_min(1e-30)) if corrected else None,
        momentum_angle=angle(z, direction) if corrected else None,
        top_block_angle=angle(a, block) if corrected else None,
        rounding_fraction=float((rounded-delta).norm()/delta.norm().clamp_min(1e-30)),
        returned_frobenius_over_radius=float(delta.norm()/radius),
        radius_error=abs(float(norm2(point)/radius)-1),
        target_reference='own_state_cluster' if policy == 'half_output_of_cluster' else 'own_state_half_d0')
    return point, info
