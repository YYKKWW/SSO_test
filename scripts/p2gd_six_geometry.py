"""Six same-state controls: shared radial return, separate amplitude controls."""
import torch

from p2gd_budget_geometry import match_returned, norm2, radial
from p2gd_elr_geometry import aus, repair

ARMS = [
    dict(name='half_d0', direction='half_d0', metric='reference', fraction=1.),
    dict(name='half_output_of_cluster', direction='half_d0', metric='output', fraction=1.),
    *[dict(name=name, direction=name, metric='spectral', fraction=1.)
      for name in ('ambient', 'top1', 'cluster_cone', 'finite_momentum')],
]


@torch.no_grad()
def candidates(x, m, d0, u, s, v, radius, alpha, h, baseline=None):
    baseline = radial(x+.5*alpha*d0, radius) if baseline is None else baseline
    x_norm, output_norm = norm2(x), (h@x.T).norm()
    if output_norm <= 1e-20:
        raise ValueError('Zero calibration output')
    targets = dict(spectral=float(norm2(baseline-x)/x_norm),
                   output=float((h@(baseline-x).T).norm()/output_norm))
    points = {'half_d0': baseline}
    details = {'half_d0': (.5*d0, dict(effective_alpha=alpha, target=0., measured=0.,
        tolerance=0., relative_error=0., evaluations=0, matching_enforced=False,
        metric='reference', return_kind='radial', reference_reused=True, norm_svd_calls=0))}
    for name in ('ambient', 'top1', 'cluster_cone', 'finite_momentum'):
        fn = lambda coefficient: repair(name, d0, m, u, s, v, radius, coefficient)
        d, point, info = match_returned(x, fn, radius, alpha, targets['spectral'], 'spectral', h)
        points[name], details[name] = point, (d, info)
    # Match this component's realized cone output, not a global empirical ratio.
    output_target = float((h@(points['cluster_cone']-x).T).norm()/output_norm)
    d, point, info = match_returned(x, lambda coefficient: .5*d0, radius,
                                  alpha, output_target, 'output', h)
    points['half_output_of_cluster'] = point
    details['half_output_of_cluster'] = d, info
    records = []
    for arm in ARMS:
        name = arm['name']; d, info = details[name]; point = points[name]; delta = point-x
        block = u.T@d@v
        block = (block+block.T)*.5
        records.append(dict(method=name, direction=arm['direction'], budget_fraction=1.,
            target_reference='cluster_cone' if name=='half_output_of_cluster' else 'half_d0',
            **info, alpha_ratio=info['effective_alpha']/alpha,
            returned_spectral_elr=float(norm2(delta)/x_norm),
            local_output_elr=float((h@delta.T).norm()/output_norm), actual_aus=aus(x,point),
            radius_error=abs(float(norm2(point)/radius)-1),
            returned_frobenius_over_radius=float(delta.norm()/radius),
            momentum_work=float((m*delta).sum()),
            return_correction_fro=float((delta-info['effective_alpha']*d).norm()),
            cluster_normal_max_eigenvalue=float(torch.linalg.eigvalsh(block)[-1])))
    return points, records, targets
