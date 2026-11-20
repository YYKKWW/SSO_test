"""Diagnostic matching of complete spectral/output movement, never loss."""
import math

import torch

from p2gd_elr_geometry import aus, repair

DIRECTIONS = ('half_d0', 'finite_d0', 'cluster_cone', 'finite_momentum')
ARMS = [dict(name='half_d0', direction='half_d0', metric='reference', fraction=1.)]
for metric in ('spectral', 'output'):
    for fraction, tag in ((1., '100'), (.95, '095')):
        for direction in DIRECTIONS:
            if direction != 'half_d0' or fraction != 1.:
                ARMS.append(dict(name=f'{metric}_{tag}_{direction}', direction=direction,
                                 metric=metric, fraction=fraction))


def norm2(x):
    return torch.linalg.svdvals(x, **({'driver':'gesvd'} if x.is_cuda else {}))[0]


def radial(y, radius):
    n = norm2(y)
    if not torch.isfinite(n) or n <= 0:
        raise ValueError('Undefined radial return')
    return y*(radius/n)


def match_returned(x, direction_at, radius, nominal_alpha, target, metric, h=None,
                   rtol=2e-4, atol=2e-7, max_scale=64., iterations=40):
    if target <= 0 or nominal_alpha <= 0 or not math.isfinite(target):
        raise ValueError('Positive finite target and coefficient required')
    if metric == 'spectral':
        denominator = norm2(x)
    elif metric == 'output' and h is not None:
        denominator = (h@x.T).norm()
    else:
        raise ValueError('Spectral or output metric with calibration inputs required')
    if denominator <= 1e-20:
        raise ValueError('Zero reference signal')
    calls = 0
    tolerance = atol+rtol*target

    def evaluate(alpha):
        nonlocal calls
        calls += 1
        d = direction_at(alpha)
        returned = radial(x+alpha*d, radius)
        delta = returned-x
        measured = float((norm2(delta) if metric=='spectral' else (h@delta.T).norm())/denominator)
        if not math.isfinite(measured):
            raise ValueError('Nonfinite returned displacement')
        return alpha, d, returned, measured

    lo, hi = 0., nominal_alpha
    best = evaluate(hi)
    while best[3] < target-tolerance and hi < max_scale*nominal_alpha:
        lo, hi = hi, min(2*hi, max_scale*nominal_alpha)
        best = evaluate(hi)
    if best[3] < target-tolerance:
        raise RuntimeError('Returned target not bracketed within coefficient cap')
    for _ in range(iterations):
        if abs(best[3]-target) <= tolerance:
            break
        alpha = (lo+hi)*.5
        current = evaluate(alpha)
        if current[3] < target:
            lo = alpha
        else:
            hi = alpha
        if abs(current[3]-target) < abs(best[3]-target):
            best = current
    alpha, d, returned, value = best
    if abs(value-target) > tolerance:
        raise RuntimeError('Returned metric match failed; no silent fallback')
    return d, returned, dict(effective_alpha=alpha, target=target, measured=value,
        tolerance=tolerance, relative_error=abs(value-target)/target, evaluations=calls,
        matching_enforced=True, metric=metric, return_kind='radial', reference_reused=False,
        norm_svd_calls=(1 if metric=='spectral' else 0)+calls*(2 if metric=='spectral' else 1))


@torch.no_grad()
def candidates(x, m, d0, u, s, v, radius, alpha, h, baseline=None):
    baseline = radial(x+.5*alpha*d0, radius) if baseline is None else baseline
    x_norm = norm2(x)
    output_norm = (h@x.T).norm()
    if output_norm <= 1e-20:
        raise ValueError('Zero calibration output')
    targets = dict(spectral=float(norm2(baseline-x)/x_norm),
                   output=float((h@(baseline-x).T).norm()/output_norm))
    points, records = {}, []
    for arm in ARMS:
        name, method, metric, fraction = (arm[k] for k in ('name','direction','metric','fraction'))
        if name == 'half_d0':
            d, point = .5*d0, baseline
            info = dict(effective_alpha=alpha, target=0., measured=0., tolerance=0.,
                relative_error=0., evaluations=0, matching_enforced=False, metric=metric,
                return_kind='radial', reference_reused=True, norm_svd_calls=0)
        else:
            fn = lambda coefficient: repair(method, d0, m, u, s, v, radius, coefficient)
            d, point, info = match_returned(x, fn, radius, alpha,
                targets[metric]*fraction, metric, h)
        delta = point-x
        points[name] = point
        records.append(dict(method=name, direction=method, budget_fraction=fraction,
            **info, alpha_ratio=info['effective_alpha']/alpha,
            returned_spectral_elr=float(norm2(delta)/x_norm),
            local_output_elr=float((h@delta.T).norm()/output_norm),
            actual_aus=aus(x,point), radius_error=abs(float(norm2(point)/radius)-1),
            returned_frobenius_over_radius=float(delta.norm()/radius),
            momentum_work=float((m*delta).sum()),
            return_correction_fro=float((delta-info['effective_alpha']*d).norm())))
    return points, records, targets
