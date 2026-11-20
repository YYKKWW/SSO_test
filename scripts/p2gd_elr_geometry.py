"""Small-block repairs and geometry-only angular matching for a research replay.

The return is radial, not a metric projection. AUS is invariant to positive
radial scaling, so coefficient search uses only small eigensystems and norms.
"""
import math

import torch


METHODS = ('half_d0', 'ambient', 'top1', 'cluster_cone',
           'finite_momentum', 'anchored_momentum', 'finite_d0', 'anchored_d0')


def aus(x, y):
    return float((x/x.norm().clamp_min(1e-30)-y/y.norm().clamp_min(1e-30)).norm())


def repair(method, d0, momentum, u, s, v, radius, alpha):
    if alpha <= 0:
        raise ValueError('Repair coefficient must be positive')
    if method == 'half_d0':
        return .5*d0
    z = -.5*d0.norm()*momentum/momentum.norm().clamp_min(1e-30)
    if method == 'ambient':
        return z
    if method == 'top1':
        normal = u[:, :1]@v[:, :1].T
        return z-(normal*z).sum()*normal
    if method.endswith('_d0'):
        z = .5*d0
    a = u.T@z@v
    a = (a+a.T)*.5
    if method.startswith('anchored_'):
        b = torch.zeros_like(a)
        if len(s)>1:
            values, vectors = torch.linalg.eigh(a[1:, 1:])
            b[1:, 1:] = (vectors*values.clamp(max=0))@vectors.T
    elif method == 'cluster_cone' or method.startswith('finite_'):
        offset = torch.diag((radius-s)/alpha) if method.startswith('finite_') else torch.zeros_like(a)
        values, vectors = torch.linalg.eigh(a-offset)
        values = values.clamp(max=0)
        values[-1] = 0
        b = (vectors*values)@vectors.T+offset
    else:
        raise ValueError(method)
    return z+u@(b-a)@v.T


def match_aus(x, direction_at, alpha, target, rtol=2e-3, atol=2e-6,
              max_scale=64., iterations=40):
    """Rebuild alpha-dependent repairs; no objective values or full SVD calls."""
    if target <= 0 or alpha <= 0:
        raise ValueError('Positive angular target and coefficient required')
    tolerance = atol+rtol*target
    calls = 0

    def evaluate(coefficient):
        nonlocal calls
        calls += 1
        direction = direction_at(coefficient)
        trial = x+coefficient*direction
        value = aus(x, trial)
        if not math.isfinite(value) or float(trial.norm()) == 0:
            raise ValueError('Undefined angular candidate')
        return coefficient, direction, trial, value

    lo, hi = 0., alpha
    best = evaluate(hi)
    while best[3] < target-tolerance and hi < alpha*max_scale:
        lo, hi = hi, min(hi*2, alpha*max_scale)
        best = evaluate(hi)
    if best[3] < target-tolerance:
        raise RuntimeError('Angular target not bracketed within coefficient cap')
    for _ in range(iterations):
        if abs(best[3]-target) <= tolerance:
            break
        mid = (lo+hi)*.5
        current = evaluate(mid)
        if current[3] < target:
            lo = mid
        else:
            hi = mid
        if abs(current[3]-target) < abs(best[3]-target):
            best = current
    coefficient, direction, trial, measured = best
    if abs(measured-target) > tolerance:
        raise RuntimeError('Angular match failed; do not treat it as a valid comparison')
    return direction, trial, dict(effective_alpha=coefficient, target=target,
        measured=measured, tolerance=tolerance, relative_error=abs(measured-target)/target,
        evaluations=calls, metric='aus', matching_enforced=True, return_kind='radial',
        reference_reused=False, search_full_svd_calls=0)
