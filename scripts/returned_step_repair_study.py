"""Same-state MCSD-TP/P2GD-type radial audit with post-return ELR matching.

Only geometry determines the matching coefficient. Full SVD is an offline
oracle here; none of these quadratic objectives is an LLM loss measurement.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from audit_corner_momentum import synthetic_states
from local_anchored_cone_radial import anchored_project, unit
from local_cone_lmo_feasibility import cone
from local_step_aware_hybrid import linear_direction, model_project, sym, tangent

METHODS = ('half_d0', 'ambient', 'top1', 'cluster_cone',
           'finite_momentum', 'anchored_momentum', 'finite_d0', 'anchored_d0')


class State:
    def __init__(self, x, m, radius, alpha, d0=None, window=2e-4):
        self.radius, self.alpha = float(radius), float(alpha)
        self.u, s, vh = torch.linalg.svd(x, full_matrices=False)
        self.v = vh.T
        self.x = x * (radius/s[0])
        self.s = s * (radius/s[0])
        self.m = m
        self.q = int(((1-self.s/radius) <= window).sum())
        self.true_q = int(((1-self.s/radius) <= 1e-10).sum())
        self.d0 = linear_direction(m, self.u[:, :1], self.v[:, :1])[0] if d0 is None else d0.clone()
        self.z = -.5*self.d0.norm()*unit(m)

    def direction(self, method, alpha):
        u, s, v = self.u[:, :self.q], self.s[:self.q], self.v[:, :self.q]
        if method == 'half_d0':
            return .5*self.d0
        if method == 'ambient':
            return self.z
        if method == 'top1':
            return tangent(self.z, u[:, :1], v[:, :1])
        if method == 'cluster_cone':
            return cone(self.z, u, v)
        if method.startswith('finite_'):
            # The model must use the candidate coefficient, not the nominal LR.
            z = .5*self.d0 if method == 'finite_d0' else self.z
            return model_project(z, u, s, v, self.radius, alpha)
        if method.startswith('anchored_'):
            z = .5*self.d0 if method == 'anchored_d0' else self.z
            return anchored_project(z, u, v)
        raise ValueError(method)

    def evaluate(self, method, alpha, metric):
        if alpha == 0:
            return dict(alpha=0., direction=torch.zeros_like(self.x), returned=self.x.clone(), length=0.)
        d = self.direction(method, alpha)
        y = self.x + alpha*d
        n = torch.linalg.svdvals(y)[0]
        if not torch.isfinite(n) or n <= 0:
            raise ValueError('undefined radial return')
        returned = y*(self.radius/n)
        delta = returned-self.x
        if metric == 'spectral':
            length = float(torch.linalg.svdvals(delta)[0]/self.radius)
        elif metric == 'frobenius':
            length = float(delta.norm()/self.radius)
        elif metric == 'aus':
            length = float((unit(returned)-unit(self.x)).norm())
        else:
            raise ValueError(metric)
        return dict(alpha=float(alpha), direction=d, returned=returned, length=length)


def match(state, method, target, metric, rtol=1e-5, max_scale=64., iterations=36):
    """Bracket/root search on geometric length only; verify, never assume success.

    A degenerate eigenspace selection can make a candidate path discontinuous.
    A residual that does not converge is a failed match, not a small-loss win.
    """
    if target <= 1e-12:
        return state.evaluate(method, 0., metric), dict(matched=False, reason='zero_reference', evaluations=0)
    tolerance = max(1e-12, target*rtol)
    lo, hi, calls = 0., state.alpha, 0
    best = state.evaluate(method, hi, metric)
    calls += 1
    while best['length'] < target-tolerance and hi < state.alpha*max_scale:
        lo, hi = hi, min(hi*2, state.alpha*max_scale)
        best = state.evaluate(method, hi, metric)
        calls += 1
    if best['length'] < target-tolerance:
        return best, dict(matched=False, reason='no_bracket_within_cap', evaluations=calls)
    for _ in range(iterations):
        if abs(best['length']-target) <= tolerance:
            break
        mid = (lo+hi)*.5
        candidate = state.evaluate(method, mid, metric)
        calls += 1
        if candidate['length'] < target:
            lo = mid
        else:
            hi = mid
        if abs(candidate['length']-target) < abs(best['length']-target):
            best = candidate
    success = abs(best['length']-target) <= tolerance
    return best, dict(matched=success, reason='matched' if success else 'residual_not_converged', evaluations=calls)


def metrics(state, result, gradient=None):
    x, d, returned = state.x, result['direction'], result['returned']
    alpha = result['alpha']
    delta = returned-x
    values = torch.linalg.svdvals(returned)
    h = sym(state.u[:, :state.true_q].T@d@state.v[:, :state.true_q])
    out = dict(alpha=alpha, alpha_ratio=alpha/state.alpha,
        returned_spectral_elr=float(torch.linalg.svdvals(delta)[0]/state.radius),
        returned_frobenius_elr=float(delta.norm()/state.radius),
        aus=float((unit(returned)-unit(x)).norm()),
        trial_frobenius_elr=float(alpha*d.norm()/state.radius),
        radial_change_fraction=float((delta-alpha*d).norm())/max(float(alpha*d.norm()), 1e-30),
        radius_error=abs(float(values[0]/state.radius)-1),
        returned_gap=float(1-values[1]/values[0]),
        tangent_residual=float(torch.linalg.eigvalsh(h)[-1])/max(float(d.norm()), 1e-30),
        momentum_work=float((state.m*delta).sum()),
        loss_change=None, gradient_work=None, trial_gradient_work=None, return_gradient_work=None)
    if gradient is not None:
        out.update(loss_change=float((gradient*delta).sum()+.5*delta.square().sum()),
            gradient_work=float((gradient*delta).sum()),
            trial_gradient_work=float((gradient*(alpha*d)).sum()),
            return_gradient_work=float((gradient*(delta-alpha*d)).sum()))
    return out


def audit(state, gradient, metadata, metric_names=('spectral', 'frobenius', 'aus')):
    rows = []
    for metric in metric_names:
        reference = state.evaluate('half_d0', state.alpha, metric)
        target = reference['length']
        for method in METHODS:
            nominal = state.evaluate(method, state.alpha, metric)
            result, info = match(state, method, target, metric)
            row = dict(metadata, q=state.q, true_q=state.true_q, method=method, metric=metric,
                target=target, relative_match_error=abs(result['length']-target)/max(target, 1e-30),
                **info, **metrics(state, result, gradient),
                nominal=metrics(state, nominal, gradient))
            rows.append(row)
    return rows


def summarize(rows):
    summary = []
    cases = defaultdict(list)
    for row in rows:
        cases[row['group'], row['state_id'], row['metric']].append(row)
    common = {key for key, group in cases.items() if len(group)==len(METHODS) and all(r['matched'] for r in group)}
    by = defaultdict(list)
    for row in rows:
        by[row['group'], row['metric'], row.get('alignment'), row['method']].append(row)
    for (group, metric, alignment, method), selected in sorted(by.items(), key=lambda item:str(item[0])):
        paired = [r for r in selected if (group, r['state_id'], metric) in common]
        differences = []
        for r in paired:
            ref = next(v for v in cases[group, r['state_id'], metric] if v['method']=='half_d0')
            if r['loss_change'] is not None:
                differences.append(r['loss_change']-ref['loss_change'])
        loss_rows = [r for r in paired if r['loss_change'] is not None]
        summary.append(dict(group=group, metric=metric, alignment=alignment, method=method,
            cases=len(selected), matched=sum(r['matched'] for r in selected), common_cases=len(paired),
            loss_cases=len(loss_rows), loss_increases=sum(r['loss_change']>1e-12 for r in loss_rows),
            mean_loss_change=sum(r['loss_change'] for r in loss_rows)/len(loss_rows) if loss_rows else None,
            mean_loss_difference_vs_half=sum(differences)/len(differences) if differences else None,
            wins_vs_half=sum(d < -1e-10 for d in differences),
            ties_vs_half=sum(abs(d) <= 1e-10 for d in differences),
            max_successful_match_error=max((r['relative_match_error'] for r in selected if r['matched']), default=None),
            mean_alpha_ratio=sum(r['alpha_ratio'] for r in paired)/len(paired) if paired else None))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--skip-toy', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    start, rows = time.perf_counter(), []
    if not args.skip_toy:
        for i, (meta, x, m, g, alpha) in enumerate(synthetic_states()):
            state = State(x, m, 1., alpha)
            group = 'exact_corner' if state.true_q>1 else ('near_simple' if state.q>1 else 'safe_simple')
            rows.extend(audit(state, g, dict(meta, group=group, state_id=str(i))))
        print('toy complete', flush=True)
    if args.snapshots:
        files = sorted(args.snapshots.glob('sample_*.pt'))
        selected = []
        for path in files:
            data = torch.load(path, map_location='cpu', weights_only=True)
            if data['record']['near']:
                selected.append((path, data))
        for path, data in selected[:args.limit]:
            state = State(data['current'].double(), data['momentum'].double(),
                float(data['radius']), float(data['alpha']), data['d0'].double())
            rows.extend(audit(state, None, dict(group='real_snapshot', state_id=path.stem,
                step=data['step'], shape=list(state.x.shape), alignment=None)))
            print('snapshot', path.name, flush=True)
    report = dict(rows=rows, summary=summarize(rows), seconds=time.perf_counter()-start,
        protocol=dict(window=2e-4, reference='exact radial of X + alpha * 0.5 * D0',
            metrics=['returned spectral norm / R', 'returned Frobenius norm / R', 'AUS'],
            search='geometry only, alpha <=64 nominal, finite-step repair rebuilt at each alpha'),
        limitations=['Constructed quadratic losses are not LLM losses.',
            'Real snapshots contain no raw gradient or full model; only geometry/momentum work is reported.',
            'All methods are evaluated at the same exact-radial-normalized X with full FP64 geometry.',
            'Common matched cases are reported separately from failures; search never consults loss.',
            'Forced one-step counterfactuals do not establish stochastic convergence or training superiority.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for row in report['summary']:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
