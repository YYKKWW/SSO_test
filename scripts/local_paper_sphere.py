"""Exact small spectral spheres: paper trajectories and matched six-way returns.

CPU FP64 only. Geometry matching never reads the objective. The trajectory
reference uses EMA, a deterministic certified step, and exact metric PGD.
The six-way replay reuses the H20 diagnostic's actual direction/matching code.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from p2gd_budget_geometry import norm2, radial
from p2gd_six_geometry import ARMS, candidates

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('pgd', 'hybrid_exact', 'hybrid_half', 'cluster_radial', 'ambient_radial', 'true_cone_radial')


def spectrum(x):
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    return u, s, vh.transpose(-2, -1)


def project(x):
    u, s, v = spectrum(x)
    s = s.clamp(max=1.)
    s[..., 0] = 1.
    return (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)


def radial_batch(x):
    return x / torch.linalg.svdvals(x)[..., 0, None, None]


def tangent(z, u, v):
    h = u[..., :, :1] @ v[..., :, :1].transpose(-2, -1)
    return z - (z*h).sum((-2, -1), keepdim=True)*h


def polar(z):
    u, s, v = spectrum(z)
    # Zero singular directions need not be filled; this remains an exact LMO.
    active = (s > 1e-13*s[..., :1]).to(s.dtype)
    return (u*active.unsqueeze(-2)) @ v.transpose(-2, -1)


def cone_batch(z, u, v, counts):
    out = z.clone()
    for q in counts.unique().tolist():
        mask = counts == q
        uq, vq, zz = u[mask, :, :q], v[mask, :, :q], z[mask]
        a = uq.transpose(-2, -1) @ zz @ vq
        a = (a+a.transpose(-2, -1))*.5
        values, vectors = torch.linalg.eigh(a)
        values = values.clamp(max=0)
        values[:, -1] = 0
        b = (vectors*values.unsqueeze(-2)) @ vectors.transpose(-2, -1)
        out[mask] = zz + uq @ (b-a) @ vq.transpose(-2, -1)
    return out


def loss(x, a):
    return .5*((x-a)**2).sum((-2, -1))


def square_residual(x, gradient):
    """Actual-edge limiting residual; no near-corner snapping tolerance."""
    w, g = x.diagonal(dim1=-2, dim2=-1), gradient.diagonal(dim1=-2, dim2=-1)
    corner = w[:, 0].abs() == w[:, 1].abs()
    axis = w.abs().argmax(-1)
    edge = g.gather(-1, (1-axis)[:, None]).squeeze(-1).abs()
    quadrant = (w.sign()*g).clamp(min=0).norm(dim=-1)
    at_corner = torch.minimum(quadrant, g.abs().min(-1).values)
    return torch.where(corner, at_corner, edge)


def problems(n, family, seeds):
    xs, targets = [], []
    for seed in range(seeds):
        rng = torch.Generator().manual_seed(81100+seed)
        if n == 2:
            target = {'edge': [2., 0.], 'corner': [2., 2.],
                      'inside': [.7, .2], 'stationary_corner': [.8, 1.]}[family]
            start = [1., 1.] if family == 'stationary_corner' else [-1., -1.]
            xs.append(torch.diag(torch.tensor(start)))
            targets.append(torch.diag(torch.tensor(target)))
        else:
            u = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
            v = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
            ui = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
            vi = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
            s = torch.linspace(.7, .1, n)
            if family == 'edge':
                s[0] = 1.4
            elif family in ('corner', 'capacity6'):
                s[:(2 if family == 'corner' else 6)] = 1.3
            start = torch.linspace(.65, .1, n)
            start[:2] = 1.
            xs.append((ui*start)@vi.T)
            targets.append((u*s)@v.T)
    return torch.stack(xs), torch.stack(targets)


def noise_sequence(n, steps, seeds):
    values = []
    for seed in range(seeds):
        rng = torch.Generator().manual_seed(130100+seed)
        if n == 2:
            value = torch.diag_embed(torch.randn(steps, n, generator=rng))/math.sqrt(n)
        else:
            value = torch.randn(steps, n, n, generator=rng)/n
        values.append(value)
    return torch.stack(values, dim=1)


def run_trajectories(seeds, square_horizons, matrix_steps, emit):
    rows = []
    for n in (2, 16, 32):
        families = ('edge', 'corner', 'inside', 'stationary_corner' if n == 2 else 'capacity6')
        horizons = square_horizons if n == 2 else [matrix_steps]
        for steps in horizons:
            noise = noise_sequence(n, steps, seeds)
            for family in families:
                x0, a = problems(n, family, seeds)
                fstar = loss(project(a), a)
                for noise_rms in (0., .05, .2):
                    for method in METHODS:
                        delta = .1
                        alpha = min(delta/(4*math.sqrt(n)), .5, steps**(-.75))
                        beta = 1-steps**(-.5)
                        x, m = x0.clone(), torch.zeros_like(x0)
                        near_count = torch.zeros(seeds, dtype=torch.int64)
                        uphill = torch.zeros_like(near_count)
                        max_radius = torch.zeros(seeds)
                        max_safe_ratio = torch.zeros(seeds)
                        max_defect = torch.zeros(seeds)
                        work = torch.zeros(seeds)
                        tracking = torch.zeros(seeds)
                        best_residual2 = torch.full((seeds,), float('inf'))
                        trace = []
                        for t in range(steps):
                            g = x-a
                            if n == 2:
                                best_residual2 = torch.minimum(best_residual2, square_residual(x, g)**2)
                            m = beta*m+(1-beta)*(g+noise_rms*noise[t])
                            u, s, v = spectrum(x)
                            near = s[:, 0]-s[:, 1] < delta
                            d0 = tangent(polar(tangent(-m, u, v)), u, v)
                            if method == 'pgd':
                                d = -m
                                updated = project(x+alpha*d)
                            else:
                                d = d0*(.5 if method == 'hybrid_half' else 1.)
                                if method in ('hybrid_exact', 'hybrid_half'):
                                    d[near] = -m[near]
                                else:
                                    z = -m/m.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
                                    tolerance = 1e-12 if method == 'true_cone_radial' else delta
                                    counts = ((s[:, :1]-s) <= tolerance).sum(-1)
                                    d[near] = (cone_batch(z, u, v, counts) if method != 'ambient_radial' else z)[near]
                                updated = radial_batch(x+alpha*d)
                                if method in ('hybrid_exact', 'hybrid_half') and near.any():
                                    updated[near] = project((x+alpha*d)[near])
                                safe_ratio = alpha*d.norm(dim=(-2, -1))/delta
                                max_safe_ratio = torch.maximum(max_safe_ratio, torch.where(~near, safe_ratio, 0.))
                            e = updated-x
                            uphill += loss(updated, a) > loss(x, a)+1e-12
                            near_count += near
                            max_radius = torch.maximum(max_radius, (torch.linalg.svdvals(updated)[:, 0]-1).abs())
                            max_defect = torch.maximum(max_defect, (e-alpha*d).norm(dim=(-2, -1)))
                            work += (g*e).sum((-2, -1))
                            tracking += ((m-g)**2).sum((-2, -1))
                            x = updated
                            if (t+1) % 32 == 0 or t == steps-1:
                                trace.append(dict(step=t+1, mean_excess=float((loss(x, a)-fstar).mean()),
                                    mean_radius_error=float((torch.linalg.svdvals(x)[:, 0]-1).abs().mean())))
                        r = square_residual(x, x-a) if n == 2 else (x-project(x-.01*(x-a))).norm(dim=(-2, -1))/.01
                        if n == 2:
                            best_residual2 = torch.minimum(best_residual2, r*r)
                        final = loss(x, a)-fstar
                        for seed in range(seeds):
                            rows.append(dict(n=n, family=family, horizon=steps, noise_rms=noise_rms,
                                method=method, seed=seed, alpha=alpha, beta=beta,
                                final_excess=float(final[seed]), final_residual=float(r[seed]),
                                residual_kind='limiting_normal' if n == 2 else 'projection_mapping_alpha_0.01',
                                best_residual_squared=float(best_residual2[seed]) if n == 2 else None,
                                near_rate=float(near_count[seed])/steps, uphill_steps=int(uphill[seed]),
                                max_radius_error=float(max_radius[seed]), max_safe_perturbation_over_delta=float(max_safe_ratio[seed]),
                                max_return_defect=float(max_defect[seed]), total_true_gradient_work=float(work[seed]),
                                mean_tracking_error_squared=float(tracking[seed])/steps))
                        emit('trajectory', dict(n=n, family=family, horizon=steps, noise=noise_rms,
                            method=method, mean_final_excess=float(final.mean()), trace=trace))
                print(f'trajectories n={n} T={steps} family={family}', flush=True)
    return rows


def run_matched(seeds):
    rows, failures = [], []
    for n in (16, 32):
        for q in (1, 2, 4, 6):
            for exact in ((False,) if q == 1 else (False, True)):
                for seed in range(seeds):
                    rng = torch.Generator().manual_seed(9400+seed+100*n+1000*q)
                    u = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
                    v = torch.linalg.qr(torch.randn(n, n, generator=rng))[0]
                    s = torch.linspace(.7, .1, n)
                    s[:q] = 1. if exact else torch.linspace(1., 1.-1e-4, q)
                    if q == 1:
                        s[1] = 1.-3e-4
                    x = (u*s)@v.T
                    g = torch.randn(n, n, generator=rng)
                    g /= g.norm()
                    orth = torch.randn(n, n, generator=rng)
                    orth -= (orth*g).sum()*g
                    orth /= orth.norm()
                    # Independent local inputs with the same anisotropic covariance.
                    scales = torch.linspace(.2, 2., n)
                    h = torch.randn(128, n, generator=rng)*scales
                    h2 = torch.randn(128, n, generator=rng)*scales
                    a = x-g
                    for cosine in (1., .5):
                        m = cosine*g+math.sqrt(1-cosine*cosine)*orth
                        d0 = tangent(polar(tangent(-m, u, v)), u, v)
                        for alpha in (.005, .015):
                            context = dict(n=n, q=q, true_multiplicity=q if exact else 1,
                                exact=exact, seed=seed, cosine=cosine, alpha=alpha)
                            try:
                                points, geometry, _ = candidates(x, m, d0, u[:, :q], s[:q], v[:, :q], 1., alpha, h)
                            except (RuntimeError, ValueError) as exc:
                                failures.append(dict(**context, error=str(exc)))
                                continue
                            by = {r['method']:r for r in geometry}
                            for name, point in points.items():
                                e = point-x
                                rows.append(dict(**context, method=name, loss_change=float(loss(point,a)-loss(x,a)),
                                    difference_vs_half=float(loss(point,a)-loss(points['half_d0'],a)),
                                    true_gradient_work=float((g*e).sum()), curvature_cost=float(.5*(e*e).sum()),
                                    independent_output_elr=float((h2@e.T).norm()/(h2@x.T).norm()),
                                    **{k:val for k,val in by[name].items() if k!='method'}))
                print(f'matched n={n} q={q} exact={exact}', flush=True)
    return rows, failures


def summarize(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, key), count=len(group), **{f:sum(r[f] for r in group)/len(group) for f in fields})
            for key, group in groups.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--square-horizons', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--matrix-steps', type=int, default=512)
    parser.add_argument('--part', choices=('all', 'trajectories', 'matched'), default='all')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = [Path(__file__), Path(__file__).with_name('p2gd_six_geometry.py'),
               Path(__file__).with_name('p2gd_budget_geometry.py'), Path(__file__).with_name('p2gd_elr_geometry.py')]
    config = dict(seeds=args.seeds, square_horizons=args.square_horizons, matrix_steps=args.matrix_steps,
        dtype='float64', device='cpu', torch=torch.__version__, part=args.part,
        trajectory_momentum='current-sample EMA, beta=1-T^(-1/2)',
        trajectory_alpha='min(0.1/(4*sqrt(n)), 0.5, T^(-3/4))',
        noise='centered Gaussian; expected squared Frobenius norm = noise_rms^2; common random numbers',
        near_threshold=.1, cluster_threshold=.1, true_multiplicity_tolerance=1e-12, radius=1.,
        stationarity='exact actual-edge limiting residual only for diagonal square; matrix mapping is a diagnostic',
        matched_note='Numerical cluster threshold 2e-4; q supplied from constructed spectrum; full capacity, no top4 cap',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    with (args.output/'progress.jsonl').open('w') as stream:
        def emit(kind, row):
            stream.write(json.dumps(dict(kind=kind, **row))+'\n')
            stream.flush()
        if args.part in ('all', 'trajectories'):
            rows = run_trajectories(args.seeds, args.square_horizons, args.matrix_steps, emit)
            (args.output/'trajectories.json').write_text(json.dumps(rows, indent=2)+'\n')
            summary = summarize(rows, ('n','family','horizon','noise_rms','method'),
                ('final_excess','final_residual','near_rate','uphill_steps','max_radius_error',
                 'max_safe_perturbation_over_delta','mean_tracking_error_squared'))
            (args.output/'trajectory_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if args.part in ('all', 'matched'):
            rows, failures = run_matched(args.seeds)
            (args.output/'matched.json').write_text(json.dumps(dict(rows=rows, failures=failures), indent=2)+'\n')
    (args.output/'COMPLETED.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
        part=args.part), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), seconds=time.perf_counter()-started)))


if __name__ == '__main__':
    main()
