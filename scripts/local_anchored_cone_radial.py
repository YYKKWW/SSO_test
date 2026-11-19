"""FP64 checks for an anchored cone face and radial return; research only."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from audit_corner_momentum import synthetic_states
from local_cone_lmo_feasibility import cone
from local_step_aware_hybrid import linear_direction, sym, tangent


def unit(x):
    n = x.norm()
    return x / n if float(n) else torch.zeros_like(x)


def anchored_project(z, u, v):
    """Frobenius projection: H(D)<=0 and H(D)e1=0; first pair is the anchor."""
    a = sym(u.T @ z @ v)
    b = torch.zeros_like(a)
    if a.shape[0] > 1:
        vals, vecs = torch.linalg.eigh(a[1:, 1:])
        b[1:, 1:] = (vecs * vals.clamp(max=0)) @ vecs.T
    return z + u @ (b-a) @ v.T


def radial(y, radius=1.):
    return y * (radius / torch.linalg.svdvals(y)[0])


def algebra_checks():
    rng = torch.Generator().manual_seed(20261118)
    worst = defaultdict(float)
    cases = 0
    for shape in ((6, 4), (4, 6), (8, 8)):
        r = min(shape)
        for radius in (.1, 1., 10.):
            for q in sorted({1, 2, r}):
                for repeat in range(12):
                    u = torch.linalg.qr(torch.randn(shape[0], r, generator=rng))[0]
                    v = torch.linalg.qr(torch.randn(shape[1], r, generator=rng))[0]
                    s = radius * torch.linspace(.65, .2, r)
                    s[:q] = radius * (1-torch.linspace(0, .015, q))
                    if repeat % 2:
                        s[:q] = radius
                    x = (u*s) @ v.T
                    uq, vq = u[:, :q], v[:, :q]
                    z = torch.randn(shape, generator=rng)
                    d = anchored_project(z, uq, vq)
                    hd = sym(uq.T @ d @ vq)
                    identity = abs(float((z*d).sum()-d.square().sum()))
                    assert identity < 1e-10
                    assert float(torch.linalg.eigvalsh(hd)[-1]) < 1e-11
                    assert float(hd[:, 0].norm()) < 1e-11
                    assert float(d.norm()) <= float(z.norm())+1e-11
                    torch.testing.assert_close(anchored_project(d, uq, vq), d)
                    torch.testing.assert_close(anchored_project(2.3*z, uq, vq), 2.3*d)
                    torch.testing.assert_close(tangent(d, uq, vq), tangent(z, uq, vq))
                    z2 = torch.randn(shape, generator=rng)
                    d2 = anchored_project(z2, uq, vq)
                    assert float((d-d2).norm()) <= float((z-z2).norm())+1e-11
                    torch.testing.assert_close(anchored_project(torch.zeros_like(z), uq, vq), torch.zeros_like(z))
                    # Rotations that keep the anchor fixed preserve the cone face.
                    rotation = torch.eye(q)
                    if q > 1:
                        rotation[1:, 1:] = torch.linalg.qr(torch.randn(q-1, q-1, generator=rng))[0]
                    torch.testing.assert_close(anchored_project(z, uq@rotation, vq@rotation), d)
                    d = unit(d)
                    h, gamma = .02*radius, float(radius-s[q]) if q < r else radius
                    alpha = .2*h
                    model = torch.diag(s[:q])+alpha*sym(uq.T@d@vq)
                    model_error = abs(float(torch.linalg.eigvalsh(model)[-1])-radius)
                    assert model_error < 1e-10
                    y = x+alpha*d
                    excess = float(torch.linalg.svdvals(y)[0])-radius
                    bound = 2*alpha**2*float(torch.linalg.svdvals(d)[0])**2/gamma
                    assert excess >= -1e-11 and excess <= bound+1e-11
                    return_bound = r**.5*bound
                    residual = float((radial(y, radius)-y).norm())
                    assert residual <= return_bound+1e-10
                    worst['return_bound_ratio'] = max(worst['return_bound_ratio'], residual/max(return_bound, 1e-30))
                    g = torch.randn(shape, generator=rng)
                    m = g + (0., .1, 10.)[repeat % 3] * torch.randn(shape, generator=rng)
                    p = anchored_project(-unit(m), uq, vq)
                    rproxy = float(tangent(g, uq, vq).norm())
                    gbar = max(1., float(g.norm()))
                    rhs = -rproxy**2/gbar + 2*float((m-g).norm())
                    assert float((g*p).sum()) <= rhs+1e-10
                    worst['projection_identity_abs_error'] = max(worst['projection_identity_abs_error'], identity)
                    worst['finite_model_abs_error'] = max(worst['finite_model_abs_error'], model_error)
                    cases += 1
    return dict(cases=cases, **worst)


def near_plateau_counterexamples():
    rows = []
    g = torch.diag(torch.tensor([2., 1.]))
    eye = torch.eye(2)
    for eps in (1e-2, 1e-3, 1e-4):
        x = torch.diag(torch.tensor([1., 1-eps]))
        alpha = eps**2
        for name, d in (("unanchored_cluster_cone", cone(-unit(g), eye, eye)),
                        ("anchored_cluster_cone", anchored_project(-unit(g), eye, eye))):
            delta = radial(x+alpha*d)-x
            change = float((g*delta).sum()+.5*delta.square().sum())
            if name.startswith('unanchored'):
                assert change > 0
            else:
                assert change < 0
            rows.append(dict(eps=eps, alpha=alpha, method=name, delta_f=change,
                delta_f_over_alpha=change/alpha,
                return_error_over_alpha=float((delta-alpha*d).norm())/alpha))
    return rows


def nonuniform_true_tangent():
    rows = []
    d = torch.tensor([[0., 1.], [1., 0.]])
    for eps in (.1, .03, .01, .003, .001):
        x = torch.diag(torch.tensor([1., 1-eps]))
        alpha = .1*eps
        y = x+alpha*d
        rows.append(dict(gap=eps, alpha=alpha,
                         return_error_over_alpha_squared=float((radial(y)-y).norm())/alpha**2))
    assert rows[-1]['return_error_over_alpha_squared'] > 50*rows[0]['return_error_over_alpha_squared']
    return rows


def paired_toy_checks():
    rows = []
    for index, (meta, x, m, g, alpha) in enumerate(synthetic_states()):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        x = x/s[0]
        s = s/s[0]
        v = vh.T
        q = int(((1-s) <= 2e-4).sum())
        d0 = linear_direction(m, u[:, :1], v[:, :1])[0]
        d = anchored_project(-unit(m), u[:, :q], v[:, :q])
        if float(d.norm()) > 1e-12:
            d = unit(d)*(.5*d0.norm())
        else:
            d = torch.zeros_like(d)
        delta = radial(x+alpha*d)-x
        change = float((g*delta).sum()+.5*delta.square().sum())
        rows.append(dict(id=str(index), **meta, alpha=alpha, q=q,
            true_q=int(((1-s) <= 1e-10).sum()), delta_f=change,
            trial_length=float((alpha*d).norm()), returned_length=float(delta.norm())))
    summary = []
    for alignment in (1., .5):
        selected = [v for v in rows if v['true_q']>1 and v['alignment']==alignment]
        summary.append(dict(alignment=alignment, cases=len(selected),
            decreased=sum(v['delta_f'] < -1e-12 for v in selected),
            unchanged=sum(abs(v['delta_f']) <= 1e-12 for v in selected),
            increased=sum(v['delta_f'] > 1e-12 for v in selected),
            zero_trial=sum(v['trial_length'] <= 1e-12 for v in selected),
            mean_delta_f=sum(v['delta_f'] for v in selected)/len(selected)))
    return dict(summary=summary, rows=rows,
        note="One arbitrary exact leading pair anchors each face. Half-D0 trial matching, not a convergence test or raw stochastic P2GD.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    report = dict(algebra=algebra_checks(), near_plateau=near_plateau_counterexamples(),
                  nonuniform_tangent=nonuniform_true_tangent(), toy=paired_toy_checks())
    eye = torch.eye(2)
    g = torch.diag(torch.tensor([1., 0.]))
    report['b_stationarity_obstruction'] = dict(
        anchored_norm=float(anchored_project(-g, eye, eye).norm()),
        full_cone_norm=float(cone(-g, eye, eye).norm()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='toy'}, indent=2))
    print(json.dumps(report['toy']['summary'], indent=2))


if __name__ == '__main__':
    main()
