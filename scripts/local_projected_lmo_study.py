"""Small spectral-sphere projected-LMO study with a separate model safeguard.

CPU FP64, exact spectra/oracles, EMA momentum, predetermined steps. Reuses
the production direction helper. The safeguarded variant is NOT the original
algorithm and may perform two projections per step.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from local_cone_lmo_feasibility import (
    ROOT, cone, polar, spectral_cone_lmo_direction, sphere_project, tangent,
)

METHODS = ("pgd", "mcsd_project", "mcsd_tp_project", "hybrid_pgd_project",
           "hybrid_p2gd_project", "hybrid_lmo_project", "hybrid_mcsd_lmo_project",
           "hybrid_cluster_lmo_project", "guarded_hybrid_lmo_project")
ALPHA = .02
BETA = .9
KAPPA = 1.5
GAP = .02


def loss(x, a):
    return float(.5*((x-a)**2).sum())


def candidate_direction(x, m, method):
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    gap = float(1-s[1]/s[0])
    q = int(((s[0]-s)/s[0] <= 1e-10).sum())
    near = method not in ("pgd", "mcsd_project", "mcsd_tp_project") and gap <= GAP
    if method == "pgd":
        return -m, False, q
    if not near:
        projected = tangent(-m, u[:, 0], vh.T[:, 0])
        if method in ("mcsd_project", "hybrid_mcsd_lmo_project"):
            return polar(projected), False, q
        return tangent(polar(projected), u[:, 0], vh.T[:, 0]), False, q
    if method == "hybrid_pgd_project":
        return -m, True, q
    if method == "hybrid_cluster_lmo_project":
        q = int(((s[0]-s)/s[0] <= .05).sum())
    uq, vq = u[:, :q], vh.T[:, :q]
    if method == "hybrid_p2gd_project":
        return cone(-m, uq, vq), True, q
    return spectral_cone_lmo_direction(-m, uq, vq, polar), True, q


def guarded_return(x, m, d, alpha, lipschitz=1., kappa=KAPPA):
    candidate = sphere_project(x+alpha*d)
    baseline = sphere_project(x-alpha*m)
    dc, db = candidate-x, baseline-x
    model_c = (m*dc).sum()+.5*lipschitz*(dc*dc).sum()
    model_b = (m*db).sum()+.5*lipschitz*(db*db).sum()
    accepted = bool(model_c <= model_b and torch.linalg.vector_norm(dc) <= kappa*torch.linalg.vector_norm(db))
    return (candidate if accepted else baseline), accepted, dict(
        candidate_model=float(model_c), baseline_model=float(model_b),
        baseline_displacement=float(torch.linalg.vector_norm(db)),
        candidate_displacement=float(torch.linalg.vector_norm(dc)))


def projection_counterexample():
    rows = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        x = torch.diag(torch.tensor([1., 1-eps], dtype=torch.float64))
        z = torch.tensor([[4., 1.], [1., 0.]], dtype=torch.float64)
        d, near, _ = candidate_direction(x, -z, "hybrid_lmo_project")
        alpha = eps
        returned = sphere_project(x+alpha*d)
        safeguarded, accepted, _ = guarded_return(x, -z, d, alpha, lipschitz=0.)
        rows.append(dict(eps=eps, step=alpha, near=near, direction=d.tolist(),
            derivative=float((-z*d).sum()),
            actual_change_over_step=float((-z*(returned-x)).sum()/alpha),
            return_error_over_step=float(torch.linalg.vector_norm(returned-x-alpha*d)/alpha),
            guard_accepted=accepted,
            guard_change_over_step=float((-z*(safeguarded-x)).sum()/alpha)))
    return rows


def make_problem(case):
    gen = torch.Generator().manual_seed(73100+case)
    u = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    family = ("inside", "outside", "plateau", "near_plateau")[case % 4]
    if family == "inside":
        a = (u*torch.tensor([.7,.5,.3,.1], dtype=torch.float64)) @ v.T
    elif family == "outside":
        a = (u*torch.tensor([1.4,.8,.4,.2], dtype=torch.float64)) @ v.T
    elif family == "plateau":
        a = (u*torch.tensor([1.3,1.3,.5,.2], dtype=torch.float64)) @ v.T
    else:
        a = (u*torch.tensor([1.3,1.2999,.5,.2], dtype=torch.float64)) @ v.T
    # Distinct starting orientation, so this is not only singular-value fitting.
    ui = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    vi = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    x = (ui*torch.tensor([1.,1.,.65,.3], dtype=torch.float64)) @ vi.T
    return family, x, a


def diagnostics(x, a, alpha):
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    q = int(((s[0]-s)/s[0] <= 1e-10).sum())
    g = x-a
    return dict(loss=loss(x,a), radius_error=abs(float(s[0])-1),
        cone_stationarity=float(torch.linalg.vector_norm(cone(-g,u[:,:q],vh.T[:,:q]))),
        projection_mapping=float(torch.linalg.vector_norm(x-sphere_project(x-alpha*g))/alpha),
        gap=float(1-s[1]/s[0]), top_multiplicity=q)


def run_study(steps, cases):
    rows = []
    for case in range(cases):
        family, x0, a = make_problem(case)
        fstar = loss(sphere_project(a), a)
        gen = torch.Generator().manual_seed(93200+case)
        noise = torch.randn(steps, 4, 4, generator=gen, dtype=torch.float64)/4
        for stochastic in (False, True):
            for schedule in ("constant", "diminishing"):
                for method in METHODS:
                    x, m = x0.clone(), torch.zeros_like(x0)
                    trace = [dict(step=0, **diagnostics(x,a,ALPHA))]
                    near_count = accepted_count = uphill_count = 0
                    projections = 0
                    for t in range(steps):
                        alpha = ALPHA if schedule == "constant" else ALPHA*(1+t/200)**(-.6)
                        sample = x-a
                        if stochastic:
                            sample = sample+.1*noise[t]/(t+1)**.6
                        m = BETA*m+(1-BETA)*sample
                        d, near, _ = candidate_direction(x, m, method)
                        near_count += near
                        if method.startswith("guarded"):
                            updated, accepted, _ = guarded_return(x,m,d,alpha)
                            accepted_count += accepted
                            projections += 2
                        else:
                            updated = sphere_project(x+alpha*d)
                            projections += 1
                        uphill_count += loss(updated,a) > loss(x,a)+1e-12
                        x = updated
                        if (t+1) % 50 == 0 or t+1 == steps:
                            trace.append(dict(step=t+1, **diagnostics(x,a,alpha)))
                    rows.append(dict(case=case, family=family, stochastic=stochastic,
                        schedule=schedule, method=method, fstar=fstar,
                        final_excess=trace[-1]["loss"]-fstar,
                        final_mapping=trace[-1]["projection_mapping"],
                        final_cone_stationarity=trace[-1]["cone_stationarity"],
                        near_rate=near_count/steps,
                        acceptance_rate=accepted_count/steps if method.startswith("guarded") else None,
                        projections=projections, uphill_steps=uphill_count, trace=trace))
    summaries = []
    for stochastic in (False, True):
        for schedule in ("constant", "diminishing"):
            for method in METHODS:
                group = [r for r in rows if (r['stochastic'],r['schedule'],r['method']) == (stochastic,schedule,method)]
                summaries.append(dict(stochastic=stochastic, schedule=schedule, method=method,
                    mean_excess=sum(r['final_excess'] for r in group)/len(group),
                    max_excess=max(r['final_excess'] for r in group),
                    mean_mapping=sum(r['final_mapping'] for r in group)/len(group),
                    max_mapping=max(r['final_mapping'] for r in group),
                    mean_near_rate=sum(r['near_rate'] for r in group)/len(group),
                    mean_acceptance=sum(r['acceptance_rate'] for r in group)/len(group) if method.startswith('guarded') else None,
                    max_radius_error=max(t['radius_error'] for r in group for t in r['trace'])))
    return dict(rows=rows, summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--cases', type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.cases < 1:
        parser.error('positive steps and cases required')
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    output = dict(config=dict(steps=args.steps, cases=args.cases, shape=[4,4],
        radius=1., alpha=ALPHA, beta=BETA, gap_threshold=GAP, cluster_threshold=.05,
        kappa=KAPPA, lipschitz=1., dtype='float64', device='cpu',
        torch=torch.__version__, near_multiplier=1.,
        stochastic_noise='Gaussian Frobenius RMS .1/(t+1)^.6; variance summable',
        momentum='EMA only, not Nesterov', exact_top_tolerance=1e-10,
        note='No line search. Near-cluster arm is explicitly heuristic. Guard is an additional algorithm.'),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/local_projected_lmo_study.py', 'scripts/local_cone_lmo_feasibility.py',
            'Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cone_lmo.py')},
        counterexample=projection_counterexample(), experiment=run_study(args.steps,args.cases))
    output['elapsed_seconds'] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(counterexample=output['counterexample'],
                         summary=output['experiment']['summary'], output=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
