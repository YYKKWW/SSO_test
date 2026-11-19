"""CPU-only spectral-sphere geometry checks and paired stochastic toy runs.

Uses the production selected-face LMO; exact spectra isolate geometry from
estimation. No line search, remote jobs, production changes, or GPU claims.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers.spectral_cone_lmo import spectral_cone_lmo_direction
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz


def sym(a):
    return (a + a.T) / 2


def unit(a):
    return a / torch.linalg.vector_norm(a).clamp_min(1e-30)


def polar(a):
    if not torch.count_nonzero(a):
        return torch.zeros_like(a)
    u, _, vh = torch.linalg.svd(a, full_matrices=False)
    return u @ vh


def l1_lmo(a):
    out = torch.zeros_like(a).reshape(-1)
    index = a.abs().argmax()
    out[index] = a.reshape(-1)[index].sign()
    return out.reshape_as(a)


def cone(z, u, v):
    a = sym(u.T @ z @ v)
    vals, vecs = torch.linalg.eigh(a)
    vals = vals.clamp(max=0)
    vals[-1] = 0
    return z + u @ ((vecs * vals) @ vecs.T - a) @ v.T


def tangent(z, u, v):
    return z - (u @ z @ v) * torch.outer(u, v)


def sphere_project(y):
    u, s, vh = torch.linalg.svd(y, full_matrices=False)
    clipped = s.clamp(max=1)
    clipped[0] = 1
    return (u * clipped) @ vh


def radial(y):
    return y / torch.linalg.svdvals(y)[0]


def violation(d, u, v):
    return float(torch.linalg.eigvalsh(sym(u.T @ d @ v))[-1])


def mean(values):
    return sum(values) / len(values)


def algebra(seed):
    gen = torch.Generator().manual_seed(seed)
    oracles = {
        "spectral": (polar, lambda x: torch.linalg.svdvals(x).sum(),
                     lambda x: torch.linalg.matrix_norm(x, ord=2)),
        "frobenius": (unit, torch.linalg.vector_norm, torch.linalg.vector_norm),
        "entry_l1": (l1_lmo, lambda x: x.abs().max(), lambda x: x.abs().sum()),
    }
    output = []
    for m, n in ((8, 6), (6, 8), (16, 12)):
        for q in (1, 2, 4, min(m, n)):
            for repetition in range(12):
                u = torch.linalg.qr(torch.randn(m, q, generator=gen, dtype=torch.float64))[0]
                v = torch.linalg.qr(torch.randn(n, q, generator=gen, dtype=torch.float64))[0]
                z = torch.randn(m, n, generator=gen, dtype=torch.float64)
                projected = cone(z, u, v)
                for name, (oracle, dual, norm) in oracles.items():
                    d = spectral_cone_lmo_direction(z, u, v, oracle)
                    # Construct the same selected face to check the exact identity.
                    av, aq = torch.linalg.eigh(sym(u.T @ z @ v))
                    bv = av.clamp(max=0)
                    bv[-1] = 0
                    b = (aq * bv) @ aq.T
                    p = (aq * (bv < 0).double()) @ aq.T
                    s = oracle(projected)
                    c = p @ sym(u.T @ s @ v) @ p
                    cv, cq = torch.linalg.eigh(sym(c))
                    positive = (cq * cv.clamp(min=0)) @ cq.T
                    lhs = (z * d).sum()
                    rhs = (projected * s).sum() - (b * positive).sum()
                    row = dict(shape=[m, n], q=q, repetition=repetition, norm=name,
                               cone_error=abs(violation(d, u, v)),
                               identity_error=float(abs(lhs-rhs)),
                               descent_margin=float(lhs-dual(projected)),
                               output_norm=float(norm(d)),
                               fro_contraction=float(torch.linalg.vector_norm(d)-torch.linalg.vector_norm(s)),
                               normalized_cone_error=abs(violation(d / norm(d).clamp_min(1e-30), u, v)))
                    assert row["cone_error"] < 1e-10
                    assert row["identity_error"] < 1e-10
                    assert row["descent_margin"] > -1e-10
                    assert row["fro_contraction"] < 1e-10
                    output.append(row)
    summaries = {}
    for name in oracles:
        rows = [r for r in output if r["norm"] == name]
        summaries[name] = dict(samples=len(rows),
            max_cone_error=max(r["cone_error"] for r in rows),
            max_identity_error=max(r["identity_error"] for r in rows),
            min_descent_margin=min(r["descent_margin"] for r in rows),
            max_output_norm=max(r["output_norm"] for r in rows),
            norm_ball_violations=sum(r["output_norm"] > 1+1e-10 for r in rows))
    return dict(summary=summaries, rows=output)


def counterexamples():
    eye = torch.eye(3, dtype=torch.float64)
    z = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64))
    # True multiplicity 3, but only two vectors are supplied to the algorithm.
    d = spectral_cone_lmo_direction(z, eye[:, :2], eye[:, :2], polar)
    omitted = dict(direction=d.tolist(), model_cone=violation(d, eye[:, :2], eye[:, :2]),
                   true_cone=violation(d, eye, eye))
    # Simple top value: treating the second as a repeated value makes H11<0.
    d2 = spectral_cone_lmo_direction(z[:2, :2], eye[:2, :2], eye[:2, :2], polar)
    near = dict(singular_values=[1., .9999], direction=d2.tolist(),
                model_cone=violation(d2, eye[:2, :2], eye[:2, :2]),
                true_cone=float(d2[0, 0]))
    uniform = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        x = torch.diag(torch.tensor([1., 1-eps], dtype=torch.float64))
        dt = torch.diag(torch.tensor([0., 1.], dtype=torch.float64))
        alpha = 2*eps
        y = x + alpha*dt
        uniform.append(dict(eps=eps, alpha=alpha,
            radial_error_over_step=float(torch.linalg.vector_norm(radial(y)-y)/alpha),
            projection_error_over_step=float(torch.linalg.vector_norm(sphere_project(y)-y)/alpha),
            linear_objective_change_over_step=float((torch.diag(torch.tensor([-2., -1.], dtype=torch.float64))*(radial(y)-x)).sum()/alpha),
            predicted_linear_change_over_step=-1.))
    # At a stationary point, unbiased noise need not yield an unbiased unit LMO.
    return dict(omitted_top_direction=omitted, near_cluster_not_true_cone=near,
                nonuniform_return=uniform,
                normalized_noise=dict(values=[1., -9.], probabilities=[.9, .1],
                                      mean=0., mean_sign=.8))


def local_return(seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.diag(torch.tensor([1., 1., .7, .4], dtype=torch.float64))
    u = torch.eye(4, dtype=torch.float64)[:, :2]
    z = torch.randn(4, 4, generator=gen, dtype=torch.float64)
    d = unit(spectral_cone_lmo_direction(z, u, u, polar))
    rows = []
    for alpha in (1e-2, 1e-3, 1e-4, 1e-5):
        y = x + alpha*d
        rows.append(dict(alpha=alpha,
            radial_error_over_step=float(torch.linalg.vector_norm(radial(y)-y)/alpha),
            project_error_over_step=float(torch.linalg.vector_norm(sphere_project(y)-y)/alpha)))
    return rows


METHODS = ("mcsd_tp", "hybrid_pgd_raw", "hybrid_pgd_unit", "hybrid_p2gd_project",
           "hybrid_p2gd_radial", "hybrid_lmo_project", "hybrid_lmo_radial", "hybrid_ambient_radial")


def trajectories(seed, steps, cases, *, plateau=False):
    rows = []
    for case in range(cases):
        gen = torch.Generator().manual_seed(seed+case)
        m, n = 16, 12
        u = torch.linalg.qr(torch.randn(m, n, generator=gen, dtype=torch.float64))[0]
        v = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))[0]
        s = torch.linspace(.8, .2, n, dtype=torch.float64)
        s[:(2 if case % 2 == 0 else 4)] = 1
        x0 = (u*s) @ v.T
        a = torch.randn(m, n, generator=gen, dtype=torch.float64)
        a = a / torch.linalg.svdvals(a)[0] * (.7 if case % 3 == 0 else 1.4)
        if plateau:
            target_s = s*.65
            target_s[:(2 if case % 2 == 0 else 4)] = 1.3
            a = (u*target_s) @ v.T
        fstar = float(.5*((sphere_project(a)-a)**2).sum())
        noise = torch.randn(steps, m, n, generator=gen, dtype=torch.float64) / math.sqrt(m*n)
        for noise_scale in (0., .1):
            for rule in ("literal", "matched_fro"):
                for method in METHODS:
                    x, buf = x0.clone(), torch.zeros_like(x0)
                    near_count = violations = uphill = 0
                    min_stationarity = float("inf")
                    for t in range(steps):
                        grad = x-a
                        sample = grad+noise_scale*noise[t]
                        buf = .9*buf+.1*sample
                        momentum = .1*sample+.9*buf
                        ux, sx, vxh = torch.linalg.svd(x, full_matrices=False)
                        gap = float(1-sx[1]/sx[0])
                        true_q = int(((sx[0]-sx)/sx[0] <= 1e-10).sum())
                        stationarity = float(torch.linalg.vector_norm(cone(-grad, ux[:, :true_q], vxh.T[:, :true_q])))
                        min_stationarity = min(min_stationarity, stationarity)
                        near = method != "mcsd_tp" and gap <= 5e-5
                        near_count += near
                        use_project = False
                        if not near:
                            normal_u, normal_v = ux[:, 0], vxh.T[:, 0]
                            d = tangent(polar(tangent(-momentum, normal_u, normal_v)), normal_u, normal_v)
                        elif method in ("hybrid_pgd_raw", "hybrid_pgd_unit", "hybrid_ambient_radial"):
                            d = -momentum if method == "hybrid_pgd_raw" else -unit(momentum)
                            use_project = method != "hybrid_ambient_radial"
                        else:
                            q = min(4, int(((sx[0]-sx)/sx[0] <= 2e-3).sum()))
                            uq, vq = ux[:, :q], vxh.T[:, :q]
                            if "lmo" in method:
                                d = spectral_cone_lmo_direction(-unit(momentum), uq, vq, polar)
                            else:
                                d = cone(-unit(momentum), uq, vq)
                            use_project = method.endswith("project")
                        violations += abs(violation(d, ux[:, :true_q], vxh.T[:, :true_q])) > 1e-8
                        if rule == "matched_fro":
                            d = unit(d)
                        # Shared preset cosine schedule, no data-dependent line search.
                        alpha = .015*(.1+.9*(1+math.cos(math.pi*t/(steps-1)))/2)
                        y = x+alpha*d
                        returned = sphere_project(y) if use_project else radial(y)
                        uphill += float(.5*((returned-a)**2).sum()) > float(.5*((x-a)**2).sum())+1e-12
                        x = returned
                    ux, sx, vxh = torch.linalg.svd(x, full_matrices=False)
                    q = int(((sx[0]-sx)/sx[0] <= 1e-10).sum())
                    rows.append(dict(case=case, target="plateau_attractor" if plateau else ("inside" if case % 3 == 0 else "outside"),
                        noise=noise_scale, rule=rule, method=method,
                        objective_excess=float(.5*((x-a)**2).sum())-fstar,
                        final_stationarity=float(torch.linalg.vector_norm(cone(a-x, ux[:, :q], vxh.T[:, :q]))),
                        min_stationarity=min_stationarity, near_rate=near_count/steps,
                        true_cone_violations=violations, uphill_steps=uphill,
                        radius_error=abs(float(sx[0])-1)))
    summaries = []
    for noise in (0., .1):
        for rule in ("literal", "matched_fro"):
            for method in METHODS:
                group = [r for r in rows if r["noise"] == noise and r["rule"] == rule and r["method"] == method]
                reference = [r for r in rows if r["noise"] == noise and r["rule"] == rule and r["method"] == "hybrid_pgd_raw"]
                summaries.append(dict(noise=noise, rule=rule, method=method, cases=len(group),
                    mean_excess=mean([r["objective_excess"] for r in group]),
                    mean_final_stationarity=mean([r["final_stationarity"] for r in group]),
                    mean_near_rate=mean([r["near_rate"] for r in group]),
                    wins_vs_raw_pgd=sum(r["objective_excess"] < b["objective_excess"]-1e-10 for r,b in zip(group,reference)),
                    max_radius_error=max(r["radius_error"] for r in group)))
    return dict(summary=summaries, rows=rows)


def width256_audit(seed):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for shape in ((256, 256), (768, 256), (256, 768)):
        m, n = shape
        r = min(m, n)
        u = torch.linalg.qr(torch.randn(m, r, generator=gen, dtype=torch.float64))[0]
        v = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=torch.float64))[0]
        for true_q in (2, 6):
            s = torch.linspace(.95, .1, r, dtype=torch.float64)
            s[:true_q] = 1
            x = (u*s) @ v.T
            z = torch.randn(m, n, generator=gen, dtype=torch.float64)
            d = unit(spectral_cone_lmo_direction(z, u[:, :true_q], v[:, :true_q], polar))
            for alpha in (.0015, .015):
                y = x+alpha*d
                exact = torch.linalg.svdvals(y)
                for k in (10, 20):
                    uh, sh, vh = warm_block8_ritz(y.float(), iterations=k, initial_v=v[:, :8].float())
                    estimate = float(1-sh[1]/sh[0])
                    truth = float(1-exact[1]/exact[0])
                    rows.append(dict(shape=list(shape), true_q=true_q, alpha=alpha, iterations=k,
                        gap_abs_error=abs(estimate-truth), exact_trial_gap=truth,
                        estimated_trial_gap=estimate,
                        post_radius_relative_error=float(exact[0]/sh[0])-1,
                        routing_disagreement=(estimate <= 5e-5) != (truth <= 5e-5)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 2 or args.cases < 1:
        parser.error("steps must be at least 2 and cases must be positive")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    result = dict(config=dict(seed=20261106, steps=args.steps, cases=args.cases,
        python=sys.version, torch=torch.__version__,
        device="cpu", geometry_dtype="float64", trajectory_spectra="exact",
        trajectory_lmo="exact polar", warm_audit_dtype="float32", beta=.9,
        schedule="cosine .015 to .0015", near_scale=1., gap=5e-5, cluster=2e-3,
        note="Toy geometry study, not a reproduction of c8 LLM runs. No tuning or backtracking."),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "scripts/local_cone_lmo_feasibility.py",
            "Megatron-LM/emerging_optimizers/orthogonalized_optimizers/spectral_cone_lmo.py",
            "Megatron-LM/emerging_optimizers/orthogonalized_optimizers/warm_block_ritz.py")},
        algebra=algebra(20261106), counterexamples=counterexamples(),
        fixed_point_return=local_return(20261106),
        trajectories=trajectories(20261106, args.steps, args.cases),
        plateau_trajectories=trajectories(20261106, args.steps, args.cases, plateau=True),
        width256_audit=width256_audit(20261106))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"algebra": result["algebra"]["summary"],
                      "trajectories": len(result["trajectories"]["rows"])+len(result["plateau_trajectories"]["rows"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
