"""Offline cone-angle and complete-return audit, never an LLM loss oracle.

Stored FP32 snapshots are audited in FP64. Counterfactuals share the radially
normalized X, stored Nesterov M and stored D0. A near cluster defines a cone
at a nearby lifted plateau, not the true tangent cone of a simple-top X.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch

from local_cone_lmo_feasibility import cone
from local_step_aware_hybrid import linear_direction, sym, tangent


def cosine(a, b):
    denom = float(a.norm()*b.norm())
    return None if denom <= 1e-30 else max(-1., min(1., float((a*b).sum())/denom))


def angle(a, b):
    c = cosine(a, b)
    return None if c is None else math.degrees(math.acos(c))


def scaled(d, length):
    n = float(d.norm())
    return torch.zeros_like(d) if n <= 1e-30 else d*(length/n)


def cone_metrics(z, u, v):
    q = cone(z, u, v)
    a = sym(u.T@z@v)
    values = torch.linalg.eigvalsh(a)
    n = float(z.norm())
    dist = float((z-q).norm())/max(n, 1e-30)
    top_norm = float(a.norm())
    top_dist = float((z-q).norm())/max(top_norm, 1e-30)
    # Projection onto a closed cone optimizes length along each admissible ray.
    degrees = math.degrees(math.asin(min(1., dist))) if n else None
    return dict(angle_degrees=degrees, distance_fraction=dist,
        top_block_angle_degrees=math.degrees(math.asin(min(1., top_dist))) if top_norm else None,
        retained_fraction=float(q.norm())/max(n, 1e-30),
        top_block_fraction=float(a.norm())/max(n, 1e-30),
        lambda_max_over_norm=float(values[-1])/max(n, 1e-30),
        lambda_min_over_norm=float(values[0])/max(n, 1e-30),
        positive_top_eigenvalues=int((values > 1e-12*max(n, 1e-30)).sum()))


def directions(m, d0, u, v):
    z = -m
    ref = float(d0.norm())
    ambient_ref = ref if ref > 1e-12 else min(m.shape)**.5
    result = {"d0": (d0, "radial"), "half_d0": (.5*d0, "radial")}
    for rho, label in ((.25,"025"),(.5,"050"),(1.,"100")):
        result["ambient_"+label] = (scaled(z, rho*ambient_ref), "radial")
    result["top1_050"] = (scaled(tangent(z,u[:,:1],v[:,:1]), .5*ref), "radial")
    result["cluster_cone_050"] = (scaled(cone(z,u,v), .5*ref), "radial")
    result["ambient_050_project"] = (result["ambient_050"][0], "sphere_projection")
    result["cluster_cone_050_project"] = (result["cluster_cone_050"][0], "sphere_projection")
    return result


@torch.no_grad()
def audit_state(x, m, radius, alpha, d0=None, *, gradient=None, window=2e-4):
    x, m = x.double(), m.double()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    pre_bias = float(s[0]/radius-1)
    factor = radius/s[0]
    x, s = x*factor, s*factor
    v = vh.T
    relative = (s[0]-s)/s[0]
    true_q = int((relative <= 1e-10).sum())
    cluster_q = int((relative <= window).sum())
    ut, vt = u[:,:true_q], v[:,:true_q]
    uc, vc = u[:,:cluster_q], v[:,:cluster_q]
    if d0 is None:
        d0 = linear_direction(m,u[:,:1],v[:,:1])[0]
        d0_source = "exact_polar_top1_mcsd_tp"
    else:
        d0 = d0.double()
        d0_source = "stored_production_d0"
    info = dict(shape=list(x.shape), radius=radius, alpha=alpha,
        pre_radius_error=pre_bias, gap=float(relative[1]), true_q=true_q,
        cluster_q=cluster_q, d0_source=d0_source,
        cluster_lift_relative_fro=float((radius-s[:cluster_q]).norm()/x.norm()),
        negative_momentum_true_cone=cone_metrics(-m,ut,vt),
        negative_momentum_cluster_cone=cone_metrics(-m,uc,vc),
        positive_momentum_cluster_cone=cone_metrics(m,uc,vc),
        gradient_momentum_cosine=cosine(gradient,m) if gradient is not None else None)
    rows = []
    for method, (d, return_rule) in directions(m,d0,uc,vc).items():
        y = x+alpha*d
        uy, sy, vhy = torch.linalg.svd(y, full_matrices=False)
        if return_rule == "radial":
            returned = y*(radius/sy[0])
            values = sy*(radius/sy[0])
        else:
            values = sy.clamp(max=radius).clone()
            values[0] = radius
            returned = (uy*values)@vhy
        delta = returned-x
        h = float(torch.linalg.eigvalsh(sym(ut.T@d@vt))[-1])
        radial_first = d-(h/radius)*x
        gain = -float((m*delta).sum())
        trial_gain = -alpha*float((m*d).sum())
        cutoff = 1e-12*max(float(m.norm()*delta.norm()),1e-30)
        gap_to_inactive = float(s[0]-s[true_q]) if true_q < len(s) else None
        row = dict(method=method, return_rule=return_rule,
            trial_fro_length=float((alpha*d).norm()), returned_fro_length=float(delta.norm()),
            actual_fro_elr=float(delta.norm()/x.norm()),
            actual_spectral_elr=float(torch.linalg.svdvals(delta)[0]/radius),
            returned_over_d0_trial=float(delta.norm())/max(alpha*float(d0.norm()),1e-30),
            return_rotation_degrees=angle(d,delta),
            momentum_return_cosine=cosine(-m,delta),
            momentum_trial_gain=trial_gain, momentum_return_gain=gain,
            momentum_ascent=gain < -cutoff,
            return_momentum_penalty=float((m*(returned-y)).sum()),
            true_cone_signed_residual=h/max(float(d.norm()),1e-30),
            radial_first_order_gain=-alpha*float((m*radial_first).sum()),
            radial_model_relative_error=float((delta/alpha-radial_first).norm())/max(float(d.norm()),1e-30),
            step_over_inactive_gap=alpha*float(torch.linalg.svdvals(d)[0])/gap_to_inactive if gap_to_inactive else None,
            returned_gap=float(1-values[1]/values[0]),
            quadratic_loss_change=None, quadratic_loss_increased=None)
        if gradient is not None:
            # Exact change for f(W)=0.5*||W-(X-G)||_F^2, not an LLM loss.
            change=float((gradient*delta).sum()+.5*delta.square().sum())
            row.update(quadratic_loss_change=change, quadratic_loss_increased=change>1e-12)
        rows.append(row)
    return info, rows


def aggregate(states, rows, predicate=lambda state:True):
    ids = {s["id"] for s in states if predicate(s)}
    output = []
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["state_id"] in ids and r["method"] == method]
        if not selected:
            continue
        def avg(key):
            a=[r[key] for r in selected if r[key] is not None]
            return mean(a) if a else None
        valid_loss=[r for r in selected if r["quadratic_loss_increased"] is not None]
        output.append(dict(method=method, states=len(selected),
            momentum_ascent_count=sum(r["momentum_ascent"] for r in selected),
            mean_momentum_return_cosine=avg("momentum_return_cosine"),
            mean_return_rotation_degrees=avg("return_rotation_degrees"),
            mean_returned_over_d0_trial=avg("returned_over_d0_trial"),
            momentum_gain_ratio_of_sums=sum(r["momentum_return_gain"] for r in selected)/max(sum(r["momentum_trial_gain"] for r in selected),1e-30),
            quadratic_loss_increase_count=sum(r["quadratic_loss_increased"] for r in valid_loss) if valid_loss else None,
            mean_quadratic_loss_change=avg("quadratic_loss_change")))
    return output


def synthetic_states():
    for seed in range(4):
        rng=torch.Generator().manual_seed(17300+seed)
        u=torch.linalg.qr(torch.randn(16,12,dtype=torch.float64,generator=rng))[0]
        v=torch.linalg.qr(torch.randn(12,12,dtype=torch.float64,generator=rng))[0]
        for q,gap in ((1,0.),(2,0.),(4,0.),(2,1e-4),(4,1e-4)):
            s=torch.linspace(.7,.15,12,dtype=torch.float64)
            s[:q]=1-torch.linspace(0,gap,q,dtype=torch.float64)
            x=(u*s)@v.T
            for bias in (-1.,0.,1.):
                g=torch.randn(x.shape,dtype=torch.float64,generator=rng)
                g=g+4*bias*float(g.norm())*(u[:,:q]@v[:,:q].T)/q**.5
                g=g/g.norm()
                noise=torch.randn(x.shape,dtype=torch.float64,generator=rng)
                noise=noise-(noise*g).sum()*g
                noise=noise/noise.norm()
                for alignment in (1.,.5):
                    m=alignment*g+(1-alignment**2)**.5*noise
                    for alpha in (.0015,.0075,.015):
                        yield dict(seed=seed,planned_q=q,planned_gap=gap,gradient_top_bias=bias,
                                   alignment=alignment),x,m,g,alpha


def radial_ascent_example():
    x=torch.eye(2,dtype=torch.float64)
    g=torch.diag(torch.tensor([-2.,-1.],dtype=torch.float64))
    d=-g/g.norm()
    alpha=.015
    y=x+alpha*d
    radial=y/torch.linalg.svdvals(y)[0]
    delta=radial-x
    q=cone(-g,x,x)
    return dict(x=x.tolist(),gradient=g.tolist(),alpha=alpha,
        raw_direction_pairing=float((g*d).sum()),
        radial_first_order_pairing=1/math.sqrt(5),
        quadratic_loss_change=float((g*delta).sum()+.5*delta.square().sum()),
        cone_projection_norm=float(q.norm()),
        explanation="At X=I2 the positive diagonal antigradient projects to zero. Radial return moves away from this constrained quadratic minimizer.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    real_states,real_rows=[],[]
    files=sorted(args.snapshots.glob("sample_*.pt"))
    if not files:
        raise RuntimeError("no snapshots")
    for file in files:
        data=torch.load(file,map_location="cpu",weights_only=True)
        info, rows=audit_state(data["current"],data["momentum"],float(data["radius"]),
                              float(data["alpha"]),data["d0"])
        info.update(id=file.stem,source=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            step=data["step"],selection=data["selection"],component=data["component_label"],
            cached_gap=data["record"]["current_gap"],cached_near=data["record"]["near"])
        real_states.append(info)
        real_rows.extend(dict(state_id=info["id"],**r) for r in rows)
        print(f"audited {file.name}",flush=True)
    toy_states,toy_rows=[],[]
    for index,(meta,x,m,g,alpha) in enumerate(synthetic_states()):
        info,rows=audit_state(x,m,1.,alpha,gradient=g)
        info.update(id=str(index),**meta)
        toy_states.append(info)
        toy_rows.extend(dict(state_id=str(index),**r) for r in rows)
    report=dict(reference_job="4442918_2",window=2e-4,true_multiplicity_relative_tolerance=1e-10,
        real_states=real_states,real_rows=real_rows,synthetic_states=toy_states,synthetic_rows=toy_rows,
        real_near_summary=aggregate(real_states,real_rows,lambda s:s["cluster_q"]>1),
        toy_corner_aligned_summary=aggregate(toy_states,toy_rows,lambda s:s["true_q"]>1 and s["alignment"]==1.),
        toy_corner_noisy_summary=aggregate(toy_states,toy_rows,lambda s:s["true_q"]>1 and s["alignment"]==.5),
        counterexample=radial_ascent_example(),
        limitations=["Real snapshots contain Nesterov M and D0, not raw gradients, full model or evaluation batches; no real LLM loss is computed.",
            "All counterfactuals start from a common exact-radial-normalized copy. This is not a replay of the original warm trajectory.",
            "The near-cluster cone is the true Bouligand cone at a nearby lifted plateau, not at a simple-top current X.",
            "Snapshots are scheduled shapes and early near calls, not an unbiased sample of training.",
            "Toy loss is a constructed quadratic with controlled momentum-gradient alignment, not the LLM objective or a convergence proof.",
            "Nine operations are forced one-step counterfactuals, not nine newly trained hybrid methods. All decompositions are offline."])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("real_near_summary","toy_corner_aligned_summary","toy_corner_noisy_summary","counterexample")},indent=2))


if __name__=="__main__":
    main()
