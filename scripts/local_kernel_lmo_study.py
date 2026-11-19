"""Exact-spectrum research reference; no production changes or external jobs.

The new near branch preserves the finite-step model's symmetric top block
and adds a bounded MCSD direction in its kernel. Diagnostics, including the
coupling certificate, are offline and never select a direction or a step.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

import local_step_aware_hybrid as base

METHODS = ("MCSD-TP", "A2", "A4", "A4-budget", "K-LMO-0.5", "K-LMO-1",
           "A2-input-1.5", "A2-input-2")
PROFILE_CONTROL = "MCSD-TP/A2-profile"
SHAPES = ((4, 4), (16, 12), (12, 16))
PROTOCOLS = ("certified_ema", "practical_nesterov")


def direction(m, spec, radius, alpha, q, method):
    if method in base.METHODS:
        return base.direction(m, spec, radius, alpha, q, method)
    if method not in METHODS:
        raise ValueError("unknown method")
    if q == 1:
        return base.direction(m, spec, radius, alpha, q, "MCSD-TP")
    u, s, v = spec
    u, s, v = u[:, :q], s[:q], v[:, :q]
    nu = torch.linalg.vector_norm(m)
    info = dict(q=q, lmo_svd=0, small_eigh=0, accepted=None)
    if nu == 0:
        return torch.zeros_like(m), info
    if method.startswith("A2-input-"):
        scale = float(method.removeprefix("A2-input-"))
        info["small_eigh"] = 1
        return base.model_project(-scale * m / nu, u, s, v, radius, alpha), info
    df = base.model_project(-m / nu, u, s, v, radius, alpha)
    info["small_eigh"] = 1
    nf = torch.linalg.vector_norm(df)
    if nf == 0:
        return df, info
    if method == "A4-budget":
        candidate = base.polar(df)
        nc = torch.linalg.vector_norm(candidate)
        if nc == 0:
            return df, info
        candidate = candidate * torch.minimum(torch.ones_like(nc), nf / nc)
        candidate = base.model_project(candidate, u, s, v, radius, alpha)
        accept = bool((m * (candidate - df)).sum() <= 0 and
                      torch.linalg.vector_norm(candidate) <= 2 * nf)
        info.update(lmo_svd=1, small_eigh=2, accepted=accept)
        return (candidate if accept else df), info
    theta = .5 if method == "K-LMO-0.5" else 1.
    tangent, count = base.linear_direction(m, u, v)
    nt = torch.linalg.vector_norm(tangent)
    # A scalar momentum check also protects against an inexact future LMO.
    accept = bool(nt > 0 and (m * tangent).sum() <= 0)
    info.update(lmo_svd=count, accepted=accept)
    if accept:
        return df + theta * nf * (tangent / nt), info
    return df, info


def coupling_bound(spec, d, q, radius, alpha):
    """Dilation cross-block norm from a q-by-q Gram matrix (audit only).

    The certificate assumes exact paired bases and a feasible model block.
    A full-matrix operator norm here is separately an offline diagnostic.
    """
    u, s, v = spec
    u, v = u[:, :q], v[:, :q]
    h = base.sym(u.T @ d @ v)
    dv, dtu = d @ v, d.T @ u
    gram = base.sym(.5 * (dv.T @ dv + dtu.T @ dtu) - h @ h)
    c2 = max(0., float(torch.linalg.eigvalsh(gram)[-1]))
    op = float(torch.linalg.matrix_norm(d, ord=2))
    gap = base.gamma(s, q, radius)
    sep = gap - alpha * op
    root = math.sqrt(sep * sep + 4 * alpha * alpha * c2)
    # Rationalize only for a positive denominator to avoid cancellation.
    omega = 2 * alpha * alpha * c2 / (root + sep) if root + sep > 0 else .5 * (root - sep)
    old = 2 * alpha * alpha * op * op / gap if gap > 0 and alpha * op <= gap / 2 else None
    return dict(coupling=math.sqrt(c2), direction_op=op, complement_separation=sep,
                overshoot_bound=omega, old_overshoot_bound=old,
                gram_min=float(torch.linalg.eigvalsh(gram)[0]))


def run(x0, target, weights, meta, noise, method, protocol, steps, profile=None):
    x = x0.clone()
    spec = base.spectrum(x)
    stored = torch.zeros_like(x)
    radius, r = meta["radius"], min(x.shape)
    counts, sums = Counter(), defaultdict(float)
    maximum = defaultdict(float)
    trace, samples, norms = [], [], []
    audited = {0, steps // 2, steps - 1}
    last_near = False
    for t in range(steps):
        alpha, h, gain = base.settings(protocol, t, steps, radius, r)
        g = weights * (x - target)
        sample = g + .05 * radius * noise[t]
        if protocol == "certified_ema":
            stored = (1 - gain) * stored + gain * sample
            m = stored
        else:
            stored = .9 * stored + .1 * sample
            m = .9 * stored + .1 * sample
        q = base.window_q(spec[1], radius, h)
        if method == PROFILE_CONTROL:
            d, info = base.direction(m, spec, radius, alpha, 1, "MCSD-TP")
            nd = float(torch.linalg.vector_norm(d))
            if nd:
                d = d * (profile[t] / nd)
            elif profile[t] > 1e-14:
                counts["profile_zero_direction"] += 1
            maximum["profile_length_error"] = max(maximum["profile_length_error"],
                abs(float(torch.linalg.vector_norm(d)) - profile[t]))
        else:
            d, info = direction(m, spec, radius, alpha, q, method)
        used_q = info["q"]
        near = method not in ("MCSD-TP", PROFILE_CONTROL) and used_q > 1
        counts["near"] += near
        counts["repeat_near"] += near and last_near
        last_near = near
        counts["q_gt4"] += used_q > 4
        counts["q_gt8"] += used_q > 8
        counts["lmo_svd"] += info["lmo_svd"]
        counts["small_eigh"] += info["small_eigh"]
        if info["accepted"] is not None:
            counts["guard_calls"] += 1
            counts["guard_accept"] += info["accepted"]
        nd = float(torch.linalg.vector_norm(d))
        norms.append(nd)
        old = base.objective(x, target, weights)
        y = x + alpha * d
        updated, next_spec, trial_op = base.radial_cached(y, radius)
        change = base.objective(updated, target, weights) - old
        prediction = alpha * float((g * d).sum())
        step = float(torch.linalg.vector_norm(updated - x)) / radius
        counts["uphill"] += change > 1e-12
        counts["near_uphill"] += near and change > 1e-12
        counts["prediction_reversal"] += prediction < -1e-12 and change > 1e-12
        sums["returned_length"] += step
        sums["trial_length"] += alpha * nd / radius
        if near:
            df, _ = base.direction(m, spec, radius, alpha, q, "A2")
            maximum["reference_guard_violation"] = max(maximum["reference_guard_violation"],
                float((m * (d - df)).sum()))
            model = torch.diag(spec[1][:q]) + alpha * base.sym(spec[0][:, :q].T @ d @ spec[2][:, :q])
            maximum["model_error"] = max(maximum["model_error"],
                abs(float(torch.linalg.eigvalsh(model)[-1]) - radius))
        if t in audited:
            bound = coupling_bound(spec, d, used_q, radius, alpha)
            overshoot = trial_op - radius
            counts["coupling_audits"] += 1
            counts["coupling_bound_violation"] += overshoot > bound["overshoot_bound"] + 1e-10
            if bound["old_overshoot_bound"] is not None and bound["old_overshoot_bound"] > 0:
                sums["bound_ratio"] += bound["overshoot_bound"] / bound["old_overshoot_bound"]
                counts["bound_ratio"] += 1
            full_bound = prediction + bound["overshoot_bound"] / radius * max(0., -float((g * y).sum()))
            full_bound += .5 * float(weights.max()) * float((updated - x).square().sum())
            counts["complete_bound_violation"] += change > full_bound + 1e-9
            record = dict(step=t, q=used_q, actual_change=change, full_bound=full_bound,
                          trial_overshoot=overshoot, **bound)
            if near:
                control, _ = base.direction(m, spec, radius, alpha, q, "MCSD-TP")
                norm_control = float(torch.linalg.vector_norm(control))
                if norm_control > 1e-14:
                    control = control * (nd / norm_control)
                    yc, _, _ = base.radial_cached(x + alpha * control, radius)
                    record["matched_change_delta"] = base.objective(updated, target, weights) - base.objective(yc, target, weights)
                # Rebuild each direction when testing a larger selected cluster.
                needed = None
                for j in range(q, r + 1):
                    dj, _ = direction(m, spec, radius, alpha, j, method)
                    if 2 * alpha * float(torch.linalg.matrix_norm(dj, ord=2)) <= base.gamma(spec[1], j, radius):
                        needed = j
                        break
                record["rebuilt_q"] = needed
            samples.append(record)
            maximum["radius_error"] = max(maximum["radius_error"],
                abs(float(torch.linalg.matrix_norm(updated, ord=2)) / radius - 1))
        x, spec = updated, next_spec
        if t % 50 == 0 or t == steps - 1:
            trace.append(dict(step=t + 1, loss=base.objective(x, target, weights),
                              q=used_q, returned_length=step))
    final = base.objective(x, target, weights)
    return dict(**meta, method=method, protocol=protocol, steps=steps, final_loss=final,
                excess=None if meta["fstar"] is None else final-meta["fstar"],
                counts=dict(counts), sums=dict(sums), maximum=dict(maximum), trace=trace,
                audits=samples, _norms=norms)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["protocol"], row["method"]].append(row)
    summaries = []
    for (protocol, method), group in groups.items():
        counts = Counter()
        for row in group:
            counts.update(row["counts"])
        known = [row["excess"] for row in group if row["excess"] is not None]
        steps = sum(row["steps"] for row in group)
        near_samples = [s for row in group for s in row["audits"] if "matched_change_delta" in s]
        capacities = [s["rebuilt_q"] for row in group for s in row["audits"] if s.get("rebuilt_q") is not None]
        summaries.append(dict(protocol=protocol, method=method, instances=len(group),
            known_optimum_instances=len(known), mean_excess=sum(known)/len(known),
            near_rate=counts["near"]/steps, uphill_rate=counts["uphill"]/steps,
            near_uphill_rate=counts["near_uphill"]/max(1,counts["near"]),
            mean_trial_length=sum(row["sums"]["trial_length"] for row in group)/steps,
            mean_returned_length=sum(row["sums"]["returned_length"] for row in group)/steps,
            counts=dict(counts), max_model_error=max(row["maximum"].get("model_error",0.) for row in group),
            max_profile_error=max(row["maximum"].get("profile_length_error",0.) for row in group),
            max_guard_violation=max(row["maximum"].get("reference_guard_violation",0.) for row in group),
            mean_bound_ratio=sum(row["sums"].get("bound_ratio",0.) for row in group)/max(1,counts["bound_ratio"]),
            matched_near_samples=len(near_samples),
            matched_near_mean_delta=sum(s["matched_change_delta"] for s in near_samples)/max(1,len(near_samples)),
            rebuilt_audits=len(capacities), rebuilt_gt8=sum(q>8 for q in capacities),
            max_rebuilt_q=max(capacities,default=None)))
    lookup = {(row["seed"],tuple(row["shape"]),row["protocol"],row["method"]):row for row in rows}
    pairs = []
    for protocol in PROTOCOLS:
        for method in (*METHODS, PROFILE_CONTROL):
            if method == "A2":
                continue
            diffs = [row["final_loss"]-lookup[(row["seed"],tuple(row["shape"]),protocol,"A2")]["final_loss"]
                     for row in rows if row["protocol"]==protocol and row["method"]==method]
            pairs.append(dict(protocol=protocol, method=method, reference="A2", pairs=len(diffs),
                mean_delta=sum(diffs)/len(diffs), wins=sum(d < -1e-10 for d in diffs),
                ties=sum(abs(d)<=1e-10 for d in diffs), losses=sum(d > 1e-10 for d in diffs)))
    return summaries, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=2000)
    parser.add_argument("--seeds",type=int,default=16)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.seeds < 4:
        parser.error("at least two steps and four seeds required")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)
    rows = []
    start = time.perf_counter()
    for shape in SHAPES:
        for seed in range(args.seeds):
            x,target,weights,meta = base.make_case(seed,shape)
            gen = torch.Generator().manual_seed(83700+seed+100*shape[0]+shape[1])
            noise = torch.randn(args.steps,*shape,generator=gen)/math.sqrt(x.numel())
            for protocol in PROTOCOLS:
                for method in METHODS:
                    row = run(x,target,weights,meta,noise,method,protocol,args.steps)
                    profile = row.pop("_norms")
                    rows.append(row)
                    if method == "A2":
                        control = run(x,target,weights,meta,noise,PROFILE_CONTROL,protocol,args.steps,profile)
                        control.pop("_norms")
                        rows.append(control)
        print(json.dumps(dict(shape=shape,trajectories=len(rows),elapsed=time.perf_counter()-start)),flush=True)
    summary,pairs = summarize(rows)
    result = dict(config=dict(steps=args.steps,seeds=args.seeds,shapes=SHAPES,
        methods=(*METHODS,PROFILE_CONTROL),protocols=PROTOCOLS,dtype="FP64",device="CPU",
        noise="Paired iid Gaussian entries / sqrt(m*n), sample noise variance .05^2 * R^2",
        profile_control="Offline A2 trial-Frobenius-length replay, not a deployable online optimizer",
        note="Same synthetic families as prior study; not independent LLM evidence; exact spectra and polar LMOs"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-start,summary=summary,pairs=pairs,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),trajectories=len(rows),elapsed=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
