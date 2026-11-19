"""Exact-spectrum CPU reference for the handoff's MCSD-TP/A1/A2/A4.

Research only: no training imports, line search, external jobs or production
changes. All oracle and diagnostic SVD costs are counted separately.
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

METHODS = ("MCSD-TP", "A1", "A2", "A4")
SHAPES = ((4, 4), (16, 12), (12, 16), (32, 24), (24, 32))
PROTOCOLS = ("certified_det", "certified_ema", "practical_nesterov")


def sym(a):
    return (a + a.T) * .5


def tangent(z, u, v):
    return z - u @ sym(u.T @ z @ v) @ v.T


def polar(z):
    if not torch.count_nonzero(z):
        return torch.zeros_like(z)
    u, s, vh = torch.linalg.svd(z, full_matrices=False)
    # The zero extension is also an LMO; discard roundoff-sized singular modes.
    active = s > 64 * torch.finfo(s.dtype).eps * s[0]
    return (u * active) @ vh


def linear_direction(m, u, v):
    z = tangent(-m, u, v)
    if torch.linalg.vector_norm(z) <= 64 * torch.finfo(m.dtype).eps * torch.linalg.vector_norm(m):
        return torch.zeros_like(m), 0
    return tangent(polar(z), u, v), 1


def model_project(z, u, s, v, radius, alpha):
    if alpha <= 0 or radius <= 0:
        raise ValueError("positive effective alpha and radius required")
    h = sym(u.T @ z @ v)
    a = h - torch.diag((radius - s) / alpha)
    w, o = torch.linalg.eigh(a)
    clipped = w.clamp(max=0)
    clipped[-1] = 0
    b = (o * clipped) @ o.T
    return z + u @ (b - a) @ v.T


def direction(m, spectrum, radius, alpha, q, method):
    u, s, v = spectrum
    if method not in METHODS or not 1 <= q <= len(s):
        raise ValueError("invalid method or cluster capacity")
    if method == "MCSD-TP":
        q = 1
    uq, sq, vq = u[:, :q], s[:q], v[:, :q]
    info = dict(q=q, lmo_svd=0, small_eigh=0, accepted=None, protection_margin=None)
    if method in ("MCSD-TP", "A1") or q == 1:
        d, count = linear_direction(m, uq, vq)
        info["lmo_svd"] = count
        return d, info
    nu = torch.linalg.vector_norm(m)
    if nu == 0:
        return torch.zeros_like(m), info
    reference = model_project(-m / nu, uq, sq, vq, radius, alpha)
    info["small_eigh"] = 1
    if method == "A2":
        return reference, info
    candidate = model_project(polar(reference), uq, sq, vq, radius, alpha)
    info.update(lmo_svd=int(bool(torch.count_nonzero(reference))), small_eigh=2)
    margin = float((m * (reference - candidate)).sum())
    info.update(accepted=margin >= 0, protection_margin=margin)
    return (candidate if margin >= 0 else reference), info


def spectrum(x):
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    return u, s, vh.T


def radial_cached(y, radius):
    u, s, v = spectrum(y)
    if s[0] <= 0:
        raise ValueError("radial return undefined at zero")
    scale = radius / s[0]
    return y * scale, (u, s * scale, v), float(s[0])


def sphere_project(x, radius):
    u, s, v = spectrum(x)
    s = s.clamp(max=radius)
    s[0] = radius
    return (u * s) @ v.T


def window_q(s, radius, h):
    return max(1, int(((radius - s) <= h + 32 * torch.finfo(s.dtype).eps * radius).sum()))


def gamma(s, q, radius):
    return float(radius - s[q]) if q < len(s) else radius


def exact_metrics(g, spec, radius):
    u, s, v = spec
    k = int(((radius - s) <= 1e-10 * radius).sum())
    uq, vq = u[:, :k], v[:, :k]
    clarke = torch.linalg.vector_norm(tangent(g, uq, vq))
    # E=0 gives the true multiplicity cone, within the stated FP64 tolerance.
    cone = model_project(-g, uq, torch.full_like(s[:k], radius), vq, radius, 1.)
    return dict(clarke_residual=float(clarke), b_residual=float(torch.linalg.vector_norm(cone)),
                multiplicity=k, relative_gap=float(1 - s[1] / s[0]))


def objective(x, target, weights):
    return float(.5 * (weights * (x - target).square()).sum())


def make_case(seed, shape, radius=1.):
    gen = torch.Generator().manual_seed(2026111100 + seed * 97 + shape[0] * 13 + shape[1])
    m, n = shape
    r = min(shape)
    def bases():
        return (torch.linalg.qr(torch.randn(m, r, generator=gen))[0],
                torch.linalg.qr(torch.randn(n, r, generator=gen))[0])
    u, v = bases()
    k = (1, 2, min(4, r), r)[(seed // 4) % 4]
    dense = seed % 2 == 0
    s = torch.linspace(.9999, .98, r) if dense else torch.linspace(.85, .15, r)
    s[:k] = 1
    x = (u * (radius * s)) @ v.T
    u, v = bases()
    family = ("inside", "outside", "plateau_target", "anisotropic")[seed % 4]
    a = torch.linspace(.7, .15, r)
    if family in ("outside", "anisotropic"):
        a[0] = 1.4
    if family == "plateau_target":
        a[:min(4, r)] = 1.3
    target = (u * (radius * a)) @ v.T
    weights = torch.ones_like(x)
    if family == "anisotropic":
        weights = .5 + 4.5 * torch.rand(m, n, generator=gen)
    fstar = None if family == "anisotropic" else objective(sphere_project(target, radius), target, weights)
    return x, target, weights, dict(seed=seed, shape=shape, radius=radius, family=family,
                                   initial_multiplicity=k, dense_tail=dense, fstar=fstar)


def settings(protocol, t, steps, radius, r):
    if protocol.startswith("certified"):
        decay = 1 + t / 100
        h = .05 * radius * decay ** -.3
        alpha = .2 * .05 * radius / (2 * math.sqrt(r)) * decay ** -.9
        return alpha, h, .5 * decay ** -.5
    warm = min(1., (t + 1) / 20)
    cosine = .1 + .9 * .5 * (1 + math.cos(math.pi * t / max(1, steps - 1)))
    return .0075 * radius * warm * cosine, .0002 * radius, .1


def capacity_audit(m, spec, radius, alpha, q, method):
    u, s, v = spec
    r = len(s)
    d, _ = direction(m, spec, radius, alpha, q, method)
    norm = float(torch.linalg.matrix_norm(d, ord=2))
    fixed_need = next((j for j in range(1, r + 1) if 2 * alpha * norm <= gamma(s, j, radius)), None)
    rebuilt_need = None
    attempts = 0
    if method != "MCSD-TP":
        for j in range(q, r + 1):
            candidate, _ = direction(m, spec, radius, alpha, j, method)
            attempts += 1
            if 2 * alpha * float(torch.linalg.matrix_norm(candidate, ord=2)) <= gamma(s, j, radius):
                rebuilt_need = j
                break
    return dict(selected_q=q, fixed_direction_gap_requirement=fixed_need,
                rebuilt_min_q=rebuilt_need, rebuild_attempts=attempts,
                note="Offline audit only; MCSD-TP is never silently replaced by a cluster method.")


def amplitude_audit(x, m, spec, target, weights, radius, alpha, q):
    """Counterfactuals on an A4 state, NOT extra steps in its trajectory.

    Match only the top-1 baseline's length. Never rescale the finite-step
    model direction after projection, which would invalidate that model.
    """
    base, _ = direction(m, spec, radius, alpha, q, "MCSD-TP")
    base_norm = float(torch.linalg.vector_norm(base))
    old_loss = objective(x, target, weights)
    rows = []
    for method in METHODS:
        d, _ = direction(m, spec, radius, alpha, q, method)
        norm = float(torch.linalg.vector_norm(d))
        returned, _, _ = radial_cached(x + alpha * d, radius)
        matched_change = matched_trial = None
        if base_norm > 1e-14:
            matched = base * (norm / base_norm)
            matched_return, _, _ = radial_cached(x + alpha * matched, radius)
            matched_change = objective(matched_return, target, weights) - old_loss
            matched_trial = float(torch.linalg.vector_norm(alpha * matched)) / radius
        rows.append(dict(method=method, actual_change=objective(returned,target,weights)-old_loss,
                         trial_length=alpha*norm/radius, matched_top1_change=matched_change,
                         matched_top1_trial_length=matched_trial))
    return rows


def trajectory(x0, target, weights, meta, noise, method, protocol, steps):
    x = x0.clone()
    spec = spectrum(x)
    m = torch.zeros_like(x)
    b = torch.zeros_like(x)
    radius, r = meta["radius"], min(x.shape)
    sums, maxima, counts = defaultdict(float), defaultdict(float), Counter()
    q_hist, valid_counts, traces, capacities, amplitudes = Counter(), Counter(), [], [], []
    lmo_svd = small_eigh = 0
    audit_steps = {0, steps // 2, steps - 1}
    last_near = False
    longest_burst = burst = 0
    start = time.perf_counter()
    for t in range(steps):
        alpha, h, gain = settings(protocol, t, steps, radius, r)
        g = weights * (x - target)
        sample = g if protocol == "certified_det" else g + .05 * radius * noise[t]
        if protocol == "certified_det":
            m = sample
        elif protocol == "certified_ema":
            m = (1 - gain) * m + gain * sample
        else:
            b = .9 * b + .1 * sample
            m = .9 * b + .1 * sample
        q_window = window_q(spec[1], radius, h)
        d, info = direction(m, spec, radius, alpha, q_window, method)
        q = info["q"]
        q_hist[q] += 1
        lmo_svd += info["lmo_svd"]
        small_eigh += info["small_eigh"]
        near = method != "MCSD-TP" and q > 1
        counts["near"] += near
        counts["repeat"] += near and last_near
        burst = burst + 1 if near else 0
        longest_burst = max(longest_burst, burst)
        last_near = near
        counts["q_window_gt4"] += q_window > 4
        counts["q_window_gt8"] += q_window > 8
        if info["accepted"] is not None:
            counts["guard_calls"] += 1
            counts["guard_accept"] += info["accepted"]
        if t in audit_steps:
            capacities.append(dict(step=t, **capacity_audit(m, spec, radius, alpha, q_window, method)))
            if method == "A4":
                amplitudes.append(dict(step=t,q=q_window,rows=amplitude_audit(
                    x,m,spec,target,weights,radius,alpha,q_window)))
        d_op = float(torch.linalg.matrix_norm(d, ord=2))
        d_fro = float(torch.linalg.vector_norm(d))
        gam = gamma(spec[1], q, radius)
        delta = alpha * d_op
        # A top-1 control at a true plateau has Gamma=0: no finite bound.
        bound_ok = gam > 0 and delta <= gam / 2 * (1 + 1e-10)
        y = x + alpha * d
        updated, next_spec, y_norm = radial_cached(y, radius)
        displacement = updated - x
        residual = updated - y
        change = objective(updated, target, weights) - objective(x, target, weights)
        predicted = float(alpha * (g * d).sum())
        momentum_predicted = float(alpha * (m * d).sum())
        residual_norm = float(torch.linalg.vector_norm(residual))
        return_bound = 2 * math.sqrt(r) * alpha ** 2 * d_op ** 2 / gam if gam > 0 else None
        model_error = abs(float(torch.linalg.eigvalsh(
            torch.diag(spec[1][:q]) + alpha * sym(spec[0][:, :q].T @ d @ spec[2][:, :q]))[-1]) - radius)
        full_upper = (predicted + float(torch.linalg.vector_norm(g)) * return_bound
                      + .5 * float(weights.max()) * float(displacement.square().sum())) if gam > 0 else None
        tol = 1e-11 * radius ** 2
        counts["bound_ok"] += bound_ok
        counts["zero_gamma"] += gam <= 0
        counts["uphill"] += change > tol
        counts["prediction_reversed"] += predicted < -tol and change > tol
        counts["certified_uphill"] += bound_ok and change > tol
        counts["full_upper_negative"] += bound_ok and full_upper < -tol
        counts["upper_violation"] += bound_ok and change > full_upper + tol
        counts["return_bound_violation"] += bound_ok and residual_norm > return_bound + 1e-10 * radius
        vals = dict(loss_change=change, gradient_prediction=predicted, momentum_prediction=momentum_predicted,
                    trial_fro=alpha * d_fro / radius, actual_fro=float(torch.linalg.vector_norm(displacement)) / radius,
                    trial_spectral=delta / radius, direction_spectral=d_op,
                    return_relative=residual_norm / max(alpha * d_fro, 1e-30),
                    return_absolute=residual_norm / radius, condition_ratio=2 * delta / gam if gam > 0 else None,
                    model_error=model_error / radius, full_upper=full_upper,
                    norm_excess=(y_norm - radius) / radius)
        for key, value in vals.items():
            if value is None:
                continue
            sums[key] += value
            valid_counts[key] += 1
            maxima[key] = max(maxima[key], abs(value))
        if t % 25 == 0 or t == steps - 1:
            metrics = exact_metrics(weights * (updated - target), next_spec, radius)
            # Fresh SVD is an OFFLINE radius/cache check, not part of the update.
            fresh_s = torch.linalg.svdvals(updated)
            metrics["radius_error"] = abs(float(fresh_s[0]) / radius - 1)
            metrics["cache_error"] = float((fresh_s - next_spec[1]).abs().max()) / radius
            traces.append(dict(step=t + 1, loss=objective(updated, target, weights), q=q,
                               q_window=q_window, alpha=alpha, h=h, bound_ok=bound_ok, **vals, **metrics))
        x, spec = updated, next_spec
    final = objective(x, target, weights)
    return dict(**meta, method=method, protocol=protocol, steps=steps, final_loss=final,
                final_excess=None if meta["fstar"] is None else final - meta["fstar"],
                counts=dict(counts), q_histogram=dict(q_hist), max_burst=longest_burst,
                means={k: v / valid_counts[k] for k, v in sums.items()}, maxima=dict(maxima),
                diagnostic_valid_counts=dict(valid_counts),
                final_metrics=traces[-1], trace=traces, capacity=capacities, amplitude_audit=amplitudes,
                calls=dict(initial_weight_svd=1, trial_weight_svd=steps, direction_lmo_svd=lmo_svd,
                           direction_small_eigh=small_eigh, diagnostic_per_step_svdvals=steps,
                           cache_audit_svdvals=len(traces), capacity_audits=len(capacities),
                           note="Capacity and amplitude audits include additional oracle/SVD calls; wall time includes all diagnostics."),
                seconds=time.perf_counter() - start)


def counterexamples():
    rows = []
    for eps in (1e-2, 1e-4, 1e-6):
        x = torch.diag(torch.tensor([1., 1. - eps]))
        m = -torch.tensor([[4., 1.], [1., 0.]])
        spec = spectrum(x)
        for method in METHODS:
            d, info = direction(m, spec, 1., eps, 2, method)
            returned, _, _ = radial_cached(x + eps * d, 1.)
            rows.append(dict(eps=eps, method=method, q=info["q"],
                             predicted=float((m * d).sum()),
                             actual_over_alpha=float((m * (returned - x)).sum()) / eps,
                             return_over_alpha=float(torch.linalg.vector_norm(returned - x - eps * d)) / eps))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["method"])].append(row)
    result = []
    for (protocol, method), group in groups.items():
        n = sum(r["steps"] for r in group)
        known = [r["final_excess"] for r in group if r["final_excess"] is not None]
        capacities = [a["rebuilt_min_q"] for r in group for a in r["capacity"] if a["rebuilt_min_q"] is not None]
        total = lambda key: sum(r["counts"].get(key, 0) for r in group)
        result.append(dict(protocol=protocol, method=method, trajectories=len(group),
            mean_excess=sum(known)/len(known), known_optimum_count=len(known),
            bound_rate=total("bound_ok")/n, uphill_rate=total("uphill")/n,
            prediction_reversal_rate=total("prediction_reversed")/n,
            certified_uphill=total("certified_uphill"), upper_violations=total("upper_violation"),
            return_bound_violations=total("return_bound_violation"),
            near_rate=total("near")/n, repeat_rate=total("repeat")/max(1,total("near")),
            guard_accept_rate=total("guard_accept")/max(1,total("guard_calls")) if method=="A4" else None,
            mean_trial_fro=sum(r["means"]["trial_fro"] for r in group)/len(group),
            mean_actual_fro=sum(r["means"]["actual_fro"] for r in group)/len(group),
            window_gt4=total("q_window_gt4")/n,
            rebuilt_q_gt4=sum(q>4 for q in capacities)/max(1,len(capacities)),
            rebuilt_q_gt8=sum(q>8 for q in capacities)/max(1,len(capacities)),
            rebuilt_audits=len(capacities), max_rebuilt_q=max(capacities,default=None),
            max_radius_error=max(p["radius_error"] for r in group for p in r["trace"])))
    paired = []
    lookup = {(r["seed"],tuple(r["shape"]),r["protocol"],r["method"]):r for r in rows}
    for protocol in PROTOCOLS:
        for a, b in (("A4","MCSD-TP"),("A4","A1"),("A4","A2"),("A2","A1")):
            diffs = [r["final_loss"]-lookup[(r["seed"],tuple(r["shape"]),protocol,b)]["final_loss"]
                     for r in rows if r["method"]==a and r["protocol"]==protocol]
            paired.append(dict(protocol=protocol, left=a, right=b, pairs=len(diffs),
                               mean_delta=sum(diffs)/len(diffs), wins=sum(v < -1e-10 for v in diffs),
                               losses=sum(v > 1e-10 for v in diffs)))
    return result, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.steps < 2 or args.seeds < 4:
        parser.error("at least two steps and four seeds required")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)
    rows=[]; start=time.perf_counter()
    for shape in SHAPES:
        for seed in range(args.seeds):
            x,target,weights,meta=make_case(seed,shape)
            gen=torch.Generator().manual_seed(73200+seed+100*shape[0]+shape[1])
            noise=(torch.randint(0,2,(args.steps,*shape),generator=gen)*2-1).double()/math.sqrt(x.numel())
            for protocol in PROTOCOLS:
                for method in METHODS:
                    rows.append(trajectory(x,target,weights,meta,noise,method,protocol,args.steps))
        print(json.dumps(dict(shape=shape,trajectories=len(rows),elapsed=time.perf_counter()-start)),flush=True)
    summary,paired=summarize(rows)
    report=dict(config=dict(steps=args.steps,seeds=args.seeds,shapes=SHAPES,methods=METHODS,
        protocols=PROTOCOLS,dtype="FP64",device="CPU",radius=1.,torch=torch.__version__,
        noise="Shared Rademacher Frobenius norm .05; conditionally unbiased and bounded",
        routing="Full window, q=1 uses MCSD-TP; no fixed-gap routing or q cap",
        note="Finite trajectories test inequalities, not asymptotic convergence; no original LLM loss is measured."),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counterexamples=counterexamples(),summary=summary,paired=paired,rows=rows,
        elapsed_seconds=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),summary=summary,paired=paired),indent=2))


if __name__=="__main__":
    main()
