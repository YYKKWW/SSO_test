"""Opt-in H20 pilot adapter. Existing optimizer source is not modified.

Exact arms retain the FULL paired spectrum, not the host's default top eight.
Warm arms have an explicitly uncertified eight-dimensional cache. A complete
window routes the new hybrid; this differs from the old independent gap gate.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(os.environ.get("MEGATRON_PATH", Path(__file__).resolve().parents[1]/"Megatron-LM"))))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host
from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz
from local_step_aware_hybrid import model_project, tangent, sym

ARMS = (
    ("control", "exact", "spel_pgd_dist"),
    ("a2", "exact", "spel_pgd_dist"),
    ("a2_1p5", "exact", "spel_pgd_dist"),
    ("kernel1", "exact", "spel_pgd_dist"),
    ("a2", "warm_block8", "spel_pgd_dist"),
    ("kernel1", "warm_block8", "spel_pgd_dist"),
    ("baseline", "reference", "muon_ball_dist"),
    ("baseline", "reference", "spectral_ball_dist"),
)


@contextmanager
def pilot_dispatch(adapter):
    """Install variable-rank pilot updates and diagnostics in this process only."""
    from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial
    optimizer = host.SpELPGDSameProjection
    previous_update = spectral_cluster_radial.compute_cluster_radial_update
    previous_record = optimizer._record_update_info

    def record(instance, info, **kwargs):
        # The legacy logger starts with ranks 1-4; the pilot has no such cap.
        if info.cluster_dimension:
            instance.cluster_rank_counts.setdefault(info.cluster_dimension, 0)
        return previous_record(instance, info, **kwargs)

    spectral_cluster_radial.compute_cluster_radial_update = adapter
    optimizer._record_update_info = record
    try:
        yield
    finally:
        spectral_cluster_radial.compute_cluster_radial_update = previous_update
        optimizer._record_update_info = previous_record


def mcsd_direction(m, u, v, steps):
    p = -tangent(m.float(), u, v)
    length = torch.linalg.vector_norm(p)
    if float(length) <= 1e-12:
        return torch.zeros_like(p)
    return tangent(host.msign(p / length, steps=steps).float(), u, v)


def candidate(m, cache, radius, alpha, q, method, steps=8):
    u, s, v = cache.u_basis[:, :q], cache.s_basis[:q], cache.v_basis[:, :q]
    if method == "control" or q == 1:
        return mcsd_direction(m, u[:, :1], v[:, :1], steps), 0.
    n = torch.linalg.vector_norm(m)
    if float(n) <= 1e-12:
        return torch.zeros_like(m), 0.
    scale = 1.5 if method == "a2_1p5" else 1.
    f = model_project(-scale*m/n, u, s, v, radius, alpha)
    if method != "kernel1":
        return f, 0.
    t = mcsd_direction(m, u, v, steps)
    nt = torch.linalg.vector_norm(t)
    accepted = float(nt) > 1e-12 and float((m*t).sum()) <= 0
    return (f + torch.linalg.vector_norm(f)*t/nt if accepted else f), float(accepted)


class PilotAdapter:
    def __init__(self, method, solver, window=2e-4, iterations=10):
        if method not in ("control", "a2", "a2_1p5", "kernel1") or solver not in ("exact", "warm_block8"):
            raise ValueError("invalid pilot method or solver")
        self.method, self.solver, self.window, self.iterations = method, solver, window, iterations
        self.counts, self.sums, self.q_hist = Counter(), defaultdict(float), Counter()
        self.last = None

    def direction(self, m, cache, radius, alpha, q, steps):
        return candidate(m, cache, radius, alpha, q, self.method, steps)

    def trial_return(self, y, u, s, vh, radius):
        factor = radius/s[0]
        return y*factor, s*factor

    def decompose(self, x, previous=None):
        if self.solver == "exact":
            self.counts["full_svd"] += 1
            return host.accurate_svd_fp32(x)
        self.counts["small_ritz_svd"] += 1
        return warm_block8_ritz(x, iterations=self.iterations,
                               initial_v=None if previous is None else previous.v_basis)

    def cache(self, u, s, vh, radius):
        values = s.detach().float().clone()
        values.clamp_(max=radius)
        values[0] = radius
        exact = self.solver == "exact"
        cache = host._spectral_point_cache_from_decomposition(u, values, vh,
            is_exact=exact, bounds_certified=exact, **({} if exact else dict(gap_lower=0.,gap_upper=1.)))
        return replace(cache, u_basis=u.detach().float().clone(),
                       v_basis=vh.T.detach().float().clone(), s_basis=values)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        matrix = args[0] if args else kwargs["W"]
        with utils.fp32_matmul_precision("highest"), torch.autocast(device_type=matrix.device.type, enabled=False):
            return self._update(*args, **kwargs)

    def _update(self, W, M, target_radius, power_iteration_steps, msign_steps, *,
                 effective_lr, current_spectral_cache=None, tp_group=None, **kw):
        from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import _tp_world_and_rank
        if _tp_world_and_rank(tp_group)[0] != 1:
            raise ValueError("pilot requires TP=1")
        self.last = None
        r = min(W.shape)
        expected = r if self.solver == "exact" else min(8,r)
        cache = current_spectral_cache
        hit = (cache is not None and cache.is_exact == (self.solver == "exact")
               and cache.s_basis is not None and len(cache.s_basis) == expected
               and host._spectral_point_cache_is_compatible(cache,W,target_radius))
        if not hit:
            u,s,vh = self.decompose(W)
            if float(s[0]) <= 0:
                raise ValueError("zero initialization has no radial return")
            W.copy_(W.float()*(target_radius/s[0]))
            cache = self.cache(u,s*(target_radius/s[0]),vh,target_radius)
            self.counts["bootstrap_solves"] += 1
        relative = (cache.s_basis[0]-cache.s_basis)/cache.s_basis[0]
        active = max(1,int((relative <= self.window).sum()))
        q = 1 if self.method == "control" else active
        near = self.method != "control" and q > 1
        censored = len(cache.s_basis) < r and active == len(cache.s_basis)
        if effective_lr <= 0:
            return torch.zeros_like(W),host.SpELPGDUpdateInfo(
                branch="zero_lr",rel_gap=cache.rel_gap,sigma1=cache.sigma1,gap_sigma1=cache.sigma1,
                sigma2=cache.sigma2,pre_retract_bias=0.,trial_sigma=cache.sigma1,post_retract_bias=0.,
                projection_mode="cluster_radial",projection_rank=0,gap_probed=True,
                spectral_cache_hit=hit,next_spectral_cache=cache)
        x,m = W.float(),M.float()
        alpha = float(effective_lr)
        d,accepted = self.direction(m,cache,target_radius,alpha,q,msign_steps)
        y = x+alpha*d
        u,s,vh = self.decompose(y,cache)
        if float(s[0]) <= 0:
            raise ValueError("zero trial has no radial return")
        returned, returned_values = self.trial_return(y,u,s,vh,target_radius)
        next_cache = self.cache(u,returned_values,vh,target_radius)
        trial_length = float(torch.linalg.vector_norm(alpha*d))/target_radius
        radial = float(torch.linalg.vector_norm(returned-y))/target_radius
        displacement = returned-x
        h = sym(cache.u_basis[:,:q].T@d@cache.v_basis[:,:q])
        model = torch.diag(cache.s_basis[:q])+alpha*h
        # One diagnostic eigenvalue solve per near call; no accept/reject gate.
        model_error = abs(float(torch.linalg.eigvalsh(model)[-1])-target_radius)/target_radius if near else 0.
        record = dict(q=q,active=active,near=near,censored=censored,kernel_accepted=accepted,
            trial_length=trial_length,returned_length=float(torch.linalg.vector_norm(displacement))/target_radius,
            radial_length=radial,model_error=model_error,
            momentum_trial=float((m*(alpha*d)).sum()),momentum_returned=float((m*displacement).sum()),
            current_gap=cache.rel_gap,returned_gap=next_cache.rel_gap)
        self.counts["updates"] += 1
        self.counts["trial_solves"] += 1
        self.counts["near"] += near
        self.counts["censored_window"] += censored
        self.counts["window_gt4"] += active > 4
        self.counts["window_gt8"] += active > 8
        self.counts["kernel_accepted"] += int(accepted)
        self.q_hist[str(q)] += 1
        for key in ("trial_length","returned_length","radial_length","model_error","momentum_trial","momentum_returned"):
            self.sums[key] += record[key]
        self.last = dict(record=record,current=x,momentum=m,direction=d,trial=y,returned=returned,
                         cache=cache,radius=target_radius,alpha=alpha)
        info=host.SpELPGDUpdateInfo(branch="pgd" if near else "spel",rel_gap=cache.rel_gap,
            sigma1=cache.sigma1,gap_sigma1=cache.sigma1,sigma2=cache.sigma2,pre_retract_bias=0.,
            trial_sigma=float(s[0]),post_retract_bias=0. if self.solver=="exact" else float("nan"),
            projection_mode="cluster_radial",projection_rank=q if near else 0,gap_probed=True,
            next_v=next_cache.v1,spectral_cache_hit=hit,next_spectral_cache=next_cache,
            cluster_dimension=q if near else 0,cluster_cap_overflow=censored,cluster_step_multiplier=1.,
            radial_change_ratio=radial/max(trial_length,1e-30),
            approximate_mode="warm_block8_ritz" if self.solver!="exact" else "none",
            approximate_spectral_iterations=self.iterations if self.solver!="exact" else 0,
            approximate_subspace_dimension=expected if self.solver!="exact" else 0)
        return (x-returned)/alpha,info

    def report(self):
        return dict(method=self.method,solver=self.solver,window=self.window,
                    geometry_precision="fp32_highest",lmo_precision="bf16_ns8",
                    counts=dict(self.counts),sums=dict(self.sums),q_hist=dict(self.q_hist))


if __name__ == "__main__":
    import argparse,json
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(dict(zip(("method","solver","optimizer"),ARMS[args.arm]))))
