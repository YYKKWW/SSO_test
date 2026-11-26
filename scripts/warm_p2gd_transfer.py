"""Persistent block8/K10 transfer of the small-step P2GD-type direction.

Only the near direction differs between arms. No full matrix SVD, norm-SVD,
accuracy fallback, or returned-step coefficient search is used online.
The finite numerical cluster is not the exact tangent cone in general.
"""
from collections import Counter
from dataclasses import replace
import math

import torch

from direct_top_repair import DirectTopAdapter
from kernel_lmo_h20 import mcsd_direction
from local_step_aware_hybrid import sym

MODES = ("cone", "top1", "ambient")


def near_direction(m, d0, u, v, mode, ratio):
    """Normalize raw momentum, scale once, then optionally remove top block.

    Do not normalize again after repair. All three policies use the same raw
    direction and retain zero D0 as zero (no artificial fallback length).
    """
    if mode not in MODES or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("valid mode and finite positive ratio required")
    norm = torch.linalg.vector_norm(m)
    if float(norm) == 0:
        return torch.zeros_like(m)
    z = -m / norm * (ratio * torch.linalg.vector_norm(d0))
    if mode == "ambient":
        return z
    if mode == "top1":
        normal = u[:, :1] @ v[:, :1].T
        return z - (normal * z).sum() * normal
    a = sym(u.T @ z @ v)
    values, vectors = torch.linalg.eigh(a)
    values = values.clamp(max=0)
    values[-1] = 0
    b = (vectors * values) @ vectors.T
    return z + u @ (b - a) @ v.T


class WarmP2GDAdapter(DirectTopAdapter):
    def __init__(self, mode, *, ratio=.075, capacity=6, window=2e-4):
        if mode not in MODES or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("invalid warm-transfer policy or ratio")
        if capacity not in (2, 4, 6, 8):
            raise ValueError("capacity must fit the eight-vector cache")
        super().__init__("none", "warm_block8", window)
        self.near_mode, self.ratio, self.capacity = mode, ratio, capacity
        self.used_q = Counter()
        self._previous_near = {}

    def direction(self, m, cache, radius, alpha, q, steps):
        self._d0 = mcsd_direction(m, cache.u_basis[:, :1], cache.v_basis[:, :1], steps)
        if q == 1:
            return self._d0, 0.
        used = min(q, self.capacity)
        return near_direction(m, self._d0, cache.u_basis[:, :used],
                              cache.v_basis[:, :used], self.near_mode, self.ratio), 0.

    def _update(self, *args, **kwargs):
        update, info = super()._update(*args, **kwargs)
        if self.last is None:
            return update, info
        state, c = self.last, self.last["cache"]
        record = state["record"]
        near, active = record["near"], record["active"]
        used = min(active, self.capacity) if self.near_mode == "cone" else 1
        overflow = near and active > self.capacity
        key = (kwargs.get("param_name"), kwargs.get("component_label"), tuple(state["current"].shape))
        self.counts["near_repeats"] += int(near and self._previous_near.get(key, False))
        self._previous_near[key] = near
        self.counts["direction_cap_overflow"] += int(overflow)
        if near:
            self.used_q[str(used)] += 1
        record.update(near_mode=self.near_mode, raw_momentum_ratio=self.ratio,
                      direction_dimension=used, direction_cap_overflow=overflow,
                      actual_elr_matched=False)
        # Retain approximate current-X spectrum for sparse, post-training audits.
        state.update(cached_u=c.u_basis, cached_s=c.s_basis, cached_v=c.v_basis)
        return update, replace(info, cluster_dimension=used if near else 0,
                               cluster_cap_overflow=overflow or record["censored"])

    def report(self):
        return dict(super().report(), study="warm_p2gd_transfer", method=self.near_mode,
                    raw_momentum_ratio=self.ratio, direction_capacity=self.capacity,
                    direction_q_hist=dict(self.used_q), iterations=self.iterations,
                    near_normalization="frobenius_momentum_then_ratio_times_d0_fro",
                    zero_d0_reference="zero", return_rule="approximate_radial_both_branches",
                    coefficient_search=False, actual_elr_matched=False,
                    online_full_svd_fallback=False, post_correction_rescaling=False)
