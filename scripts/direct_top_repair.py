"""Same MCSD-TP D0, three optional top-block repairs, shared radial return.

The multiple-root repair uses the full selected cone, not the older
selected-face LMO pipeline. The finite-step repair retains spectral slack.
Neither repair is a full-matrix sphere projection or a loss-based safeguard.
"""
import torch

from kernel_lmo_h20 import PilotAdapter, mcsd_direction
from local_step_aware_hybrid import model_project, sym

MODES = ("none", "multiple", "finite_step")


def repair(d0, u, s, v, radius, alpha, mode):
    """Do not renormalize or rescale after this operation."""
    if mode not in MODES or alpha <= 0 or radius <= 0:
        raise ValueError("valid mode and positive radius/step required")
    if mode == "none" or len(s) == 1:
        return d0
    values = torch.full_like(s, radius) if mode == "multiple" else s
    return model_project(d0, u, values, v, radius, alpha)


def shared_directions(m, spectrum, radius, alpha, q, *, exact_lmo=False):
    """Compute D0 once, then reuse that exact tensor in all counterfactuals."""
    u, s, v = spectrum
    if exact_lmo:
        from local_step_aware_hybrid import linear_direction
        d0, _ = linear_direction(m, u[:, :1], v[:, :1])
    else:
        d0 = mcsd_direction(m.float(), u[:, :1].float(), v[:, :1].float(), 8).to(m)
    return d0, {mode: repair(d0, u[:, :q], s[:q], v[:, :q], radius, alpha, mode)
                for mode in MODES}


class DirectTopAdapter(PilotAdapter):
    def __init__(self, mode, solver="exact", window=2e-4):
        if mode not in MODES or solver not in ("exact", "warm_block8"):
            raise ValueError("invalid direct-top mode or spectral solver")
        if solver != "exact" and mode != "none":
            raise ValueError("top repairs require the full FP32 spectrum")
        super().__init__("a2", solver, window=window)
        self.method = mode
        self._d0 = None

    def direction(self, m, cache, radius, alpha, q, steps):
        self._d0 = mcsd_direction(m, cache.u_basis[:, :1], cache.v_basis[:, :1], steps)
        d = repair(self._d0, cache.u_basis[:, :q], cache.s_basis[:q],
                   cache.v_basis[:, :q], radius, alpha, self.method)
        return d, 0.

    def _update(self, *args, **kwargs):
        self._d0 = None
        result = super()._update(*args, **kwargs)
        if self.last is None:
            return result
        state = self.last
        d0, d = self._d0, state["direction"]
        alpha, radius = state["alpha"], state["radius"]
        denom = float(d0.norm())
        correction = float((d-d0).norm())
        fields = dict(d0_trial_length=alpha*denom/radius,
                      correction_length=alpha*correction/radius,
                      correction_ratio=correction/max(denom, 1e-30),
                      direction_ratio=float(d.norm())/max(denom, 1e-30))
        state["record"].update(fields)
        state["d0"] = d0.detach()
        self.counts["repair_calls"] += int(state["record"]["near"] and self.method in ("multiple", "finite_step"))
        for key, value in fields.items():
            self.sums[key] += value
            if state["record"]["near"]:
                self.sums["near_"+key] += value
        return result

    def report(self):
        return dict(super().report(), study="direct_top_repair", d0="mcsd_tp_bf16_ns8",
                    near_counter="window eligibility, including the uncorrected control",
                    post_correction_rescaling=False)
