"""Exact-cache comparison: MCSD-TP, spectral-slack repair, and normalized PGD.

PGD's trial Frobenius length is rho times the same-step MCSD-TP D0 length.
This explicitly chosen variable step is NOT the historical pgd_lr_scale.
Only PGD uses a full spectral-sphere projection; other arms return radially.
Opt-in ambient_return="radial" keeps the same near direction but scales the
whole trial matrix. That arm is ambient-radial, not projected gradient descent.
"""
from dataclasses import replace

import torch

from direct_top_repair import DirectTopAdapter, repair
from kernel_lmo_h20 import mcsd_direction

RATIOS = {"pgd_half": .5, "pgd_full": 1.}
MODES = ("none", "finite_step", *RATIOS)
ARMS = tuple((mode, "exact", "spel_pgd_dist") for mode in MODES) + (
    ("baseline", "reference", "muon_ball_dist"),
    ("baseline", "reference", "spectral_ball_dist"),
)


def pgd_reference_norm(d0):
    n = torch.linalg.vector_norm(d0)
    # A vanishing tangent direction must not disable the ambient fallback.
    return n if float(n)>1e-12 else n.new_tensor(min(d0.shape)**.5)


def matched_pgd_direction(m, d0, ratio):
    if ratio <= 0:
        raise ValueError("positive PGD trial-length ratio required")
    n = torch.linalg.vector_norm(m)
    if float(n) == 0:
        return torch.zeros_like(m)
    return (-m/n)*(ratio*pgd_reference_norm(d0))


def sphere_from_svd(u, s, vh, radius):
    """Frobenius projection onto the sphere, not the closed spectral ball."""
    if radius <= 0:
        raise ValueError("positive radius required")
    values = s.clamp(max=radius).clone()
    values[0] = radius
    return (u*values)@vh, values


def shared_candidates(m, d0, u, s, v, radius, alpha, q):
    directions = {
        "none": d0,
        "finite_step": repair(d0,u[:,:q],s[:q],v[:,:q],radius,alpha,"finite_step"),
    }
    directions.update({mode:matched_pgd_direction(m,d0,ratio) if q>1 else d0
                       for mode,ratio in RATIOS.items()})
    return directions


class DirectPKAdapter(DirectTopAdapter):
    def __init__(self, mode, solver="exact", window=2e-4, *, ambient_return="sphere_projection"):
        if mode not in (*MODES, "tp_half"):
            raise ValueError("invalid direct-PK method")
        if ambient_return not in ("sphere_projection", "radial"):
            raise ValueError("ambient_return must be sphere_projection or radial")
        if solver != "exact" and (ambient_return != "radial" or mode == "finite_step"):
            raise ValueError("warm spectra are supported only for uncorrected radial arms")
        if mode == "tp_half" and ambient_return != "radial":
            raise ValueError("near-step attenuation is a radial-only control")
        super().__init__("none",solver,window)
        self.method = mode
        self.ambient_return = ambient_return
        self._is_ambient = False
        self._is_pgd = False
        self._is_attenuated = False
        self._projection_clipped = 0
        self._previous = {}

    def direction(self, m, cache, radius, alpha, q, steps):
        self._d0 = mcsd_direction(m,cache.u_basis[:,:1],cache.v_basis[:,:1],steps)
        self._is_ambient = q>1 and self.method in RATIOS
        self._is_pgd = self._is_ambient and self.ambient_return == "sphere_projection"
        self._is_attenuated = q>1 and self.method == "tp_half"
        if self._is_attenuated:
            return .5*self._d0,0.
        if self._is_ambient:
            return matched_pgd_direction(m,self._d0,RATIOS[self.method]),0.
        if self.method == "finite_step":
            return repair(self._d0,cache.u_basis[:,:q],cache.s_basis[:q],
                          cache.v_basis[:,:q],radius,alpha,"finite_step"),0.
        return self._d0,0.

    def trial_return(self, y, u, s, vh, radius):
        self._projection_clipped = 0
        if not self._is_pgd:
            return super().trial_return(y,u,s,vh,radius)
        self._projection_clipped = int((s>radius).sum())
        return sphere_from_svd(u,s,vh,radius)

    def _update(self, *args, **kwargs):
        update,info = super()._update(*args,**kwargs)
        if self.last is None:
            return update,info
        state = self.last
        record = state["record"]
        pgd = self._is_pgd
        record.update(return_kind="sphere_projection" if pgd else "radial",
                      pgd=pgd,projection_clipped=self._projection_clipped,
                      ambient_near=self._is_ambient,
                      tp_near_attenuated=self._is_attenuated,
                      # The inherited radial_length is the general return correction.
                      return_correction_length=record["radial_length"])
        key=(kwargs.get("param_name"),kwargs.get("component_label"),tuple(state["current"].shape))
        previous_near,previous_pgd,previous_ambient=self._previous.get(key,(False,False,False))
        self._previous[key]=(record["near"],pgd,self._is_ambient)
        self.counts["near_repeats"] += int(record["near"] and previous_near)
        self.counts["ambient_near_calls"] += int(self._is_ambient)
        self.counts["ambient_near_repeats"] += int(self._is_ambient and previous_ambient)
        self.counts["ambient_radial_calls"] += int(self._is_ambient and not pgd)
        self.counts["tp_near_attenuation_calls"] += int(self._is_attenuated)
        self.counts["pgd_calls"] += int(pgd)
        self.counts["pgd_zero_d0_reference"] += int(pgd and float(state["d0"].norm())<=1e-12)
        self.counts["pgd_repeats"] += int(pgd and previous_pgd)
        self.counts["projection_clipped_total"] += self._projection_clipped
        self.counts["pgd_returned_plateau"] += int(pgd and info.next_spectral_cache.rel_gap==0.)
        for prefix,enabled in (("near_",record["near"]),("pgd_",pgd)):
            if enabled:
                for name in ("trial_length","returned_length","radial_length","momentum_trial","momentum_returned"):
                    self.sums[prefix+name] += record[name]
        if pgd:
            info=replace(info,branch="pgd",projection_mode="full_svd",
                         projection_rank=self._projection_clipped,
                         pgd_projection_clipped_count=self._projection_clipped,
                         pgd_post_rel_gap=info.next_spectral_cache.rel_gap)
        else:
            info=replace(info,branch="spel")
        return update,info

    def report(self):
        result=super().report()
        result.update(study="direct_top_pgd_pk",seed=2345,
                      pgd_normalization="frobenius_before_projection",
                      pgd_trial_length_ratio=RATIOS.get(self.method,0.),
                      pgd_length_reference="current MCSD-TP D0 before return",
                      zero_d0_reference="sqrt(min(matrix shape)) if D0 norm <= 1e-12",
                      return_rule="full_svd_projection_on_pgd_else_radial",
                      old_radial_length_field="general return correction, including PGD")
        result.update(ambient_return=self.ambient_return,
                      ambient_trial_length_ratio=RATIOS.get(self.method,0.),
                      ambient_direction_normalization="frobenius_before_return")
        if self.ambient_return == "radial":
            result.update(return_rule="radial_both_branches",pgd_trial_length_ratio=0.,
                          pgd_normalization="not_applicable",
                          configured_mode=self.method,
                          method=self.method.replace("pgd_", "ambient_radial_"))
        result.update(trial_length_matching="Frobenius norm relative to same-state D0",
                      actual_elr_matched=False,
                      approximate_iterations=self.iterations if self.solver=="warm_block8" else None,
                      online_full_svd_fallback=False)
        if self.method == "tp_half":
            result.update(method="mcsd_tp_near_half",tp_near_trial_length_ratio=.5,
                          ambient_direction_normalization="not_applicable",
                          trial_length_matching="near D=0.5*D0; safe D=D0; no direction change")
        return result
