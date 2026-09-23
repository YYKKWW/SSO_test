"""MCSD and MCSD-TP for three matrix constraint geometries.

This optimizer owns only complete logical matrices.  A physical QKV or gated
FC1 parameter is partitioned into logical components before each update.
Auxiliary parameters belong to a separate AdamW optimizer.
"""

import logging
from typing import Iterable, Literal

import torch

from .manifold_geometry import (
    GeometryName,
    GeometryParameters,
    constraint_defect,
    initialize_matrix,
    partial_polar,
    return_to_manifold,
    spectral_tangent_normal,
    tangent_project,
)
from .manifold_layout import components_for_parameter, validate_layout
from .manifold_polar import polar_express_msign
from .manifold_spectral import leading_triplet, spectral_return_pi

MethodName = Literal["mcsd", "mcsd_tp"]
logger = logging.getLogger(__name__)


@torch.no_grad()
def prepare_manifold_model(
    model: torch.nn.Module,
    *,
    geometry: GeometryName,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attention_heads: int,
    num_query_groups: int,
    kv_channels: int,
    initialize: bool,
    polar_mode: Literal["exact", "ns"] = "exact",
) -> list[dict]:
    """Attach component layout and optionally initialize before BF16 conversion.

    On a fresh run, save an FP32 copy for Megatron's master-weight creation.
    A resumed run must instead load master weights and radii from checkpoint.
    """
    manifest = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim != 2:
            continue
        components = components_for_parameter(
            name,
            tuple(param.shape),
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
        )
        if not components:
            continue
        validate_layout(components, param.shape[0])
        specs = []
        for component in components:
            item = {"label": component.label, "rows": component.rows}
            if initialize:
                if param.dtype != torch.float32:
                    raise ValueError("manifold initialization requires FP32 raw weights")
                raw = component.read(param.data)
                initial, values = initialize_matrix(raw, geometry, polar_mode=polar_mode)
                component.write(param.data, initial)
                item.update(radius=values.radius, scale=values.scale)
            specs.append(item)
            manifest.append({"parameter": name, "shape": tuple(param.shape), **item})
        param.manifold_spec = specs
        if initialize:
            param.manifold_initial_fp32 = param.data.detach().clone()
    if not manifest:
        raise ValueError("no supported hidden matrices found for manifold optimizer")
    return manifest


class ManifoldMCSD(torch.optim.Optimizer):
    """Current-sample ambient EMA with exact partial-polar LMO and return."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float,
        geometry: GeometryName,
        method: MethodName,
        momentum_beta: float = 0.95,
        gap_warning: float = 1e-4,
        lmo_mode: Literal["exact", "ns"] = "ns",
        stiefel_return_mode: Literal["exact", "ns"] = "ns",
        spectral_solver: Literal["exact", "pi_topk"] = "pi_topk",
        power_steps: int = 10,
        topk_rank: int = 8,
        spectral_audit_interval: int = 200,
    ) -> None:
        if geometry not in ("frobenius", "spectral", "stiefel"):
            raise ValueError(f"unsupported geometry: {geometry}")
        if method not in ("mcsd", "mcsd_tp"):
            raise ValueError(f"unsupported method: {method}")
        if lr < 0 or not 0 <= momentum_beta < 1:
            raise ValueError("invalid learning rate or momentum")
        if lmo_mode not in ("exact", "ns") or stiefel_return_mode not in ("exact", "ns"):
            raise ValueError("unknown matrix sign implementation")
        if (
            spectral_solver not in ("exact", "pi_topk")
            or power_steps < 1
            or topk_rank < 1
            or spectral_audit_interval < 1
        ):
            raise ValueError("invalid spectral solver settings")
        defaults = dict(
            lr=lr,
            geometry=geometry,
            method=method,
            momentum_beta=momentum_beta,
            gap_warning=gap_warning,
            lmo_mode=lmo_mode,
            stiefel_return_mode=stiefel_return_mode,
            spectral_solver=spectral_solver,
            power_steps=power_steps,
            topk_rank=topk_rank,
            spectral_audit_interval=spectral_audit_interval,
            weight_decay=0.0,
            wd_mult=0.0,
        )
        super().__init__(params, defaults)
        self.last_step_stats: dict[str, float | int] = {}

    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Initialize states also when Megatron prepares sharded checkpoints."""
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            state = self.state[param]
            if "components" in state:
                continue
            specs = getattr(param, "manifold_spec", None)
            if not specs:
                raise ValueError("matrix lacks manifold component metadata")
            components = []
            for spec in specs:
                rows = tuple(spec["rows"])
                component = {
                    "label": spec["label"],
                    "rows": rows,
                    "radius": float(spec.get("radius", float("nan"))),
                    "scale": float(spec.get("scale", float("nan"))),
                    "momentum": torch.zeros(
                        (len(rows), param.shape[1]), dtype=torch.float32, device=param.device
                    ),
                    "update_count": 0,
                }
                components.append(component)
            state["components"] = components

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        decisions = 0
        warnings = 0
        max_defect = 0.0
        min_gap = float("inf")
        audited = 0
        defect_measured = 0
        max_update_count = 0
        for group in self.param_groups:
            if group.get("wd_mult", 0.0) != 0 or group.get("weight_decay", 0.0) != 0:
                raise ValueError("constrained matrices must have zero weight decay")
            self._init_group(group)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dtype != torch.float32 or param.ndim != 2:
                    raise ValueError("ManifoldMCSD requires FP32 master matrices")
                grad = param.grad.detach().float()
                for component in self.state[param]["components"]:
                    rows = torch.tensor(component["rows"], device=param.device)
                    w = param.data.index_select(0, rows)
                    g = grad.index_select(0, rows)
                    values = GeometryParameters(
                        group["geometry"], component["radius"], component["scale"]
                    )
                    if not torch.isfinite(torch.tensor((values.radius, values.scale))).all():
                        raise ValueError("missing initialized radii; load optimizer checkpoint")
                    momentum = component["momentum"]
                    momentum.mul_(group["momentum_beta"]).add_(
                        g, alpha=1.0 - group["momentum_beta"]
                    )
                    use_spectral_pi = (
                        group["geometry"] == "spectral"
                        and group["spectral_solver"] == "pi_topk"
                    )
                    if group["geometry"] == "spectral":
                        if use_spectral_pi:
                            _, u, v = leading_triplet(w, steps=group["power_steps"])
                            normal = torch.outer(u, v)
                        else:
                            normal = spectral_tangent_normal(w)

                        def project_spectral(vector):
                            return vector - torch.sum(normal * vector) * normal

                        q = project_spectral(momentum)
                    else:
                        q = tangent_project(w, momentum, values)
                    direction = -(
                        partial_polar(q)
                        if group["lmo_mode"] == "exact"
                        else polar_express_msign(q)
                    )
                    if group["method"] == "mcsd_tp":
                        direction = (
                            project_spectral(direction)
                            if group["geometry"] == "spectral"
                            else tangent_project(w, direction, values)
                        )
                    trial = w + group["lr"] * values.scale * direction
                    return_mode = (
                        "radial"
                        if group["geometry"] == "spectral" and group["method"] == "mcsd_tp"
                        else "projection"
                    )
                    if use_spectral_pi:
                        updated = spectral_return_pi(
                            trial,
                            values.radius,
                            mode=return_mode,
                            steps=group["power_steps"],
                            rank=group["topk_rank"],
                        )
                        info = None
                    else:
                        updated, info = return_to_manifold(
                            trial,
                            values,
                            mode=return_mode,
                            gap_stop=group["gap_warning"],
                            polar_mode=group["stiefel_return_mode"],
                        )
                    param.data.index_copy_(0, rows, updated)
                    component["update_count"] += 1
                    max_update_count = max(max_update_count, component["update_count"])
                    decisions += 1
                    if use_spectral_pi and (
                        component["update_count"] % group["spectral_audit_interval"] == 0
                    ):
                        sigma = torch.linalg.svdvals(updated)
                        measured_gap = (1.0 - sigma[1] / sigma[0]).item()
                        min_gap = min(min_gap, measured_gap)
                        warnings += int(measured_gap <= group["gap_warning"])
                        max_defect = max(
                            max_defect, abs(sigma[0].item() / values.radius - 1.0)
                        )
                        audited += 1
                        defect_measured += 1
                    elif not use_spectral_pi:
                        max_defect = max(max_defect, constraint_defect(updated, values))
                        defect_measured += 1
                    if info is not None and info.returned_relative_gap is not None:
                        min_gap = min(min_gap, info.returned_relative_gap)
                        warnings += int(info.top_gap_below_warning)
        self.last_step_stats = {
            "step": max_update_count,
            "components": decisions,
            "spectral_near_tie": warnings,
            "spectral_gap_audits": audited,
            "minimum_relative_gap": min_gap if min_gap != float("inf") else float("nan"),
            "maximum_constraint_defect": max_defect if defect_measured else float("nan"),
        }
        if audited and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.info("manifold spectral audit: %s", self.last_step_stats)
        return loss
