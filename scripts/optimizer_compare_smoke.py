#!/usr/bin/env python
"""Small GPU smoke test for Megatron SpEL and SpectralBall optimizers.

This intentionally avoids Megatron's GPT pretraining entrypoint because this
workspace does not include the full GPT builder/model stack.  It still calls
the Megatron optimizer wrappers on a tiny Megatron-like module so QKV head
splitting, FC1 splitting, nonlinear Adam params, backward, and step are tested.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_DIR = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_DIR) not in sys.path:
    sys.path.insert(0, str(MEGATRON_DIR))

# Keep smoke tests independent of Triton/torch.compile availability.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def install_optimizer_only_megatron_namespace() -> None:
    """Bypass broken package __init__ files for optimizer-only imports.

    This Megatron checkout is missing the full GPT/model stack.  Importing
    megatron.core or megatron.core.transformer normally executes package
    __init__ files that import those missing modules.  The optimizer wrappers
    only need specific submodules, so expose the relevant directories as
    namespace packages for this smoke test.
    """

    package_paths = {
        "megatron": MEGATRON_DIR / "megatron",
        "megatron.core": MEGATRON_DIR / "megatron" / "core",
        "megatron.core.distributed": MEGATRON_DIR / "megatron" / "core" / "distributed",
        "megatron.core.transformer": MEGATRON_DIR / "megatron" / "core" / "transformer",
    }
    for package_name, package_path in package_paths.items():
        module = sys.modules.get(package_name)
        if module is None or not hasattr(module, "__path__"):
            module = ModuleType(package_name)
            module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
            module.__package__ = package_name
            sys.modules[package_name] = module

        if "." in package_name:
            parent_name, attr_name = package_name.rsplit(".", 1)
            parent = sys.modules[parent_name]
            setattr(parent, attr_name, module)


def install_absl_logging_fallback() -> None:
    """Provide the tiny absl.logging surface used by optimizer files if absent."""

    try:
        from absl import logging as _unused_logging  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import logging as py_logging

    absl_module = ModuleType("absl")
    logging_module = ModuleType("absl.logging")
    for name in ("debug", "info", "warning", "error", "exception", "critical", "fatal"):
        setattr(logging_module, name, getattr(py_logging, name))
    setattr(absl_module, "logging", logging_module)
    sys.modules.setdefault("absl", absl_module)
    sys.modules.setdefault("absl.logging", logging_module)


def install_emerging_optimizer_namespace() -> None:
    """Avoid importing orthogonalized_optimizers.__init__, which needs Triton."""

    package_name = "emerging_optimizers.orthogonalized_optimizers"
    package_path = MEGATRON_DIR / "emerging_optimizers" / "orthogonalized_optimizers"
    module = sys.modules.get(package_name)
    if module is None or not hasattr(module, "__path__"):
        module = ModuleType(package_name)
        module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        module.__package__ = package_name
        sys.modules[package_name] = module

    import emerging_optimizers

    emerging_optimizers.orthogonalized_optimizers = module


def install_triton_kernel_fallback() -> None:
    """Let muon_utils import without Triton; patched NS step uses torch.addmm."""

    import emerging_optimizers

    module_name = "emerging_optimizers.triton_kernels"
    module = sys.modules.get(module_name)
    if module is None:
        module = ModuleType(module_name)
        module.HAS_TRITON_340 = False
        sys.modules[module_name] = module
    emerging_optimizers.triton_kernels = module


def patch_newton_schulz_to_torch_fallback() -> None:
    """Use the non-Triton Newton-Schulz step for this tiny smoke test."""

    muon_utils = importlib.import_module(
        "emerging_optimizers.orthogonalized_optimizers.muon_utils"
    )

    def newton_schulz_step_torch(
        x: torch.Tensor,
        a: float,
        b: float,
        c: float,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ) -> torch.Tensor:
        gram = x @ x.mT
        if tp_group is not None:
            torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        update = torch.addmm(gram, gram, gram, alpha=c, beta=b)
        return torch.addmm(x, update, x, alpha=1.0, beta=a)

    muon_utils.newton_schulz_step_tsyrk = newton_schulz_step_torch


def import_smoke_optimizers():
    """Import smoke optimizers without package-level Triton imports."""

    install_absl_logging_fallback()
    install_emerging_optimizer_namespace()
    install_triton_kernel_fallback()
    patch_newton_schulz_to_torch_fallback()
    from emerging_optimizers.orthogonalized_optimizers.spectral_ball import SpectralBall
    from emerging_optimizers.orthogonalized_optimizers.spel import SpEL
    from emerging_optimizers.orthogonalized_optimizers.spel_pgd_same_projection import (
        SpELPGDSameProjection,
    )

    package = sys.modules["emerging_optimizers.orthogonalized_optimizers"]
    package.SpectralBall = SpectralBall
    package.SpEL = SpEL
    package.SpELPGDSameProjection = SpELPGDSameProjection
    return SpectralBall, SpEL, SpELPGDSameProjection


def bare_optimizer_name(optimizer_name: str) -> str:
    """Map legacy layer-wise names to the emerging optimizer registry key."""
    return optimizer_name[: -len("_dist")] if optimizer_name.endswith("_dist") else optimizer_name


class TinyMegatronLikeModel(nn.Module):
    """Tiny module with parameter names that exercise Megatron optimizer splits."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_query_groups: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_query_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_query_groups")

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.kv_channels = hidden_size // num_attention_heads
        self.kv_total = num_query_groups * self.kv_channels

        qkv_out = hidden_size + 2 * self.kv_total
        self.linear_qkv = nn.Linear(hidden_size, qkv_out, bias=False)
        self.kv_proj = nn.Linear(2 * self.kv_total, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_fc1 = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, 1)

        # Keep output in the Adam side of the Megatron wrappers.
        self.output.weight.is_embedding_or_output_parameter = True

        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            kv_channels=self.kv_channels,
            gated_linear_unit=True,
            ffn_hidden_size=ffn_hidden_size,
            context_parallel_size=1,
            num_moe_experts=0,
            expert_model_parallel_size=1,
            moe_ffn_hidden_size=ffn_hidden_size,
        )
        self.ddp_config = SimpleNamespace(
            use_megatron_fsdp=False,
            use_distributed_optimizer=False,
            num_distributed_optimizer_instances=1,
        )

        self._tag_parameters()
        self._init_weights()

    @property
    def qkv_split_shapes(self) -> tuple[int, int, int]:
        q_dim_per_group = (
            self.num_attention_heads // self.num_query_groups
        ) * self.kv_channels
        return (q_dim_per_group, self.kv_channels, self.kv_channels)

    @property
    def fc1_split_shapes(self) -> tuple[int, int]:
        return (self.ffn_hidden_size, self.ffn_hidden_size)

    def _tag_parameters(self) -> None:
        for name, param in self.named_parameters():
            param.param_name = name
            if name == "linear_qkv.weight":
                param.is_qkv = True
            if name == "linear_fc1.weight":
                param.is_fc1 = True

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "norm.weight" in name:
                nn.init.ones_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        qkv = self.linear_qkv(x)
        q, k, v = torch.split(
            qkv, [self.hidden_size, self.kv_total, self.kv_total], dim=-1
        )
        h = torch.tanh(q + self.kv_proj(torch.cat([k, v], dim=-1)))
        h = self.norm(h)
        gate, up = self.linear_fc1(h).chunk(2, dim=-1)
        h = self.linear_fc2(F.silu(gate) * up)
        pred = self.output(h).squeeze(-1)
        return F.mse_loss(pred, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["direct", "wrapper", "both"], default="direct")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["spectral_ball_dist", "spel"],
        choices=[
            "spectral_ball",
            "spectral_ball_dist",
            "spel",
            "spel_dist",
            "spel_pgd",
            "spel_pgd_dist",
        ],
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--ffn-hidden-size", type=int, default=64)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-query-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--msign-steps", type=int, default=2)
    parser.add_argument("--power-iteration-steps", type=int, default=3)
    parser.add_argument("--solver-max-iterations", type=int, default=10)
    parser.add_argument("--solver-tolerance-f", type=float, default=2e-4)
    parser.add_argument("--radius-mode", default="spectral_mup")
    parser.add_argument("--scale-mode", default="spectral_mup")
    parser.add_argument("--retract-mode", default="hard")
    parser.add_argument(
        "--spel-projection-mode",
        default="retraction",
        choices=["retraction", "exact", "topk"],
    )
    parser.add_argument("--spel-projection-rank", type=int, default=1)
    parser.add_argument("--qkv-split-mode", default="head", choices=["component", "group", "head"])
    parser.add_argument("--spel-pgd-branch-mode", default="auto", choices=["auto", "spel", "pgd"])
    parser.add_argument("--spel-pgd-gap-threshold-rel", type=float, default=5e-3)
    parser.add_argument("--spel-pgd-sigma2-power-iteration-steps", type=int, default=3)
    parser.add_argument(
        "--spel-pgd-direction-normalization",
        default="none",
        choices=["none", "fro"],
    )
    parser.add_argument(
        "--spel-pgd-projection-mode",
        default="fallback_exact",
        choices=[
            "fallback_exact",
            "fallback_retraction",
            "fallback_topk",
            "shared_exact",
            "shared_retraction",
            "shared_topk",
        ],
    )
    parser.add_argument("--spel-pgd-projection-rank", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="outputs/optimizer_compare_smoke.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_arg)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return device


def init_distributed_for_wrapper() -> None:
    install_optimizer_only_megatron_namespace()
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if backend == "gloo":
            rendezvous_path = Path(tempfile.gettempdir()) / f"optimizer_smoke_{os.getpid()}"
            if rendezvous_path.exists():
                rendezvous_path.unlink()
            init_method = f"file:///{rendezvous_path.as_posix()}"
        else:
            init_method = "tcp://127.0.0.1:29591"
        dist.init_process_group(backend=backend, init_method=init_method, rank=0, world_size=1)
    else:
        dist.init_process_group(backend=backend, init_method="env://")

    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            num_distributed_optimizer_instances=1,
            create_gloo_process_groups=False,
        )


def cleanup_distributed() -> None:
    if not dist.is_initialized():
        return
    try:
        install_optimizer_only_megatron_namespace()
        from megatron.core import parallel_state

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
    except Exception as exc:
        print(f"warning: could not destroy Megatron model parallel state: {exc}", flush=True)
    finally:
        dist.destroy_process_group()


def make_model(args: argparse.Namespace, device: torch.device) -> TinyMegatronLikeModel:
    model = TinyMegatronLikeModel(
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
    )
    return model.to(device)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(args.batch_size, args.seq_length, args.hidden_size, device=device)
    target = torch.sin(x[..., 0]) + 0.1 * torch.cos(x[..., 1])
    return x, target


def param_stats(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 2 and (
                name in {
                    "linear_qkv.weight",
                    "linear_fc1.weight",
                    "linear_fc2.weight",
                    "kv_proj.weight",
                }
            ):
                stats[f"{name}:spectral_norm"] = float(
                    torch.linalg.matrix_norm(param.detach().float(), ord=2).item()
                )
    return stats


def finite_or_raise(model: nn.Module, label: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"{label}: non-finite parameter detected in {name}")
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"{label}: non-finite gradient detected in {name}")


def run_direct(
    optimizer_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    SpectralBall, SpEL, SpELPGDSameProjection = import_smoke_optimizers()
    bare_name = bare_optimizer_name(optimizer_name)

    set_seed(args.seed)
    model = make_model(args, device)
    x, target = make_batch(args, device)

    linear_params = []
    nonlinear_params = []
    for _name, param in model.named_parameters():
        is_output = getattr(param, "is_embedding_or_output_parameter", False)
        if param.dim() == 2 and not is_output:
            linear_params.append(param)
        else:
            nonlinear_params.append(param)

    common = dict(
        lr=args.lr,
        momentum_beta=args.momentum,
        weight_decay=args.weight_decay,
        use_nesterov=True,
        weight_decay_method="decoupled",
        fp32_matmul_prec="medium",
        power_iteration_steps=args.power_iteration_steps,
        msign_steps=args.msign_steps,
        radius_mode=args.radius_mode,
        scale_mode=args.scale_mode,
        retract_mode=args.retract_mode,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=model.qkv_split_shapes,
        qkv_split_mode=args.qkv_split_mode,
        split_fc1=True,
        is_fc1_fn=lambda p: getattr(p, "is_fc1", False),
        fc1_split_shapes=model.fc1_split_shapes,
        split_moe_experts=False,
        pg_collection=None,
        tp_mode="duplicated",
    )
    param_groups = [{"params": linear_params, "wd_mult": 0.0}]
    if bare_name == "spel":
        opt = SpEL(
            param_groups,
            projection_mode=args.spel_projection_mode,
            projection_rank=args.spel_projection_rank,
            **common,
        )
    elif bare_name == "spel_pgd":
        opt = SpELPGDSameProjection(
            param_groups,
            branch_mode=args.spel_pgd_branch_mode,
            gap_threshold_rel=args.spel_pgd_gap_threshold_rel,
            sigma2_power_iteration_steps=args.spel_pgd_sigma2_power_iteration_steps,
            pgd_direction_normalization=args.spel_pgd_direction_normalization,
            projection_mode=args.spel_pgd_projection_mode,
            projection_rank=args.spel_pgd_projection_rank,
            **common,
        )
    elif bare_name == "spectral_ball":
        opt = SpectralBall(
            param_groups,
            solver="bisection",
            solver_tolerance_f=args.solver_tolerance_f,
            solver_max_iterations=args.solver_max_iterations,
            radius_scaler=1.0,
            **common,
        )
    else:
        raise ValueError(f"Unsupported optimizer for smoke test: {optimizer_name}")
    adam = torch.optim.AdamW(nonlinear_params, lr=args.lr, weight_decay=args.weight_decay)

    losses = []
    start = time.time()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        loss = model(x, target)
        loss.backward()
        finite_or_raise(model, f"direct:{optimizer_name}")
        opt.step()
        adam.step()
        losses.append(float(loss.detach().cpu().item()))
    elapsed = time.time() - start

    final_loss = float(model(x, target).detach().cpu().item())
    return {
        "mode": "direct",
        "optimizer": optimizer_name,
        "ok": math.isfinite(final_loss),
        "losses": losses,
        "final_loss": final_loss,
        "elapsed_sec": elapsed,
        "param_stats": param_stats(model),
    }


def make_optimizer_config(optimizer_name: str, args: argparse.Namespace):
    install_optimizer_only_megatron_namespace()
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    bare_name = bare_optimizer_name(optimizer_name)
    return OptimizerConfig(
        optimizer=bare_name,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        clip_grad=args.clip_grad,
        fp16=False,
        bf16=False,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=optimizer_name.endswith("_dist"),
        spectral_ball_momentum=args.momentum,
        spectral_ball_use_nesterov=True,
        spectral_ball_split_qkv=True,
        spectral_ball_qkv_split_mode=args.qkv_split_mode,
        spectral_ball_split_fc1=True,
        spectral_ball_split_moe_experts=False,
        spectral_ball_msign_steps=args.msign_steps,
        spectral_ball_solver="bisection",
        spectral_ball_solver_tolerance_f=args.solver_tolerance_f,
        spectral_ball_solver_max_iterations=args.solver_max_iterations,
        spectral_ball_radius_mode=args.radius_mode,
        spectral_ball_radius_scaler=1.0,
        spectral_ball_power_iteration_steps=args.power_iteration_steps,
        spectral_ball_scale_mode=args.scale_mode,
        spectral_ball_retract_mode=args.retract_mode,
        spel_momentum=args.momentum,
        spel_use_nesterov=True,
        spel_split_qkv=True,
        spel_qkv_split_mode=args.qkv_split_mode,
        spel_split_fc1=True,
        spel_split_moe_experts=False,
        spel_msign_steps=args.msign_steps,
        spel_radius_mode=args.radius_mode,
        spel_power_iteration_steps=args.power_iteration_steps,
        spel_scale_mode=args.scale_mode,
        spel_retract_mode=args.retract_mode,
        spel_projection_mode=args.spel_projection_mode,
        spel_projection_rank=args.spel_projection_rank,
        spel_pgd_momentum=args.momentum,
        spel_pgd_use_nesterov=True,
        spel_pgd_split_qkv=True,
        spel_pgd_qkv_split_mode=args.qkv_split_mode,
        spel_pgd_split_fc1=True,
        spel_pgd_split_moe_experts=False,
        spel_pgd_msign_steps=args.msign_steps,
        spel_pgd_radius_mode=args.radius_mode,
        spel_pgd_power_iteration_steps=args.power_iteration_steps,
        spel_pgd_scale_mode=args.scale_mode,
        spel_pgd_retract_mode=args.retract_mode,
        spel_pgd_use_pgd_fallback=True,
        spel_pgd_branch_mode=args.spel_pgd_branch_mode,
        spel_pgd_gap_threshold_rel=args.spel_pgd_gap_threshold_rel,
        spel_pgd_sigma2_power_iteration_steps=args.spel_pgd_sigma2_power_iteration_steps,
        spel_pgd_pgd_direction_normalization=args.spel_pgd_direction_normalization,
        spel_pgd_projection_mode=args.spel_pgd_projection_mode,
        spel_pgd_projection_rank=args.spel_pgd_projection_rank,
    )


def run_wrapper(
    optimizer_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    import_smoke_optimizers()
    install_optimizer_only_megatron_namespace()
    init_distributed_for_wrapper()
    from megatron.core.optimizer import _get_megatron_emerging_optimizer

    set_seed(args.seed)
    model = make_model(args, device)
    x, target = make_batch(args, device)
    config = make_optimizer_config(optimizer_name, args)
    optimizer = _get_megatron_emerging_optimizer(config, [model], config_overrides={})

    losses = []
    step_returns = []
    start = time.time()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, target)
        loss.backward()
        finite_or_raise(model, f"wrapper:{optimizer_name}")
        step_result = optimizer.step()
        step_returns.append(str(step_result))
        losses.append(float(loss.detach().cpu().item()))
    elapsed = time.time() - start

    final_loss = float(model(x, target).detach().cpu().item())
    return {
        "mode": "wrapper",
        "optimizer": optimizer_name,
        "ok": math.isfinite(final_loss),
        "losses": losses,
        "final_loss": final_loss,
        "step_returns": step_returns,
        "elapsed_sec": elapsed,
        "param_stats": param_stats(model),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    results: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "args": vars(args),
        "tests": [],
    }

    try:
        for optimizer_name in args.optimizers:
            if args.mode in ("direct", "both"):
                result = run_direct(optimizer_name, args, device)
                print(json.dumps(result, indent=2), flush=True)
                results["tests"].append(result)
            if args.mode in ("wrapper", "both"):
                result = run_wrapper(optimizer_name, args, device)
                print(json.dumps(result, indent=2), flush=True)
                results["tests"].append(result)
    finally:
        cleanup_distributed()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)

    failures = [r for r in results["tests"] if not r.get("ok")]
    if failures:
        raise SystemExit(f"{len(failures)} optimizer smoke tests failed")


if __name__ == "__main__":
    main()
