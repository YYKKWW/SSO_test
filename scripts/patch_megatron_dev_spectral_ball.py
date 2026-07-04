#!/usr/bin/env python3
"""Expose SpectralBall/SSO as spectral_ball_dist in a Megatron-LM dev checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


SPECTRAL_BALL_CONFIG_TO_KWARGS = '''

def _spectral_ball_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to SpectralBall constructor kwargs."""
    model_cfg = model_chunks[0].config
    kwargs = _kwargs_from_config(SpectralBall, "spectral_ball", config)
    kwargs["momentum_beta"] = config.spectral_ball_momentum
    kwargs["weight_decay_method"] = "decoupled" if config.decoupled_weight_decay else "l2"
    kwargs["fp32_matmul_prec"] = "medium"
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_cfg)
    kwargs["is_fc1_fn"] = lambda p: getattr(p, "is_fc1", False)
    kwargs["fc1_split_shapes"] = (
        (model_cfg.ffn_hidden_size, model_cfg.ffn_hidden_size)
        if getattr(model_cfg, "gated_linear_unit", False)
        else None
    )
    kwargs["is_grouped_moe_fn"] = lambda p: getattr(p, "is_grouped_moe", False)
    kwargs["pg_collection"] = pg_collection
    kwargs["tp_mode"] = "duplicated"
    return kwargs
'''


SPECTRAL_BALL_REGISTRY_ENTRY = '''        "spectral_ball": EmergingOptimizerEntry(
            optimizer_cls=SpectralBall,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_spectral_ball_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
'''


SPECTRAL_BALL_CONFIG_FIELDS = '''    # Spectral Ball / SSO.
    spectral_ball_momentum: float = 0.9
    """Momentum coefficient for Spectral Ball optimizer."""

    spectral_ball_use_nesterov: bool = True
    """Whether to use Nesterov-style momentum in Spectral Ball."""

    spectral_ball_split_qkv: bool = True
    """Whether to split QKV parameters for Spectral Ball optimizer."""

    spectral_ball_qkv_split_mode: str = "component"
    """QKV split mode for Spectral Ball optimizer: component, group, or head."""

    spectral_ball_split_fc1: bool = True
    """Whether to split FC1 gate/up projections for Spectral Ball optimizer."""

    spectral_ball_split_moe_experts: bool = True
    """Whether to split grouped MoE expert weights for Spectral Ball optimizer."""

    spectral_ball_msign_steps: int = 8
    """Number of matrix-sign iterations for Spectral Ball."""

    spectral_ball_solver: str = 'bisection'
    """Solver for the Spectral Ball Lagrange multiplier."""

    spectral_ball_solver_tolerance_f: float = 2e-4
    """Function tolerance for the Spectral Ball solver."""

    spectral_ball_solver_max_iterations: int = 20
    """Maximum iterations for the Spectral Ball solver."""

    spectral_ball_radius_mode: str = 'spectral_mup'
    """Target spectral radius mode for Spectral Ball."""

    spectral_ball_radius_scaler: float = 1.0
    """Multiplier for the Spectral Ball target radius."""

    spectral_ball_power_iteration_steps: int = 10
    """Power iteration steps used to estimate spectral norm in Spectral Ball."""

    spectral_ball_scale_mode: str = 'spectral_mup'
    """Update scaling mode for Spectral Ball."""

    spectral_ball_retract_mode: str = 'hard'
    """Spectral Ball retraction mode: hard or dynamic."""

    spectral_ball_retract_alpha: float = 0.05
    """Step size for dynamic Spectral Ball retraction."""

'''


SPECTRAL_BALL_CLI_ARGS = '''    group.add_argument('--spectral-ball-momentum', type=float, default=0.9, help='Momentum coefficient for Spectral Ball optimizer')
    group.add_argument('--spectral-ball-use-nesterov', action='store_true', default=True, help='Use Nesterov-style momentum in Spectral Ball')
    group.add_argument(
        '--spectral-ball-no-split-qkv',
        action='store_false',
        default=True,
        dest='spectral_ball_split_qkv',
        help='Disable QKV splitting for Spectral Ball optimizer',
    )
    group.add_argument(
        '--spectral-ball-qkv-split-mode',
        type=str,
        default='component',
        choices=['component', 'group', 'head'],
        help='QKV split mode for Spectral Ball: component, group, or head',
    )
    group.add_argument(
        '--spectral-ball-no-split-fc1',
        action='store_false',
        default=True,
        dest='spectral_ball_split_fc1',
        help='Disable FC1 gate/up splitting for Spectral Ball optimizer',
    )
    group.add_argument(
        '--spectral-ball-no-split-moe-experts',
        action='store_false',
        default=True,
        dest='spectral_ball_split_moe_experts',
        help='Disable grouped MoE expert splitting for Spectral Ball optimizer',
    )
    group.add_argument('--spectral-ball-msign-steps', type=int, default=8, help='Matrix-sign iterations for Spectral Ball')
    group.add_argument('--spectral-ball-solver', type=str, default='bisection', choices=['bisection'], help='Solver for Spectral Ball Lagrange multiplier')
    group.add_argument('--spectral-ball-solver-tolerance-f', type=float, default=2e-4, help='Function tolerance for Spectral Ball solver')
    group.add_argument('--spectral-ball-solver-max-iterations', type=int, default=20, help='Maximum Spectral Ball solver iterations')
    group.add_argument(
        '--spectral-ball-radius-mode',
        type=str,
        default='spectral_mup',
        choices=['spectral_mup', 'identity', 'initialize'],
        help='Target radius mode for Spectral Ball',
    )
    group.add_argument('--spectral-ball-radius-scaler', type=float, default=1.0, help='Target radius multiplier for Spectral Ball')
    group.add_argument('--spectral-ball-power-iteration-steps', type=int, default=10, help='Power iteration steps for Spectral Ball spectral norm')
    group.add_argument(
        '--spectral-ball-scale-mode',
        type=str,
        default='spectral_mup',
        choices=['align_adamw_rms', 'spectral_mup', 'shape_scaling'],
        help='Update scaling mode for Spectral Ball',
    )
    group.add_argument(
        '--spectral-ball-retract-mode',
        type=str,
        default='hard',
        choices=['hard', 'dynamic'],
        help='Retraction mode for Spectral Ball',
    )
    group.add_argument('--spectral-ball-retract-alpha', type=float, default=0.05, help='Dynamic Spectral Ball retraction step size')

'''


def patch_file(path: Path, transforms: list[tuple[str, str]]) -> bool:
    text = path.read_text()
    original = text
    for old, new in transforms:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"Marker not found in {path}: {old[:100]!r}")
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text)
        return True
    return False


def patch_emerging(root: Path) -> bool:
    path = root / "megatron/core/optimizer/emerging_optimizers.py"
    text = path.read_text()
    original = text

    if "from emerging_optimizers.orthogonalized_optimizers.spectral_ball import SpectralBall" not in text:
        text = text.replace(
            "    from emerging_optimizers.orthogonalized_optimizers.spel import SpEL\n",
            "    from emerging_optimizers.orthogonalized_optimizers.spel import SpEL\n"
            "    from emerging_optimizers.orthogonalized_optimizers.spectral_ball import SpectralBall\n",
            1,
        )

    if "SpectralBall = object" not in text:
        text = text.replace("    SpEL = object\n", "    SpEL = object\n    SpectralBall = object\n", 1)

    if "def _spectral_ball_config_to_kwargs" not in text:
        marker = "\n\ndef _spel_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:\n"
        if marker not in text:
            raise RuntimeError("Could not find _spel_config_to_kwargs marker")
        text = text.replace(marker, SPECTRAL_BALL_CONFIG_TO_KWARGS + marker, 1)

    if '"spectral_ball": EmergingOptimizerEntry' not in text:
        marker = '        "spel": EmergingOptimizerEntry(\n'
        if marker not in text:
            raise RuntimeError("Could not find SpEL registry marker")
        text = text.replace(marker, SPECTRAL_BALL_REGISTRY_ENTRY + marker, 1)

    if text != original:
        path.write_text(text)
        return True
    return False


def patch_optimizer_config(root: Path) -> bool:
    path = root / "megatron/core/optimizer/optimizer_config.py"
    text = path.read_text()
    if "spectral_ball_momentum" in text:
        return False
    marker = "    # SpEL.\n"
    if marker not in text:
        raise RuntimeError("Could not find SpEL config marker")
    path.write_text(text.replace(marker, SPECTRAL_BALL_CONFIG_FIELDS + marker, 1))
    return True


def patch_arguments(root: Path) -> bool:
    path = root / "megatron/training/arguments.py"
    text = path.read_text()
    original = text

    if "'spectral_ball'" not in text:
        text = text.replace(
            "choices=['adam', 'sgd', 'muon', 'dist_muon', 'spel', 'spel_dist', 'lion', 'soap', 'adaptive_muon'],",
            "choices=['adam', 'sgd', 'muon', 'dist_muon', 'spectral_ball', 'spectral_ball_dist', 'spel', 'spel_dist', 'lion', 'soap', 'adaptive_muon'],",
            1,
        )

    if "--spectral-ball-momentum" not in text:
        marker = "    group.add_argument('--spel-momentum', type=float, default=0.9, help='Momentum coefficient for SpEL optimizer')\n"
        if marker not in text:
            raise RuntimeError("Could not find SpEL CLI marker")
        text = text.replace(marker, SPECTRAL_BALL_CLI_ARGS + marker, 1)

    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("megatron_root", type=Path)
    args = parser.parse_args()

    root = args.megatron_root.expanduser().resolve()
    changed = {
        "emerging_optimizers.py": patch_emerging(root),
        "optimizer_config.py": patch_optimizer_config(root),
        "arguments.py": patch_arguments(root),
    }
    for name, did_change in changed.items():
        print(f"{name}: {'patched' if did_change else 'already ok'}")


if __name__ == "__main__":
    main()
