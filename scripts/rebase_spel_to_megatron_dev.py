#!/usr/bin/env python3
"""Apply the local SpEL patch set to a clean Megatron-LM dev checkout.

The patch is intentionally narrow: it keeps NVIDIA Megatron's current emerging
optimizer plumbing and adds SpEL as another registry entry.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    text = read(path)
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


def insert_after(path: Path, anchor: str, insert: str) -> None:
    text = read(path)
    if insert.strip() in text:
        return
    if anchor not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {anchor[:120]!r}")
    write(path, text.replace(anchor, anchor + insert, 1))


def copy_vendored_emerging_optimizers(source: Path | None, target: Path) -> None:
    if source is None:
        return
    src = source / "emerging_optimizers"
    if not src.exists():
        src = source / "Megatron-LM" / "emerging_optimizers"
    if not src.exists():
        raise RuntimeError(f"Cannot find emerging_optimizers under {source}")
    dst = target / "emerging_optimizers"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def patch_optimizer_package_detection(root: Path) -> None:
    path = root / "megatron/core/optimizer/__init__.py"
    old = """try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _eo_ver = tuple(int(x) for x in _pkg_version('emerging-optimizers').split('.')[:2])
except (ImportError, PackageNotFoundError):
    _eo_ver = (0, 0)

HAVE_EMERGING_OPTIMIZERS = _eo_ver >= (0, 2)
"""
    new = """try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _eo_ver = tuple(int(x) for x in _pkg_version('emerging-optimizers').split('.')[:2])
except (ImportError, PackageNotFoundError):
    try:
        import emerging_optimizers as _vendored_emerging_optimizers  # noqa: F401

        # The SpEL tree vendors emerging_optimizers in the repository instead of
        # installing a wheel with dist-info metadata.
        _eo_ver = (0, 2)
    except ImportError:
        _eo_ver = (0, 0)

HAVE_EMERGING_OPTIMIZERS = _eo_ver >= (0, 2)
"""
    replace_once(path, old, new)

    replace_once(
        path,
        """if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion
""",
        """if HAVE_EMERGING_OPTIMIZERS:
    try:
        from emerging_optimizers.scalar_optimizers import Lion
    except ImportError:
        Lion = None
""",
    )

    old_tag = """            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
"""
    new_tag = """            param.param_name = name
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if 'linear_fc1.weight' in name and len(param.shape) == 2:
                param.is_fc1 = True
            if 'experts.weight1' in name or 'experts.weight2' in name:
                param.is_grouped_moe = True
                try:
                    param.num_local_experts = (
                        model_chunk.config.num_moe_experts
                        // model_chunk.config.expert_model_parallel_size
                    )
                    param.moe_ffn_hidden_size = model_chunk.config.moe_ffn_hidden_size
                    param.is_gated = model_chunk.config.gated_linear_unit
                except Exception:
                    param.is_grouped_moe = False
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
"""
    replace_once(path, old_tag, new_tag)


def patch_emerging_optimizer_registry(root: Path) -> None:
    path = root / "megatron/core/optimizer/emerging_optimizers.py"
    old = """try:
    from emerging_optimizers import registry
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object
"""
    new = """try:
    import emerging_optimizers as _vendored_eopt  # noqa: F401

    try:
        from emerging_optimizers import registry
    except ImportError:
        registry = None

    from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
        OrthogonalizedOptimizer,
    )
    from emerging_optimizers.orthogonalized_optimizers.spel import SpEL

    try:
        from emerging_optimizers.orthogonalized_optimizers import AdaptiveMuon
    except ImportError:
        AdaptiveMuon = object

    try:
        from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
    except ImportError:
        from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor

    try:
        from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
            NSCoeffT,
            newton_schulz_tp,
        )
    except ImportError:
        from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
            newton_schulz_tp,
        )

        NSCoeffT = Literal["simple", "quintic", "polar_express", "aol"]

    try:
        from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    except ImportError:
        pass
    try:
        from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import
    except ImportError:
        pass

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object
    SpEL = object
"""
    replace_once(path, old, new)

    insert_after(
        path,
        """def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    \"\"\"Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs.\"\"\"
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs
""",
        """

def _spel_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    \"\"\"Convert OptimizerConfig to SpEL constructor kwargs.\"\"\"
    model_cfg = model_chunks[0].config
    kwargs = _kwargs_from_config(SpEL, "spel", config)
    kwargs["momentum_beta"] = config.spel_momentum
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
""",
    )

    old_registry = """        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
"""
    new_registry = """        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "spel": EmergingOptimizerEntry(
            optimizer_cls=SpEL,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_spel_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
"""
    replace_once(path, old_registry, new_registry)

    replace_once(
        path,
        """if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
""",
        """if HAVE_EMERGING_OPTIMIZERS and registry is not None:
    for eopt_name in registry.get_optimizer_name_list():
""",
    )


def patch_optimizer_config(root: Path) -> None:
    path = root / "megatron/core/optimizer/optimizer_config.py"
    anchor = """    muon_scalar_optimizer: str = 'adam'
    \"\"\"Optimizer for nonlinear parameters (embeddings, biases, norms) when using muon.
    One of 'adam' or 'lion'. Defaults to 'adam'.\"\"\"

"""
    insert = """    # SpEL.
    spel_momentum: float = 0.9
    \"\"\"Momentum coefficient for SpEL optimizer.\"\"\"

    spel_use_nesterov: bool = True
    \"\"\"Whether to use Nesterov-style momentum in SpEL.\"\"\"

    spel_split_qkv: bool = True
    \"\"\"Whether to split QKV parameters for SpEL optimizer.\"\"\"

    spel_qkv_split_mode: str = "component"
    \"\"\"QKV split mode for SpEL optimizer: component, group, or head.\"\"\"

    spel_split_fc1: bool = True
    \"\"\"Whether to split FC1 gate/up projections for SpEL optimizer.\"\"\"

    spel_split_moe_experts: bool = True
    \"\"\"Whether to split grouped MoE expert weights for SpEL optimizer.\"\"\"

    spel_msign_steps: int = 8
    \"\"\"Number of Newton-Schulz iterations for the SpEL matrix sign step.\"\"\"

    spel_radius_mode: str = 'spectral_mup'
    \"\"\"Target spectral radius mode for SpEL.\"\"\"

    spel_power_iteration_steps: int = 10
    \"\"\"Power iteration steps used to estimate spectral norm in SpEL.\"\"\"

    spel_scale_mode: str = 'spectral_mup'
    \"\"\"Update scaling mode for SpEL.\"\"\"

    spel_retract_mode: str = 'hard'
    \"\"\"SpEL retraction mode: hard or dynamic.\"\"\"

    spel_retract_alpha: float = 0.05
    \"\"\"Step size for dynamic SpEL retraction.\"\"\"

"""
    insert_after(path, anchor, insert)


def patch_arguments(root: Path) -> None:
    path = root / "megatron/training/arguments.py"
    replace_once(
        path,
        "choices=['adam', 'sgd', 'muon', 'dist_muon', 'lion', 'soap', 'adaptive_muon'],",
        "choices=['adam', 'sgd', 'muon', 'dist_muon', 'spel', 'spel_dist', 'lion', 'soap', 'adaptive_muon'],",
    )

    insert_after(
        path,
        """        if args.optimizer == 'dist_muon':
            warn_rank_0(
                "optimizer='dist_muon' is deprecated. "
                "Use --optimizer muon --use-distributed-optimizer instead."
            )
            args.optimizer = 'muon'
            args.use_layer_wise_distributed_optimizer = True
""",
        """
        if args.optimizer.endswith('_dist'):
            bare_optimizer_name = args.optimizer[: -len('_dist')]
            warn_rank_0(
                f"optimizer='{args.optimizer}' maps to --optimizer {bare_optimizer_name} "
                "with layer-wise distributed optimizer enabled."
            )
            args.optimizer = bare_optimizer_name
            args.use_layer_wise_distributed_optimizer = True
""",
    )

    insert_after(
        path,
        """    group.add_argument(
        '--muon-scalar-optimizer',
        type=str,
        default='adam',
        choices=['adam', 'lion'],
        help='Optimizer for scalar parameters (embeddings, biases, norms) '
        'when using muon. Defaults to adam.',
    )
""",
        """
    group.add_argument('--spel-momentum', type=float, default=0.9, help='Momentum coefficient for SpEL optimizer')
    group.add_argument('--spel-use-nesterov', action='store_true', default=True, help='Use Nesterov-style momentum in SpEL')
    group.add_argument(
        '--spel-no-split-qkv',
        action='store_false',
        default=True,
        dest='spel_split_qkv',
        help='Disable QKV splitting for SpEL optimizer',
    )
    group.add_argument(
        '--spel-qkv-split-mode',
        type=str,
        default='component',
        choices=['component', 'group', 'head'],
        help='QKV split mode for SpEL: component, group, or head',
    )
    group.add_argument(
        '--spel-no-split-fc1',
        action='store_false',
        default=True,
        dest='spel_split_fc1',
        help='Disable FC1 gate/up splitting for SpEL optimizer',
    )
    group.add_argument(
        '--spel-no-split-moe-experts',
        action='store_false',
        default=True,
        dest='spel_split_moe_experts',
        help='Disable grouped MoE expert splitting for SpEL optimizer',
    )
    group.add_argument('--spel-msign-steps', type=int, default=8, help='Newton-Schulz iterations for SpEL matrix sign')
    group.add_argument(
        '--spel-radius-mode',
        type=str,
        default='spectral_mup',
        choices=['spectral_mup', 'identity', 'initialize'],
        help='Target radius mode for SpEL',
    )
    group.add_argument('--spel-power-iteration-steps', type=int, default=10, help='Power iteration steps for SpEL spectral norm')
    group.add_argument(
        '--spel-scale-mode',
        type=str,
        default='spectral_mup',
        choices=['align_adamw_rms', 'spectral_mup', 'shape_scaling'],
        help='Update scaling mode for SpEL',
    )
    group.add_argument(
        '--spel-retract-mode',
        type=str,
        default='hard',
        choices=['hard', 'dynamic'],
        help='Retraction mode for SpEL',
    )
    group.add_argument('--spel-retract-alpha', type=float, default=0.05, help='Dynamic SpEL retraction step size')
""",
    )


def patch_utils(root: Path) -> None:
    path = root / "megatron/core/utils.py"
    insert_after(
        path,
        """    std = sigma / (math.sqrt(multiplier * num_layers) * math.sqrt(width_mult))
    return functools.partial(torch.nn.init.normal_, mean=0.0, std=std)
""",
        """

def spectral_mup_init_method_normal(sigma):
    \"\"\"Spectral MuP initialization for 2D linear weights.\"\"\"

    def init_(tensor):
        with torch.no_grad():
            torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
            if len(tensor.shape) != 2:
                return tensor

            d_out, d_in = tensor.shape
            if d_out > 50000 or d_in > 50000:
                return tensor

            tensor_fp32 = tensor.detach().float()
            spectral_norm = torch.linalg.matrix_norm(tensor_fp32, ord=2)
            if torch.isfinite(spectral_norm) and spectral_norm > 0:
                tensor.mul_(math.sqrt(d_out / d_in) / spectral_norm)
            return tensor

    return init_


def get_qkv_init_method(config):
    \"\"\"Return an initializer that can split fused QKV weights.\"\"\"
    if not getattr(config, "split_qkv_init", True):
        return config.init_method

    split_mode = getattr(config, "split_qkv_init_mode", "group")
    qkv_split_shapes = [
        config.num_attention_heads // config.num_query_groups * config.kv_channels,
        config.kv_channels,
        config.kv_channels,
    ]

    def inner(tensor):
        with torch.no_grad():
            out_dim, in_dim = tensor.shape
            split_sum = sum(qkv_split_shapes)
            num_groups = out_dim // split_sum
            tensor_view = tensor.view(num_groups, split_sum, in_dim)

            if split_mode == "group":
                for group_idx in range(num_groups):
                    for part in torch.split(tensor_view[group_idx], qkv_split_shapes, dim=0):
                        config.init_method(part)
            elif split_mode == "component":
                for part in torch.split(tensor_view, qkv_split_shapes, dim=1):
                    config.init_method(part.reshape(-1, in_dim))
            elif split_mode == "head":
                heads_per_group = config.num_attention_heads // config.num_query_groups
                for group_idx in range(num_groups):
                    q_part, k_part, v_part = torch.split(
                        tensor_view[group_idx], qkv_split_shapes, dim=0
                    )
                    for head in q_part.view(heads_per_group, config.kv_channels, in_dim):
                        config.init_method(head)
                    config.init_method(k_part)
                    config.init_method(v_part)
            else:
                raise ValueError(f"Invalid split_qkv_init_mode: {split_mode}")

    return inner


def get_fc1_init_method(config):
    \"\"\"Return an initializer that can split gated FC1 into gate/up blocks.\"\"\"
    if not getattr(config, "split_fc1_init", True):
        return config.init_method

    def inner(tensor):
        with torch.no_grad():
            if getattr(config, "gated_linear_unit", False) and tensor.shape[0] % 2 == 0:
                gate, up = tensor.chunk(2, dim=0)
                config.init_method(gate)
                config.init_method(up)
            else:
                config.init_method(tensor)

    return inner
""",
    )


def patch_transformer_config(root: Path) -> None:
    path = root / "megatron/core/transformer/transformer_config.py"
    replace_once(
        path,
        """    mup_scaled_init_method_normal,
    scaled_init_method_normal,
)
""",
        """    mup_scaled_init_method_normal,
    scaled_init_method_normal,
    spectral_mup_init_method_normal,
)
""",
    )

    insert_after(
        path,
        """    embedding_init_method_std: Optional[float] = None
    \"\"\"
    Standard deviation of the zero mean normal for the default initialization method for the 
    embedding layer. If None, will be set to init_method_std. Setting this to a value around
    1.0 may avoid loss spikes in training. Setting this to any value will also skip applying
    weight decay on embedding weights to avoid shrinkage towards zero.
    See https://arxiv.org/abs/2312.16903 for more details.
    \"\"\"
""",
        """
    split_qkv_init: bool = True
    \"\"\"Whether to initialize fused QKV components separately.\"\"\"

    split_qkv_init_mode: str = "group"
    \"\"\"QKV initialization split mode: group, component, or head.\"\"\"

    split_fc1_init: bool = True
    \"\"\"Whether to initialize gated FC1 gate/up projections separately.\"\"\"

    split_expert_init: bool = True
    \"\"\"Whether to initialize grouped MoE expert weights separately.\"\"\"

    spectral_mup_init: bool = False
    \"\"\"Use spectral MuP initialization for 2D linear weights.\"\"\"
""",
    )

    replace_once(
        path,
        """        if self.init_method is None:
            if self.use_mup:
                # MuP: scale std by 1/sqrt(width_mult).
                self.init_method = init_method_normal(
                    self.init_method_std / math.sqrt(self.mup_width_mult)
                )
            else:
                self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            if self.use_mup:
                # MuP: depth and width scaling for output layers.
                self.output_layer_init_method = mup_scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    self.mup_width_mult,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )
            else:
                self.output_layer_init_method = scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )
""",
        """        if self.spectral_mup_init and not self.use_cpu_initialization:
            warnings.warn(
                "spectral_mup_init is most faithful with use_cpu_initialization=True.",
                UserWarning,
            )

        if self.init_method is None:
            if self.spectral_mup_init:
                init_std = (
                    self.init_method_std / math.sqrt(self.mup_width_mult)
                    if self.use_mup
                    else self.init_method_std
                )
                self.init_method = spectral_mup_init_method_normal(init_std)
            elif self.use_mup:
                # MuP: scale std by 1/sqrt(width_mult).
                self.init_method = init_method_normal(
                    self.init_method_std / math.sqrt(self.mup_width_mult)
                )
            else:
                self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            if self.spectral_mup_init:
                self.output_layer_init_method = spectral_mup_init_method_normal(
                    self.init_method_std
                )
            elif self.use_mup:
                # MuP: depth and width scaling for output layers.
                self.output_layer_init_method = mup_scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    self.mup_width_mult,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )
            else:
                self.output_layer_init_method = scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )
""",
    )


def patch_transformer_modules(root: Path) -> None:
    attention = root / "megatron/core/transformer/attention.py"
    replace_once(
        attention,
        """    get_pg_rank,
    get_pg_size,
    is_fa_min_version,
""",
        """    get_pg_rank,
    get_pg_size,
    get_qkv_init_method,
    is_fa_min_version,
""",
    )
    replace_once(
        attention,
        "            init_method=not_none(self.config.init_method),\n",
        "            init_method=not_none(get_qkv_init_method(self.config)),\n",
    )

    mlp = root / "megatron/core/transformer/mlp.py"
    replace_once(
        mlp,
        """    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
""",
        """    get_fc1_init_method,
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
""",
    )
    replace_once(
        mlp,
        "            init_method=not_none(self.config.init_method),\n",
        "            init_method=not_none(get_fc1_init_method(self.config)),\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    root = args.target.expanduser().resolve()
    source = args.source.expanduser().resolve() if args.source else None
    if not (root / "megatron").exists():
        raise RuntimeError(f"{root} does not look like a Megatron-LM checkout")

    copy_vendored_emerging_optimizers(source, root)
    patch_optimizer_package_detection(root)
    patch_emerging_optimizer_registry(root)
    patch_optimizer_config(root)
    patch_arguments(root)
    patch_utils(root)
    patch_transformer_config(root)
    patch_transformer_modules(root)
    print(f"Applied SpEL rebase patch to {root}")


if __name__ == "__main__":
    main()
