#!/usr/bin/env python3
"""Small import smoke test for the SpEL Megatron rebase."""

from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from emerging_optimizers.orthogonalized_optimizers.spel import SpEL


def main() -> None:
    names = sorted(_EMERGING_OPTIMIZERS.keys())
    print("emerging_optimizers:", names)
    assert "spel" in _EMERGING_OPTIMIZERS, names
    assert _EMERGING_OPTIMIZERS["spel"].optimizer_cls is SpEL
    print("spel_class:", SpEL.__name__)

    import sys
    from megatron.training.arguments import parse_args

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "pretrain_gpt.py",
            "--optimizer",
            "spel_dist",
            "--spel-momentum",
            "0.9",
            "--spel-use-nesterov",
            "--spel-qkv-split-mode",
            "head",
            "--spel-msign-steps",
            "8",
            "--spel-radius-mode",
            "spectral_mup",
            "--spel-power-iteration-steps",
            "10",
            "--spel-scale-mode",
            "spectral_mup",
            "--spel-retract-mode",
            "hard",
            "--spel-retract-alpha",
            "0.05",
            "--split-qkv-init-mode",
            "head",
            "--spectral-mup-init",
        ]
        args = parse_args()
    finally:
        sys.argv = old_argv

    assert args.optimizer == "spel_dist"
    assert args.spel_qkv_split_mode == "head"
    assert args.split_qkv_init_mode == "head"
    assert args.spectral_mup_init is True
    print("argparse_spel: ok")


if __name__ == "__main__":
    main()
