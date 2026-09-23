"""CPU-only launch contract; does not require Slurm or training data."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/manifold" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch, run = load_module("launch"), load_module("run")


def config(*extra):
    return launch.resolve_config(launch.parser().parse_args([
        "--method", "manifold_mcsd_tp", "--geometry", "frobenius", *extra
    ]))


def test_three_billion_uses_real_data_and_full_horizon():
    cfg = config("--stage", "main")
    assert cfg["train_iters"] == 11445
    assert cfg["actual_tokens"] == 3000238080
    assert "olmo_mix_1124_3b" in cfg["train_prefix"]
    assert cfg["training"]["nesterov"] is False


@pytest.mark.parametrize("width,heads,groups", [(256,4,2), (384,6,3), (512,8,4)])
def test_dense_scales(width, heads, groups):
    cfg = config("--width", str(width))
    assert cfg["model"]["num_attention_heads"] == heads
    assert cfg["model"]["num_query_groups"] == groups
    assert cfg["model"]["ffn_hidden_size"] == 3*width


def test_invalid_mixes_and_budgets():
    with pytest.raises(ValueError):
        config("--steps", "101")
    with pytest.raises(ValueError):
        config("--stage", "main", "--steps", "100")
    with pytest.raises(ValueError):
        config("--nesterov")
    with pytest.raises(ValueError):
        config("--method", "manifold_imuon")
    with pytest.raises(ValueError):
        config("--lr", "nan")


def test_resume_does_not_discard_optimizer_or_shorten_schedule():
    cfg = config("--stage", "main", "--resume", "/tmp/ckpt", "--exit-interval", "2")
    cfg["run_dir"] = "/tmp/run"
    args = run.training_args(cfg)
    assert args[args.index("--lr-decay-iters")+1] == "11445"
    assert "--load" in args and "--no-load-optim" not in args and "--no-load-rng" not in args
    assert args[args.index("--ckpt-format")+1] == "torch"


def test_native_spectral_baselines_have_nesterov_by_default():
    cfg = config("--method", "manifold_sso", "--geometry", "spectral")
    assert cfg["training"]["nesterov"] is True
    assert cfg["training"]["momentum_beta"] == .9


def test_resume_rejects_changed_method_and_learning_rate(tmp_path):
    original = config("--stage", "main")
    (tmp_path / "resolved_config.json").write_text(json.dumps(original))
    continued = config("--stage", "main", "--resume", str(tmp_path / "checkpoints"))
    launch.validate_resume(continued)
    continued["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="training.lr"):
        launch.validate_resume(continued)
    continued["training"]["lr"] = original["training"]["lr"]
    continued["method"] = "manifold_muonh"
    with pytest.raises(ValueError, match="method"):
        launch.validate_resume(continued)
