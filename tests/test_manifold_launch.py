"""CPU-only launch contract; does not require Slurm or training data."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/manifold" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch, run = load_module("launch"), load_module("run")


def test_resume_comparison_detects_nested_momentum_and_dtype_changes():
    import torch

    check = load_module("check_resume")
    left = {"components": [{"radius": 1.2, "momentum": torch.eye(2)}]}
    right = {"components": [{"radius": 1.2, "momentum": torch.eye(2)}]}
    assert not check.differences(left, right)
    right["components"][0]["momentum"][0, 0] += .1
    assert check.differences(left, right) == ["root.components[0].momentum: tensor"]
    right["components"][0]["momentum"] = torch.eye(2).bfloat16()
    assert check.differences(left, right)


def test_framework_models_are_not_ignored_as_weights():
    paths = [
        "Megatron-LM/megatron/core/models/gpt/gpt_model.py",
        "Megatron-LM/megatron/training/models/gpt.py",
        "Megatron-LM/megatron/core/tokenizers/text/models/gpt_tokenizer.py",
    ]
    for path in paths:
        assert (ROOT / path).is_file()
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 1


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


def test_rolling_checkpoint_retention_uses_native_megatron_option():
    cfg = config("--stage", "main", "--save-interval", "2000", "--save-retain-interval", "12000")
    cfg["run_dir"] = "/tmp/run"
    args = run.training_args(cfg)
    assert args[args.index("--save-retain-interval") + 1] == "12000"
    assert cfg["train_iters"] < cfg["training"]["save_retain_interval"]
    assert args[args.index("--save-interval") + 1] == "2000"


@pytest.mark.parametrize("save,retain", [(0, 12000), (2000, 0), (2000, -1), (2000, 2100)])
def test_invalid_checkpoint_retention(save, retain):
    with pytest.raises(ValueError, match="retention"):
        config("--stage", "main", "--save-interval", str(save), "--save-retain-interval", str(retain))


def test_retention_remains_opt_in_for_historical_configs():
    cfg = config("--stage", "main")
    cfg["run_dir"] = "/tmp/run"
    assert "--save-retain-interval" not in run.training_args(cfg)


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


def test_deferred_resume_requires_successful_exact_producer(tmp_path):
    (tmp_path / "job_id.txt").write_text("12345\n")
    cfg = config("--resume", str(tmp_path / "checkpoints"))
    with pytest.raises(ValueError, match="exact producer"):
        launch.validate_checkpoint_dependency(cfg)
    cfg["dependency"] = "afterany:12345"
    with pytest.raises(ValueError):
        launch.validate_checkpoint_dependency(cfg)
    cfg["dependency"] = "afterok:23456"
    with pytest.raises(ValueError):
        launch.validate_checkpoint_dependency(cfg)
    cfg["dependency"] = "afterok:12345:23456"
    launch.validate_checkpoint_dependency(cfg)


def test_resume_comparison_is_only_enabled_for_controlled_smoke():
    with pytest.raises(ValueError, match="verification"):
        config("--verify-against-run", "/tmp/control")
    cfg = config("--resume", "/tmp/ckpt", "--save-interval", "2", "--verify-against-run", "/tmp/control")
    assert cfg["verify_against_run"] == str(Path("/tmp/control").resolve())
