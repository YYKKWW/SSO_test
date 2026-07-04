#!/usr/bin/env python3
"""Smoke test for Spectral-Sphere-Optimizer/sso.py."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def load_sso(repo_root: Path):
    sso_path = repo_root / "Spectral-Sphere-Optimizer" / "sso.py"
    spec = importlib.util.spec_from_file_location("sso_module", sso_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {sso_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SSO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    torch.manual_seed(1234)
    SSO = load_sso(args.repo_root)

    weight = torch.nn.Parameter(torch.randn(32, 16, device=device, dtype=torch.float32) * 0.02)
    x = torch.randn(64, 16, device=device)
    target = torch.randn(64, 32, device=device)
    opt = SSO([weight], lr=1e-3, momentum=0.9, weight_decay=0.01, power_iter_steps=3)

    losses = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        pred = x @ weight.t()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("losses:", " ".join(f"{v:.6f}" for v in losses))
    print("weight_norm:", float(weight.norm().detach().cpu()))


if __name__ == "__main__":
    main()
