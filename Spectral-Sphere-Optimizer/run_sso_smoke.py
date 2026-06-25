import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sso import SSO


class TinyRegressionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim, bias=False))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small CUDA smoke test for the SSO optimizer.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def require_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    device = require_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = TinyRegressionModel(args.input_dim, args.hidden_dim, args.output_dim, args.depth).to(device)
    target = nn.Linear(args.input_dim, args.output_dim, bias=False).to(device)
    for param in target.parameters():
        param.requires_grad_(False)

    optimizer = SSO(model.parameters(), lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    last_loss = None

    print(
        json.dumps(
            {
                "event": "start",
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "args": vars(args) | {
                    "output_dir": str(args.output_dir),
                    "checkpoint_dir": str(args.checkpoint_dir),
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, args.input_dim, device=device)
        with torch.no_grad():
            y = target(x)

        pred = model(x)
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                peak_mem_mb = 0.0
            elapsed = time.perf_counter() - start_time
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": step,
                        "loss": last_loss,
                        "elapsed_sec": round(elapsed, 3),
                        "peak_cuda_mem_mb": round(peak_mem_mb, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    result = {
        "status": "ok",
        "steps": args.steps,
        "final_loss": last_loss,
        "elapsed_sec": elapsed,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    result_path = args.output_dir / "sso_smoke_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint_path = args.checkpoint_dir / "sso_smoke_last.pt"
    torch.save({"model": model.state_dict(), "result": result}, checkpoint_path)

    print(json.dumps({"event": "finished", "result": result, "result_path": str(result_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
