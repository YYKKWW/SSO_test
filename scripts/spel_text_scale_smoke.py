"""Run a small text-training smoke test with the SpEL optimizer.

This is a standalone PyTorch path for validating the h256/l28/lr1e-3 scale
from the collected Megatron log while the bundled Megatron tree is incomplete.
"""

import argparse
import math
import os
import random
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from emerging_optimizers.orthogonalized_optimizers.spel import SpEL


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class GPTBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True, bias=False
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_hidden_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        ffn_hidden_size: int,
        num_layers: int,
        num_heads: int,
        seq_length: int,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_length, hidden_size)
        self.layers = nn.ModuleList(
            [GPTBlock(hidden_size, ffn_hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.seq_length = seq_length

    def forward(self, idx: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
        batch_size, seq_length = idx.shape
        pos = torch.arange(seq_length, device=idx.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        mask = torch.ones(seq_length, seq_length, device=idx.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            if use_checkpoint and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return self.lm_head(self.final_norm(x))


def download_text(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, path)


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_length: int, device: torch.device):
    max_start = tokens.numel() - seq_length - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[s : s + seq_length] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1 : s + seq_length + 1] for s in starts]).to(device)
    return x, y


def build_optimizers(model: nn.Module, args):
    spel_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "token_embedding" not in name and "lm_head" not in name:
            spel_params.append(param)
        else:
            adam_params.append(param)

    optimizers = []
    if spel_params:
        optimizers.append(
            SpEL(
                spel_params,
                lr=args.lr,
                momentum_beta=args.spel_momentum,
                weight_decay=args.weight_decay,
                use_nesterov=True,
                power_iteration_steps=args.spel_power_iteration_steps,
                msign_steps=args.spel_msign_steps,
                radius_mode="spectral_mup",
                scale_mode="spectral_mup",
                retract_mode="hard",
            )
        )
    if adam_params:
        optimizers.append(
            torch.optim.AdamW(
                adam_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
            )
        )
    return optimizers, len(spel_params), len(adam_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/text_smoke")
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--ffn-hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--spel-momentum", type=float, default=0.9)
    parser.add_argument("--spel-msign-steps", type=int, default=8)
    parser.add_argument("--spel-power-iteration-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data_dir) / "tinyshakespeare.txt"
    download_text(data_path, args.data_url)
    text = data_path.read_bytes()
    tokens = torch.tensor(list(text), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharGPT(
        vocab_size=256,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_attention_heads,
        seq_length=args.seq_length,
    ).to(device)
    optimizers, spel_count, adam_count = build_optimizers(model, args)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(
        "config="
        f"h{args.hidden_size}_l{args.num_layers}_ffn{args.ffn_hidden_size}_"
        f"seq{args.seq_length}_heads{args.num_attention_heads}_lr{args.lr}"
    )
    print(f"tokens={tokens.numel()} params={num_params} spel_params={spel_count} adam_params={adam_count}")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(tokens, args.batch_size, args.seq_length, device)
        logits = model(x, use_checkpoint=not args.no_checkpoint)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_gb = math.nan
        print(f"step={step} loss={loss.item():.6f} grad_norm={float(grad_norm):.6f} peak_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
