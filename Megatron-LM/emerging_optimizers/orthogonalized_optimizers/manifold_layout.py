"""Logical transformer matrices inside Megatron fused parameter storage."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MatrixComponent:
    """One full logical out-by-in matrix and its physical row indices."""

    label: str
    rows: tuple[int, ...]

    def read(self, physical: torch.Tensor) -> torch.Tensor:
        """Gather this component in its logical row order."""
        return physical.index_select(0, torch.tensor(self.rows, device=physical.device))

    def write(self, physical: torch.Tensor, logical: torch.Tensor) -> None:
        """Write a logical component into its original physical rows."""
        physical.index_copy_(0, torch.tensor(self.rows, device=physical.device), logical)


def components_for_parameter(
    name: str,
    shape: tuple[int, int],
    *,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attention_heads: int,
    num_query_groups: int,
    kv_channels: int,
) -> tuple[MatrixComponent, ...]:
    """Map Q/K/V, O, gate/up, and down without assuming contiguous QKV thirds.

    Megatron SelfAttention reads the fused QKV output as
    ``[group, q_heads_per_group + k + v, head_dim]``.  The returned Q rows
    concatenate the Q slice of every group; K and V do likewise.
    """
    if len(shape) != 2 or num_attention_heads % num_query_groups:
        raise ValueError("invalid matrix shape or GQA configuration")
    if name.endswith("linear_qkv.weight"):
        q_per_group = num_attention_heads // num_query_groups * kv_channels
        stride = q_per_group + 2 * kv_channels
        expected = (num_query_groups * stride, hidden_size)
        if shape != expected:
            raise ValueError(f"QKV shape {shape} does not match expected {expected}")
        row_sets: list[list[int]] = [[], [], []]
        for group in range(num_query_groups):
            start = group * stride
            row_sets[0].extend(range(start, start + q_per_group))
            row_sets[1].extend(range(start + q_per_group, start + q_per_group + kv_channels))
            row_sets[2].extend(range(start + q_per_group + kv_channels, start + stride))
        return tuple(
            MatrixComponent(label, tuple(rows))
            for label, rows in zip(("attention.q", "attention.k", "attention.v"), row_sets)
        )
    if name.endswith("linear_proj.weight"):
        if shape != (hidden_size, hidden_size):
            raise ValueError(f"attention output shape mismatch: {shape}")
        return (MatrixComponent("attention.o", tuple(range(shape[0]))),)
    if name.endswith("linear_fc1.weight"):
        expected = (2 * ffn_hidden_size, hidden_size)
        if shape != expected:
            raise ValueError(f"gated FC1 shape {shape} does not match expected {expected}")
        return (
            MatrixComponent("mlp.gate", tuple(range(ffn_hidden_size))),
            MatrixComponent("mlp.up", tuple(range(ffn_hidden_size, 2 * ffn_hidden_size))),
        )
    if name.endswith("linear_fc2.weight"):
        expected = (hidden_size, ffn_hidden_size)
        if shape != expected:
            raise ValueError(f"FC2 shape {shape} does not match expected {expected}")
        return (MatrixComponent("mlp.down", tuple(range(shape[0]))),)
    return ()


def validate_layout(components: tuple[MatrixComponent, ...], physical_rows: int) -> None:
    """Require that each physical row belongs to exactly one logical block."""
    indices = [row for component in components for row in component.rows]
    if len(indices) != physical_rows or sorted(indices) != list(range(physical_rows)):
        raise ValueError("logical components must partition the physical rows exactly")
