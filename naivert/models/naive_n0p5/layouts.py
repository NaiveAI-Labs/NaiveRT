"""Rotary tables shared by FP8 checkpoint loading and prefill checks."""
from __future__ import annotations

import torch


def rope_table(theta: float, rope_dim: int, max_seq_len: int) -> torch.Tensor:
    """Plain rotary table [max_seq_len, rope_dim] as interleaved (cos, sin) pairs."""
    frequencies = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).reshape(max_seq_len, rope_dim)
