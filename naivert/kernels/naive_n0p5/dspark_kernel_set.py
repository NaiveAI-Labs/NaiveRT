"""Prepared extension inventory for the fixed Naive N0.5 draft chain."""
from dataclasses import dataclass
from typing import Any

from . import dspark_attention, dspark_context, dspark_head_markov, dspark_mlp


@dataclass(frozen=True)
class DraftKernelSet:
    devices: tuple[int, ...]
    context: Any
    attention: Any
    mlp: Any
    head_markov: Any
    operation_ids: tuple[str, ...] = (
        dspark_context.OPERATION_ID, dspark_attention.OPERATION_ID,
        dspark_mlp.OPERATION_ID, dspark_head_markov.OPERATION_ID,
    )


def prepare_draft(devices) -> DraftKernelSet:
    """Compile and configure normal entries before capture; caller enables P2P."""
    devices = tuple(int(d) for d in devices)
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError("draft requires eight distinct CUDA devices")
    return DraftKernelSet(
        devices, dspark_context.prepare_context(devices),
        dspark_attention.prepare_attention(devices), dspark_mlp.prepare_mlp(devices),
        dspark_head_markov.prepare_head_markov(devices),
    )
