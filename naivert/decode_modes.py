"""Decode modes as data: plan, required kernel bundles, and the decode buffer table.

A model package instantiates DecodeModeSpec for each mode it supports.  The buffer table says how each of
the decode ABI pointers is backed -- owned, borrowed, or null -- as expressions
over the runtime spec.  This module is layer-neutral: models and runtime
both import it.
"""

from __future__ import annotations

from dataclasses import dataclass

from naivert.model import MtpPlan, Seq1Plan


@dataclass(frozen=True)
class RuntimeBufferAbi:
    """How one decode ABI pointer is backed; expressions are over the runtime spec."""

    name: str
    storage: str  # owned | borrowed | null
    rows_expression: str
    bytes_expression: str
    mtp_only: bool = False


@dataclass(frozen=True)
class RuntimeMemoryAbi:
    buffers: tuple[RuntimeBufferAbi, ...]

    @property
    def pointer_fields(self) -> tuple[str, ...]:
        return tuple(buffer.name for buffer in self.buffers)


POINTER_FIELDS = (
    "control_state",
    "input_tokens",
    "input_hidden",
    "target_logits",
    "target_tokens",
    "target_last_hidden",
    "accepted_tokens",
    "accepted_count",
    "accepted_last_hidden",
    "output_tokens",
    "main_cache",
    "draft_tokens",
    "draft_hidden",
    "mtp_cache",
)
MTP_ONLY_FIELDS = frozenset({"draft_tokens", "draft_hidden", "mtp_cache"})


def runtime_memory_abi(*, mtp: bool) -> RuntimeMemoryAbi:
    buffers = []
    for name in POINTER_FIELDS:
        mtp_only = name in MTP_ONLY_FIELDS
        if mtp_only and not mtp:
            storage, rows, size = "null", "0", "0"
        elif name in {"main_cache", "mtp_cache"}:
            storage, rows, size = "borrowed", "active_rows", "0"
        elif name == "control_state":
            storage, rows, size = "owned", "1", "control_size"
        elif name == "accepted_count":
            storage, rows, size = "owned", "1", "4"
        elif name == "accepted_last_hidden":
            storage, rows, size = "owned", "1", "row_stride"
        elif name in {"draft_tokens", "draft_hidden"}:
            storage, rows, size = "owned", "speculative_depth", "row_stride * speculative_depth"
        else:
            storage, rows, size = "owned", "active_rows", "row_stride * active_rows"
        buffers.append(RuntimeBufferAbi(name, storage, rows, size, mtp_only))
    return RuntimeMemoryAbi(buffers=tuple(buffers))


def mailbox_layer_tags(layer: int) -> tuple[int, int]:
    """Per-layer mailbox generation tags: (attention all-reduce, FFN all-reduce).

    Two tags per layer keep consecutive all-reduces of one step apart; the low byte
    of the generation carries the tag, the replay epoch sits above it (see
    naivert/common/tp8_mailbox.cuh).
    """
    return 2 * int(layer) + 2, 2 * int(layer) + 3


def seq1_last_cur_pos(prompt_tokens: int, max_decode_tokens: int) -> int:
    """cur_pos of the last decode step: prefill commits prompt[:-1], the first
    step runs at len(prompt) - 1, and each further step advances by one."""
    return int(prompt_tokens) - 1 + int(max_decode_tokens) - 1


def seq1_request_fits_short_path(prompt_tokens: int, max_decode_tokens: int,
                                 short_max_cur_pos: int | None) -> bool:
    """True when every step of the request stays inside the fast short-context
    kernel's validated range, so the request may run on the short graph set."""
    if short_max_cur_pos is None:
        return False
    return seq1_last_cur_pos(prompt_tokens, max_decode_tokens) <= int(short_max_cur_pos)


@dataclass(frozen=True)
class DecodeModeSpec:
    mode_id: str
    plan: Seq1Plan | MtpPlan
    required_operations: tuple[str, ...]
    runtime_memory_abi: RuntimeMemoryAbi

    @property
    def decode_plan_id(self) -> str:
        return self.plan.plan_id


__all__ = [
    "mailbox_layer_tags",
    "DecodeModeSpec",
    "MTP_ONLY_FIELDS",
    "POINTER_FIELDS",
    "RuntimeBufferAbi",
    "RuntimeMemoryAbi",
    "runtime_memory_abi",
]
