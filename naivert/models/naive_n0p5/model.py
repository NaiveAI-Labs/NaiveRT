"""Current Naive N0.5 FP8 model geometry, independent of checkpoint identity.

Checkpoint metadata is validated by checkpoint_fp8.NaiveNCheckpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

from naivert.model import (
    AttentionLayout,
    HybridModelSpec,
    MoeRoutingSpec,
    SparseIndexerSpec,
)

# config.json hybrid_layer_pattern: 0 = full attention (+ DSA indexer), 1 = sliding window.
_HYBRID_LAYER_PATTERN = (
    0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
)
FULL_DSA = "full_dsa"
SLIDING_WINDOW = "sliding_window"
#: The operand contract the kernel prefill executor (naivert.prefill.executor) was accepted
#: against: original Naive N0.5 bf16/fp8 operands, fp32 router, W8A8 experts. A rank namespace
#: claiming it carries the keys naivert.prefill.operands builds.
PRECISION_ID = "n0p5-w8a8-fp32-expert-completion-v1"

NAIVE_N0P5_HYBRID = HybridModelSpec(
    layer_kinds=tuple(SLIDING_WINDOW if bit else FULL_DSA for bit in _HYBRID_LAYER_PATTERN),
    attention_layouts={
        FULL_DSA: AttentionLayout(
            kind=FULL_DSA,
            head_count=64,
            kv_head_count=4,
            head_dim=192,
            v_head_dim=128,
            rope_dim=64,  # partial_rotary_factor 0.334 of 192
            rope_theta=1.0e7,
            cache_roles=("ki", "k", "v"),
            cache_widths=(128, 192, 128),  # per token, per rank (one KV head per rank)
            value_scale=0.707,
        ),
        SLIDING_WINDOW: AttentionLayout(
            kind=SLIDING_WINDOW,
            head_count=64,
            kv_head_count=8,
            head_dim=192,
            v_head_dim=128,
            rope_dim=64,
            rope_theta=1.0e4,
            cache_roles=("k_window", "v_window"),
            cache_widths=(192, 128),
            sliding_window=128,
            sink_bias=True,
            value_scale=0.707,
        ),
    },
    indexer=SparseIndexerSpec(head_count=16, head_dim=128, rope_dim=64, top_k=2048),
    indexer_layer_kinds=(FULL_DSA,),
    routing=MoeRoutingSpec(
        score_func="sigmoid",
        group_count=1,
        limited_group_count=1,
        route_scale=1.0,
        normalize_top_k=True,
        shared_expert_count=0,
    ),
    norm_eps=1e-5,
    weight_dtype="fp8",
)

@dataclass(frozen=True)
class NaiveNGeometry:
    """Dimensions shared by the serving loader and compiled TP8 kernels.

    Indexer heads are replicated on each rank; attention query heads and expert
    intermediate channels are sharded. Cache capacity is chosen at startup.
    """

    rank_count: int = 8
    main_layer_count: int = 48
    dense_layer_count: int = 1
    hidden_size: int = 4096
    dense_intermediate_size: int = 2048  # per rank
    vocab_size: int = 152576
    vocab_shard_size: int = 19072
    routed_expert_count: int = 256
    total_expert_count: int = 256
    selected_expert_count: int = 8
    stop_token_ids: tuple[int, ...] = (151645,)


NAIVE_N0P5_GEOMETRY = NaiveNGeometry()

__all__ = ["FULL_DSA", "SLIDING_WINDOW", "PRECISION_ID", "NAIVE_N0P5_HYBRID",
           "NAIVE_N0P5_GEOMETRY", "NaiveNGeometry"]
