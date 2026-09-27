"""Compile-time geometry the TP8 kernel builds take from the Naive N spec."""
from __future__ import annotations

from dataclasses import dataclass

from .model import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY


@dataclass(frozen=True)
class AllreduceGeometry:
    expert_down_outputs_per_tile: int
    down_outputs_per_tile: int
    unproj_outputs_per_tile: int
    unproj_projection_k: int


_full = NAIVE_N0P5_HYBRID.attention_layouts["full_dsa"]
ALLREDUCE_GEOMETRY = AllreduceGeometry(
    expert_down_outputs_per_tile=64,  # 64 tiles = 16 CTA records x 4 warps
    down_outputs_per_tile=64,
    unproj_outputs_per_tile=64,       # 64 tiles, one 0x100 record each
    unproj_projection_k=_full.local_heads(NAIVE_N0P5_GEOMETRY.rank_count) * _full.v_head_dim,  # 8 x 128
)

# expert_select scores the 256 routed experts as eight warp-owned groups of 32.
# Naive N routes with n_group = 1 (no group limit), so every warp group survives.
_ROUTER_WARP_GROUPS = 8
_ROUTING = NAIVE_N0P5_HYBRID.routing
if _ROUTER_WARP_GROUPS % _ROUTING.group_count:
    raise ValueError(f"the router's {_ROUTER_WARP_GROUPS} warp groups cannot express n_group={_ROUTING.group_count}")
ROUTE_KEPT_GROUPS = _ROUTER_WARP_GROUPS // _ROUTING.group_count * _ROUTING.limited_group_count  # 8
ROUTE_SCALE = float(_ROUTING.route_scale)  # 1.0
NORM_EPS = float(NAIVE_N0P5_HYBRID.norm_eps)  # 1e-5

for _width in (
    ALLREDUCE_GEOMETRY.expert_down_outputs_per_tile,
    ALLREDUCE_GEOMETRY.down_outputs_per_tile,
    ALLREDUCE_GEOMETRY.unproj_outputs_per_tile,
):
    if NAIVE_N0P5_GEOMETRY.hidden_size % _width:
        raise ValueError(f"tile width {_width} does not divide hidden {NAIVE_N0P5_GEOMETRY.hidden_size}")


def geometry_cuda_cflags(*, sm_count: int, expert_weight_format: int = 0) -> tuple[str, ...]:
    """-D flags of the Naive N geometry; expert_weight_format 0 = fp8 block-128, 2 = mxfp8."""
    spec = NAIVE_N0P5_GEOMETRY
    g = ALLREDUCE_GEOMETRY
    return (
        f"-DNAIVERT_TP_RANKS={spec.rank_count}",
        f"-DNAIVERT_HIDDEN={spec.hidden_size}",
        f"-DNAIVERT_SM_COUNT={int(sm_count)}",
        f"-DNAIVERT_ROUTED_EXPERTS={spec.routed_expert_count}",
        f"-DNAIVERT_SHARED_EXPERTS={spec.total_expert_count - spec.routed_expert_count}",
        f"-DNAIVERT_SELECTED_EXPERTS={spec.selected_expert_count}",
        f"-DNAIVERT_ROUTE_KEPT_GROUPS={ROUTE_KEPT_GROUPS}",
        f"-DNAIVERT_ROUTE_SCALE={ROUTE_SCALE!r}",
        f"-DNAIVERT_NORM_EPS={NORM_EPS!r}",
        f"-DNAIVERT_DENSE_INTER={spec.dense_intermediate_size}",
        f"-DNAIVERT_VOCAB_SHARD={spec.vocab_shard_size}",
        f"-DNAIVERT_EXPERT_WEIGHT_FORMAT={int(expert_weight_format)}",
        f"-DNAIVERT_EXPERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE={g.expert_down_outputs_per_tile}",
        f"-DNAIVERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE={g.down_outputs_per_tile}",
        f"-DNAIVERT_UNPROJ_O_ALLREDUCE_OUTPUTS_PER_TILE={g.unproj_outputs_per_tile}",
        f"-DNAIVERT_UNPROJ_O_ALLREDUCE_PROJECTION_K={g.unproj_projection_k}",
    )


__all__ = ["ALLREDUCE_GEOMETRY", "NORM_EPS", "ROUTE_KEPT_GROUPS", "ROUTE_SCALE", "AllreduceGeometry", "geometry_cuda_cflags"]
