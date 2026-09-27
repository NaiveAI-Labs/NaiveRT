"""Model selection for the per-kernel GPU tests.

The tests run the shared model_ops kernels at one model's geometry (Naive N in this tree). Every number comes from the model package.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class KernelModel:
    name: str
    ranks: int
    hidden: int
    routed_experts: int
    shared_experts: int
    selected_experts: int
    expert_inter: int          # per rank, per expert
    dense_inter: int           # per rank
    vocab_shard: int
    eps: float
    route_scale: float
    route_kept_groups: int     # of the router's 8 groups (8 = no group limit)
    weights: str               # routed-expert weight format: "fp8" (block-128) or "mx" (mxfp8)
    expert_down_tile: int
    down_tile: int
    unproj_tile: int
    unproj_k: int
    load_extension: Callable[[], Any]

    @property
    def slots(self) -> int:
        return self.selected_experts + self.shared_experts

    @property
    def total_experts(self) -> int:
        return self.routed_experts + self.shared_experts


def select(name: str, weights: str = "fp8") -> KernelModel:
    if name == "naive_n0p5":
        from naivert.kernels.naive_n0p5.model_ops import load_extension
        from naivert.models.naive_n0p5.kernel_geometry import ALLREDUCE_GEOMETRY, NORM_EPS, ROUTE_KEPT_GROUPS, ROUTE_SCALE
        from naivert.models.naive_n0p5.model import NAIVE_N0P5_GEOMETRY as spec
    else:
        raise ValueError(f"unknown model {name!r}")
    if weights == "mx":
        if name != "naive_n0p5":
            raise ValueError(f"{weights} experts are a Naive N build")
        load_extension = functools.partial(load_extension, weights=weights)
    elif weights != "fp8":
        raise ValueError(f"unknown weight format {weights!r}")
    return KernelModel(
        name=name,
        ranks=spec.rank_count,
        hidden=spec.hidden_size,
        routed_experts=spec.routed_expert_count,
        shared_experts=spec.total_expert_count - spec.routed_expert_count,
        selected_experts=spec.selected_expert_count,
        expert_inter=256,
        dense_inter=spec.dense_intermediate_size,
        vocab_shard=spec.vocab_shard_size,
        eps=float(NORM_EPS),
        route_scale=float(ROUTE_SCALE),
        route_kept_groups=int(ROUTE_KEPT_GROUPS),
        weights=weights,
        expert_down_tile=ALLREDUCE_GEOMETRY.expert_down_outputs_per_tile,
        down_tile=ALLREDUCE_GEOMETRY.down_outputs_per_tile,
        unproj_tile=ALLREDUCE_GEOMETRY.unproj_outputs_per_tile,
        unproj_k=ALLREDUCE_GEOMETRY.unproj_projection_k,
        load_extension=load_extension,
    )


def add_model_argument(parser) -> None:
    parser.add_argument("--model", choices=("naive_n0p5",), default="naive_n0p5",
                        help="whose geometry and model_ops build to run")
    parser.add_argument("--weights", choices=("fp8", "mx"), default="fp8",
                        help="routed-expert weight format (fp8 = block-128, mx = mxfp8)")


def dequant(q, s):
    """fp8 block-128 (q, scale) -> f32 weights, for any leading batch dims."""
    return q.float() * s.float().repeat_interleave(128, -2).repeat_interleave(128, -1)


def ulp_diff_bf16(a, b):
    return (a.view(__import__("torch").int16).int() - b.view(__import__("torch").int16).int()).abs()


def close_bf16(out, ref, scale: float, rel: float = 0.02):
    """Elementwise |out - ref| <= max(rel * scale, one bf16 ulp of ref).

    The all-reduce outputs carry the residual, whose bf16 rounding (one ulp of a value near 16
    is 0.0625) can exceed a few percent of the much smaller projection contribution.
    Returns (ok, max_abs_diff, max_excess_ratio).
    """
    import torch
    out, ref = out.float(), ref.float()
    ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    thr = torch.maximum(torch.full_like(ref, rel * scale), ulp)
    diff = (out - ref).abs()
    return bool((diff <= thr).all()), float(diff.max()), float((diff / thr).max())
