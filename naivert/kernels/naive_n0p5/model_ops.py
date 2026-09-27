"""The shared model_ops sources (MoE, dense MLP, head, argmax) built with Naive N's geometry."""

from __future__ import annotations

import functools
from pathlib import Path

from naivert.kernel_variant import KernelVariantSpec
from naivert.kernels.model_ops import body
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags
from naivert.models.naive_n0p5.kernel_specs import MODEL_OPS, MODEL_OPS_MX, MODEL_OPS_W8A8

BUILD = body.ModelOpsBuild(
    model="naive_n0p5",
    geometry_cuda_cflags=geometry_cuda_cflags,
    variant=MODEL_OPS,
)
# Routed experts as mxfp8 (naivert/kernels/packing.py pack_gate_up_mx / pack_down_mx).
BUILD_MX = body.ModelOpsBuild(
    model="naive_n0p5_mx",
    geometry_cuda_cflags=functools.partial(geometry_cuda_cflags, expert_weight_format=2),
    variant=MODEL_OPS_MX,
)
BUILDS = {"fp8": BUILD, "mx": BUILD_MX}


def _w8a8_geometry(**kwargs):
    return (*geometry_cuda_cflags(**kwargs), "-DNAIVERT_EXPERT_DOWN_W8A8=1")


BUILD_W8A8 = body.ModelOpsBuild(
    model="naive_n0p5_w8a8_down", geometry_cuda_cflags=_w8a8_geometry, variant=MODEL_OPS_W8A8,
)


def _build_manifest(*, expected_variant: KernelVariantSpec | None = None, weights: str = "fp8", expert_down="bf16"):
    return body._build_manifest(expected_variant=expected_variant, build=_select_build(weights, expert_down))


def _select_build(weights, expert_down):
    if expert_down not in ("bf16", "w8a8") or (expert_down == "w8a8" and weights != "fp8"):
        raise ValueError("native W8A8 down requires block128 fp8 expert weights")
    return BUILD_W8A8 if expert_down == "w8a8" else BUILDS[weights]


def load_extension(
    verbose: bool = False,
    *,
    build_directory: str | Path | None = None,
    expected_variant: KernelVariantSpec | None = None,
    weights: str = "fp8",
    expert_down: str = "bf16",
):
    """Load the Naive N model_ops extension; ``weights`` selects the expert weight format build."""
    if weights not in BUILDS:
        raise ValueError(f"unknown expert weight format {weights!r} (fp8 or mx)")
    return body.load_extension(
        verbose,
        build_directory=build_directory,
        expected_variant=expected_variant,
        build=_select_build(weights, expert_down),
    )


__all__ = ["BUILD", "BUILD_MX", "BUILD_W8A8", "BUILDS", "load_extension"]
