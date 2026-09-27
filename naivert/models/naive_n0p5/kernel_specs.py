"""Pure-data kernel facts accepted by the Naive N0.5 model package."""

from __future__ import annotations

from naivert.kernel_variant import KernelVariantSpec

# The shared model_ops sources (MoE, dense MLP, head, argmax) compiled with
# Naive N's geometry (naivert/kernels/naive_n0p5/model_ops.py).
MODEL_OPS = KernelVariantSpec(
    variant_id="naive-n0.5.model-ops.sm103a.abi1",
)
# The same sources with NAIVERT_EXPERT_WEIGHT_FORMAT=2: routed experts as mxfp8 (e4m3, ue8m0 per 32).
MODEL_OPS_MX = KernelVariantSpec(
    variant_id="naive-n0.5.model-ops-mx.sm103a.abi1",
)

MODEL_OPS_W8A8 = KernelVariantSpec(
    variant_id="naive-n0.5.model-ops-w8a8-down.sm103a.abi4",
)

__all__ = ["MODEL_OPS", "MODEL_OPS_MX", "MODEL_OPS_W8A8"]
