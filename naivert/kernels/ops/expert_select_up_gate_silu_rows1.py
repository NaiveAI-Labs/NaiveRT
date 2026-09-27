"""Op descriptor: MoE expert selection + up/gate projection + SiLU, one row (plain and PDL consumer)."""

from __future__ import annotations

from naivert.kernels.opgen import Binding, Launch, OpDescriptor, TensorParam, Variant

# The kernel traits carry the model geometry (naivert/common/model_geometry.cuh);
# the binding checks against them rather than repeating a model's numbers.
_TRAITS = "naivert_replica_expert_select_up_gate_silu"
_CONSTANTS = {
    "kRows": "1",
    "kHidden": f"{_TRAITS}::kHidden",
    "kRoutedExperts": f"{_TRAITS}::kRoutedExperts",
    "kTotalExperts": f"{_TRAITS}::kTotalExperts",
    "kSelectedExperts": f"{_TRAITS}::kSelectedExperts",
    "kOutputSlots": f"{_TRAITS}::kOutputSlots",
    "kOutputGroups": f"{_TRAITS}::kOutputGroups",
    "kOutputsPerGroup": f"{_TRAITS}::kOutputsPerGroup",
    "kPages": f"{_TRAITS}::kPages",
    "kPageK": f"{_TRAITS}::kPageK",
    "kPackedPageStride": "naivert_replica_expert_select_up_gate_silu::kPackedPageStride",
}
_PARAMS = (
    TensorParam(
        "hidden", dtype="bf16", field="x", ptr_type="const __nv_bfloat16*",
        numel="kRows * kHidden", rows_source=True, rows_divisor="kHidden",
    ),
    TensorParam("scores", dtype="f32", field="scores", ptr_type="const float*", numel="kRows * kRoutedExperts"),
    TensorParam("bias", dtype="f32", field="bias", ptr_type="const float*", min_numel="kRoutedExperts"),
    TensorParam(
        "packed_weights", dtype="fp8", field="packed_weights", ptr_type="const __nv_fp8_e4m3*",
        min_numel="kTotalExperts * kOutputGroups * kPages * kPackedPageStride",
    ),
    TensorParam(
        "up_gate", dtype=None, field="y", ptr_type="naivert::ExpertActivation*",
        dtype_expr="naivert::kExpertActivationFp32 ? at::kFloat : at::kBFloat16",
        min_numel="kRows * kOutputSlots * kOutputGroups * kOutputsPerGroup",
    ),
    TensorParam("expert_probs", dtype="f32", field="expert_probs", ptr_type="float*", min_numel="kRows * kSelectedExperts"),
    TensorParam("expert_indices", dtype="i32", field="expert_indices", ptr_type="int*", min_numel="kRows * kSelectedExperts"),
)
_FILLS = (
    ("hidden", "kHidden"),
    ("routed_experts", "kRoutedExperts"),
    ("total_experts", "kTotalExperts"),
    ("selected_experts", "kSelectedExperts"),
    ("output_slots", "kOutputSlots"),
    ("output_groups", "kOutputGroups"),
    ("outputs_per_group", "kOutputsPerGroup"),
    ("seq_len", "kRows"),
    ("pages", "kPages"),
    ("page_k", "kPageK"),
    ("total_tiles", "kOutputSlots * kOutputGroups"),
    ("x_stride", "kHidden"),
    ("y_slot_stride", "kOutputGroups * kOutputsPerGroup"),
    ("y_group_stride", "kOutputsPerGroup"),
    ("packed_group_stride", "kOutputGroups"),
    ("packed_page_stride", "kPackedPageStride"),
    ("packed_expert_stride", "kOutputGroups * kPages * kPackedPageStride"),
    ("route_scale", "naivert::geometry::kRouteScale"),
    ("progress.total_tiles", "kOutputSlots * kOutputGroups"),
    ("progress.active_ctas", "kOutputSlots * kOutputGroups"),
)

OP = OpDescriptor(
    name="expert_select_up_gate_silu_rows1",
    source="naivert/kernels/ops/expert_select_up_gate_silu_rows1.py",
    namespace="naivert_replica_expert_select_up_gate_silu",
    args_struct="ExpertSelectUpGateSiluArgs",
    args_include="naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_args.hpp",
    header_includes=("naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_traits.cuh",),
    body_includes=("naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_body.cuh",),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_expert_select_up_gate_silu_rows1_kernel",
            body="ExpertSelectUpGateSiluBody",
            run_call="ExpertSelectUpGateSiluBody::run<false>(args, shared)",
            shared_align=128,
            launch_bounds="kBlockThreads, 1",
        ),
        Variant(
            "rows1_pdl_consumer",
            kernel="naivert_expert_select_up_gate_silu_rows1_pdl_consumer_kernel",
            body="ExpertSelectUpGateSiluBody",
            run_call="ExpertSelectUpGateSiluBody::run<true>(args, shared)",
            shared_align=128,
            launch_bounds="kBlockThreads, 1",
        ),
    ),
    launch=Launch(grid="args.total_tiles", block="kBlockThreads", shared_bytes="kSharedBytes"),
    constants=_CONSTANTS,
    params=_PARAMS,
    fills=_FILLS,
    bindings=(
        Binding("launch_expert_select_up_gate_silu_rows1", variants_by_rows={1: "rows1"}),
        Binding("launch_expert_select_up_gate_silu_rows1_pdl_consumer", variants_by_rows={1: "rows1_pdl_consumer"}),
    ),
)
