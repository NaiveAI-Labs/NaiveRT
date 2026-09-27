"""Op descriptor: post-attention RMSNorm + dense up/gate projection (rows1, rows8, rows8 PDL consumer)."""

from __future__ import annotations

from naivert.kernels.opgen import Binding, IntParam, Launch, OpDescriptor, TensorParam, Variant

_NS = "naivert_replica_rmsnorm_up_gate"

OP = OpDescriptor(
    name="rmsnorm_up_gate",
    source="naivert/kernels/ops/rmsnorm_up_gate.py",
    namespace=_NS,
    # the kernel symbols stay where the hand-written shim defined them
    kernel_namespace="naivert_naive_n0p5_model_ops",
    args_struct="RMSNormUpGateArgs",
    args_include="naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_args.hpp",
    header_includes=("naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_traits.cuh",),
    body_includes=(
        "naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_body.cuh",
        "naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_rows8_body.cuh",
    ),
    entry_includes=("naivert/pdl/grid_dependency.cuh",),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_rmsnorm_up_gate_rows1_kernel",
            body="RMSNormUpGateBody",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
        ),
        # One eight-row launch for verify-8 (all eight N columns of the m16n8k16 used).
        Variant(
            "rows8",
            kernel="naivert_rmsnorm_up_gate_rows8_kernel",
            body="RMSNormUpGateRows8Body",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
            shared_bytes="kRows8LaunchSharedBytes",
        ),
        # Under PDL: triggers its dependent (the dense down) at the top, waits for its primary
        # (the attention tail) before reading the activations; the weight pages stream meanwhile.
        Variant(
            "rows8_pdl_consumer",
            kernel="naivert_rmsnorm_up_gate_rows8_pdl_consumer_kernel",
            body="RMSNormUpGateRows8PdlBody",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
            prologue="naivert::pdl::trigger_dependents<true>();",
            shared_bytes="kRows8LaunchSharedBytes",
        ),
    ),
    launch=Launch(
        # The production shape runs on the whole persistent grid (idle CTAs exit).
        grid=(
            "args.total_tiles == naivert_replica_rmsnorm_up_gate::kInter / "
            "naivert_replica_rmsnorm_up_gate::kOutputsPerTile"
            " ? naivert::geometry::kSmCount : args.total_tiles"
        ),
        block="kBlockThreads",
        shared_bytes="kSharedBytes",
        guard="args.total_tiles > 0",
    ),
    # The compact weight workspace: one 0x8020-byte pane per (output tile, page).
    constants={
        "kPackedWorkspaceBytes": (
            "int64_t{naivert_replica_rmsnorm_up_gate::kInter / "
            "naivert_replica_rmsnorm_up_gate::kOutputsPerTile} * "
            "int64_t{naivert_replica_rmsnorm_up_gate::kPages} * "
            "int64_t{naivert_replica_rmsnorm_up_gate::kPaneVisibleBytes}"
        ),
    },
    params=(
        TensorParam(
            "unproj_o", dtype=None, field="unproj_o", ptr_type="const __nv_bfloat16*",
            rows_source=True, rows_divisor="dim",
        ),
        TensorParam("post_norm", dtype="f32", field="post_norm", ptr_type="const float*"),
        TensorParam("w_gate", dtype=None, field="w_gate", ptr_type="const __nv_fp8_e4m3*"),
        TensorParam("w_gate_scale", dtype=None, field="w_gate_scale", ptr_type="const __nv_bfloat16*"),
        TensorParam("w_up", dtype=None, field="w_up", ptr_type="const __nv_fp8_e4m3*"),
        TensorParam("w_up_scale", dtype=None, field="w_up_scale", ptr_type="const __nv_bfloat16*"),
        TensorParam(
            "packed_workspace", dtype="u8", field="workspace_base", ptr_type="std::uint8_t*",
            min_numel="kPackedWorkspaceBytes",
        ),
        TensorParam(
            "up_gate", dtype=None, field="up_gate", ptr_type="__nv_bfloat16*", min_numel="active_rows * inter_dim",
        ),
        IntParam("dim", field="dim", must_equal="naivert_replica_rmsnorm_up_gate::kHidden"),
        IntParam("inter_dim", field="inter_dim", must_equal="naivert_replica_rmsnorm_up_gate::kInter"),
    ),
    fills=(
        ("total_tiles", "(inter_dim + op::kOutputsPerTile - 1) / op::kOutputsPerTile"),
        ("active_ctas", "args.total_tiles"),
    ),
    bindings=(
        Binding("launch_rmsnorm_up_gate", variants_by_rows={1: "rows1", 8: "rows8"}),
        Binding("launch_rmsnorm_up_gate_pdl_consumer", variants_by_rows={8: "rows8_pdl_consumer"}),
    ),
)
