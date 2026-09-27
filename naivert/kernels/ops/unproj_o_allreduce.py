"""Op descriptor: attention output projection + in-kernel TP8 all-reduce (rows1, plain and PDL consumer).

The accepted-row variant (device-side row selection from the verify/commit count) has
entry logic of its own and stays hand-written in unproj_o_allreduce_accepted_row_entry.cu.
"""

from __future__ import annotations

from naivert.kernels.opgen import (
    Binding,
    IntParam,
    Launch,
    OpDescriptor,
    TensorListParam,
    TensorParam,
    Variant,
)

_ROWS = {1: "rows1"}
_ROWS_PDL = {1: "rows1_pdl_consumer"}

OP = OpDescriptor(
    name="unproj_o_allreduce",
    source="naivert/kernels/ops/unproj_o_allreduce.py",
    namespace="naivert_replica_unproj_o_allreduce",
    args_struct="UnprojOAllreduceArgs",
    args_include="naivert/body/unproj_o_allreduce/unproj_o_allreduce_args.hpp",
    header_includes=("naivert/body/unproj_o_allreduce/unproj_o_allreduce_traits.cuh",),
    body_includes=("naivert/body/unproj_o_allreduce/unproj_o_allreduce_body.cuh",),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_unproj_allreduce_rows1_kernel",
            body="UnprojOAllreduceBody",
            launch_bounds="kBlockThreads, 1",
            run_call="UnprojOAllreduceBody::run<false>(args, shared)",
            shared_align=128,
        ),
        Variant(
            "rows1_pdl_consumer",
            kernel="naivert_unproj_allreduce_rows1_pdl_consumer_kernel",
            body="UnprojOAllreduceBody",
            launch_bounds="kBlockThreads, 1",
            run_call="UnprojOAllreduceBody::run<true>(args, shared)",
            shared_align=128,
        ),
    ),
    launch=Launch(grid="naivert::geometry::kSmCount", block="kBlockThreads", shared_bytes="kSharedBytes"),
    constants={
        "kHidden": "naivert_replica_unproj_o_allreduce::kHidden",
        "kProjectionK": "naivert_replica_unproj_o_allreduce::kProjectionK",
        "kOutputsPerTile": "naivert_replica_unproj_o_allreduce::kOutputsPerTile",
    },
    params=(
        TensorParam(
            "proj_o", dtype=None, field="proj_o", ptr_type="const __nv_bfloat16*",
            rows_source=True, rows_divisor="kProjectionK",
        ),
        TensorParam("weight", dtype=None, field="weight", ptr_type="const __nv_fp8_e4m3*"),
        TensorParam("scales", dtype=None, field="scales", ptr_type="const __nv_bfloat16*"),
        TensorParam(
            "residual", dtype=None, field="residual", ptr_type="const __nv_bfloat16*", min_numel="active_rows * kHidden",
        ),
        TensorParam("output", dtype=None, field="output", ptr_type="__nv_bfloat16*", min_numel="active_rows * kHidden"),
        TensorListParam(
            "mailbox", field="mailbox_peers", count=8, element="const std::uint8_t*", min_nbytes="active_rows * 0x70000",
        ),
        IntParam("rank", field="rank", equals_device=True),
        IntParam("mailbox_tag", field="mailbox_tag", low=1, high=0xFFFFFFFF),
        TensorParam(
            "replay_iteration", dtype="u8", field="step_block_base", ptr_type="const std::uint8_t*",
            min_nbytes="8", optional=True,
        ),
    ),
    fills=(
        ("total_tiles", "kHidden / kOutputsPerTile"),
        ("active_ctas", "args.total_tiles"),
        ("hidden", "kHidden"),
        ("projection_k", "kProjectionK"),
        ("outputs_per_tile", "kOutputsPerTile"),
    ),
    bindings=(
        Binding("launch_unproj_allreduce", variants_by_rows=_ROWS, omit=("replay_iteration",)),
        Binding("launch_unproj_allreduce_replay", variants_by_rows=_ROWS),
        Binding("launch_unproj_allreduce_pdl_consumer", variants_by_rows=_ROWS_PDL, omit=("replay_iteration",)),
        Binding("launch_unproj_allreduce_replay_pdl_consumer", variants_by_rows=_ROWS_PDL),
    ),
)
