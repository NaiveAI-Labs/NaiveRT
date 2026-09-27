"""Op descriptor: dense down projection + in-kernel TP8 all-reduce (rows1, rows8, rows8 PDL consumer)."""

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

_NS = "naivert_replica_down_allreduce"

OP = OpDescriptor(
    name="down_allreduce",
    source="naivert/kernels/ops/down_allreduce.py",
    namespace=_NS,
    kernel_namespace="naivert_naive_n0p5_model_ops",
    args_struct="DownAllreduceArgs",
    args_include="naivert/body/down_allreduce/down_allreduce_args.hpp",
    header_includes=("naivert/body/down_allreduce/down_allreduce_traits.cuh",),
    body_includes=(
        "naivert/body/down_allreduce/down_allreduce_body.cuh",
        "naivert/body/down_allreduce/down_allreduce_rows8_body.cuh",
    ),
    entry_includes=("naivert/pdl/grid_dependency.cuh",),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_down_allreduce_rows1_kernel",
            body="DownAllreduceBody",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
        ),
        # One eight-row launch for verify-8 (four mailbox slabs per row, one tag).
        Variant(
            "rows8",
            kernel="naivert_down_allreduce_rows8_kernel",
            body="DownAllreduceRows8Body",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
            shared_bytes="kRows8SharedBytes",
        ),
        # Under PDL: triggers its dependent (the next attention kernel) at the top, waits for
        # its primary (the rows8 up-gate) before reading the activations and the residual.
        Variant(
            "rows8_pdl_consumer",
            kernel="naivert_down_allreduce_rows8_pdl_consumer_kernel",
            body="DownAllreduceRows8PdlBody",
            shared_align=128,
            launch_bounds=f"{_NS}::kBlockThreads, 1",
            prologue="naivert::pdl::trigger_dependents<true>();",
            shared_bytes="kRows8SharedBytes",
        ),
    ),
    launch=Launch(grid="naivert::geometry::kSmCount", block="kBlockThreads", shared_bytes="kSharedBytes"),
    params=(
        TensorParam(
            "up_gate", dtype=None, field="up_gate", ptr_type="const __nv_bfloat16*",
            rows_source=True, rows_divisor="inter",
        ),
        TensorParam("w_down", dtype=None, field="w_down", ptr_type="const __nv_fp8_e4m3*"),
        TensorParam("w_down_scale", dtype=None, field="w_down_scale", ptr_type="const __nv_bfloat16*"),
        TensorParam(
            "residual", dtype=None, field="residual", ptr_type="const __nv_bfloat16*", min_numel="active_rows * hidden",
        ),
        TensorParam("output", dtype=None, field="exp_out", ptr_type="__nv_bfloat16*", min_numel="active_rows * hidden"),
        TensorParam("step_block", dtype=None, field="step_block_base", ptr_type="const std::uint8_t*", min_nbytes="12"),
        TensorParam(
            "replay_generation", dtype=None, field="replay_generation", ptr_type="const std::int32_t*", min_nbytes="4",
        ),
        IntParam("generation_salt", field="generation_salt", low=0, high=0xFF),
        TensorListParam(
            "mailbox", field="mailbox_peers", count=8, element="const std::uint8_t*",
            min_nbytes="active_rows * 4 * 0x70000",
        ),
        IntParam("rank", field="rank", equals_device=True),
        IntParam("hidden", field="hidden"),
        IntParam("inter", field="inter"),
    ),
    fills=(
        ("tp_result", "args.exp_out"),
        ("total_tiles", "(hidden + op::kOutputsPerTile - 1) / op::kOutputsPerTile"),
        ("active_ctas", "args.total_tiles"),
        ("experts", "1"),
        ("outputs_per_tile", "op::kOutputsPerTile"),
    ),
    bindings=(
        Binding("launch_down_allreduce", variants_by_rows={1: "rows1", 8: "rows8"}),
        Binding("launch_down_allreduce_pdl_consumer", variants_by_rows={8: "rows8_pdl_consumer"}),
    ),
)
