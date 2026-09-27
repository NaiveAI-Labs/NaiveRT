"""Op descriptor: MoE expert selection + up/gate projection + SiLU, eight rows in one expert-major
launch (plain and PDL primary)."""

from __future__ import annotations

from naivert.kernels.opgen import Binding, Launch, OpDescriptor, Variant
from naivert.kernels.ops.expert_select_up_gate_silu_rows1 import _CONSTANTS, _FILLS, _PARAMS

# kSmCount persistent CTAs take (distinct expert, output group) items round-robin.
_FILLS8 = tuple(
    (name, "naivert::geometry::kSmCount") if name in ("total_tiles", "progress.total_tiles", "progress.active_ctas")
    else (name, value)
    for name, value in _FILLS
)

OP = OpDescriptor(
    name="expert_select_up_gate_silu_rows8",
    source="naivert/kernels/ops/expert_select_up_gate_silu_rows8.py",
    namespace="naivert_mtp_expert_select_rows8",
    args_struct="ExpertSelectUpGateSiluArgs",
    args_include="naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_args.hpp",
    header_includes=("naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_traits.cuh",),
    body_includes=("naivert/body/expert_select_up_gate_silu_rows8_body.cuh",),
    entry_includes=("naivert/pdl/grid_dependency.cuh",),
    entry_usings=(
        "using naivert_replica_expert_select_up_gate_silu::ExpertSelectUpGateSiluArgs;",
        "using naivert_replica_expert_select_up_gate_silu::kBlockThreads;",
    ),
    variants=(
        Variant(
            "rows8",
            kernel="naivert_expert_select_up_gate_silu_rows8_kernel",
            body="ExpertSelectUpGateSiluRows8Body",
            launch_bounds="kBlockThreads, 1",
            shared_align=128,
        ),
        Variant(
            "rows8_pdl_primary",
            kernel="naivert_expert_select_up_gate_silu_rows8_pdl_primary_kernel",
            body="ExpertSelectUpGateSiluRows8Body",
            launch_bounds="kBlockThreads, 1",
            shared_align=128,
            prologue="naivert::pdl::trigger_dependents<true>();",
            run_call="ExpertSelectUpGateSiluRows8Body::run<true>(args, shared)",
        ),
    ),
    launch=Launch(grid="args.total_tiles", block="kBlockThreads", shared_bytes="kRows8SharedBytes"),
    constants={**_CONSTANTS, "kRows": "8"},
    params=_PARAMS,
    fills=_FILLS8,
    bindings=(
        Binding("launch_expert_select_up_gate_silu_rows8", variants_by_rows={8: "rows8"}),
        Binding("launch_expert_select_up_gate_silu_rows8_pdl_producer", variants_by_rows={8: "rows8_pdl_primary"}),
    ),
)
