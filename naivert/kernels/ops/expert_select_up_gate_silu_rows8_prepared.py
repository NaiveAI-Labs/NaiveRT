"""Naive N W8A8 rows8 gate/up consuming the router's prepared activation image."""
from dataclasses import replace
from naivert.kernels.opgen import Binding, Launch, TensorParam, Variant
from naivert.kernels.ops.expert_select_up_gate_silu_rows8 import OP as _BASE

OP = replace(_BASE,
    name="expert_select_up_gate_silu_rows8_prepared",
    source="naivert/kernels/ops/expert_select_up_gate_silu_rows8_prepared.py",
    namespace="naivert_prepared_select", args_struct="Args", entry_usings=(),
    args_include="naivert/common/prepared_mlp_input_args.hpp",
    variants=(Variant("rows8", kernel="naivert_expert_select_up_gate_silu_rows8_prepared_kernel",
        body="naivert_mtp_expert_select_rows8::ExpertSelectUpGateSiluRows8Body", launch_bounds="384, 1", shared_align=128,
        run_call="naivert_mtp_expert_select_rows8::ExpertSelectUpGateSiluRows8Body::run<false, naivert::kExpertActivationFp32>(args.base, shared, args.prepared_x)"),
      # Waits for the router (the prepared image and the scores) before the bulk copy, triggers early.
      Variant("rows8_pdl_primary", kernel="naivert_expert_select_up_gate_silu_rows8_prepared_pdl_primary_kernel",
        body="naivert_mtp_expert_select_rows8::ExpertSelectUpGateSiluRows8Body", launch_bounds="384, 1", shared_align=128, prologue="naivert::pdl::trigger_dependents<true>();",
        run_call="naivert_mtp_expert_select_rows8::ExpertSelectUpGateSiluRows8Body::run<true, naivert::kExpertActivationFp32>(args.base, shared, args.prepared_x)"),),
    launch=Launch(grid="args.base.total_tiles", block="384",
                  shared_bytes="naivert_mtp_expert_select_rows8::kRows8SharedBytes"),
    params=tuple(replace(p, field="base."+p.field) for p in _BASE.params)+(TensorParam(
        "prepared_x", dtype=None, field="prepared_x", ptr_type="const unsigned char*",
        dtype_expr="naivert::kExpertActivationFp32 ? at::kByte : at::ScalarType::Undefined",
        numel="naivert::prepared_mlp::kBytes", alignment=16),),
    fills=tuple(("base."+n,v) for n,v in _BASE.fills),
    bindings=(Binding("launch_expert_select_up_gate_silu_rows8_prepared",variants_by_rows={8:"rows8"}),
              Binding("launch_expert_select_up_gate_silu_rows8_prepared_pdl_producer",variants_by_rows={8:"rows8_pdl_primary"}),),
)
