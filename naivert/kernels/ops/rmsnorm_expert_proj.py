"""Op descriptor: post-attention RMSNorm + MoE router projection (rows1, rows1 PDL producer, rows8)."""

from __future__ import annotations

from naivert.kernels.opgen import (
    Binding,
    IntParam,
    Launch,
    OpDescriptor,
    TensorParam,
    Variant,
)

OP = OpDescriptor(
    name="rmsnorm_expert_proj",
    source="naivert/kernels/ops/rmsnorm_expert_proj.py",
    namespace="naivert_replica_rmsnorm_expert_proj",
    args_struct="RMSNormExpertProjArgs",
    args_include="naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_args.hpp",
    header_includes=("naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_traits.cuh",),
    body_includes=("naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_body.cuh",),
    entry_includes=("naivert/pdl/grid_dependency.cuh",),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_rmsnorm_expert_proj_rows1_kernel",
            body="RMSNormExpertProjBody",
            run_call="RMSNormExpertProjBody::run(args, shared)",
            shared_align=128,
            launch_bounds="kRmsThreads, 1",
        ),
        Variant(
            "rows1_pdl_producer",
            kernel="naivert_rmsnorm_expert_proj_rows1_pdl_producer_kernel",
            body="RMSNormExpertProjBody",
            run_call="RMSNormExpertProjBody::run(args, shared)",
            shared_align=128,
            launch_bounds="kRmsThreads, 1",
            prologue="naivert::pdl::trigger_dependents<true>();",
        ),
        Variant(
            "rows8",
            kernel="naivert_rmsnorm_expert_proj_rows8_kernel",
            body="RMSNormExpertProjRows8Body",
            launch_bounds="kRmsThreads, 1",
            run_call="RMSNormExpertProjRows8Body::run<true>(args, shared)",
        ),
        Variant(
            "rows8_pdl_consumer",
            kernel="naivert_rmsnorm_expert_proj_rows8_pdl_consumer_kernel",
            body="RMSNormExpertProjRows8Body",
            launch_bounds="kRmsThreads, 1",
            prologue="naivert::pdl::trigger_dependents<true>();",
            run_call="RMSNormExpertProjRows8Body::run<true>(args, shared)",
        ),
    ),
    launch=Launch(grid="kRmsPersistentCtas", block="kRmsThreads", shared_bytes="kRmsSharedBytes"),
    params=(
        TensorParam(
            "hidden", dtype="bf16", field="hidden", ptr_type="const __nv_bfloat16*",
            rows_source=True, rows_divisor="hidden_dim",
        ),
        TensorParam("gamma", dtype="f32", field="gamma", ptr_type="const float*", min_numel="hidden_dim"),
        TensorParam(
            "router_weight", dtype=None, field="router_weight", ptr_type="const naivert::RouterWeight*",
            dtype_expr="naivert::kExpertActivationFp32 ? at::kFloat : at::kBFloat16",
            min_numel="experts * hidden_dim",
        ),
        TensorParam("scores", dtype="f32", field="scores", ptr_type="float*", min_numel="active_rows * experts"),
        TensorParam(
            "x_mlp", dtype="bf16", field="x_mlp", ptr_type="__nv_bfloat16*", min_numel="active_rows * hidden_dim",
        ),
        IntParam("hidden_dim", field="hidden_dim", must_equal="naivert_replica_rmsnorm_expert_proj::kRmsHiddenDim"),
        IntParam("experts", field="experts", must_equal="naivert_replica_rmsnorm_expert_proj::kRmsExperts"),
    ),
    rows_field="tokens",
    bindings=(
        Binding("launch_rmsnorm_expert_proj", variants_by_rows={1: "rows1", 8: "rows8"}),
        Binding("launch_rmsnorm_expert_proj_pdl_producer", variants_by_rows={1: "rows1_pdl_producer", 8: "rows8_pdl_consumer"}),
    ),
)
