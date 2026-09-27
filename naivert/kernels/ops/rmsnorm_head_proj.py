"""Op descriptor: final RMSNorm + lm_head projection (rows1, rows1 PDL consumer, rows8, rows8 PDL consumer)."""

from __future__ import annotations

from naivert.kernels.opgen import (
    Binding,
    IntParam,
    Launch,
    OpDescriptor,
    PointerListParam,
    TensorParam,
    Variant,
)

OP = OpDescriptor(
    name="rmsnorm_head_proj",
    source="naivert/kernels/ops/rmsnorm_head_proj.py",
    namespace="naivert_replica_rmsnorm_head_proj",
    args_struct="RMSNormHeadProjArgs",
    args_include="naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_args.hpp",
    header_includes=("naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_traits.cuh",),
    body_includes=(
        "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_body.cuh",
        "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_rows_body.cuh",
    ),
    variants=(
        Variant(
            "rows1",
            kernel="naivert_rmsnorm_head_proj_rows1_kernel",
            body="RMSNormHeadProjBody",
            maxnreg=80,
            run_call="RMSNormHeadProjBody::run<false>(args, shared)",
            shared_align=128,
        ),
        Variant(
            "rows1_pdl_consumer",
            kernel="naivert_rmsnorm_head_proj_rows1_pdl_consumer_kernel",
            body="RMSNormHeadProjBody",
            maxnreg=80,
            run_call="RMSNormHeadProjBody::run<true>(args, shared)",
            shared_align=128,
        ),
        # One eight-row launch for verify-8 (48 KiB panes: eight hidden pages + the weight page).
        Variant(
            "rows8",
            kernel="naivert_rmsnorm_head_proj_rows8_kernel",
            body="RMSNormHeadProjRows8Body",
            maxnreg=128,
            run_call="RMSNormHeadProjRows8Body::run<false>(args, shared)",
            shared_bytes="kRows8SharedBytes",
        ),
        # Under PDL the consumers wait for the primary (the last expert_down) before reading
        # exp_out; the helper warps stream the weight pages meanwhile.
        Variant(
            "rows8_pdl_consumer",
            kernel="naivert_rmsnorm_head_proj_rows8_pdl_consumer_kernel",
            body="RMSNormHeadProjRows8Body",
            maxnreg=128,
            run_call="RMSNormHeadProjRows8Body::run<true>(args, shared)",
            shared_bytes="kRows8SharedBytes",
        ),
    ),
    launch=Launch(
        grid="kGridCtas",
        block="kBlockThreads",
        shared_bytes="kSharedBytes",
        carveout=100,
        fixed_fields={"total_tiles": "kOutputTiles", "active_ctas": "kGridCtas"},
        guard="args.dim == kHidden && args.vocab == kVocabShard",
    ),
    params=(
        TensorParam(
            "exp_out", dtype="bf16", field="exp_out", ptr_type="const __nv_bfloat16*",
            rows_source=True, rows_divisor="hidden",
        ),
        TensorParam("final_norm", dtype="f32", field="final_norm", ptr_type="const float*", min_numel="hidden"),
        TensorParam(
            "lm_head", dtype="bf16", field="lm_head", ptr_type="const __nv_bfloat16*", min_numel="hidden * vocab",
        ),
        TensorParam("logits_out", dtype="f32", field="logits_out", ptr_type="float*", min_numel="active_rows * vocab"),
        TensorParam(
            "last_hidden_states", dtype="bf16", field="last_hidden_states", ptr_type="__nv_bfloat16*",
            min_numel="active_rows * hidden",
        ),
        TensorParam(
            "workspace", dtype="u8", field="workspace_base", ptr_type="std::uint8_t*", min_numel="active_rows * 64",
        ),
        IntParam("hidden", field="dim", must_equal="naivert_replica_rmsnorm_head_proj::kHidden"),
        IntParam("vocab", field="vocab", must_equal="naivert_replica_rmsnorm_head_proj::kVocabShard"),
        PointerListParam(
            "sampler_inbox_pointers", field="sampler_inbox_peers", count=8, element="const std::uint8_t*",
        ),
        IntParam("tp_rank", field="tp_rank", equals_device=True),
    ),
    bindings=(
        Binding("launch_rmsnorm_head_proj_rows1", variants_by_rows={1: "rows1", 8: "rows8"}),
        Binding("launch_rmsnorm_head_proj_rows1_pdl_consumer",
                variants_by_rows={1: "rows1_pdl_consumer", 8: "rows8_pdl_consumer"}),
    ),
)
