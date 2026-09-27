"""Naive N W8A8 rows8 router with a reusable K128-quantized activation image."""
from dataclasses import replace
from naivert.kernels.opgen import Binding, Launch, TensorParam, Variant
from naivert.kernels.ops.rmsnorm_expert_proj import OP as _BASE

OP = replace(_BASE,
    name="rmsnorm_expert_proj_prepared",
    source="naivert/kernels/ops/rmsnorm_expert_proj_prepared.py",
    namespace="naivert_prepared_router", args_struct="Args",
    args_include="naivert/common/prepared_mlp_input_args.hpp",
    variants=(Variant("rows8", kernel="naivert_rmsnorm_expert_proj_prepared_rows8_kernel",
        body="naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjRows8Body", launch_bounds="384, 1",
        run_call="naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjRows8Body::run<true, naivert::kExpertActivationFp32>(args.base, shared, args.prepared_x)"),
      # The PDL form waits for the attention tail (as the base variant) and triggers its dependents early;
      # the dependent selector's wait_for_primary covers the prepared image (griddepcontrol.wait = full completion).
      Variant("rows8_pdl_consumer", kernel="naivert_rmsnorm_expert_proj_prepared_rows8_pdl_consumer_kernel",
        body="naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjRows8Body", launch_bounds="384, 1", prologue="naivert::pdl::trigger_dependents<true>();",
        run_call="naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjRows8Body::run<true, naivert::kExpertActivationFp32>(args.base, shared, args.prepared_x)"),),
    launch=Launch(grid="naivert_replica_rmsnorm_expert_proj::kRmsPersistentCtas",
                  block="384", shared_bytes="naivert_replica_rmsnorm_expert_proj::kRmsSharedBytes"),
    params=tuple(replace(p, field="base."+p.field, optional=True) if p.name == "x_mlp"
                 else replace(p, field="base."+p.field) for p in _BASE.params)+(TensorParam(
        "prepared_x", dtype=None, field="prepared_x", ptr_type="unsigned char*",
        dtype_expr="naivert::kExpertActivationFp32 ? at::kByte : at::ScalarType::Undefined",
        numel="naivert::prepared_mlp::kBytes", alignment=16),),
    rows_field="base.tokens",
    bindings=(Binding("launch_rmsnorm_expert_proj_prepared", variants_by_rows={8:"rows8"}),
              Binding("launch_rmsnorm_expert_proj_prepared_pdl_producer", variants_by_rows={8:"rows8_pdl_consumer"}),
      # The prepared selector consumes only the quantized packet. Keep the
      # original bindings for callers that also need normalized BF16 output.
      Binding("launch_rmsnorm_expert_proj_prepared_only", variants_by_rows={8:"rows8"}, omit=("x_mlp",)),
      Binding("launch_rmsnorm_expert_proj_prepared_only_pdl_producer", variants_by_rows={8:"rows8_pdl_consumer"}, omit=("x_mlp",))),
)

# Resolve gamma alignment before capture. Keep the existing public bindings
# and 16-byte-aligned path; both explicit variants are canonicalized as well.
_wide_body = "naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjRowsBody<8, true>"
_wide = tuple(replace(v, name=v.name + "_gamma_aligned",
    kernel=v.kernel.removesuffix("_kernel") + "_gamma_aligned_kernel",
    body=_wide_body, run_call=v.run_call.replace(v.body, _wide_body)) for v in OP.variants)
OP = replace(OP, variants=(*OP.variants, *_wide), launch=replace(OP.launch,
    variant_selector="""(variant == Variant::kRows8 || variant == Variant::kRows8GammaAligned)
      ? ((reinterpret_cast<std::uintptr_t>(args.base.gamma) & 31u) == 0
          ? Variant::kRows8GammaAligned : Variant::kRows8)
      : ((variant == Variant::kRows8PdlConsumer || variant == Variant::kRows8PdlConsumerGammaAligned)
          ? ((reinterpret_cast<std::uintptr_t>(args.base.gamma) & 31u) == 0
              ? Variant::kRows8PdlConsumerGammaAligned : Variant::kRows8PdlConsumer)
          : variant)"""))
