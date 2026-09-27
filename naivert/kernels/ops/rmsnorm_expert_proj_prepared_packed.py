"""Explicit original+half-swapped FP32 router weight container for verify-8."""
from dataclasses import replace
from naivert.kernels.ops.rmsnorm_expert_proj_prepared import OP as _BASE


def packed_variant(v):
    wide = "true" if v.name.endswith("_gamma_aligned") else "false"
    body = ("naivert_replica_rmsnorm_expert_proj::"
            f"RMSNormExpertProjRowsBody<8, {wide}, naivert::kExpertActivationFp32>")
    return replace(v, body=body, run_call=v.run_call.replace(v.body, body))


OP = replace(_BASE,
    name="rmsnorm_expert_proj_prepared_packed",
    source="naivert/kernels/ops/rmsnorm_expert_proj_prepared_packed.py",
    namespace="naivert_prepared_packed_router",
    entry_usings=("using Args = naivert_prepared_router::Args;",),
    variants=tuple(packed_variant(v) for v in _BASE.variants),
    params=tuple(replace(p, min_numel=None, numel="2 * experts * hidden_dim",
                        dtype_expr="naivert::kExpertActivationFp32 ? at::kFloat : at::ScalarType::Undefined")
                 if p.name == "router_weight" else p for p in _BASE.params),
    bindings=tuple(replace(b, name=b.name.replace("_prepared", "_prepared_packed"))
                   for b in _BASE.bindings),
)
