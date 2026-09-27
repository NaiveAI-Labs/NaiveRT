"""Executor namespaces beside this tree's W8A8 decode weights: the operands of
``naivert.prefill.operands`` (shared caches, bf16 projections, fp32 router, EP8 experts in
DeepGEMM's masked layout) -- the same loader, kept as the native executor's entry point."""
from naivert.prefill.operands import kernel_prefill_weights as native_weights  # noqa: F401

__all__ = ["native_weights"]
