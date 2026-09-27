"""Prefill executors and cache adapters for Naive N0.5 on eight GPU streams.

``native.executor.NativePrefill`` is the serving path: fixed 4,096-row chunks,
CUDA graph capture, and tensor- and expert-parallel kernel execution.
``executor`` and ``draft_features`` retain the bucketed reference path used by
GPU comparison harnesses. ``operands`` shares cache tensors with decode/verify
and prepares prefill weights. ``draft_features`` also publishes the target
residual rows consumed by the speculative draft.

Kernel bindings live in ``naivert.kernels.naive_n0p5.prefill``. Numerical
helpers shared by these executors live in ``ops``.
"""
