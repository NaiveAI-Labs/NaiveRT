"""The naive_ops_lib ``dsa.cuda`` modules naivert's prefill binds (see ``prefill_ext/manifest.json``).

Upstream's ``__init__`` imports the decode, heuristic and context-parallel top-k kernels,
which naivert does not copy, so this package exports nothing.
"""
