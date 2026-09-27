# Tests

- `host/`: CPU-only contracts and behaviour, the `pytest` default (`python3 -m pytest tests/host -q`): kernel
  manifests and source inventories, package layering, the model registry, the session engine's prefill-path
  decisions, and the Messages API server against a fake engine.
- `contracts/`: runnable host checks of model geometry, mailbox formulas, and residual-export constants,
  plus golden data. Run `python3 -m pytest tests/contracts -q`. The GPU checks in this directory are
  placeholders and always skip; see the [contract status](contracts/README.md).
- `support/`: helpers shared by the host tests and the GPU harnesses.
- `gpu/`: harnesses that need the eight GPUs, the checkpoints named in `naivert/models/registry.py` and the CUDA
  environment (`NAIVERT_CUDA_HOME`, `NAIVERT_CUTLASS_INCLUDE`): `attn_dsa/`, `attn_swa/`, `attn_fused/` (attention
  kernels against torch references), `naive_n0p5_ops/` (model_ops kernels), `naive_n0p5_draft/` (draft kernels, the
  persistent round, request and session sets, acceptance), `naive_n0p5_e2e/` (decode and verify end to end),
  `prefill_native/` (the 4096-row prefill executor), `serve/` (checks against a running server).

Gates: decode `tests/gpu/naive_n0p5_draft/round/run.py --persistent`; acceptance
`tests/gpu/naive_n0p5_draft/round/request_set.py --acceptance-only --draft-pdl --prefill native`; served endpoints
`tests/gpu/serve/check_sessions_http.py`, `check_sse.py`, `check_sessions_hitrate.py`, `check_sampling_http.py`,
`check_resume_policy.py`.
