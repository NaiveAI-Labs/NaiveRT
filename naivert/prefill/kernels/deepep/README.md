# DeepEP dispatch source

This is DeepEP 1.2.1 from the installed source archive used by the prefill audit.
The archive has no Git metadata; `manifest.json` records its upstream file hashes
and every adapted file hash. The MIT license is retained in `LICENSE`.

All 13 files under `csrc/kernels/` are unchanged. The host patch:

- Fills the existing peer-pointer tables directly for eight devices in one
  process, instead of reopening that process's own CUDA IPC allocations.
- Enables peer access and guards the owning device during setup and teardown.
- Releases the Python GIL during dispatch and collective cleanup.
- Allows explicit destruction after the owner has drained **every** peer,
  including cleanup after incomplete setup.
- Uses module-local pybind types and the name `naivert_deep_ep_cpp`, so an
  installed `deep_ep_cpp` can coexist.
- Builds intra-node sources only, without NVSHMEM, and explicitly device-links
  the relocatable CUDA objects.

The runtime loader in `naivert/prefill/native/deepep.py` verifies source hashes,
builds into the torch extension cache, and loads that binary by its exact path.
It does not import the installed `deep_ep` package or initialize distributed
process groups. The cache key includes source hashes, PyTorch, Python, CUDA,
architecture and compiler selection. Set `NAIVERT_CUDA_HOME`/`CUDA_HOME` and
`TORCH_CUDA_ARCH_LIST` as for the rest of the B300 runtime (default `10.3a`).

`NativePrefill` uses this dispatch by default. `NAIVERT_PREFILL_DISPATCH=nccl`
selects the prior packed-row implementation for comparisons. The return path
still uses the existing per-expert BF16 outputs and fixed-slot FP32 weighting.
The FP8 values and packed integer UE8M0 scales are transported without conversion.

The measured configuration uses 128 SMs and 805,316,608 communication bytes per
GPU, plus the library's 32 MiB workspace. Whole-chunk graphs include dispatch.
When only segment graphs are enabled, expert processing stays eager because
DeepEP allocates fresh receive tensors outside those graphs.

Call `NativePrefill.close()` when its owner shuts down. It drains streams,
destroys captured graphs, frees peer buffers and stops the launch thread pool.
Calls after close are rejected; repeated close is safe. Do not depend on Python
destructors to run multi-device cleanup.

Reproduce vendoring with `tools/vendor_prefill_deepep.py --source <DeepEP tree>`;
add `--check` to verify the checked-in source and manifest. Only the recorded
upstream host and device hashes are accepted. `host_adapter.patch` records all
host-source/build changes. This README is local documentation, not upstream code.
