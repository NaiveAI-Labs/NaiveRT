# FlashInfer mixed communication source

Pinned FlashInfer 0.6.12 source for BF16 numerical collectives. `manifest.json`
records every source hash and the upstream identity. The CUDA sources and
template instantiations are unchanged; `generator.patch` extracts the upstream
host generator and redirects its paths. The source uses Apache-2.0 (`LICENSE`).

`tools/vendor_prefill_collectives.py --source EXPORT --check` reproduces this
snapshot. EXPORT contains `csrc/`, `include/`, the wheel's `LICENSE`, and
`upstream_jit_comm.py` copied from `flashinfer/jit/comm.py`.

`naivert.prefill.native.collectives` owns in-process peer pointers and scratch.
It uses FlashInfer's existing JIT builder, TVM-FFI, CUDA and NVSHMEM development
libraries, with explicit architecture flags during device linking. No binary or
generated instantiation source is stored in the repository. No NVSHMEM context
or distributed process group is initialized for this intra-node path.
The tested environment uses CUDA 13.2, FlashInfer 0.6.12, TVM-FFI 0.1.11 and
`nvidia-nvshmem-cu13` 3.4.5. Its shared host library and static device library are
still build/load dependencies even though the intra-node path does not initialize NVSHMEM.

These are numerical collectives. Upstream canonicalizes adjacent BF16 negative
zero pairs to positive zeros because the packed negative-zero word is its empty
buffer sentinel. Do not use this path to transport packed FP8 or scale bytes;
the host adapter delegates non-BF16 tensors to the original communication owner.

Runtime scratch must outlive captured graphs. `NativePrefill.close()` drains
the streams and resets its graphs before releasing this owner.
