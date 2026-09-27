# FlashInfer joint Q/K RoPE

Unchanged FlashInfer 0.6.12 CUDA sources, TVM-FFI bindings and all local header
dependencies. `manifest.json` pins every upstream file by SHA256; the license
is Apache-2.0. `tools/vendor_prefill_rope.py --source EXPORT --check` reproduces
the snapshot from the installed wheel's `data/` tree, `LICENSE`, and
`jit/rope.py` exported as `upstream_jit_rope.py`.

`naivert.prefill.native.rope` uses FlashInfer's JIT builder and direct TVM-FFI
entry. Include paths put these vendored headers before the installed wheel's
headers. No generated binary is checked into Git. The tested toolchain is
CUDA 13.2, FlashInfer 0.6.12, Torch 2.11.0+cu130 and TVM-FFI 0.1.11 on B300.

The native executor applies the table-based, interleaved-pair operator to Q
and K together: eight Q heads, one K head, head dimension 192, rotary dimension
64. Inputs are strided views of the QKV projection; outputs are separate
contiguous BF16 tensors. This head dimension uses upstream's general
`RopeQuantize` implementation with BF16 outputs and scale 1; it does not
quantize activations to FP8. The existing table is rearranged once at setup
from interleaved cos/sin pairs to separate cos and sin halves.

The DSA indexer retains its existing RoPE kernel because its query site has
a different FMA spelling. This integration changes only attention Q/K.
