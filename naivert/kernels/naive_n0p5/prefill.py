"""Kernel bindings shared by the reference and native prefill executors.

Functions take explicit tensors and launch on the current CUDA device and stream.
Callers own model weights, cache positions, streams, and workspaces. The reference
executor uses ``STAGE_BINDINGS`` to describe its stage inventory; native prefill
also uses optimized grouped-GEMM, dispatch, RoPE, and indexer helpers in this module.
The inventory is descriptive metadata, not an execution schedule.

Bindings use installed sgl_kernel, DeepGEMM, and cuBLAS kernels; vendored indexer,
top-k, sparse attention, and FA4 sources; and NaiveRT layout, quantization, routing,
and cache helpers. TP communication is provided separately by ``TP8Comm`` and the
native executor's collective adapter. The serving engine obtains prompt logits
from the final decode step rather than a separate prefill head runtime.

Importing this module creates no CUDA context. Call ``check_versions()`` and
``prepare_device(device)`` before the first kernel call or graph capture.
Preparation loads per-device DeepGEMM runtime state and compiles the top-k module.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
from typing import Iterable, NamedTuple

import torch


class PrefillKernelError(RuntimeError):
    """A prefill kernel is missing, mis-versioned, unprepared, or refused its operands."""


# ------------------------------------------------------------------------------ versions
#: Validated dependency versions, by distribution name.  ``check_versions`` refuses anything else.
PINNED_DISTRIBUTIONS: dict[str, str] = {
    "sglang-kernel": "0.4.4",
    "sgl-deep-gemm": "0.1.4",
    "nvidia-cutlass-dsl": "4.5.2",
    "triton": "3.6.0",
    "ttfs": "0.6.3",
}
#: Held to their metapackage's pin: ``nvidia-cutlass-dsl`` installs no files itself; the
#: ``cutlass`` package the FA4 CuTe copy imports comes from these, required ``==4.5.2``.
PINNED_COMPONENTS: dict[str, str] = {
    "nvidia-cutlass-dsl-libs-base": "4.5.2",
    "nvidia-cutlass-dsl-libs-cu13": "4.5.2",
}
#: Additional pinned dependencies used by the kernels below:
#: their kernel definitions compile into kernels this path launches.  ``sgl_kernel.rmsnorm`` and
#: ``fused_add_rmsnorm`` hand the call to flashinfer's CuTe DSL norms, and the FA4 copy builds
#: its kernels from quack's CuTe DSL helpers (``copy_utils``, ``layout_utils``, ``ParamsBase``).
EXCEPTION_PINS: dict[str, str] = {
    "flashinfer-python": "0.6.12",
    "quack-kernels": "0.5.0",
}
#: Recorded next to the pins, not enforced: torch, and the loaders underneath (DeepGEMM's
#: runtime is a tvm-ffi module, the radix top-k compiles through cuda-bindings' NVRTC, and the
#: FA4 copy reshapes with einops).
RECORDED_DISTRIBUTIONS: tuple[str, ...] = ("torch", "apache-tvm-ffi", "cuda-bindings", "einops")


def installed_distributions(names: Iterable[str]) -> dict[str, str]:
    """``{distribution: version}``, ``"missing"`` for one that is not installed."""
    found = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "missing"
    return found


def check_versions() -> dict[str, str]:
    """Refuse unless every pinned distribution is installed at its pin; return all versions."""
    pins = {**PINNED_DISTRIBUTIONS, **PINNED_COMPONENTS, **EXCEPTION_PINS}
    found = installed_distributions(pins)
    wrong = {name: f"{found[name]} (pinned {want})"
             for name, want in pins.items() if found[name] != want}
    if wrong:
        raise PrefillKernelError(f"prefill kernel distributions off their pins: {wrong}")
    return {**found, **installed_distributions(RECORDED_DISTRIBUTIONS)}


# ----------------------------------------------------------------------------- constants
# Shapes for one TP8 rank of Naive N0.5.
HIDDEN = 4096
VOCAB = 152576
RANKS = 8
VOCAB_SHARD = VOCAB // RANKS                    # 19072
ATTN_HEADS = 8                                  # 64 query heads / TP8
KV_HEADS = 1                                    # DSA max(1, 4 // 8), SWA 8 // 8
HEAD_DIM = 192
V_HEAD_DIM = 128
QKV_ROWS = ATTN_HEADS * HEAD_DIM + KV_HEADS * (HEAD_DIM + V_HEAD_DIM)     # 1856
O_PROJ_COLS = ATTN_HEADS * V_HEAD_DIM                                     # 1024
ROPE_DIM = 64
DSA_ROPE_THETA = 1.0e7
SWA_ROPE_THETA = 1.0e4
SWA_WINDOW = (127, 0)                           # sliding_window 128
ATTN_SCALE = HEAD_DIM ** -0.5
VALUE_SCALE = 0.707
NORM_EPS = 1.0e-5
INDEX_HEADS = 16
INDEX_DIM = 128
INDEX_TOPK = 2048
INDEX_QKW_ROWS = INDEX_HEADS * INDEX_DIM + INDEX_DIM + INDEX_HEADS       # 2192
INDEX_WEIGHT_SCALE = INDEX_HEADS ** -0.5
INDEX_LOGITS_BUDGET_MB = 512                   # the per-card logits workspace
_PAD_LAST = torch.iinfo(torch.int32).max        # the -1 padding's sort key
DENSE_INTERMEDIATE = 16384 // RANKS             # 2048
EXPERTS = 256
EXPERTS_PER_RANK = EXPERTS // RANKS             # 32
TOPK = 8
EXPERT_INTERMEDIATE = 2048
QUANT_GROUP = 128
QUANT_EPS = 1e-10
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)     # 448.0

_EXT = "naivert.kernels.naive_n0p5.prefill_ext"
_FA4 = _EXT + ".fa4_cute.interface"
_TOPK = _EXT + ".naive_ops.dsa.triton.topk"
_TOPK_NVRTC = _EXT + ".naive_ops.dsa.cuda.indexer_topk"
_LOGITS = _EXT + ".naive_ops.dsa.triton.logits"
_LOGITS_GLUON = _EXT + ".naive_ops.dsa.gluon.logits_prefill_sm100"
_SPARSE = _EXT + ".naive_ops.dsa.triton.sparse_attn"
_SHARED = "naivert.prefill.ops"
_ADAPTERS = "naivert.prefill.adapters"
_COMMS = "naivert.prefill.comms"


@functools.lru_cache(maxsize=None)
def _module(name: str):
    """Import a kernel module on first use; afterwards a dict lookup."""
    return importlib.import_module(name)


def _on_current_device(*tensors: torch.Tensor | None) -> None:
    index = torch.cuda.current_device()
    for t in tensors:
        if t is not None and (t.device.type != "cuda" or t.device.index != index):
            raise PrefillKernelError(
                f"a {tuple(t.shape)} operand is on {t.device} but cuda:{index} is current; "
                "the kernels launch on the current device")


def _expect(t: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple | None = None) -> None:
    if t.dtype is not dtype or (shape is not None and tuple(t.shape) != tuple(shape)):
        want = dtype if shape is None else f"{tuple(shape)} {dtype}"
        raise PrefillKernelError(f"{name} must be {want}, got {tuple(t.shape)} {t.dtype}")


def _ceil4(n: int) -> int:
    return (int(n) + 3) // 4 * 4


# ------------------------------------------------------------------- per-device runtime
class DeepGemmRuntime(NamedTuple):
    module: object          # the tvm-ffi module of this device's _C.so copy
    path: Path
    sha256: str


_DEEP_GEMM: dict[int, DeepGemmRuntime] = {}


def deep_gemm_cache_root() -> Path:
    """Where the per-device ``_C.so`` copies live: ``$NAIVERT_DEEP_GEMM_CACHE`` or ~/.cache."""
    env = os.environ.get("NAIVERT_DEEP_GEMM_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "naivert" / "deep_gemm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_copy(src: Path, sha256: str, index: int) -> Path:
    """A byte-identical copy of ``src`` for device ``index``, verified by its sha256."""
    dst = deep_gemm_cache_root() / sha256[:16] / f"cuda{index}" / src.name
    if dst.is_file() and _sha256(dst) == sha256:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{src.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    if _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise PrefillKernelError(f"copy of {src} to {tmp} does not hash to {sha256}")
    os.replace(tmp, dst)
    return dst


def prepare_device(device: torch.device | int | str) -> DeepGemmRuntime:
    """Load ``device``'s kernel runtimes; call before its first stage call and any capture.

    DeepGEMM keeps its JIT runtime state per loaded ``_C.so``, and that state binds to the
    device of the first kernel launched through it: every other device then fails with CUDA
    error 400.  One process drives all eight cards here, and ``dlopen`` hands back the module
    already loaded for a path's inode, so each device loads its own byte-identical copy of the
    wheel's ``_C.so`` (the wheel is not modified; the copy's sha256 is checked against it).
    PDL is enabled on each copy while its device is current.  A graph capturing these kernels
    must pass its device's own ``stream=`` to ``torch.cuda.graph``: the class-level default
    capture stream is created on whichever device captured first.

    The radix top-k compiles one NVRTC module per device context (``_IndexerTopKKernels.get``);
    that compile runs here too.
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda" or dev.index is None:
        raise PrefillKernelError(f"prepare_device wants an indexed CUDA device, got {dev}")
    ready = _DEEP_GEMM.get(dev.index)
    if ready is not None:
        return ready
    check_versions()
    import deep_gemm
    from deep_gemm.cuda_helpers import find_cuda_home
    import tvm_ffi

    pkg = Path(deep_gemm.__file__).resolve().parent
    src = pkg / "_C.so"
    sha256 = _sha256(src)
    path = _device_copy(src, sha256, dev.index)
    module = tvm_ffi.load_module(str(path))
    module.init(str(pkg), find_cuda_home())
    with torch.cuda.device(dev):
        module.set_pdl(True)
        _module(_TOPK_NVRTC)._IndexerTopKKernels.get()
    runtime = DeepGemmRuntime(module=module, path=path, sha256=sha256)
    _DEEP_GEMM[dev.index] = runtime
    return runtime


def deep_gemm_module(device: torch.device):
    """The DeepGEMM runtime ``prepare_device`` loaded for ``device``."""
    runtime = _DEEP_GEMM.get(torch.device(device).index)
    if runtime is None:
        raise PrefillKernelError(f"prepare_device({device}) has not run")
    return runtime.module


# --------------------------------------------------------------------- embedding, norms
def embed(tokens: torch.Tensor, shard: torch.Tensor, *, first_row: int,
          out: torch.Tensor | None = None) -> torch.Tensor:
    """E1: one rank's vocab-parallel lookup, ``[T]`` ids -> ``[T, 4096]`` bf16.

    ``shard`` is the rank's ``[19072, 4096]`` rows of the table (``embedding[r * 19072 :
    (r + 1) * 19072]``) and ``first_row`` the first token id it holds; a token outside the
    shard gets a zero row.  E2 sums the ranks' rows (``TP8Comm.all_reduce_``), and every token
    lies in exactly one shard.
    """
    _on_current_device(tokens, shard, out)
    local = tokens.to(torch.int64) - int(first_row)
    outside = (local < 0) | (local >= int(shard.shape[0]))
    rows = torch.index_select(shard, 0, local.masked_fill(outside, 0), out=out)
    return rows.masked_fill_(outside.unsqueeze(-1), 0)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = NORM_EPS,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """D1 / F2: ``sgl_kernel.rmsnorm``.  The weight must have ``x``'s dtype (bf16 here)."""
    _on_current_device(x, weight, out)
    _expect(weight, "rmsnorm weight", x.dtype)
    return _module("sgl_kernel").rmsnorm(x, weight, eps, out=out)


def add_rmsnorm_(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *,
                 eps: float = NORM_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """S1 / S9 / D24 / D24': in place, ``residual += x`` then ``x = rmsnorm(residual)``."""
    _on_current_device(x, residual, weight)
    _expect(weight, "fused_add_rmsnorm weight", x.dtype)
    _module("sgl_kernel").fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual


# ------------------------------------------------------------------------ projections
def linear(x: torch.Tensor, weight: torch.Tensor, *, out: torch.Tensor | None = None
           ) -> torch.Tensor:
    """S2 S8 D2 D5 D23 M1 M3: ``x @ weight.T``, the cuBLAS GEMM ``F.linear`` launches."""
    _on_current_device(x, weight, out)
    if x.dtype is not weight.dtype:
        raise PrefillKernelError(f"linear of {x.dtype} rows by a {weight.dtype} weight")
    return torch.mm(x, weight.mT, out=out)


def router_logits(x: torch.Tensor, gate: torch.Tensor, *, out: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """X1: ``[T, 4096]`` -> fp32 ``[T, 256]``; the gate is fp32 and TF32 must be off."""
    _on_current_device(x, gate, out)
    _expect(gate, "router gate", torch.float32, (EXPERTS, HIDDEN))
    if torch.backends.cuda.matmul.allow_tf32:
        raise PrefillKernelError("router_logits requires TF32 matmul disabled")
    return torch.mm(x.float(), gate.mT, out=out)


# ----------------------------------------------------------------------- SWA attention
def split_qkv(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S3: ``[T, 1856]`` -> views q ``[T, 8, 192]``, k ``[T, 1, 192]``, v ``[T, 1, 128]``."""
    if qkv.dim() != 2 or int(qkv.shape[1]) != QKV_ROWS:
        raise PrefillKernelError(f"qkv rows must be [T, {QKV_ROWS}], got {tuple(qkv.shape)}")
    t, q_end, k_end = int(qkv.shape[0]), ATTN_HEADS * HEAD_DIM, ATTN_HEADS * HEAD_DIM + HEAD_DIM
    return (qkv[:, :q_end].view(t, ATTN_HEADS, HEAD_DIM),
            qkv[:, q_end:k_end].view(t, KV_HEADS, HEAD_DIM),
            qkv[:, k_end:].view(t, KV_HEADS, V_HEAD_DIM))


def rope(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, *, imag: str,
         rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """S4 D3 D8: naivert's ``rope_fma`` over the first ``rope_dim`` dims; returns a new tensor.

    ``x`` is bf16 ``[T, H, D]`` or ``[T, D]`` with a unit last stride; ``table`` is naivert's
    fp32 (cos, sin)-interleaved ``[max_seq, 64]``.  naivert stores the q/k projections and the
    index-key norm permuted into interleaved rope order, which makes this the neox rope SGLang
    applies to the unpermuted rows.  ``imag`` picks the fma spelling of the imaginary component
    (``external_ops.DSA_ROPE_IMAG``).
    """
    _on_current_device(x, positions, table)
    return _module(_SHARED).rope_fma(x, positions, table, imag=imag, rope_dim=rope_dim)


def rope_qk(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
            table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """S4/D3 in the native executor: joint FlashInfer RoPE, separate contiguous outputs.

    BF16 Q/K use interleaved pairs, head dim 192 and rotary dim 64. The FP32
    table has cos in its first 32 columns and sin in its last 32; prepare it
    once from the model's interleaved table before graph capture. Indexer RoPE
    continues to use ``rope`` for its site-specific FMA spelling.
    """
    _on_current_device(q, k, positions, table)
    _expect(table, "joint RoPE table", torch.float32, (int(table.shape[0]), ROPE_DIM))
    return _module("naivert.prefill.native.rope").apply(q, k, positions, table)


def scale_values(v: torch.Tensor, *, scale: float = VALUE_SCALE,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """S5 / D4: ``v * attention_value_scale``."""
    _on_current_device(v, out)
    return torch.mul(v, scale, out=out)


def _cache_rows(t: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    return t.reshape((int(t.shape[0]),) + tuple(cache.shape[1:]))


def swa_kv_publish(k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor, *, start: int, valid: int) -> int:
    """S6 (host integers): the chunk's last ``min(valid, 128)`` rows into the 136-row ring."""
    _on_current_device(k_cache, v_cache, k, v)
    lw = {"k_cache": k_cache, "v_cache": v_cache}
    return _module(_ADAPTERS).swa_ring_publish(
        lw, _cache_rows(k, k_cache), _cache_rows(v, v_cache), start, valid)


def swa_kv_write(k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor, rows: torch.Tensor, src: torch.Tensor) -> None:
    """S6 (device tensors, graph form): ring rows ``rows`` <- chunk rows ``src``."""
    _on_current_device(k_cache, v_cache, k, v, rows, src)
    lw = {"k_cache": k_cache, "v_cache": v_cache}
    _module(_ADAPTERS).swa_ring_write(lw, _cache_rows(k, k_cache), _cache_rows(v, v_cache),
                                      rows, src)


def swa_kv_prefix(k_cache: torch.Tensor, v_cache: torch.Tensor, start: int,
                  k_out: torch.Tensor, v_out: torch.Tensor) -> int:
    """The ring rows a chunk at ``start`` still attends (``min(start, 127)``), into staging."""
    _on_current_device(k_cache, v_cache, k_out, v_out)
    lw = {"k_cache": k_cache, "v_cache": v_cache}
    return _module(_ADAPTERS).swa_ring_prefix_into(lw, start, k_out, v_out)


def swa_kv_prefix_rows(k_cache: torch.Tensor, v_cache: torch.Tensor, rows: torch.Tensor,
                       k_out: torch.Tensor, v_out: torch.Tensor) -> int:
    """S6 (device tensors, graph form): the ring rows ``rows`` into staging."""
    _on_current_device(k_cache, v_cache, rows, k_out, v_out)
    lw = {"k_cache": k_cache, "v_cache": v_cache}
    return _module(_ADAPTERS).swa_ring_prefix_rows(lw, rows, k_out, v_out)


def swa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
                  cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, *,
                  max_seqlen_q: int, max_seqlen_k: int, scale: float = ATTN_SCALE,
                  window: tuple[int, int] = SWA_WINDOW) -> torch.Tensor:
    """S7: FA4 CuTe (``FlashAttentionForwardSm100``), causal, sliding window, learnable sink.

    q ``[T, 8, 192]``, k ``[S, 1, 192]``, v ``[S, 1, 128]`` bf16 with the queries at the end of
    the keys; ``sink`` bf16 ``[8]``.  Called as SGLang's FA4 path calls it (``num_splits=0``,
    the LSE computed and dropped), on naivert's non-paged k/v rather than a page table.
    Returns ``[T, 8, 128]``.
    """
    _on_current_device(q, k, v, sink, cu_seqlens_q, cu_seqlens_k)
    _expect(sink, "attention sink", torch.bfloat16, (int(q.shape[1]),))
    result = _module(_FA4).flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q), max_seqlen_k=int(max_seqlen_k),
        softmax_scale=float(scale), causal=True, window_size=tuple(window),
        learnable_sink=sink, softcap=0.0, num_splits=0, return_lse=True)
    return result[0] if isinstance(result, tuple) else result


# ----------------------------------------------------------------------------- indexer
def split_indexer(qkw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """D5: ``[T, 2192]`` -> views q ``[T, 16, 128]``, key ``[T, 128]``, weights ``[T, 16]``."""
    if qkw.dim() != 2 or int(qkw.shape[1]) != INDEX_QKW_ROWS:
        raise PrefillKernelError(f"indexer rows must be [T, {INDEX_QKW_ROWS}], got {tuple(qkw.shape)}")
    t, q_end = int(qkw.shape[0]), INDEX_HEADS * INDEX_DIM
    return (qkw[:, :q_end].view(t, INDEX_HEADS, INDEX_DIM), qkw[:, q_end:q_end + INDEX_DIM],
            qkw[:, q_end + INDEX_DIM:])


def scale_index_weights(w: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """D6: ``w * 16 ** -0.5``."""
    _on_current_device(w, out)
    return torch.mul(w, INDEX_WEIGHT_SCALE, out=out)


def index_key_norm(key: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *,
                   eps: float = NORM_EPS) -> torch.Tensor:
    """D7: the index key's LayerNorm(128), naivert's ``index_key_norm`` (fp32 affine, bf16 out)."""
    _on_current_device(key, weight, bias)
    return _module(_SHARED).index_key_norm(key, weight, bias, eps=eps)


def quantize_index_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D9 / D10: bf16 ``[..., 128]`` -> e4m3 codes and one fp32 scale per 128 (amax / 448).

    Scales keep the inventory's trailing group dim: q ``[T, 16, 1]``, key ``[T, 1]``.
    """
    _on_current_device(x)
    if int(x.shape[-1]) != INDEX_DIM:
        raise PrefillKernelError(f"indexer rows are {INDEX_DIM} wide, got {tuple(x.shape)}")
    codes, scale = _module(_SHARED).quantize_e4m3_rows(x)
    return codes, scale.unsqueeze(-1)


def fold_index_weights(w: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    """D11: ``w.float() * q_scale`` -> fp32 ``[T, 16]`` (the ``[T, 16, 1]`` group dim flattened).

    The Naive N ``DSAIndexer`` (``attention/dsa_indexer.py:225``) folds no ``softmax_scale``; that
    factor belongs to the sparse-attention indexer (``attention/dsa/dsa_indexer.py``).
    """
    _on_current_device(w, q_scale)
    return w.float() * q_scale.reshape(w.shape)


def dsa_kv_append(k_cache: torch.Tensor, v_cache: torch.Tensor, ki_cache: torch.Tensor,
                  ki_scale_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  ki: torch.Tensor, ki_scale: torch.Tensor, *, start: int, valid: int,
                  plain: torch.Tensor | None = None) -> int:
    """D12 + D13 (host integers): k, v and the index key at positions ``start .. start+valid``.

    The index key lands swizzled in ``ki_cache`` (decode's image) and plain in ``plain``, the
    copy the indexer reads.  Returns ``start + valid``.
    """
    _on_current_device(k_cache, v_cache, ki_cache, ki_scale_cache, k, v, ki, ki_scale, plain)
    lw = {"k_cache": k_cache, "v_cache": v_cache, "ki_cache": ki_cache,
          "ki_scale": ki_scale_cache}
    rows = int(k.shape[0])
    return _module(_ADAPTERS).dsa_cache_append(
        lw, k.reshape(rows, HEAD_DIM), v.reshape(rows, V_HEAD_DIM),
        ki.reshape(rows, INDEX_DIM), ki_scale.reshape(rows), start, valid, plain=plain)


def dsa_kv_write(k_cache: torch.Tensor, v_cache: torch.Tensor, ki_cache: torch.Tensor,
                 ki_scale_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 ki: torch.Tensor, ki_scale: torch.Tensor, positions: torch.Tensor,
                 src: torch.Tensor, *, plain: torch.Tensor | None = None) -> None:
    """D12 + D13 (device tensors, graph form): row ``src[i]`` lands at ``positions[i]``."""
    _on_current_device(k_cache, v_cache, ki_cache, ki_scale_cache, k, v, ki, ki_scale,
                       positions, src, plain)
    lw = {"k_cache": k_cache, "v_cache": v_cache, "ki_cache": ki_cache,
          "ki_scale": ki_scale_cache}
    rows = int(k.shape[0])
    _module(_ADAPTERS).dsa_cache_write(
        lw, k.reshape(rows, HEAD_DIM), v.reshape(rows, V_HEAD_DIM),
        ki.reshape(rows, INDEX_DIM), ki_scale.reshape(rows), positions, src, plain=plain)


def dsa_kv_view(k_cache: torch.Tensor, v_cache: torch.Tensor, end: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """D14: the visible k ``[end, 1, 192]`` / v ``[end, 1, 128]`` -- views, no launch."""
    end = int(end)
    return k_cache[:end].view(end, KV_HEADS, HEAD_DIM), v_cache[:end].view(end, KV_HEADS, V_HEAD_DIM)


def dsa_index_key_view(plain: torch.Tensor, ki_scale_cache: torch.Tensor, end: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """D15: the visible plain index keys e4m3 ``[end, 128]`` and scales ``[end]`` -- views."""
    end = int(end)
    return plain[:end], ki_scale_cache[:end]


def indexer_chunk_rows(query_rows: int, width: int, *, budget_mb: int = INDEX_LOGITS_BUDGET_MB
                       ) -> int:
    """Query rows per logits chunk: the fp32 ``[chunk, width]`` workspace within the budget."""
    return max(1, min(int(query_rows), (int(budget_mb) << 20) // (4 * int(width))))


def indexer_select(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor,
                   k_scale: torch.Tensor, k_start: torch.Tensor, k_end: torch.Tensor, *,
                   max_seqlen: int, chunk: int | None = None, topk: int = INDEX_TOPK
                   ) -> torch.Tensor:
    """D17 D18 D19: indexer logits, radix top-k and epilogue -> int32 ``[T, 2048]`` indices.

    q e4m3 ``[T, 16, 128]``, fp32 weights ``[T, 16]`` (D11), the visible keys e4m3 ``[S, 128]``
    with fp32 scales ``[S]`` or ``[S, 1]``, and each query's int32 key range
    ``[k_start, k_end)``.  Every card runs its whole query (no striped CP, so no D16/D20): the
    logits run in chunks of ``chunk`` rows over an fp32 ``[chunk, width]`` workspace, ``width
    = max(max_seqlen, topk)`` as SGLang passes it.  Indices are global into the keys, each row
    ascending with its -1 padding last: the radix kernel's threads claim a row's
    output slots in arrival order (``atomicAdd`` in ``indexerTopK.h``). Unless an ordered
    epilogue is installed, rows that may see more than ``topk`` keys are sorted here and
    exact cutoff ties remain arrival-dependent. The native executor installs its stable
    epilogue, which resolves cutoff ties by key ID and already emits ascending rows.
    """
    _on_current_device(q, weights, k, k_scale, k_start, k_end)
    rows = int(q.shape[0])
    _expect(q, "indexer q", FP8_DTYPE, (rows, INDEX_HEADS, INDEX_DIM))
    _expect(weights, "indexer weights", torch.float32, (rows, INDEX_HEADS))
    _expect(k, "index keys", FP8_DTYPE, (int(k.shape[0]), INDEX_DIM))
    _expect(k_scale, "index key scales", torch.float32)
    _expect(k_start, "k_start", torch.int32, (rows,))
    _expect(k_end, "k_end", torch.int32, (rows,))
    width = max(int(max_seqlen), int(topk))
    selector = _module(_TOPK)
    from naivert.prefill.native.stable_dsa_topk import (
        SINGLE_BLOCK_MAX_KEYS, StableTopKEpilogue, tiled_indexer_select,
    )
    if width > SINGLE_BLOCK_MAX_KEYS and isinstance(
            selector._indexer_topk_prefill_epilogue_kernel, StableTopKEpilogue):
        # Compose the existing logits/radix/epilogue kernels on bounded key tiles.
        # Going through the vendor host loop would also run an unused full-span radix.
        return tiled_indexer_select(q, weights, k, k_scale, k_start, k_end,
                                    max_seqlen=width, chunk=chunk, topk=int(topk))
    if chunk is None:
        chunk = indexer_chunk_rows(rows, width)
    _values, indices = selector.indexer_qk_topk_prefill(
        q.contiguous(), k.view(-1, KV_HEADS, INDEX_DIM), weights.contiguous(), int(topk),
        k_start, k_end, int(chunk), max_seqlen=width, return_global_indices=True,
        topk_impl="indexer", k_scales=k_scale.contiguous())
    ordered = getattr(selector._indexer_topk_prefill_epilogue_kernel, "returns_sorted_indices", False)
    return indices if ordered or int(max_seqlen) <= int(topk) else _ascending_rows(indices)


def _ascending_rows(indices: torch.Tensor) -> torch.Tensor:
    """Each row of int32 ``indices`` ascending, its -1 padding last."""
    keyed = torch.where(indices < 0, _PAD_LAST, indices).sort(-1).values
    return torch.where(keyed == _PAD_LAST, -1, keyed)


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor,
                     cu_seqlens_q: torch.Tensor, cu_seqlens_kv: torch.Tensor, *,
                     max_seqlen_q: int, max_seqlen_k: int, scale: float = ATTN_SCALE,
                     selected_starts: torch.Tensor | None = None,
                     selected_ends: torch.Tensor | None = None,
                     ) -> torch.Tensor:
    """D21: Triton sparse attention over the selected keys -> bf16 ``[T, 8, 128]``.

    q ``[T, 8, 192]``, k ``[S, 1, 192]``, v ``[S, 1, 128]`` bf16, ``indices`` int32 ``[T, 2048]``
    global into k/v; no sink, as SGLang calls it for the DSA layers. Optional int32 ``[T]``
    bounds supply the number of selected slots to scan (end minus start), not KV positions.
    Valid entries must occupy that prefix of each indices row, with padding after it.
    """
    _on_current_device(q, k, v, indices, cu_seqlens_q, cu_seqlens_kv, selected_starts, selected_ends)
    _expect(indices, "sparse attention indices", torch.int32)
    if (selected_starts is None) != (selected_ends is None):
        raise PrefillKernelError("sparse selection bounds must be supplied together")
    if selected_starts is not None:
        _expect(selected_starts, "selected_starts", torch.int32, (int(q.shape[0]),))
        _expect(selected_ends, "selected_ends", torch.int32, (int(q.shape[0]),))
    o, _lse = _module(_SPARSE).sparse_attn_fwd_varlen(
        q.contiguous().unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), indices,
        cu_seqlen_q=cu_seqlens_q, cu_seqlen_kv=cu_seqlens_kv, sink=None,
        cu_seqlen_k_start=selected_starts, cu_seqlen_k_end=selected_ends,
        max_seqlen_q=int(max_seqlen_q), max_seqlen_k=int(max_seqlen_k), sm_scale=float(scale))
    return o[0]


def attn_output_rows(o: torch.Tensor) -> torch.Tensor:
    """D22: ``[T, 8, 128]`` -> ``[T, 1024]``."""
    return o.reshape(int(o.shape[0]), -1)


# --------------------------------------------------------------------------- dense MLP
def silu_and_mul(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """M2: ``sgl_kernel.silu_and_mul``, ``[T, 2n]`` -> ``[T, n]``."""
    _on_current_device(x, out)
    return _module("sgl_kernel").silu_and_mul(x, out=out)


# -------------------------------------------------------------------------------- MoE
def route(logits: torch.Tensor, bias: torch.Tensor, *, topk: int = TOPK,
          route_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """X2: ``sigmoid(logits) + bias`` selects top-8; weights are the winners' sigmoids renormalised.

    naivert's router (``external_ops.moe_route_fused``: ``sgl_kernel.moe_fused_gate`` with the
    group stage neutralised), the routing decode's ``expert_select`` computes.  Returns fp32
    weights ``[T, 8]`` and int32 expert ids ``[T, 8]``.
    """
    _on_current_device(logits, bias)
    _expect(logits, "router logits", torch.float32)
    return _module(_SHARED).moe_route_fused(logits, bias, topk=int(topk),
                                             route_scale=float(route_scale))


def ue8m0_scale_buffer(lead: tuple[int, ...], rows: int, width: int, *, device: torch.device,
                       group: int = QUANT_GROUP) -> torch.Tensor:
    """The int32 UE8M0 scale layout DeepGEMM reads: ``[*lead, rows, words]``, column-major.

    SGLang's ``create_per_token_group_quant_fp8_output_scale``: storage
    ``[*lead, words, ceil4(rows)]`` transposed, ``words = ceil4(width // group) // 4`` (four
    exponent bytes a word), so every row is a TMA-aligned column.
    """
    words = _ceil4(int(width) // int(group)) // 4
    base = torch.empty((*lead, words, _ceil4(rows)), dtype=torch.int32, device=device)
    return base.transpose(-1, -2)[..., :int(rows), :]


def quantize_dispatch_rows(x: torch.Tensor, q_out: torch.Tensor, s_out: torch.Tensor) -> None:
    """X3 (quantise): bf16 ``[T, 4096]`` -> e4m3 ``q_out`` + UE8M0 ``s_out`` per 128 columns.

    ``sgl_kernel.sgl_per_token_group_quant_fp8``
    with ``scale_ue8m0=True``; ``s_out`` comes from ``ue8m0_scale_buffer((), T, 4096)``.
    """
    _on_current_device(x, q_out, s_out)
    _module("sgl_kernel").sgl_per_token_group_quant_fp8(
        x, q_out, s_out, QUANT_GROUP, QUANT_EPS, -FP8_MAX, FP8_MAX, scale_ue8m0=True,
        fuse_silu_and_mul=False, masked_m=None, enable_v2=True)


#: X3's transport row: e4m3 codes, UE8M0 scale words, routing ids, routing weights.
DISPATCH_WORDS = _ceil4(HIDDEN // QUANT_GROUP) // 4                         # 8
DISPATCH_ROW_BYTES = HIDDEN + 4 * (DISPATCH_WORDS + 2 * TOPK)               # 4192


class DispatchRows(NamedTuple):
    """The fields of packed dispatch rows, as views of the uint8 ``[N, 4192]`` buffer."""
    codes: torch.Tensor     # e4m3 [N, 4096]
    scales: torch.Tensor    # int32 [N, 8]: UE8M0 words, row-major
    ids: torch.Tensor       # int32 [N, 8]: routed experts
    weights: torch.Tensor   # fp32 [N, 8]: routing weights


def dispatch_rows(packed: torch.Tensor) -> DispatchRows:
    """X3 (unpack): views of each field of packed rows; no launch.

    Every field starts at a multiple of four bytes, so the int32 and fp32 views are legal on
    every row of the rank's packed buffer and of the all-gathered one alike.
    """
    _expect(packed, "packed dispatch rows", torch.uint8)
    if packed.dim() != 2 or int(packed.shape[1]) != DISPATCH_ROW_BYTES or not packed.is_contiguous():
        raise PrefillKernelError(f"packed dispatch rows must be contiguous [N, {DISPATCH_ROW_BYTES}], "
                                 f"got {tuple(packed.shape)} strides {packed.stride()}")
    ids = HIDDEN + 4 * DISPATCH_WORDS
    weights = ids + 4 * TOPK
    return DispatchRows(packed[:, :HIDDEN].view(FP8_DTYPE), packed[:, HIDDEN:ids].view(torch.int32),
                        packed[:, ids:weights].view(torch.int32),
                        packed[:, weights:].view(torch.float32))


def pack_dispatch_rows(q_rows: torch.Tensor, s_rows: torch.Tensor, topk_ids: torch.Tensor,
                       topk_weights: torch.Tensor, packed: torch.Tensor) -> None:
    """X3 (pack): one rank's quantised rows and routing into its packed rows, for the gather.

    ``q_rows`` / ``s_rows`` from ``quantize_dispatch_rows``, int32 ids and fp32 weights
    ``[N, 8]`` from ``route``, ``packed`` uint8 ``[N, 4192]``.  One all-gather of the packed
    rows (``TP8Comm.all_gather_``) then moves all four fields in a single launch.  The
    quantiser writes contiguous codes only, so the fields are copied in.
    """
    _on_current_device(q_rows, s_rows, topk_ids, topk_weights, packed)
    fields = dispatch_rows(packed)
    rows = int(packed.shape[0])
    _expect(q_rows, "dispatch codes", FP8_DTYPE, (rows, HIDDEN))
    _expect(s_rows, "dispatch scales", torch.int32, (rows, DISPATCH_WORDS))
    _expect(topk_ids, "routing ids", torch.int32, (rows, TOPK))
    _expect(topk_weights, "routing weights", torch.float32, (rows, TOPK))
    fields.codes.view(torch.uint8).copy_(q_rows.view(torch.uint8))
    fields.scales.copy_(s_rows)
    fields.ids.copy_(topk_ids)
    fields.weights.copy_(topk_weights)


class ExpertSlots(NamedTuple):
    """Where each (token, k) routing pair lands in one card's masked expert layout."""
    expert: torch.Tensor    # int64 [N, K]: local expert in [0, E), or E for a pair another card owns
    slot: torch.Tensor      # int64 [N, K]: the pair's row inside its expert; 0 for foreign pairs
    masked_m: torch.Tensor  # int32 [E]: live rows per local expert
    permutation: torch.Tensor  # int64 [N*K]: expert-major pairs, foreign pairs last


def expert_slots(topk_ids: torch.Tensor, *, first_expert: int,
                 experts: int = EXPERTS_PER_RANK) -> ExpertSlots:
    """X3 (layout): the card owning experts ``[first_expert, first_expert + experts)``.

    ``topk_ids`` ``[N, K]`` covers every rank's rows in rank order (the gathered
    ``dispatch_rows(...).ids``).  An expert's rows are its
    pairs in (token, k) order, so the layout is a pure function of the routing.  Sync-free and
    static-shaped: stable expert sorting preserves (token, k) order; counts come
    from the sorted expert boundaries, with no ``nonzero``/``bincount``/``item``. Foreign pairs do not
    contend on a shared histogram counter.
    """
    _on_current_device(topk_ids)
    n, k = (int(d) for d in topk_ids.shape)
    pairs, dev = n * k, topk_ids.device
    local = topk_ids.to(torch.int64) - int(first_expert)
    owned = (local >= 0) & (local < experts)
    expert = torch.where(owned, local, experts)
    flat = expert.reshape(-1)
    order = torch.arange(pairs, device=dev, dtype=torch.int64)
    # Stable ties already preserve pair order. Sorting 33 expert buckets needs
    # one-byte keys, rather than the eight-byte unique (expert, pair) encoding.
    key_dtype = torch.uint8 if experts < 256 else torch.int64
    keys, perm = torch.sort(flat.to(key_dtype), stable=True)
    starts = torch.searchsorted(keys, torch.arange(experts + 1, device=dev, dtype=key_dtype))
    counts = starts[1:] - starts[:-1]
    rank = order - starts.index_select(0, flat.index_select(0, perm))
    slot = torch.empty_like(flat).scatter_(0, perm, rank)
    slot = torch.where(owned.reshape(-1), slot, 0).view(n, k)
    return ExpertSlots(expert, slot, counts.to(torch.int32), perm)


def scatter_to_experts(q_rows: torch.Tensor, s_rows: torch.Tensor, slots: ExpertSlots,
                       q_buf: torch.Tensor, s_buf: torch.Tensor) -> torch.Tensor:
    """X3 (copy): quantised rows into the masked layout ``[E + 1, cap, ...]``.

    ``q_rows`` e4m3 ``[N, 4096]`` and ``s_rows`` int32 ``[N, 8]``, in either scale layout: the
    gathered ``dispatch_rows(...)`` fields, or ``quantize_dispatch_rows``' own buffers.
    ``q_buf`` e4m3 ``[E + 1, cap, 4096]`` and ``s_buf`` ``ue8m0_scale_buffer((E + 1,), cap,
    4096)``. SGLang's permutation skips foreign activation rows (destination -1),
    avoiding an N*K activation expansion and contended writes to the trash row.
    Scale words retain the ATen scatter into the column-major UE8M0 layout; only
    these small foreign rows write the unread trash expert E. Return the flat
    pair-to-expert-row map, which the return path also uses.
    """
    _on_current_device(q_rows, s_rows, q_buf, s_buf, slots.expert, slots.slot)
    n, k = (int(d) for d in slots.expert.shape)
    token = torch.arange(n, device=q_rows.device).view(n, 1).expand(n, k).reshape(-1)
    e, s = slots.expert.reshape(-1), slots.slot.reshape(-1)
    dest = torch.where(e < int(q_buf.shape[0]) - 1, e * int(q_buf.shape[1]) + s, -1)
    # The unchanged upstream kernel expects contiguous rows. Pack once per
    # token, not once per token/expert pair, and copy as bytes to preserve FP8
    # encodings exactly (including NaNs).
    permute = _module('naivert.kernels.naive_n0p5.prefill_sglang.permute').deepep_permute_triton_kernel
    permute[(n,)](q_rows.contiguous().view(torch.uint8), q_buf.view(torch.uint8),
                  dest, slots.expert, s_rows, k, int(q_rows.shape[1]),
                  BLOCK_SIZE=4096, num_warps=4)
    s_buf[e, s] = s_rows.index_select(0, token)
    return dest


def scatter_to_compact_experts(q_rows: torch.Tensor, s_rows: torch.Tensor, slots: ExpertSlots,
                               q_buf: torch.Tensor, s_buf: torch.Tensor, *,
                               alignment: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter into contiguous expert blocks and return pair map, ends and aligned total.

    Each expert starts at an aligned row. DeepGEMM's psum layout stores its
    actual end, so the next start is that end rounded up to ``alignment``.
    The caller reserves at least ``N*TOPK + E*alignment`` rows: even with all
    pairs local, the last row is outside the aligned prefix. Foreign scale
    writes use that unread last row; the permutation skips foreign codes.
    All offsets depend on GPU counts and remain dynamic under graph replay.
    """
    _on_current_device(q_rows, s_rows, q_buf, s_buf, slots.masked_m)
    counts = slots.masked_m
    aligned = (counts + (alignment - 1)) // alignment * alignment
    starts = torch.cat((torch.zeros(1, device=counts.device, dtype=torch.int32),
                        aligned.cumsum(0, dtype=torch.int32)))
    ends = starts[:-1] + counts
    e, s = slots.expert.reshape(-1), slots.slot.reshape(-1)
    dest = torch.where(e < counts.numel(), starts.index_select(0, e) + s, -1)
    n, k = (int(d) for d in slots.expert.shape)
    permute = _module('naivert.kernels.naive_n0p5.prefill_sglang.permute').deepep_permute_triton_kernel
    permute[(n,)](q_rows.contiguous().view(torch.uint8), q_buf.view(torch.uint8),
                  dest, slots.expert, s_rows, k, int(q_rows.shape[1]),
                  BLOCK_SIZE=4096, num_warps=4)
    token = torch.arange(n, device=q_rows.device).view(n, 1).expand(n, k).reshape(-1)
    s_buf[dest] = s_rows.index_select(0, token)
    return dest, ends, starts[-1:]


def prepare_compact_experts(q_rows: torch.Tensor, s_rows: torch.Tensor, topk_ids: torch.Tensor,
                            q_buf: torch.Tensor, s_buf: torch.Tensor, *, first_expert: int,
                            alignment: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
    """Existing SGLang scatter: counts, pair order, destinations, ends, aligned total.

    Counts and stable pair order come from byte-key sorting. Upstream scatter
    resets aligned expert counters, then assigns rows atomically while copying
    code bytes and packed scales together. Counters finish at actual expert
    ends, directly supplying DeepGEMM's psum layout. Foreign pairs skip both
    stores. Within-expert row order may change; the returned map restores the
    original pairs and the return sum still uses fixed top-k slots.
    """
    _on_current_device(q_rows, s_rows, topk_ids, q_buf, s_buf)
    experts = EXPERTS_PER_RANK
    local = topk_ids.to(torch.int64) - first_expert
    owned = (local >= 0) & (local < experts)
    local_ids = torch.where(owned, local, -1)
    keys = torch.where(owned, local, experts).reshape(-1).to(torch.uint8)
    keys, permutation = torch.sort(keys, stable=True)
    boundaries = torch.searchsorted(keys, torch.arange(experts + 1, device=topk_ids.device,
                                                       dtype=torch.uint8))
    counts = (boundaries[1:] - boundaries[:-1]).to(torch.int32)
    aligned = (counts + (alignment - 1)) // alignment * alignment
    total = aligned.sum(dtype=torch.int32).reshape(1)
    ends = torch.empty_like(counts)
    # This upstream kernel also fills a row-to-expert table. PSUM GEMM does
    # not consume it, but retaining these stores keeps the device body intact.
    row_experts = torch.empty(q_buf.shape[0], device=topk_ids.device, dtype=torch.int32)
    scatter = _module('naivert.kernels.naive_n0p5.prefill_sglang.scatter')
    scatter._fwd_kernel_ep_scatter_1[(experts,)](
        aligned, ends, row_experts, num_experts=experts,
        BLOCK_E=alignment, BLOCK_EXPERT_NUM=32, num_warps=8)
    dest = torch.full_like(topk_ids, -1, dtype=torch.int64)
    n, k = (int(d) for d in topk_ids.shape)
    scatter._fwd_kernel_ep_scatter_2[(min(n, 8192),)](
        n, ends, q_rows.view(torch.uint8), q_rows.stride(0), q_rows.stride(1),
        s_rows, s_rows.stride(0), s_rows.stride(1),
        local_ids, local_ids.stride(0), local_ids.stride(1),
        q_buf.view(torch.uint8), q_buf.stride(0), q_buf.stride(1),
        s_buf, s_buf.stride(0), s_buf.stride(1), dest, dest.stride(0), dest.stride(1),
        topk_num=k, HIDDEN_SIZE=HIDDEN, HIDDEN_SIZE_PAD=HIDDEN,
        SCALE_HIDDEN_SIZE=DISPATCH_WORDS, SCALE_HIDDEN_SIZE_PAD=DISPATCH_WORDS,
        ATOMIC_ADD_SEM=None, IS_FP8=True, num_warps=8)
    return counts, permutation, dest.reshape(-1), ends, total


def expected_m(total_rows: int, *, topk: int = TOPK, experts: int = EXPERTS) -> int:
    """DeepGEMM's ``expected_m``: ``(rows * topk + experts) // experts`` (129 at 4096 rows)."""
    return (int(total_rows) * int(topk) + int(experts)) // int(experts)


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """DeepGEMM's ``ceil_to_ue8m0`` as SGLang's ``fp8_utils`` copies it: up to a power of two."""
    bits = x.abs().float().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    return (exp << 23).view(torch.float32)


def requantize_expert_ue8m0(codes: torch.Tensor, scales: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint block-FP8 expert weights -> codes over power-of-two block scales.

    The checkpoint's ``weight_scale_inv`` are arbitrary fp32; DeepGEMM's UE8M0 grouped GEMM
    reads one exponent per block.  At load SGLang ``eb078d76`` re-quantises every expert on this
    device class (``Fp8MoEMethod.process_weights_after_loading`` ->
    ``requant_block_scale_ue8m0_for_deepgemm`` -> ``requant_weight_ue8m0``).  This is that
    function up to its packing, expression for expression: ``block_quant_dequant`` to bf16, then
    ``quant_weight_ue8m0`` -> ``per_block_cast_to_fp8`` over the ``[-1, k]`` view -- each
    128 x 128 block's ``amax.clamp(1e-4) / 448`` rounded up to a power of two, codes
    ``(x * (1 / sf)).to(e4m3)``.  ``pack_expert_weight_scales`` packs the result.

    codes e4m3 ``[..., n, k]`` and fp32 scales ``[..., ceil(n / 128), ceil(k / 128)]`` on one
    device -> new codes and scales of the same shapes and dtypes.  With leading dimensions ``n``
    must be a multiple of 128, or the flattened view's blocks would straddle two experts.
    """
    _expect(codes, "expert codes", FP8_DTYPE)
    _expect(scales, "expert block scales", torch.float32)
    *lead, n, k = (int(d) for d in codes.shape)
    g = QUANT_GROUP
    blocks = (-(-n // g), -(-k // g))
    if tuple(int(d) for d in scales.shape) != (*lead, *blocks) or scales.device != codes.device:
        raise PrefillKernelError(
            f"expert scales must be {(*lead, *blocks)} on {codes.device} for codes "
            f"{tuple(codes.shape)}, got {tuple(scales.shape)} on {scales.device}")
    if lead and n % g:
        raise PrefillKernelError(f"stacked expert codes need rows in multiples of {g}, got {n}")
    # block_quant_dequant(weight, weight_scale_inv, [128, 128], torch.bfloat16)
    repeat = scales.repeat_interleave(g, dim=-2).repeat_interleave(g, dim=-1)
    repeat = repeat[..., :n, :k]
    dequant = (codes.to(torch.float32) * repeat).to(torch.bfloat16)
    # quant_weight_ue8m0 -> per_block_cast_to_fp8(dequant.view((-1, k)))
    flat = dequant.view((-1, k))
    m = int(flat.shape[0])
    padded = torch.zeros((-(-m // g) * g, -(-k // g) * g), dtype=flat.dtype, device=flat.device)
    padded[:m, :k] = flat
    view = padded.view(-1, g, padded.size(1) // g, g)
    amax = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)
    scaled = (view * (1.0 / sf)).to(torch.float8_e4m3fn)
    new_codes = scaled.view_as(padded)[:m, :k].contiguous().view((*lead, n, k))
    new_scales = sf.view(view.size(0), view.size(2)).view((*lead, *blocks))
    return new_codes, new_scales


def pack_expert_weight_scales(scale: torch.Tensor, *, rows: int | None = None) -> torch.Tensor:
    """fp32 power-of-two block scales ``[E, ceil(n / 128), groups]`` -> DeepGEMM's packed UE8M0.

    SGLang's ``transform_scale_ue8m0(sf, mn=n, use_torch_impl=True)``: each block row repeated
    for its ``n`` weight rows (``rows``, by default ``128 * scale.shape[1]``), then
    ``_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl``: one exponent byte per (row,
    column group), rows TMA-aligned, int32 words ``[E, n, words]`` MN-major.  Only a power of
    two is its exponent byte, so any other scale -- the checkpoint's own, before
    ``requantize_expert_ue8m0`` -- is refused rather than truncated.
    """
    _expect(scale, "expert weight scales", torch.float32)
    if scale.dim() != 3:
        raise PrefillKernelError(f"expert weight scales must be [E, rows, groups], got "
                                 f"{tuple(scale.shape)}")
    bits = scale.view(torch.int32)
    exponent = bits >> 23
    power_of_two = (bits >= 0) & ((bits & 0x7FFFFF) == 0) & (exponent >= 1) & (exponent <= 254)
    if not bool(power_of_two.all()):
        raise PrefillKernelError(
            "expert weight scales must be positive finite powers of two (UE8M0); requantize "
            "the checkpoint's block scales with requantize_expert_ue8m0 first")
    e = int(scale.shape[0])
    rows = int(scale.shape[1]) * QUANT_GROUP if rows is None else int(rows)
    if rows <= 0 or -(-rows // QUANT_GROUP) != int(scale.shape[1]):
        raise PrefillKernelError(f"{rows} weight rows do not fit {int(scale.shape[1])} block "
                                 f"rows of {QUANT_GROUP}")

    from deep_gemm.utils import align, get_tma_aligned_size

    per_row = scale.index_select(-2, torch.arange(rows, device=scale.device) // QUANT_GROUP)
    exponents = (per_row.view(torch.int32) >> 23).to(torch.uint8)
    mn, kk = int(exponents.shape[-2]), int(exponents.shape[-1])
    aligned_mn, aligned_k = get_tma_aligned_size(mn, 4), align(kk, 4)
    padded = torch.zeros((e, aligned_mn, aligned_k), device=scale.device, dtype=torch.uint8)
    padded[:, :mn, :kk] = exponents
    words = padded.view(-1).view(torch.int32).view(e, aligned_mn, aligned_k // 4)
    packed = torch.zeros((e, aligned_k // 4, aligned_mn), device=scale.device,
                         dtype=torch.int32).mT
    packed[:, :, :] = words
    return packed[:, :mn, :]


def grouped_gemm(a: torch.Tensor, a_scale: torch.Tensor, weight: torch.Tensor,
                 weight_scale: torch.Tensor, out: torch.Tensor, masked_m: torch.Tensor,
                 expected: int) -> None:
    """X4 / X6: DeepGEMM ``fp8_m_grouped_gemm_nt_masked`` on this device's runtime copy.

    a e4m3 ``[E, cap, n]`` + UE8M0 scales, weight e4m3 ``[E, m, n]`` + packed UE8M0
    (``pack_expert_weight_scales``), out bf16 ``[E, cap, m]``; only ``masked_m[e]`` rows of
    expert ``e`` are computed.  The wheel's defaults: ``compiled_dims="nk"``, UE8M0 cast on.
    """
    _on_current_device(a, a_scale, weight, weight_scale, out, masked_m)
    deep_gemm_module(a.device).m_grouped_fp8_fp4_gemm_nt_masked(
        a, a_scale, weight, weight_scale, out, masked_m, int(expected), None, None, None,
        "nk", False)


def grouped_gemm_contiguous(a: torch.Tensor, a_scale: torch.Tensor, weight: torch.Tensor,
                            weight_scale: torch.Tensor, out: torch.Tensor,
                            expert_ends: torch.Tensor, expected: int) -> None:
    """Existing DeepGEMM grouped W8A8 GEMM on compact rows with GPU psum ends.

    A and output are 2D; weights retain their expert dimension. Capacity is
    static, while expert ends identify the live rows. Zero output padding
    allows the fused quantizer to process the aligned prefix as one group.
    """
    _on_current_device(a, a_scale, weight, weight_scale, out, expert_ends)
    deep_gemm_module(a.device).m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, a_scale, weight, weight_scale, out, expert_ends,
        None, None, None, "nk", False, True, True, int(expected))


def act_quant_masked(up: torch.Tensor, q_out: torch.Tensor, s_out: torch.Tensor,
                     masked_m: torch.Tensor) -> None:
    """X5: ``silu(gate) * up`` quantised to e4m3 + UE8M0, live rows only, in one kernel.

    Uses ``sgl_per_token_group_quant_fp8(fuse_silu_and_mul=True, masked_m=...)``.
    BF16 ``[E, cap, 2n]`` becomes E4M3 ``[E, cap, n]`` plus
    ``ue8m0_scale_buffer((E,), cap, n)``.
    """
    _on_current_device(up, q_out, s_out, masked_m)
    _module("sgl_kernel").sgl_per_token_group_quant_fp8(
        up, q_out, s_out, QUANT_GROUP, QUANT_EPS, -FP8_MAX, FP8_MAX, scale_ue8m0=True,
        fuse_silu_and_mul=True, masked_m=masked_m, enable_v2=True)


def combine_partial(dn: torch.Tensor, slots: ExpertSlots, weights: torch.Tensor,
                    out: torch.Tensor) -> torch.Tensor:
    """X7 (this card): ``out[t] = sum_j weights[t, j] * dn[expert, slot]`` in fp32, j = 0..K-1.

    ``dn`` bf16 ``[E + 1, cap, 4096]`` with an all-zero trash expert ``E``, so foreign pairs
    add zero; ``weights`` fp32 ``[N, K]`` (the gathered ``dispatch_rows(...).weights``); ``out``
    fp32 ``[N, 4096]``.  A fixed order of gathers and fp32 additions, where a weighted
    ``index_add_`` over duplicate token indices accumulates in a nondeterministic order.
    The cards' partials are summed by ``TP8Comm.reduce_scatter_``, which leaves
    each rank its own rows.
    """
    _on_current_device(dn, slots.expert, slots.slot, weights, out)
    _expect(weights, "routing weights", torch.float32, tuple(slots.expert.shape))
    _expect(out, "combine partial", torch.float32)
    out.zero_()
    for j in range(int(slots.expert.shape[1])):
        rows = dn[slots.expert[:, j], slots.slot[:, j]]
        out.addcmul_(rows.float(), weights[:, j:j + 1])
    return out


# ------------------------------------------------------------------------------ output
def add_residual_(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """F1 (before its all-gather): the last layer's ``x += residual``, in place."""
    _on_current_device(x, residual)
    return x.add_(residual)


def last_row(x: torch.Tensor, valid: int) -> torch.Tensor:
    """F3: the last real row, ``[1, 4096]``."""
    return x[int(valid) - 1:int(valid)]


def last_row_into(x: torch.Tensor, last: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """F3 (device tensors, graph form): row ``last[0]`` of ``x`` into ``out`` ``[1, 4096]``.

    ``last`` is ``ChunkMeta.last`` (``valid - 1``), so which row is kept is a device value
    rather than the host slice a capture would bake in.
    """
    _on_current_device(x, last, out)
    return torch.index_select(x, 0, last, out=out)


# ------------------------------------------------------------------------ stage table
SOURCES = ("wheel", "torch", "prefill_ext", "naivert", "nccl", "removed", "none")
_P = "naivert.kernels.naive_n0p5.prefill:"
_PX = _EXT + ".naive_ops.dsa."


@dataclass(frozen=True)
class StageBinding:
    """An inventory stage: the functions implementing it and the kernels they launch."""
    functions: tuple[str, ...]      # "module:qualname"
    kernels: tuple[str, ...]        # "module:qualname" of the launched symbol
    sources: tuple[str, ...]        # subset of SOURCES
    note: str = ""


def _b(functions, kernels, sources, note=""):
    return StageBinding(tuple(functions), tuple(kernels), tuple(sources), note)


_ADD_NORM = _P + "add_rmsnorm_"
_LINEAR = _P + "linear"
_ROPE = _P + "rope"
_AR = _COMMS + ":TP8Comm.all_reduce_"
_RS = _COMMS + ":TP8Comm.reduce_scatter_"
_AG = _COMMS + ":TP8Comm.all_gather_"
_GEMM = ("torch:mm",)
_SGL_ADD_NORM = ("sgl_kernel:fused_add_rmsnorm",)
_SGL_QUANT = "sgl_kernel:sgl_per_token_group_quant_fp8"
_DG = "deep_gemm:fp8_m_grouped_gemm_nt_masked"
_LOGITS_KERNEL = (_PX + "triton.topk:indexer_qk_topk_prefill",
                  _PX + "gluon.logits_prefill_sm100:indexer_logits_mqa_prefill_gluon_kernel")

STAGE_BINDINGS: dict[str, StageBinding] = {
    "E1": _b([_P + "embed"], ["torch:index_select", "torch:Tensor.masked_fill_"], ["torch"],
             "the rank's 19072-row shard, zero rows outside it"),
    "E2": _b([_AR], ["torch.cuda.nccl:all_reduce"], ["nccl"], "bf16 [T, 4096]"),
    "S1": _b([_ADD_NORM], _SGL_ADD_NORM, ["wheel"]),
    "S1b": _b([_AG], ["torch.cuda.nccl:all_gather"], ["nccl"], "[512, 4096] -> [4096, 4096]"),
    "S2": _b([_LINEAR], _GEMM, ["torch"], "qkv [T, 4096] x [1856, 4096]^T bf16"),
    "S3": _b([_P + "split_qkv"], [], ["torch"], "views"),
    "S4": _b([_ROPE], [_SHARED + ":rope_fma"], ["naivert"], "theta 1e4, head 192"),
    "S5": _b([_P + "scale_values"], ["torch:mul"], ["torch"]),
    "S6": _b([_P + "swa_kv_publish", _P + "swa_kv_write", _P + "swa_kv_prefix",
              _P + "swa_kv_prefix_rows"],
             [_ADAPTERS + ":swa_ring_publish", _ADAPTERS + ":swa_ring_write",
              _ADAPTERS + ":swa_ring_prefix_into", _ADAPTERS + ":swa_ring_prefix_rows"],
             ["naivert"], "naivert's 136-row ring"),
    "S7": _b([_P + "swa_attention"], [_FA4 + ":flash_attn_varlen_func"], ["prefill_ext"],
             "FA4 CuTe FlashAttentionForwardSm100, window (127, 0), bf16 sink"),
    "S8": _b([_LINEAR], _GEMM, ["torch"], "o_proj [T, 1024] -> [T, 4096], not reduced"),
    "S9": _b([_RS, _ADD_NORM], ("torch.cuda.nccl:reduce_scatter",) + _SGL_ADD_NORM,
             ["nccl", "wheel"], "[4096, 4096] -> [512, 4096], then the post norm"),
    "S10–18": _b([], [], ["none"], "the MoE block: see X1-X8"),
    "S19": _b([], [], ["none"], "no kernel"),
    "D1": _b([_P + "rmsnorm"], ["sgl_kernel:rmsnorm"], ["wheel"],
             "layer 0; the other DSA layers norm as S1/S1b"),
    "D2": _b([_LINEAR], _GEMM, ["torch"]),
    "D3": _b([_ROPE], [_SHARED + ":rope_fma"], ["naivert"], "theta 1e7, head 192"),
    "D4": _b([_P + "scale_values"], ["torch:mul"], ["torch"]),
    "D5": _b([_LINEAR, _P + "split_indexer"], _GEMM, ["torch"],
             "[T, 4096] x [2192, 4096]^T bf16, split 2048 / 128 / 16"),
    "D6": _b([_P + "scale_index_weights"], ["torch:mul"], ["torch"]),
    "D7": _b([_P + "index_key_norm"], [_SHARED + ":index_key_norm"], ["naivert"]),
    "D8": _b([_ROPE], [_SHARED + ":rope_fma"], ["naivert"], "theta 1e7, head 128"),
    "D9/D10": _b([_P + "quantize_index_rows"], [_SHARED + ":quantize_e4m3_rows"],
                 ["naivert"], "one scale per 128: q [T, 16, 1], key [T, 1]"),
    "D11": _b([_P + "fold_index_weights"], ["torch:mul"], ["torch"]),
    "D12": _b([_P + "dsa_kv_append", _P + "dsa_kv_write"],
              [_ADAPTERS + ":dsa_cache_append", _ADAPTERS + ":dsa_cache_write"], ["naivert"],
              "index key: swizzled into ki_cache, plain into the indexer's copy"),
    "D13": _b([_P + "dsa_kv_append", _P + "dsa_kv_write"],
              [_ADAPTERS + ":dsa_cache_append", _ADAPTERS + ":dsa_cache_write"], ["naivert"],
              "k/v, in the same publish as D12"),
    "D14": _b([_P + "dsa_kv_view"], [], ["torch"], "views of the position-indexed cache"),
    "D15": _b([_P + "dsa_index_key_view"], [], ["torch"], "views of the plain index keys"),
    "D16": _b([], [], ["removed"], "indexer CP cancelled: every card runs its whole query"),
    "D17": _b([_P + "indexer_select"], _LOGITS_KERNEL, ["prefill_ext"],
              "gluon sm100 logits, GQA=16, BLOCK_M 8 / BLOCK_N 128"),
    "D18": _b([_P + "indexer_select"], [_PX + "cuda.indexer_topk:invoke_indexer_topk_prefill"],
              ["prefill_ext"], "topk_per_row_prefill_radix (NVRTC), one module per device"),
    "D19": _b([_P + "indexer_select"],
              [_PX + "triton.topk:_indexer_topk_prefill_epilogue_kernel", "torch:where",
               "torch:Tensor.sort"], ["prefill_ext", "torch"],
              "then each row ascending, -1 padding last, past 2048 visible keys"),
    "D20": _b([], [], ["removed"], "no indexer CP, nothing to all-gather"),
    "D21": _b([_P + "sparse_attention"], [_PX + "triton.sparse_attn:sparse_attn_fwd_varlen"],
              ["prefill_ext"], "sparse_attn_fwd_varlen_tma_kernel, BLOCK_N 64, no sink"),
    "D22": _b([_P + "attn_output_rows"], [], ["torch"], "view"),
    "D23": _b([_LINEAR], _GEMM, ["torch"]),
    "D24": _b([_AR, _ADD_NORM], ("torch.cuda.nccl:all_reduce",) + _SGL_ADD_NORM,
              ["nccl", "wheel"], "layer 0: mlp FULL"),
    "D24'": _b([_RS, _ADD_NORM], ("torch.cuda.nccl:reduce_scatter",) + _SGL_ADD_NORM,
               ["nccl", "wheel"], "layers 5..47: [4096, 4096] -> [512, 4096]"),
    "M1": _b([_LINEAR], _GEMM, ["torch"], "gate_up [T, 4096] x [4096, 4096]^T"),
    "M2": _b([_P + "silu_and_mul"], ["sgl_kernel:silu_and_mul"], ["wheel"]),
    "M3": _b([_LINEAR], _GEMM, ["torch"], "down [T, 2048] -> [T, 4096]"),
    "M4": _b([_AR], ["torch.cuda.nccl:all_reduce"], ["nccl"]),
    "X1": _b([_P + "router_logits"], _GEMM, ["torch"], "fp32 sgemm [512, 4096] x [256, 4096]^T"),
    "X2": _b([_P + "route"], [_SHARED + ":moe_route_fused", "sgl_kernel:moe_fused_gate"],
             ["naivert", "wheel"], "decode's expert_select routing"),
    "X3": _b([_P + "quantize_dispatch_rows", _P + "pack_dispatch_rows", _AG, _P + "dispatch_rows",
              _P + "expert_slots", _P + "scatter_to_experts"],
             [_SGL_QUANT, "torch.cuda.nccl:all_gather"], ["wheel", "nccl", "torch"],
             "UE8M0 per-128 quant; one all-gather of packed uint8 rows (codes, scales, ids, "
             "weights), copied into [32, cap, 4096] + masked_m"),
    "X4": _b([_P + "grouped_gemm"], [_DG], ["wheel"], "per-device _C.so copy"),
    "X5": _b([_P + "act_quant_masked"], [_SGL_QUANT], ["wheel"],
             "fuse_silu_and_mul + masked_m"),
    "X6": _b([_P + "grouped_gemm"], [_DG], ["wheel"], "per-device _C.so copy"),
    "X7": _b([_P + "combine_partial", _RS], ["torch:Tensor.addcmul_",
                                              "torch.cuda.nccl:reduce_scatter"],
             ["torch", "nccl"],
             "fixed-order fp32 weighted sum, then an fp32 reduce-scatter"),
    "X8": _b([], [], ["none"], "slice, no kernel"),
    "F1": _b([_P + "add_residual_", _AG], ["torch:Tensor.add_", "torch.cuda.nccl:all_gather"],
             ["torch", "nccl"], "layer 47's residual add, then [512, 4096] -> [4096, 4096]"),
    "F2": _b([_P + "rmsnorm"], ["sgl_kernel:rmsnorm"], ["wheel"]),
    "F3": _b([_P + "last_row", _P + "last_row_into"], ["torch:index_select"], ["torch"],
             "a view, or the graph form's device row"),
    "F4": _b([], [], ["none"], "head projection is outside this prefill executor"),
    "F5": _b([], [], ["none"], "head shard concatenation is outside this prefill executor"),
}


__all__ = [
    "ATTN_HEADS", "ATTN_SCALE", "DENSE_INTERMEDIATE", "DSA_ROPE_THETA", "DeepGemmRuntime",
    "EXPERTS", "EXPERTS_PER_RANK", "EXPERT_INTERMEDIATE", "ExpertSlots", "FP8_DTYPE", "FP8_MAX",
    "DISPATCH_ROW_BYTES", "DISPATCH_WORDS", "DispatchRows", "EXCEPTION_PINS",
    "HEAD_DIM", "HIDDEN", "INDEX_DIM", "INDEX_HEADS", "INDEX_LOGITS_BUDGET_MB",
    "INDEX_QKW_ROWS", "INDEX_TOPK", "INDEX_WEIGHT_SCALE", "KV_HEADS",
    "NORM_EPS",
    "O_PROJ_COLS", "PINNED_COMPONENTS", "PINNED_DISTRIBUTIONS", "PrefillKernelError",
    "QKV_ROWS", "QUANT_EPS",
    "QUANT_GROUP", "RANKS", "RECORDED_DISTRIBUTIONS", "ROPE_DIM", "SOURCES", "STAGE_BINDINGS",
    "SWA_ROPE_THETA", "SWA_WINDOW", "StageBinding", "TOPK", "VALUE_SCALE", "VOCAB",
    "VOCAB_SHARD", "V_HEAD_DIM", "act_quant_masked", "add_residual_", "add_rmsnorm_",
    "attn_output_rows", "check_versions",
    "combine_partial", "deep_gemm_cache_root", "deep_gemm_module", "dispatch_rows",
    "dsa_index_key_view",
    "dsa_kv_append", "dsa_kv_view", "dsa_kv_write", "embed", "expected_m", "expert_slots",
    "fold_index_weights", "grouped_gemm", "grouped_gemm_contiguous", "index_key_norm", "indexer_chunk_rows",
    "indexer_select", "installed_distributions", "last_row", "last_row_into", "linear",
    "pack_dispatch_rows",
    "pack_expert_weight_scales",
    "prepare_device", "prepare_compact_experts", "quantize_dispatch_rows", "quantize_index_rows", "requantize_expert_ue8m0",
    "rmsnorm", "rope", "rope_qk",
    "route", "router_logits", "scale_index_weights", "scale_values", "scatter_to_experts",
    "scatter_to_compact_experts",
    "silu_and_mul", "sparse_attention", "split_indexer", "split_qkv", "swa_attention",
    "swa_kv_prefix", "swa_kv_prefix_rows", "swa_kv_publish", "swa_kv_write",
    "ue8m0_scale_buffer",
]
