"""Shared prefill routing and decode-compatible attention arithmetic."""
from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
FUSED_GATE_MAX_GROUP = 32
ROPE_DIM = 64
NORM_EPS = 1e-5
INDEX_DIM = 128
E4M3_AMAX = 448.0
# Match the fused multiply-add order at each decode RoPE call site.
DSA_ROPE_IMAG = {"q": "cos", "k": "cos", "ik": "cos", "iq": "sin"}


class ExternalKernelError(RuntimeError):
    """An external prefill kernel is missing, mis-versioned, or refused its input."""


def fused_gate_groups(experts: int) -> int:
    """Return the fewest groups that fit sgl-kernel's 32-expert group limit.

    Keeping all groups makes the grouping step a no-op and preserves global
    top-k routing for Naive N."""
    return max(1, -(-int(experts) // FUSED_GATE_MAX_GROUP))


def moe_route_fused(scores: torch.Tensor, bias: torch.Tensor, *, topk: int,
                    route_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``moe_fused_gate`` with the group stage neutralised (see :func:`fused_gate_groups`)."""
    import sgl_kernel

    g = fused_gate_groups(int(scores.shape[1]))
    # `apply_routed_scaling_factor_on_output` is False by default in sgl-kernel 0.4.4, and
    # with it False the kernel *ignores* `routed_scaling_factor` entirely.  Naive N's scale is
    # 1.0 so the two are indistinguishable here, but a routed_scaling_factor of 2.5 would not be.
    return sgl_kernel.moe_fused_gate(
        scores, bias, g, g, topk, num_fused_shared_experts=0,
        routed_scaling_factor=float(route_scale),
        apply_routed_scaling_factor_on_output=True)


@triton.jit
def _neg(v):
    # A sign flip, which is what an FFMA's negated operand is.  Triton's unary minus is
    # ``0 - v``, and that is +0 where the kernel's operand is -0.
    return (v.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)


@triton.jit
def _rope_fma(x_ptr, o_ptr, t_ptr, p_ptr, n_vec, heads, s_row, s_head, ROPE: tl.constexpr,
              REST: tl.constexpr, IMAG_COS: tl.constexpr, BLOCK: tl.constexpr):
    vec = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = vec < n_vec
    row = vec // heads
    src = (x_ptr + row * s_row + (vec % heads) * s_head)[:, None]
    dst = (o_ptr + vec * (ROPE + REST))[:, None]
    ev = 2 * tl.arange(0, ROPE // 2)[None, :]
    cs = t_ptr + tl.load(p_ptr + row, mask=live, other=0)[:, None] * ROPE + ev
    m = live[:, None]
    re = tl.load(src + ev, mask=m, other=0.0).to(tl.float32)
    im = tl.load(src + ev + 1, mask=m, other=0.0).to(tl.float32)
    c = tl.load(cs, mask=m, other=0.0)
    s = tl.load(cs + 1, mask=m, other=0.0)
    real = tl.fma(re, c, _neg(im * s))
    if IMAG_COS:
        imag = tl.fma(im, c, re * s)
    else:
        imag = tl.fma(re, s, im * c)
    tl.store(dst + ev, real.to(tl.bfloat16), mask=m)
    tl.store(dst + ev + 1, imag.to(tl.bfloat16), mask=m)
    rest = ROPE + tl.arange(0, REST)[None, :]
    tl.store(dst + rest, tl.load(src + rest, mask=m), mask=m)


def rope_fma(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, *,
             imag: Literal["cos", "sin"], rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """Apply RoPE with the same FP32 FMA order and BF16 rounding as decode.

    ``imag`` selects which imaginary product stays inside the FMA. Input is
    BF16 [M, H, D] or [M, D]; table is interleaved FP32 [positions, rope_dim].
    Dimensions beyond rope_dim are copied unchanged."""
    if x.dtype is not torch.bfloat16:
        raise ExternalKernelError(f"rope_fma expects bf16, got {x.dtype}")
    if imag not in ("cos", "sin"):
        raise ExternalKernelError(f"rope_fma: imag must be 'cos' or 'sin', got {imag!r}")
    if (table.dtype is not torch.float32 or int(table.shape[1]) != rope_dim
            or tuple(table.stride()) != (rope_dim, 1)):
        raise ExternalKernelError(
            f"rope table must be contiguous fp32 [max_seq_len, {rope_dim}], got "
            f"{tuple(table.shape)} {table.dtype}")
    flat = x.unsqueeze(1) if x.dim() == 2 else x
    m, heads, dim = (int(v) for v in flat.shape)
    rest = dim - int(rope_dim)
    if rest <= 0 or rest & (rest - 1) or int(flat.stride(-1)) != 1:
        raise ExternalKernelError(f"rope_fma wants a unit-stride head with a power-of-two tail "
                                  f"past {rope_dim}, got {tuple(x.shape)} / {tuple(x.stride())}")
    if int(positions.shape[0]) != m:
        raise ExternalKernelError(f"rope positions {tuple(positions.shape)} do not match rows {m}")
    out = torch.empty((m, heads, dim), dtype=torch.bfloat16, device=x.device)
    n = m * heads
    if n:
        block = 32
        _rope_fma[(triton.cdiv(n, block),)](
            flat, out, table, positions.contiguous(), n, heads, int(flat.stride(0)),
            int(flat.stride(1)), ROPE=int(rope_dim), REST=rest, IMAG_COS=imag == "cos",
            BLOCK=block)
    return out.squeeze(1) if x.dim() == 2 else out


def _block_sum_128(v: torch.Tensor) -> torch.Tensor:
    """Sums over a 128-wide last dim in ``block_sum``'s order (``attn_common/cta_helpers.cuh``).

    With 384 threads and 128 live values, the kernel's strides 256 and 128 add zeros, stride 64
    adds ``v[t] + v[t + 64]``, warp 0 adds ``+ 32``, and the ``shfl_down`` ladder 16 .. 1
    finishes.  That is a fixed pairwise tree, replayed here one level per op so that every fp32
    rounding lands where the kernel's does.
    """
    s = v[..., :64] + v[..., 64:]
    for w in (32, 16, 8, 4, 2, 1):
        s = s[..., :w] + s[..., w:2 * w]
    return s[..., 0]


def index_key_norm(ik: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *,
                   eps: float = NORM_EPS) -> torch.Tensor:
    """Apply the decode index-key LayerNorm with a fixed reduction order.

    Two-pass FP32 mean/variance, rounded normalization, an affine FMA, then
    BF16 output. This reference preserves the cache bytes produced by decode;
    native prefill uses the fused equivalent in the prefill kernel bindings."""
    if ik.dtype is not torch.bfloat16:
        raise ExternalKernelError(f"index_key_norm expects bf16, got {ik.dtype}")
    if weight.dtype is not torch.float32 or bias.dtype is not torch.float32:
        raise ExternalKernelError(
            f"index_key_norm expects fp32 affine parameters, got {weight.dtype} / {bias.dtype}")
    if int(ik.shape[-1]) != INDEX_DIM:
        raise ExternalKernelError(f"index_key_norm is publish_ki's {INDEX_DIM}-wide norm, got "
                                  f"{tuple(ik.shape)}")
    v = ik.float()
    delta = v - (_block_sum_128(v) * (1.0 / INDEX_DIM)).unsqueeze(-1)
    rstd = torch.rsqrt(_block_sum_128(delta * delta) * (1.0 / INDEX_DIM) + float(eps))
    return torch.addcmul(bias, delta * rstd.unsqueeze(-1), weight).to(torch.bfloat16)


@triton.jit
def _quant_rows_rn(x_ptr, q_ptr, s_ptr, n_rows, AMAX: tl.constexpr, DIM: tl.constexpr,
                   BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = rows < n_rows
    at = rows[:, None] * DIM + tl.arange(0, DIM)[None, :]
    x = tl.load(x_ptr + at, mask=live[:, None], other=0.0).to(tl.float32)
    # IEEE round-to-nearest division, twice, as nvcc compiles the kernel's two `/`s.
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30), AMAX)
    q = tl.div_rn(x, scale[:, None])
    tl.store(q_ptr + at, q.to(tl.float8e4nv), mask=live[:, None])
    tl.store(s_ptr + rows, scale, mask=live)


def quantize_e4m3_rows(x: torch.Tensor, *, amax: float = E4M3_AMAX
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 [..., D] to E4M3 with one FP32 scale per row.

    Match decode's two IEEE round-to-nearest divisions: scale = amax / 448,
    code = value / scale. Reciprocal multiplication can change boundary codes."""
    if x.dtype is not torch.bfloat16:
        raise ExternalKernelError(f"quantize_e4m3_rows expects bf16, got {x.dtype}")
    dim = int(x.shape[-1])
    if dim <= 0 or dim & (dim - 1):
        raise ExternalKernelError(f"quantize_e4m3_rows wants a power-of-two row, got {tuple(x.shape)}")
    flat = x.reshape(-1, dim).contiguous()
    n = int(flat.shape[0])
    q = torch.empty(flat.shape, dtype=FP8_DTYPE, device=x.device)
    s = torch.empty(n, dtype=torch.float32, device=x.device)
    if n:
        block = 32
        _quant_rows_rn[(triton.cdiv(n, block),)](flat, q, s, n, AMAX=float(amax), DIM=dim,
                                                  BLOCK=block)
    return q.view(x.shape), s.view(x.shape[:-1])
