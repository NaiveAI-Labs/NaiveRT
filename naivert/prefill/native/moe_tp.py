"""TP-sharded MoE for the native prefill: every rank runs every routed pair through its
256-wide intermediate slice of every expert (the decode sharding), so there is no dispatch
across ranks and no expert skew; the rank's fp32 partial is reduced across ranks afterwards.

Per layer, per rank, over the N gathered rows (N = chunk tokens) and their top-8 routing:

1. ``pair_layout``      pairs (token, k) sorted by expert into a contiguous row layout, each
                        expert's rows padded to a 128 multiple (DeepGEMM's contiguous grouped
                        GEMM), ``m_indices`` -1 on padding rows.  Static shapes: the layout
                        buffer holds ``N * 8 + 256 * 127`` rows.
2. ``gather_quant``     the tokens' bf16 rows gathered into that layout, then per-token
                        group-128 e4m3 with UE8M0 scales (sgl_kernel, packed int32 layout).
3. ``grouped_gemm``     gate/up ``[rows, 4096] x [4096, 512]``, SiLU * up, quant, down
                        ``[rows, 256] x [256, 4096]`` (DeepGEMM, one runtime per device).
4. ``combine``          ``out[t] = sum_k w[t, k] * down[row(t, k)]`` in fp32, fixed k order
                        (Triton); the partial each rank contributes to the TP completion.

Weights: ``load_tp_experts`` builds ``w13`` e4m3 ``[256, 512, 4096]`` (this rank's 256 gate
rows then its 256 up rows per expert) and ``w2`` e4m3 ``[256, 4096, 256]`` from the
checkpoint's rank slices, re-quantised to UE8M0 block scales as SGLang/DeepGEMM require
(``requantize_expert_ue8m0``) and packed (``pack_expert_weight_scales``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from naivert.kernels.naive_n0p5 import prefill as P

EXPERTS, TOPK, HIDDEN, INTER = 256, 8, 4096, 256
ALIGN = 128
FP8 = torch.float8_e4m3fn


def layout_rows(tokens: int) -> int:
    """Rows of the padded contiguous layout for ``tokens`` (every expert may pad by 127)."""
    return tokens * TOPK + EXPERTS * (ALIGN - 1)


# --------------------------------------------------------------------------------- weights
def load_tp_experts(checkpoint, layer: int, rank: int, device) -> dict:
    """This rank's slices of all 256 experts in DeepGEMM's grouped layout (see module doc)."""
    device = torch.device(device)
    gate_q, gate_s, up_q, up_s, dn_q, dn_s = [], [], [], [], [], []
    for e in range(EXPERTS):
        (gq, gs), (uq, us), (dq, ds) = (checkpoint.expert(layer, e, which, rank, device, kernel_scales=False)
                                        for which in range(3))
        gate_q.append(gq); gate_s.append(gs); up_q.append(uq); up_s.append(us); dn_q.append(dq); dn_s.append(ds)
    w13 = torch.stack([torch.cat((g, u)) for g, u in zip(gate_q, up_q)])          # [256, 512, 4096] e4m3
    w13_s = torch.stack([torch.cat((g, u)) for g, u in zip(gate_s, up_s)]).float()  # [256, 4, 32]
    w2 = torch.stack(dn_q)                                                          # [256, 4096, 256]
    w2_s = torch.stack(dn_s).float()                                                # [256, 32, 2]
    if w13.dtype is not FP8 or w2.dtype is not FP8:
        raise ValueError(f"expert codes are {w13.dtype}/{w2.dtype}, expected e4m3")
    out = {}
    for name, q, s in (("w13", w13, w13_s), ("w2", w2, w2_s)):
        q, s = P.requantize_expert_ue8m0(q.contiguous(), s.contiguous())
        out[name] = (q.contiguous(), P.pack_expert_weight_scales(s, rows=int(q.shape[1])))
    return out


# ---------------------------------------------------------------------------------- layout
class PairLayout:
    """Static buffers for one rank's pair layout over ``tokens`` rows."""

    def __init__(self, tokens: int, device):
        self.tokens, self.device = int(tokens), torch.device(device)
        self.rows = layout_rows(self.tokens)
        i32 = dict(dtype=torch.int32, device=self.device)
        i64 = dict(dtype=torch.int64, device=self.device)
        self.m_indices = torch.full((self.rows,), -1, **i32)
        self.token_of_row = torch.zeros(self.rows, **i64)     # padding rows read token 0 (skipped)
        self.row_of_pair = torch.zeros(self.tokens * TOPK, **i64)
        self.counts = torch.zeros(EXPERTS, **i64)
        self._order = torch.arange(self.tokens * TOPK, **i64)
        self._ones = torch.ones(self.tokens * TOPK, **i64)

    @torch.no_grad()
    def build(self, topk_ids: torch.Tensor) -> None:
        """``topk_ids`` int32 ``[tokens, 8]`` -> m_indices, token_of_row, row_of_pair (torch ops,
        static shapes, no host sync; a Triton version is M2 work)."""
        n, k = self.tokens, TOPK
        flat = topk_ids.reshape(-1).to(torch.int64)
        perm = torch.argsort(flat * (n * k) + self._order)              # stable by (expert, pair)
        counts = torch.zeros(EXPERTS, dtype=torch.int64, device=self.device).scatter_add_(0, flat, self._ones)
        padded = (counts + (ALIGN - 1)) // ALIGN * ALIGN
        starts = torch.cumsum(padded, 0) - padded                       # padded start per expert
        group_start = torch.cumsum(counts, 0) - counts                  # start in sorted pair order
        sorted_expert = flat.index_select(0, perm)
        within = self._order - group_start.index_select(0, sorted_expert)
        dest = starts.index_select(0, sorted_expert) + within           # padded row of each sorted pair
        self.row_of_pair.scatter_(0, perm, dest)
        self.m_indices.fill_(-1)
        self.m_indices.scatter_(0, dest, sorted_expert.to(torch.int32))
        self.token_of_row.zero_()
        self.token_of_row.scatter_(0, dest, perm // k)
        self.counts.copy_(counts)


# ------------------------------------------------------------------------------ MoE stages
class TPMoE:
    """Workspaces + stages for one rank at a fixed token count."""

    def __init__(self, tokens: int, device, *, out_dtype=torch.bfloat16):  # DeepGEMM writes bf16 only
        self.tokens, self.device = int(tokens), torch.device(device)
        self.layout = PairLayout(tokens, device)
        rows = self.layout.rows
        bf16 = dict(dtype=torch.bfloat16, device=self.device)
        self.a_bf16 = torch.zeros((rows, HIDDEN), **bf16)
        self.a_q = torch.empty((rows, HIDDEN), dtype=FP8, device=self.device)
        self.a_s = P.ue8m0_scale_buffer((), rows, HIDDEN, device=self.device)
        self.up = torch.empty((rows, 2 * INTER), **bf16)
        self.act = torch.empty((rows, INTER), **bf16)
        self.act_q = torch.empty((rows, INTER), dtype=FP8, device=self.device)
        self.act_s = P.ue8m0_scale_buffer((), rows, INTER, device=self.device)
        self.down = torch.zeros((rows, HIDDEN), dtype=out_dtype, device=self.device)
        self.partial = torch.zeros((self.tokens, HIDDEN), dtype=torch.float32, device=self.device)
        self.runtime = P.prepare_device(self.device)

    def gather_quant(self, x: torch.Tensor) -> None:
        """``x`` bf16 ``[tokens, 4096]`` (the gathered, normed rows) -> a_q / a_s in the layout."""
        torch.index_select(x, 0, self.layout.token_of_row, out=self.a_bf16)
        P.quantize_dispatch_rows(self.a_bf16, self.a_q, self.a_s)

    def _grouped(self, a, a_s, w, w_s, out):
        P.deep_gemm_module(self.device).m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, a_s, w, w_s, out, self.layout.m_indices, None, None, None, "nk", False, False, True, None)

    def experts(self, weights: dict) -> None:
        (w13, w13_s), (w2, w2_s) = weights["w13"], weights["w2"]
        self._grouped(self.a_q, self.a_s, w13, w13_s, self.up)
        P.silu_and_mul(self.up, out=self.act)
        P.quantize_dispatch_rows(self.act, self.act_q, self.act_s)
        self._grouped(self.act_q, self.act_s, w2, w2_s, self.down)

    def combine(self, topk_weights: torch.Tensor) -> torch.Tensor:
        combine_rows(self.down, self.layout.row_of_pair, topk_weights, self.partial)
        return self.partial

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, weights: dict):
        self.layout.build(topk_ids)
        self.gather_quant(x)
        self.experts(weights)
        return self.combine(topk_weights)


# --------------------------------------------------------------------------------- combine
@triton.jit
def _combine_kernel(down_ptr, row_ptr, w_ptr, out_ptr, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        row = tl.load(row_ptr + t * K + k)
        w = tl.load(w_ptr + t * K + k)
        v = tl.load(down_ptr + row * H + c).to(tl.float32)
        acc = acc + v * w
    tl.store(out_ptr + t * H + c, acc)


def combine_rows(down: torch.Tensor, row_of_pair: torch.Tensor, weights: torch.Tensor, out: torch.Tensor,
                 block: int = 1024) -> torch.Tensor:
    """``out[t] = sum_k weights[t, k] * down[row_of_pair[t * 8 + k]]`` fp32, k = 0..7 in order."""
    tokens = int(out.shape[0])
    assert weights.dtype is torch.float32 and out.dtype is torch.float32 and HIDDEN % block == 0
    _combine_kernel[(tokens, HIDDEN // block)](down, row_of_pair, weights.contiguous(), out,
                                                H=HIDDEN, K=TOPK, BLOCK=block, num_warps=4)
    return out


__all__ = ["ALIGN", "PairLayout", "TPMoE", "combine_rows", "layout_rows", "load_tp_experts"]
