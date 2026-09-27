#!/usr/bin/env python3
"""Standalone check of the Naive N attn_dsa body against a torch reference.

One TP8 rank of a Naive N0.5 full-attention layer at real shapes (hidden
4096, 8 local heads x 192, kv head 192/128, 64 index heads x 128, top-k 2048)
with random bf16 weights. For every requested position T the reference fills
the k / v / ki caches for positions < T, then the kernel processes the query
at T and we compare: the fused projections, the cache rows it appended, the
top-k selection (sparse regime) and the attention output.

usage: run_attn_dsa.py [--positions 0,100,2047,2048,4095,6000] [--seed 7]
                       [--build-only] [--timing]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CUDA_HOME = os.environ.get("NAIVERT_CUDA_HOME", "/opt/cuda132-min")
# The DSA body uses cute (tcgen05 sweep); the headers live next to the repo in third_party.
# One variable governs every builder: NAIVERT_CUTLASS_INCLUDE (the runtime's cutlass_include_root), then
# the checkout beside the repository; CUTLASS_INCLUDE is honoured for older launch scripts.
from naivert.kernels.build import cutlass_include_root as _cutlass_include_root
CUTLASS_INCLUDE = os.environ.get("CUTLASS_INCLUDE") or str(_cutlass_include_root())

from naivert.kernels.toolchain import SM_COUNT  # noqa: E402
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags  # noqa: E402

GEOMETRY_FLAGS = list(geometry_cuda_cflags(sm_count=SM_COUNT))

DIM, HEADS, HEAD_DIM, ROPE, VDIM = 4096, 8, 192, 64, 128
IHEADS, IDIM, TOPK = 64, 128, 2048
THETA, EPS, VSCALE = 1.0e7, 1.0e-5, 0.707
Q_ROWS, IQ_ROWS = HEADS * HEAD_DIM, IHEADS * IDIM


PHASE_NAMES = {0: "start", 1: "normalize_x", 2: "project_fused", 3: "publish/rotate",
               4: "topk.pass0", 5: "topk.pass1", 6: "topk.compact|pass2", 7: "topk.finish0|pass3",
               10: "topk.done|digits", 11: "topk.count",
               8: "selection", 9: "attention+combine", 12: "attn.qk", 13: "attn.softmax",
               14: "attn.pv", 15: "attn.split_done", 16: "attn.combine"}


def configure_index_heads(index_heads):
    """Select the reference geometry; each test process runs one specialization."""
    if index_heads not in (16, 64):
        raise ValueError("index_heads must be 16 or 64")
    global IHEADS, IQ_ROWS
    IHEADS, IQ_ROWS = index_heads, index_heads * IDIM


def build(phase_timers: bool, index_heads: int = 64):
    from torch.utils.cpp_extension import load

    from naivert.kernels.toolchain import scoped_cuda_toolchain

    flavour = ("timers" if phase_timers else "plain") + f"_ih{index_heads}"
    head_flags = [f"-DNAIVERT_DSA_INDEX_HEADS={index_heads}"]
    build_dir = ROOT / "tests/build/attn_dsa" / flavour
    build_dir.mkdir(parents=True, exist_ok=True)
    with scoped_cuda_toolchain(CUDA_HOME, suppress_arch_flags=True):
        return load(
            name=f"naivert_attn_dsa_test_{flavour}",
            sources=[str(ROOT / "naivert/body/attn_dsa/attn_dsa_entry.cu"),
                     str(ROOT / "tests/gpu/attn_dsa/attn_dsa_bindings.cpp")],
            extra_include_paths=[str(ROOT), CUTLASS_INCLUDE],
            extra_cflags=["-O2", "-std=c++17", *GEOMETRY_FLAGS, *head_flags],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_103a,code=sm_103a",
                               f"-DNAIVERT_MLA_ROWS_ENABLE_PHASE_TIMERS={int(phase_timers)}", *GEOMETRY_FLAGS, *head_flags],
            build_directory=str(build_dir),
            verbose=os.environ.get("ATTN_DSA_VERBOSE") == "1",
        )


def phase_report(workspace, layout):
    """Deltas (us) between consecutive recorded phase timestamps of CTA 0."""
    off, nbytes = layout["phase_timers"]
    stamps = workspace[off:off + nbytes].view(torch.int64).cpu().tolist()
    recorded = [(p, t) for p, t in enumerate(stamps) if t > 0]
    recorded.sort(key=lambda pt: pt[1])
    out = []
    for (p0, t0), (p1, t1) in zip(recorded, recorded[1:]):
        out.append(f"{PHASE_NAMES.get(p0, p0)}->{PHASE_NAMES.get(p1, p1)}={(t1 - t0) / 1000:.1f}")
    if recorded:
        out.append(f"total={(recorded[-1][1] - recorded[0][1]) / 1000:.1f}")
    return " ".join(out)


# --------------------------------------------------------------- conventions

def rope_permutation():
    """HF rotate_half order (i, i+32) -> interleaved pairs (2i, 2i+1)."""
    import torch
    perm = torch.empty(ROPE, dtype=torch.long)
    perm[0::2] = torch.arange(ROPE // 2)
    perm[1::2] = torch.arange(ROPE // 2) + ROPE // 2
    return perm


def interleave_rope(x, head_dim):
    """Permute the first 64 entries of every head (last-dim groups of head_dim)."""
    shape = x.shape
    x = x.reshape(-1, head_dim).clone()
    x[:, :ROPE] = x[:, rope_permutation().to(x.device)]
    return x.reshape(shape)


def interleave_rope_rows(w, head_dim):
    """Same permutation applied to projection output rows [n * head_dim, dim]."""
    return interleave_rope(w.transpose(0, 1), head_dim).transpose(0, 1).contiguous()


def rope_table(max_pos, device):
    import torch
    freqs = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).reshape(max_pos, ROPE).contiguous()


def hf_rope(x, positions):
    """rotate_half rope on the last 64 dims of x [..., P, 64] (fp32), positions [P]."""
    import torch
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=x.device) / ROPE))
    ang = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    x1, x2 = x[..., :ROPE // 2], x[..., ROPE // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# ---------------------------------------------------------------- reference

class Layer:
    @classmethod
    def from_checkpoint(cls, checkpoint, layer, rank, device, *, training_graph=False):
        from naivert.kernels.packing import quantize_fp8_block
        from naivert.models.naive_n0p5.checkpoint_fp8 import dequant_block
        w = checkpoint.attention_hf(layer, rank, device)
        if w["kind"] != "full_dsa":
            raise ValueError("the DSA probe requires a full-attention layer")
        configure_index_heads(checkpoint.index_heads)
        obj = cls.__new__(cls)
        for key in ("wq", "wk", "wv", "wiq", "wik", "wis", "ik_w", "ik_b"):
            setattr(obj, key, w[key])
        obj.gamma = w["attn_gamma"]
        obj.value_scale = w["value_scale"]
        obj.training_graph = training_graph
        if training_graph:
            obj.wiq8 = interleave_rope_rows(obj.wiq, IDIM)
            obj.wiq8_scale = torch.ones((IHEADS,32), device=device)
            return obj
        obj.wiq8, obj.wiq8_scale = quantize_fp8_block(interleave_rope_rows(obj.wiq, IDIM))
        q, s = quantize_fp8_block(obj.wiq)
        obj.wiq_deq = dequant_block(q, s)
        return obj

    def __init__(self, device, seed):
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)

        def w(rows, scale=0.02):
            return (torch.randn(rows, DIM, generator=g) * scale).to(torch.bfloat16).to(device)

        self.wq, self.wk = w(Q_ROWS), w(HEAD_DIM)
        self.wv = (w(VDIM).float() * VSCALE).to(torch.bfloat16)  # value scale folded
        self.wiq, self.wik, self.wis = w(IQ_ROWS), w(IDIM), w(IHEADS, 0.05)
        self.gamma = (1.0 + 0.1 * torch.randn(DIM, generator=g)).to(device)
        # The kernel takes the index-query weights as e4m3 with block-128 scales (rows
        # interleaved for rope first; the interleave stays inside a 128-row block, so the
        # quantization is the same as in HF order and the reference can quantize the HF rows).
        from naivert.kernels.packing import quantize_fp8_block
        self.wiq8, self.wiq8_scale = quantize_fp8_block(interleave_rope_rows(self.wiq, IDIM))
        self.wiq8, self.wiq8_scale = self.wiq8.contiguous(), self.wiq8_scale.contiguous()
        q8, s8 = quantize_fp8_block(self.wiq)
        self.wiq_deq = (q8.float().reshape(IQ_ROWS // 128, 128, DIM // 128, 128) * s8[:, None, :, None]).reshape(IQ_ROWS, DIM)
        self.ik_w = (1.0 + 0.1 * torch.randn(IDIM, generator=g)).to(device)
        self.ik_b = (0.1 * torch.randn(IDIM, generator=g)).to(device)

    def kernel_weights(self):
        """Rope rows interleaved for the kernel; the index-key LayerNorm affine
        parameters follow the same permutation because they are per-dimension."""
        return dict(
            wq=interleave_rope_rows(self.wq, HEAD_DIM), wk=interleave_rope_rows(self.wk, HEAD_DIM),
            wv=self.wv.contiguous(), wiq=self.wiq8, wiq_scale=self.wiq8_scale,
            wik=interleave_rope_rows(self.wik, IDIM), wis=self.wis.contiguous(),
            ik_w=interleave_rope(self.ik_w, IDIM).contiguous(),
            ik_b=interleave_rope(self.ik_b, IDIM).contiguous(),
        )


def kernel_order_fused(fused_ref):
    """Reference fused vector [q | k | v | ik | iq] (HF order) -> kernel row order."""
    q, k, v, ik, iq = torch.split(fused_ref, [Q_ROWS, HEAD_DIM, VDIM, IDIM, IQ_ROWS])
    return torch.cat([interleave_rope(q, HEAD_DIM), interleave_rope(k, HEAD_DIM), v,
                      interleave_rope(ik, IDIM), interleave_rope(iq, IDIM)])


def rmsnorm(x, gamma):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma).to(torch.bfloat16)


def project(xn, w):
    return (xn.float() @ w.float().T).to(torch.bfloat16)


def reference_caches(layer, x_rows):
    """k [P,192], v [P,128], ki [P,128] (bf16, HF rope order) for x_rows [P, 4096]."""
    P = x_rows.shape[0]
    pos = torch.arange(P, device=x_rows.device)
    xn = rmsnorm(x_rows, layer.gamma)
    k = project(xn, layer.wk).float()
    k[:, :ROPE] = hf_rope(k[:, :ROPE], pos)
    v = ((xn.float() @ layer.wv.float().T) * getattr(layer, "value_scale", 1.0)).bfloat16()
    ik = project(xn, layer.wik).float()
    ik = torch.nn.functional.layer_norm(ik, (IDIM,), layer.ik_w, layer.ik_b, EPS).to(torch.bfloat16).float()
    ik[:, :ROPE] = hf_rope(ik[:, :ROPE], pos)
    return k.to(torch.bfloat16), v, ik.to(torch.bfloat16)


def reference_query(layer, x, T):
    """q [8,192] (rope applied), iq [64,128] (rope applied), w [64], fused pre-rope."""
    xn = rmsnorm(x[None], layer.gamma)
    q_raw, k_raw, v_raw = project(xn, layer.wq)[0], project(xn, layer.wk)[0], project(xn, layer.wv)[0]
    v_raw = ((xn.float() @ layer.wv.float().T) * getattr(layer, "value_scale", 1.0))[0].bfloat16()
    ik_raw = project(xn, layer.wik)[0]
    # E4m3 activations x dequantized e4m3 weights, fp32 accumulate, bf16 result.
    iq_raw = (project(xn, layer.wiq)[0] if getattr(layer, "training_graph", False) else
              (xn.to(torch.float8_e4m3fn).float() @ layer.wiq_deq.T)[0].to(torch.bfloat16))
    pos = torch.tensor([T], device=x.device)
    q = q_raw.view(HEADS, HEAD_DIM).float()
    q[:, :ROPE] = hf_rope(q[:, :ROPE], pos)
    iq = iq_raw.view(IHEADS, IDIM).float()
    iq[:, :ROPE] = hf_rope(iq[:, :ROPE], pos)
    w = (xn.float() @ layer.wis.float().T)[0] * IHEADS ** -0.5
    fused = torch.cat([q_raw, k_raw, v_raw, ik_raw, iq_raw])
    return q.to(torch.bfloat16), iq.to(torch.bfloat16), w, fused, xn[0]


def quantize_e4m3_rows(rows, min_amax=1e-30):
    """e4m3 rows with one scale per row (row amax / 448): the index-key cache format, also
    the index-query quantization (rows must already be bf16-rounded like the kernel's)."""
    rows = rows.float()
    scale = rows.abs().amax(dim=1).clamp_min(min_amax) / 448.0
    return (rows / scale[:, None]).to(torch.float8_e4m3fn), scale


def swizzle_e4m3_rows(codes, positions=None):
    """The kernel's cache / query image: row p keeps its 128 bytes with the 16-byte chunk index
    XORed by (p & 7). An involution: applying it to a kernel row gives the plain row back."""
    n = codes.shape[0]
    p = torch.arange(n, device=codes.device) if positions is None else positions.to(codes.device)
    j = torch.arange(8, device=codes.device)[None, :] ^ (p[:, None] & 7)
    u8 = codes.view(torch.uint8).reshape(n, 8, 16).gather(1, j[..., None].expand(n, 8, 16))
    return u8.reshape(n, 128).view(torch.float8_e4m3fn)


def reference_selection(iq, w, ki, visible, fp8=True, min_amax=1e-30):
    """Indices (sorted) of the top-k index logits over positions < visible; fp8=True simulates the
    kernel's e4m3 keys and queries, fp8=False is the bf16 indexer."""
    if visible <= TOPK:
        return torch.arange(visible, device=iq.device), None
    if fp8:
        ki8, ks = quantize_e4m3_rows(ki[:visible].to(torch.bfloat16), min_amax)
        iq8, qs = quantize_e4m3_rows(iq.to(torch.bfloat16), min_amax)
        scores = (ki8.float() * ks[:, None]) @ (iq8.float() * qs[:, None]).T
    else:
        scores = ki[:visible].float() @ iq.float().T
    logits = (torch.relu(scores) * w[None, :]).sum(-1)  # [visible]
    sel = torch.topk(logits, TOPK).indices
    return sel.sort().values, logits


def reference_attention(q, k, v, sel):
    """Softmax attention of q [8,192] over k/v rows sel; fp32 probs and bf16-rounded probs."""
    ks, vs = k[sel].float(), v[sel].float()
    scores = (q.float() @ ks.T) * HEAD_DIM ** -0.5  # [8, n]
    probs = torch.softmax(scores, -1)
    out_fp32 = probs @ vs
    probs_bf16 = probs.to(torch.bfloat16).float()
    out_bf16p = (probs_bf16 @ vs) / probs_bf16.sum(-1, keepdim=True)
    return out_fp32, out_bf16p


# ------------------------------------------------------------------- checks

def bf16_ulp_mismatch(a, b):
    """Elements differing by more than one bf16 ulp of the larger magnitude."""
    af, bf = a.float(), b.float()
    ulp = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(ulp)) - 7)
    return int(((af - bf).abs() > ulp * 1.001).sum()), float((af - bf).abs().max())


def run_case(ext, layer, kw, T, device, timing, max_seq_len, phases=False, repeat=0):
    import torch
    torch.manual_seed(1000 + T)
    x_rows = (torch.randn(T + 1, DIM, device=device) * 1.5).to(torch.bfloat16)
    k_ref, v_ref, ki_ref = reference_caches(layer, x_rows)
    q_ref, iq_ref, w_ref, fused_ref, xn_ref = reference_query(layer, x_rows[T], T)
    min_amax = 1e-4 if getattr(layer, "training_graph", False) else 1e-30
    sel_ref, logits_ref = reference_selection(iq_ref, w_ref, ki_ref, T + 1, min_amax=min_amax)
    sel_ref_bf16, _ = reference_selection(iq_ref, w_ref, ki_ref, T + 1, fp8=False)
    out_fp32, out_bf16p = reference_attention(q_ref, k_ref, v_ref, sel_ref)

    # Kernel buffers: caches in interleaved rope order for positions < T.
    k_cache = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(max_seq_len, VDIM, dtype=torch.bfloat16, device=device)
    ki_cache = torch.zeros(max_seq_len, IDIM, dtype=torch.float8_e4m3fn, device=device)
    ki_scale = torch.zeros(max_seq_len, dtype=torch.float32, device=device)
    if T > 0:
        k_cache[:T] = interleave_rope(k_ref[:T], HEAD_DIM)
        v_cache[:T] = v_ref[:T]
        ki_codes_plain, ki_scale[:T] = quantize_e4m3_rows(interleave_rope(ki_ref[:T], IDIM), min_amax)
        ki_cache[:T] = swizzle_e4m3_rows(ki_codes_plain)
    rope = rope_table(max_seq_len, device)
    proj_o = torch.zeros(1, HEADS, VDIM, dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device=device)
    ext.prepare_maps(workspace, kw["wq"], kw["wk"], kw["wv"], kw["wik"])
    cur_pos = torch.tensor([T], dtype=torch.int32, device=device)

    def launch():
        ext.run_scaled(x_rows[T:T + 1].contiguous(), layer.gamma, kw["wq"], kw["wk"], kw["wv"], kw["wiq"], kw["wiq_scale"],
                kw["wik"], kw["ik_w"], kw["ik_b"], kw["wis"], rope, k_cache, v_cache, ki_cache, ki_scale,
                proj_o, workspace, cur_pos, max_seq_len, 0, getattr(layer, "value_scale", 1.0))

    launch()
    torch.cuda.synchronize()

    lay = ext.layout()

    def view(name, dtype):
        off, nbytes = lay[name]
        return workspace[off:off + nbytes].view(dtype)

    report = {"T": T, "visible": T + 1}
    sparse = T + 1 > TOPK
    # 1. fused projections (pre-rope) and index weights. Below the top-k the
    # kernel skips the index-query rows and weights_proj (stale buffers).
    fused = view("fused", torch.bfloat16)
    compared = fused.numel() if sparse else lay["fused_bases"][4]
    report["fused_mismatch>1ulp"], report["fused_max_abs"] = bf16_ulp_mismatch(
        fused[:compared], kernel_order_fused(fused_ref)[:compared])
    report["fused_compared"] = int(compared)
    if sparse:
        iw = view("index_weights", torch.float32)
        report["index_weights_max_rel"] = float(
            ((iw - w_ref).abs() / w_ref.abs().clamp_min(1e-3)).max())
    # 2. cache rows appended at T (compare in interleaved order)
    report["k_row_mismatch>1ulp"], report["k_row_max_abs"] = bf16_ulp_mismatch(
        k_cache[T], interleave_rope(k_ref[T:T + 1], HEAD_DIM)[0])
    report["v_row_mismatch>1ulp"], _ = bf16_ulp_mismatch(v_cache[T], v_ref[T])
    ki_row_ref = interleave_rope(ki_ref[T:T + 1], IDIM)
    ki_row_q, ki_row_s = quantize_e4m3_rows(ki_row_ref, min_amax)
    ki_row_kernel = swizzle_e4m3_rows(ki_cache[T:T + 1], torch.tensor([T]))[0]  # plain order
    ki_codes = (ki_row_kernel.view(torch.uint8).int() - ki_row_q[0].view(torch.uint8).int()).abs()
    report["ki_row_mismatch>1ulp"] = int((ki_codes > 1).sum())
    report["ki_row_tie_flips"] = int((ki_codes == 1).sum())
    report["ki_row_scale_rel"] = float(abs(ki_scale[T] - ki_row_s[0]) / ki_row_s[0])
    report["ki_row_max_abs"] = float((ki_row_kernel.float() * ki_scale[T] - ki_row_ref[0].float()).abs().max())
    # 3. selection
    count = min(T + 1, TOPK)
    sel = view("selections", torch.int32)[:count].long()
    if logits_ref is None:
        report["selection_identity"] = bool(torch.equal(sel.sort().values, sel_ref))
    else:
        common = len(set(sel.tolist()) & set(sel_ref.tolist()))
        report["selection_overlap"] = f"{common}/{TOPK}"
        report["selection_overlap_bf16ref"] = f"{len(set(sel.tolist()) & set(sel_ref_bf16.tolist()))}/{TOPK}"
        # near-tie diagnostics: reference logit rank of the worst kernel pick
        ranks = torch.argsort(torch.argsort(logits_ref, descending=True))
        report["selection_worst_ref_rank"] = int(ranks[sel].max())
        report["selection_ref_logit_gap_at_k"] = float(
            logits_ref[sel_ref].min() - logits_ref[torch.topk(logits_ref, TOPK + 1).indices[-1]])
        report["selection_in_range"] = bool((sel >= 0).all() and (sel <= T).all())
    # 4. attention output (kernel output rows in HF head order; values unaffected by rope perm).
    #    In the sparse regime the e4m3 indexer's picks can differ from the reference at near ties
    #    (reported above), so the attention core is judged on the kernel's own picks.
    if logits_ref is not None:
        out_fp32, out_bf16p = reference_attention(q_ref, k_ref, v_ref, sel.sort().values)
    o = proj_o[0].float()
    for name, ref in (("out_vs_fp32", out_fp32), ("out_vs_bf16probs", out_bf16p)):
        err = (o - ref).abs()
        report[f"{name}_max_abs"] = float(err.max())
        report[f"{name}_max_rel"] = float((err / ref.abs().clamp_min(1e-2)).max())
    report["out_ref_scale"] = float(out_fp32.abs().max())
    report["out_nonzero"] = bool(o.abs().sum() > 0)
    # attention from kernel-side selection (isolates the attention core from near ties)
    if logits_ref is not None:
        _, own_sel_out = reference_attention(q_ref, k_ref, v_ref, sel.sort().values)
        report["out_vs_bf16probs_kernel_sel_max_abs"] = float((o - own_sel_out).abs().max())
    if repeat:
        # Bitwise run-to-run repeatability of the attention output on identical inputs.
        first = proj_o.clone()
        first_sel = view("selections", torch.int32)[:count].clone()
        differing = sel_order_differs = sel_set_differs = 0
        for _ in range(repeat):
            proj_o.zero_()
            launch()
            torch.cuda.synchronize()
            differing += int(not torch.equal(proj_o, first))
            sel_now = view("selections", torch.int32)[:count]
            sel_order_differs += int(not torch.equal(sel_now, first_sel))
            sel_set_differs += int(not torch.equal(sel_now.sort().values, first_sel.sort().values))
        report["repeat_runs"] = repeat
        report["repeat_bitwise_mismatches"] = differing
        report["repeat_selection_order_differs"] = sel_order_differs
        report["repeat_selection_set_differs"] = sel_set_differs
    if timing:
        torch.cuda.synchronize()
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = 20
        start.record()
        for _ in range(n):
            launch()
        end.record()
        torch.cuda.synchronize()
        report["us_per_step"] = round(start.elapsed_time(end) / n * 1e3, 1)
        if phases:
            workspace[lay["phase_timers"][0]:lay["phase_timers"][0] + lay["phase_timers"][1]].zero_()
            launch()
            torch.cuda.synchronize()
            report["phases_us"] = phase_report(workspace, lay)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-heads", type=int, choices=(16, 64), default=64)
    parser.add_argument("--positions", default="0,100,2047,2048,4095,6000")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--phases", action="store_true", help="build with phase timers and report them")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=0, help="relaunch N times and count bitwise output mismatches")
    args = parser.parse_args()
    if args.build_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    global torch
    import torch
    t0 = time.time()
    configure_index_heads(args.index_heads)
    ext = build(args.phases, args.index_heads)
    print(f"built extension in {time.time() - t0:.1f}s; workspace {ext.workspace_bytes() / 2**20:.1f} MiB",
          flush=True)
    if args.build_only:
        return 0
    device = torch.device("cuda")
    layer = Layer(device, args.seed)
    kw = layer.kernel_weights()
    ok = True
    for T in (int(p) for p in args.positions.split(",")):
        report = run_case(ext, layer, kw, T, device, args.timing or args.phases, args.max_seq_len,
                          phases=args.phases, repeat=args.repeat)
        print(report, flush=True)
        # Output tolerance: 1% of the output's own scale (bf16 has ~0.4% ulp).
        # Rounding flips between summation orders happen at a rate (~1 %), so the bound
        # on the fused projections scales with the number of compared elements.
        bad = (report["fused_mismatch>1ulp"] > max(8, report["fused_compared"] // 50) or
               report["k_row_mismatch>1ulp"] > max(2, HEAD_DIM // 32) or
               report["ki_row_mismatch>1ulp"] > max(2, IDIM // 32) or
               report["v_row_mismatch>1ulp"] > max(2, VDIM // 32) or
               not report["out_nonzero"] or
               report["out_vs_bf16probs_max_abs"] > 0.01 * max(report["out_ref_scale"], 1e-3))
        if "selection_identity" in report:
            bad = bad or not report["selection_identity"]
        else:
            bad = bad or not report["selection_in_range"]
        if report.get("repeat_bitwise_mismatches"):
            bad = True
        ok = ok and not bad
        print("  ->", "ok" if not bad else "MISMATCH", flush=True)
    print("ALL OK" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
