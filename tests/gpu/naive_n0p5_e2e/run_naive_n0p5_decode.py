#!/usr/bin/env python3
"""Naive N0.5 seq1 decode on 8 GPUs with random or real checkpoint weights.

usage: run_naive_n0p5_decode.py [--layers 48] [--prompt-len 8] [--steps 32] [--check-steps 4]
                          [--seed 7] [--max-seq-len 4096] [--no-graph]

The kernel path runs the Naive N model_ops build (unproj, router, expert_select, expert_down,
dense up-gate / down, head) plus the standalone attn_dsa / attn_swa test extensions, one
rank per GPU, every layer's kernels in the production order, captured per rank into a
CUDA graph and replayed in lockstep. Weights are random: bf16 attention / router / norms /
head, fp8 block-128 experts, dense MLP and o_proj packed exactly as the Naive N loader packs
them (rope rows already interleaved, dense gate/up pre-permuted for the up-gate kernel).
A torch reference walks the same per-rank weights (dequantising fp8 on the fly) with the
kernels' rounding points and greedy-decodes the same prompt; the report gives the graph
replay time per token, the full step time with the argmax handoff, and the token / logit
agreement over the checked steps. Sparse top-k is not exercised (positions stay < 2048).
Pass --checkpoint to load the identity-checked FP8 checkpoint with 16 index heads.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")

import torch  # noqa: E402

from naivert.kernels.packing import (  # noqa: E402
    dequantize_mxfp8_block32, pack_down, pack_down_mx, pack_gate_up,
    pack_gate_up_mx, pack_head, pack_unproj, permute_for_up_gate, quantize_fp8_block)
from naivert.models.naive_n0p5.kernel_geometry import ALLREDUCE_GEOMETRY, NORM_EPS, ROUTE_SCALE  # noqa: E402
from naivert.models.naive_n0p5.model import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY  # noqa: E402

RANKS = NAIVE_N0P5_GEOMETRY.rank_count
HIDDEN = NAIVE_N0P5_GEOMETRY.hidden_size
FULL = NAIVE_N0P5_HYBRID.attention_layouts["full_dsa"]
SWA = NAIVE_N0P5_HYBRID.attention_layouts["sliding_window"]
HEADS = FULL.local_heads(RANKS)                 # 8
HEAD_DIM, ROPE, VDIM = FULL.head_dim, FULL.rope_dim, FULL.v_head_dim   # 192, 64, 128
K_UNPROJ = ALLREDUCE_GEOMETRY.unproj_projection_k   # 1024
EXPERTS, TOPK, EXPERT_INTER = NAIVE_N0P5_GEOMETRY.routed_expert_count, NAIVE_N0P5_GEOMETRY.selected_expert_count, 256
DENSE_INTER = NAIVE_N0P5_GEOMETRY.dense_intermediate_size  # 2048 per rank
VOCAB_SHARD, VOCAB = NAIVE_N0P5_GEOMETRY.vocab_shard_size, NAIVE_N0P5_GEOMETRY.vocab_size
IHEADS, IDIM = NAIVE_N0P5_HYBRID.indexer.head_count, NAIVE_N0P5_HYBRID.indexer.head_dim   # 16, 128
WINDOW, WINDOW_ROWS = SWA.sliding_window, SWA.sliding_window + 8
EPS, VSCALE = float(NORM_EPS), float(FULL.value_scale)
LAYER_KINDS = NAIVE_N0P5_HYBRID.layer_kinds
MAILBOX_BYTES = 4 * 0x70000   # down_allreduce's minimum for one row


def mailbox_bytes(ops):
    """One rank's mailbox for the decode runtime: MAILBOX_BYTES, then the expert_down row."""
    return MAILBOX_BYTES + int(ops.expert_down_mailbox_row_bytes())
PANE = 0x8020


# Kernels that wait for their primary grid (cudaGridDependencySynchronize) and may therefore
# receive a programmatic edge from their predecessor in the graph.
PDL_CONSUMER_MARKERS = ("pdl_consumer", "pdl_secondary", "pdl_primary", "attn_swa_rows", "attn_dsa_rows")
# For the W8A8 verify graph only the edges into attention and expert_down pay; the prepared router /
# selector edges cost 0.1 ms per step (measured on the head). The driver picks the set per build.
PDL_W8A8_MARKERS = ("pdl_secondary", "attn_swa_rows", "attn_dsa_rows")


def torch_cudart_path():
    """The libcudart torch loaded: graph handles must be used through the same instance."""
    for line in open("/proc/self/maps", encoding="utf-8"):
        if "libcudart" in line:
            return line.split()[-1]
    return None


def install_pdl_edges(graph, markers=None):
    """Replace the linear edge into every PDL consumer kernel with a programmatic edge."""
    from naivert.runtime.cuda_backend import CudaRuntimeGraphApi
    from naivert.runtime.pdl_graph_edges import install_programmatic_edges

    api = CudaRuntimeGraphApi.load(torch_cudart_path())
    raw = graph.raw_cuda_graph()
    names = api.kernel_function_names(raw)
    positions = [i - 1 for i, name in enumerate(names)
                 if i > 0 and any(marker in name for marker in (markers or PDL_CONSUMER_MARKERS))]
    detail = install_programmatic_edges(api, raw, from_positions=positions)
    return {"kernel_nodes": len(names), "programmatic_edges": len(positions),
            "consumers": sorted({n.split("(")[0][-60:] for i, n in enumerate(names) if i - 1 in set(positions)}),
            "detail": {k: (v if isinstance(v, (int, str, bool)) else str(v)) for k, v in dict(detail).items()}}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enable_torch_tp8_peers():
    """Initialize peer access through torch, including its allocator's bookkeeping.

    Enabling peers independently in the extension leaves torch's peer table
    empty; later reference copies re-enable peers and produce CUDA error 704
    under memcheck, even though the copy itself succeeds.
    """
    for src in range(RANKS):
        with torch.cuda.device(src):
            value = torch.zeros(1, device=f"cuda:{src}")
            for dst in range(RANKS):
                if dst != src:
                    if not torch.cuda.can_device_access_peer(src, dst):
                        raise RuntimeError(f"TP8 requires peer access from GPU {src} to GPU {dst}")
                    value.to(f"cuda:{dst}")
    for rank in range(RANKS):
        torch.cuda.synchronize(rank)


# ------------------------------------------------------------------ weights
def seeded(seed: int, device):
    return torch.Generator(device=device).manual_seed(seed & 0x7FFFFFFF)


def bf16_randn(shape, seed, device, scale=0.02):
    return (torch.randn(shape, generator=seeded(seed, device), device=device) * scale).to(torch.bfloat16)


def fp8_random(shape, seed, device, scale_base=0.02 / 448):
    """Random fp8 e4m3 block-128 weight: (q [rows, cols], scale [rows/128, cols/128] bf16)."""
    g = seeded(seed, device)
    raw = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=device)
    raw[(raw & 0x7F) == 0x7F] = 0          # the NaN encodings
    q = raw.view(torch.float8_e4m3fn)
    blocks = (shape[0] // 128, shape[1] // 128)
    s = (scale_base * (0.5 + torch.rand(blocks, generator=g, device=device))).to(torch.bfloat16)
    return q, s


def dequant(q, s):
    return q.float() * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)


def mx_random(shape, seed, device, code_base=113):
    """Random mxfp8 weight: (q e4m3 [rows, cols], ue8m0 codes uint8 [rows, cols/32] around 2^-14)."""
    g = seeded(seed, device)
    raw = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=device)
    raw[(raw & 0x7F) == 0x7F] = 0          # the NaN encodings
    q = raw.view(torch.float8_e4m3fn)
    codes = (code_base + torch.randint(-1, 2, (shape[0], shape[1] // 32), generator=g, device=device)).to(torch.uint8)
    return q, codes


def rope_table(max_pos, theta, device):
    freqs = 1.0 / (theta ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).reshape(max_pos, ROPE).contiguous()


def expert_seed(layer, rank, expert, which):
    return (layer * 1000003 + rank * 10007 + expert * 101 + which) * 2654435761 % (1 << 31)


class RankWeights:
    """Everything one rank holds, packed for the kernels, plus what the reference re-derives."""

    def __init__(self, rank, layers, seed, max_seq_len, ext_dsa_ws, ext_swa_ws, experts="fp8", expert_down="bf16"):
        dev = torch.device(f"cuda:{rank}")
        self.rank, self.dev, self.layers, self.experts = rank, dev, layers, experts
        base = seed * 1_000_003
        self.embedding = bf16_randn((VOCAB, HIDDEN), base + 1, dev, 1.0)            # replicated
        self.head_gamma = 1.0 + 0.1 * torch.randn(HIDDEN, generator=seeded(base + 2, dev), device=dev)
        lm = bf16_randn((VOCAB_SHARD, HIDDEN), base + 3 + rank, dev)
        self.lm_head_ref = lm
        self.lm_head = pack_head(lm, hidden=HIDDEN)
        self.rope_full = rope_table(max_seq_len, FULL.rope_theta, dev)
        self.rope_swa = rope_table(max_seq_len, SWA.rope_theta, dev)
        self.layer = []
        for L in range(layers):
            kind = LAYER_KINDS[L]
            lseed = base + 1000 * (L + 1)
            w = {"kind": kind}
            w["attn_gamma"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=seeded(lseed + 1, dev), device=dev)
            w["ffn_gamma"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=seeded(lseed + 2, dev), device=dev)
            w["wq"] = bf16_randn((HEADS * HEAD_DIM, HIDDEN), lseed + 10 + rank, dev)
            w["wk"] = bf16_randn((HEAD_DIM, HIDDEN), lseed + 20 + rank, dev)
            w["wv"] = (bf16_randn((VDIM, HIDDEN), lseed + 30 + rank, dev).float() * VSCALE).to(torch.bfloat16)
            if kind == "full_dsa":
                w["wiq"], w["wiq_scale"] = quantize_fp8_block(bf16_randn((IHEADS * IDIM, HIDDEN), lseed + 40, dev))
                w["wik"] = bf16_randn((IDIM, HIDDEN), lseed + 41, dev)
                w["ik_w"] = 1.0 + 0.1 * torch.randn(IDIM, generator=seeded(lseed + 42, dev), device=dev)
                w["ik_b"] = 0.1 * torch.randn(IDIM, generator=seeded(lseed + 43, dev), device=dev)
                w["wis"] = bf16_randn((IHEADS, HIDDEN), lseed + 44, dev, 0.05)
                w["k_cache"] = torch.zeros(max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)
                w["v_cache"] = torch.zeros(max_seq_len, VDIM, dtype=torch.bfloat16, device=dev)
                w["ki_cache"] = torch.zeros(max_seq_len, IDIM, dtype=torch.float8_e4m3fn, device=dev)
                w["ki_scale"] = torch.zeros(max_seq_len, dtype=torch.float32, device=dev)
                w["attn_ws"] = torch.zeros(ext_dsa_ws, dtype=torch.uint8, device=dev)
            else:
                w["sink"] = 0.5 * torch.randn(HEADS, generator=seeded(lseed + 45 + rank, dev), device=dev)
                w["k_cache"] = torch.zeros(WINDOW_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
                w["v_cache"] = torch.zeros(WINDOW_ROWS, VDIM, dtype=torch.bfloat16, device=dev)
                w["attn_ws"] = torch.zeros(ext_swa_ws, dtype=torch.uint8, device=dev)
            oq, os_ = quantize_fp8_block(bf16_randn((HIDDEN, K_UNPROJ), lseed + 50 + rank, dev))
            w["o_q"], w["o_s"] = oq, os_
            w["o_packed"], w["o_scale"] = pack_unproj(oq, os_, hidden=HIDDEN, outputs_per_tile=ALLREDUCE_GEOMETRY.unproj_outputs_per_tile, input_slots=4)
            if L == 0:
                gate = bf16_randn((DENSE_INTER, HIDDEN), lseed + 60 + rank, dev)
                up = bf16_randn((DENSE_INTER, HIDDEN), lseed + 61 + rank, dev)
                slots = DENSE_INTER // EXPERT_INTER

                def parts(g, u):
                    gq, gs = zip(*(quantize_fp8_block(t) for t in g.reshape(slots, EXPERT_INTER, HIDDEN)))
                    uq, us = zip(*(quantize_fp8_block(t) for t in u.reshape(slots, EXPERT_INTER, HIDDEN)))
                    return tuple(torch.stack(t) for t in (gq, gs, uq, us))

                w["dense_gate_q"], w["dense_gate_s"], w["dense_up_q"], w["dense_up_s"] = parts(gate, up)
                native = pack_gate_up(*parts(permute_for_up_gate(gate), permute_for_up_gate(up)), hidden=HIDDEN)
                w["dense_ws"] = native.view(torch.uint8).reshape(slots * 16, HIDDEN // 1024, 32 * 1024 + 128)[:, :, :PANE].contiguous()
                down = bf16_randn((DENSE_INTER, HIDDEN), lseed + 62 + rank, dev)            # down^T
                cols = down.T.reshape(HIDDEN, slots, EXPERT_INTER).transpose(0, 1)
                dq, ds = zip(*(quantize_fp8_block(c) for c in cols))
                w["dense_down_q"], w["dense_down_s"] = torch.stack(dq), torch.stack(ds).to(torch.bfloat16)
                w["dense_down"], w["dense_down_scale"] = pack_down(w["dense_down_q"], w["dense_down_s"], hidden=HIDDEN, outputs_per_tile=ALLREDUCE_GEOMETRY.down_outputs_per_tile, inter=EXPERT_INTER)
            else:
                w["router"] = bf16_randn((EXPERTS, HIDDEN), lseed + 70, dev).to(torch.float32 if expert_down == "w8a8" else torch.bfloat16)                 # replicated
                w["router_bias"] = 0.1 * torch.randn(EXPERTS, generator=seeded(lseed + 71, dev), device=dev)
                if experts == "mx":
                    gq = torch.empty((EXPERTS, EXPERT_INTER, HIDDEN), dtype=torch.float8_e4m3fn, device=dev)
                    gs = torch.empty((EXPERTS, EXPERT_INTER, HIDDEN // 32), dtype=torch.uint8, device=dev)
                    uq, us = torch.empty_like(gq), torch.empty_like(gs)
                    dq = torch.empty((EXPERTS, HIDDEN, EXPERT_INTER), dtype=torch.float8_e4m3fn, device=dev)
                    ds = torch.empty((EXPERTS, HIDDEN, EXPERT_INTER // 32), dtype=torch.uint8, device=dev)
                    for e in range(EXPERTS):
                        gq[e], gs[e] = mx_random((EXPERT_INTER, HIDDEN), expert_seed(L, rank, e, 0), dev)
                        uq[e], us[e] = mx_random((EXPERT_INTER, HIDDEN), expert_seed(L, rank, e, 1), dev)
                        dq[e], ds[e] = mx_random((HIDDEN, EXPERT_INTER), expert_seed(L, rank, e, 2), dev)
                    # fp8-typed views of the byte blobs (the bindings check the dtype); the scale argument is unused.
                    w["expert_table"] = pack_gate_up_mx(gq, gs, uq, us, hidden=HIDDEN).view(torch.float8_e4m3fn)
                    w["expert_down"] = pack_down_mx(dq, ds, hidden=HIDDEN, outputs_per_tile=ALLREDUCE_GEOMETRY.expert_down_outputs_per_tile, inter=EXPERT_INTER).view(torch.float8_e4m3fn)
                    w["expert_down_scale"] = torch.zeros(16, dtype=torch.bfloat16, device=dev)
                else:
                    gq = torch.empty((EXPERTS, EXPERT_INTER, HIDDEN), dtype=torch.float8_e4m3fn, device=dev)
                    gs = torch.empty((EXPERTS, EXPERT_INTER // 128, HIDDEN // 128), dtype=torch.bfloat16, device=dev)
                    uq, us = torch.empty_like(gq), torch.empty_like(gs)
                    dq = torch.empty((EXPERTS, HIDDEN, EXPERT_INTER), dtype=torch.float8_e4m3fn, device=dev)
                    ds = torch.empty((EXPERTS, HIDDEN // 128, EXPERT_INTER // 128), dtype=torch.bfloat16, device=dev)
                    for e in range(EXPERTS):
                        gq[e], gs[e] = fp8_random((EXPERT_INTER, HIDDEN), expert_seed(L, rank, e, 0), dev)
                        uq[e], us[e] = fp8_random((EXPERT_INTER, HIDDEN), expert_seed(L, rank, e, 1), dev)
                        dq[e], ds[e] = fp8_random((HIDDEN, EXPERT_INTER), expert_seed(L, rank, e, 2), dev)
                    w["expert_table"] = pack_gate_up(gq, gs, uq, us, hidden=HIDDEN,
                                                          scale_dtype=torch.float32 if expert_down == "w8a8" else torch.bfloat16)
                    w["expert_down"], w["expert_down_scale"] = pack_down(dq, ds, hidden=HIDDEN, outputs_per_tile=ALLREDUCE_GEOMETRY.expert_down_outputs_per_tile, inter=EXPERT_INTER,
                                                                         layout="umma_sw128" if expert_down == "w8a8" else "qmma",
                                                                         scale_dtype=torch.float32 if expert_down == "w8a8" else torch.bfloat16)
                del gq, gs, uq, us, dq, ds
            self.layer.append(w)
        torch.cuda.synchronize(dev)

    def expert(self, L, e, which):
        shape = (HIDDEN, EXPERT_INTER) if which == 2 else (EXPERT_INTER, HIDDEN)
        if self.experts == "mx":
            return mx_random(shape, expert_seed(L, self.rank, e, which), self.dev)
        return fp8_random(shape, expert_seed(L, self.rank, e, which), self.dev)

    def dequant_expert(self, q, s):
        if self.experts == "mx":
            return dequantize_mxfp8_block32(q, s)
        return dequant(q, s)


# ------------------------------------------------------------- kernel path
class RankRuntime:
    def __init__(self, rank, weights: RankWeights, ops, ext_dsa, ext_swa, max_seq_len, mailboxes, inboxes, pdl=False,
                 fused_unproj=False, device_token=False):
        dev = weights.dev
        self.rank, self.w, self.ops, self.ext_dsa, self.ext_swa = rank, weights, ops, ext_dsa, ext_swa
        self.dev, self.max_seq_len = dev, max_seq_len
        self.mailboxes, self.inbox_ptrs = mailboxes, [int(t.data_ptr()) for t in inboxes]
        self.stream = torch.cuda.Stream(device=dev)
        layers = weights.layers
        self.pdl, self.pdl_report = pdl, None
        self.fused_unproj = fused_unproj   # O-projection + all-reduce inside the attention kernel
        # The TP8 argmax kernel writes the next token on device; the graph's embed reads it.
        self.device_token, self.inboxes = device_token, list(inboxes)
        self.token_out = torch.full((1,), -1, dtype=torch.int32, device=dev)
        self.top_p = torch.zeros(1, dtype=torch.float32, device=dev)
        self.argmax_logits = torch.zeros(VOCAB_SHARD, dtype=torch.float32, device=dev)   # binding check only
        self.token = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        # One replay epoch per step: every mailbox exchange forms its generation as
        # (epoch << 8) | layer tag, so consecutive steps never accept each other's payloads.
        self.epoch = torch.ones(1, dtype=torch.int64, device=dev)
        self.generation = torch.ones(1, dtype=torch.int32, device=dev)
        self.step_block = torch.zeros(16, dtype=torch.uint8, device=dev)
        self.resid = [torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(layers + 1)]
        self.hid_a = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.proj_o = torch.zeros(1, HEADS, VDIM, dtype=torch.bfloat16, device=dev)
        self.scores = torch.zeros(1, EXPERTS, dtype=torch.float32, device=dev)
        self.x_mlp = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.moe_up = torch.zeros(1, TOPK, 16, 16, dtype=(torch.float32 if ops.expert_activation_is_fp32() else torch.bfloat16), device=dev)
        self.probs = torch.zeros(1, TOPK, dtype=torch.float32, device=dev)
        self.indices = torch.zeros(1, TOPK, dtype=torch.int32, device=dev)
        self.dense_up = torch.zeros(1, DENSE_INTER, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(1, VOCAB_SHARD, dtype=torch.float32, device=dev)
        self.last_hidden = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.head_ws = torch.zeros(64, dtype=torch.uint8, device=dev)
        self.dummy8 = torch.zeros(16, dtype=torch.float8_e4m3fn, device=dev)
        self.dummy16 = torch.zeros(16, dtype=torch.bfloat16, device=dev)
        domain_size = int(ops.expert_down_domain_size())
        self.domains = {}
        with torch.cuda.device(dev):
            for L in range(1, layers):
                w = weights.layer[L]
                domain = torch.zeros(domain_size, dtype=torch.uint8, device=dev)
                ops.prepare_expert_down_replay_domain_epoch(
                    domain, self.moe_up[0], w["expert_down"], w["expert_down_scale"], self.probs[0],
                    self.indices[0], self.resid[L + 1][0], self.hid_a[0],
                    [mb[MAILBOX_BYTES:MAILBOX_BYTES + int(ops.expert_down_mailbox_row_bytes())] for mb in mailboxes],   # Own region
                    2 * L + 3, L, rank, self.epoch)
                self.domains[L] = domain
        self.graph = None

    def phases(self, mark: bool = False):
        # PDL variants: consumers wait for the primary grid, the router triggers early.
        _ops = self.ops
        unproj_launch = _ops.launch_unproj_allreduce_replay_pdl_consumer if self.pdl else _ops.launch_unproj_allreduce_replay
        router_launch = _ops.launch_rmsnorm_expert_proj_pdl_producer if self.pdl else _ops.launch_rmsnorm_expert_proj
        select_launch = (_ops.launch_expert_select_up_gate_silu_rows1_pdl_consumer if self.pdl
                         else _ops.launch_expert_select_up_gate_silu_rows1)
        head_launch = _ops.launch_rmsnorm_head_proj_rows1_pdl_consumer if self.pdl else _ops.launch_rmsnorm_head_proj_rows1
        """The decode step as (name, callable) phases in production order.

        With mark=True every phase is followed by a device-side progress stamp (debugging).
        """
        w, ops, r = self.w, self.ops, self.rank
        if self.device_token:
            out = [("embed", lambda: torch.index_select(w.embedding, 0, self.token_out[:1].to(torch.long),
                                                         out=self.resid[0]))]
        else:
            out = [("embed", lambda: torch.index_select(w.embedding, 0, self.token, out=self.resid[0]))]
        if mark and not hasattr(self, "marker"):
            self.marker = torch.zeros(1, dtype=torch.int32, device=self.dev)
        for L in range(w.layers):
            lw = w.layer[L]
            # With fused_unproj the attention kernel also computes hid_a = resid + allreduce(o_proj).
            tail = (lw["o_packed"], lw["o_scale"], self.hid_a, self.mailboxes, self.epoch, r, 2 * L + 2)
            if lw["kind"] == "full_dsa":
                dsa_args = lambda lw=lw, L=L: (  # noqa: E731
                    self.resid[L], lw["attn_gamma"], lw["wq"], lw["wk"], lw["wv"], lw["wiq"], lw["wiq_scale"], lw["wik"],
                    lw["ik_w"], lw["ik_b"], lw["wis"], w.rope_full, lw["k_cache"], lw["v_cache"],
                    lw["ki_cache"], lw["ki_scale"], self.proj_o, lw["attn_ws"], self.pos, self.max_seq_len, 0)
                if self.fused_unproj:
                    out.append((f"L{L}.attn_dsa+unproj", lambda a=dsa_args, t=tail, vs=lw.get("value_scale", 1.0): self.ext_dsa.run_fused_scaled(*a(), *t, vs)))
                else:
                    out.append((f"L{L}.attn_dsa", lambda a=dsa_args, vs=lw.get("value_scale", 1.0): self.ext_dsa.run_scaled(*a(), vs)))
            else:
                swa_args = lambda lw=lw, L=L: (  # noqa: E731
                    self.resid[L], lw["attn_gamma"], lw["wq"], lw["wk"], lw["wv"], w.rope_swa, lw["sink"],
                    lw["k_cache"], lw["v_cache"], self.proj_o, lw["attn_ws"], self.pos, self.max_seq_len, 0)
                if self.fused_unproj:
                    out.append((f"L{L}.attn_swa+unproj", lambda a=swa_args, t=tail, vs=lw.get("value_scale", 1.0): self.ext_swa.run_fused_scaled(*a(), *t, vs)))
                else:
                    out.append((f"L{L}.attn_swa", lambda a=swa_args, vs=lw.get("value_scale", 1.0): self.ext_swa.run_scaled(*a(), vs)))
            if not self.fused_unproj and lw["o_packed"].dtype == torch.bfloat16:
                out.append((f"L{L}.unproj_bf16", lambda lw=lw, L=L: ops.launch_checkpoint_bf16_linear(
                    self.proj_o.view(1, K_UNPROJ), lw["o_packed"], self.resid[L], self.hid_a,
                    self.mailboxes, self.epoch, r, 2 * L + 2)))
            elif not self.fused_unproj:
                out.append((f"L{L}.unproj", lambda lw=lw, L=L: unproj_launch(
                    self.proj_o.view(1, K_UNPROJ), lw["o_packed"], lw["o_scale"], self.resid[L], self.hid_a,
                    self.mailboxes, r, 2 * L + 2, self.epoch.view(torch.uint8))))
            if L == 0 and "dense_gate" in lw:
                out.append((f"L{L}.up_gate_bf16", lambda lw=lw: ops.launch_checkpoint_bf16_swiglu(
                    self.hid_a, lw["ffn_gamma"], lw["dense_gate"], lw["dense_up"], self.dense_up)))
                out.append((f"L{L}.down_bf16", lambda lw=lw, L=L: ops.launch_checkpoint_bf16_linear(
                    self.dense_up, lw["dense_down"], self.hid_a, self.resid[L + 1], self.mailboxes, self.epoch, r, 3)))
            elif L == 0:
                out.append((f"L{L}.up_gate", lambda lw=lw: ops.launch_rmsnorm_up_gate(
                    self.hid_a, lw["ffn_gamma"], self.dummy8, self.dummy16, self.dummy8, self.dummy16,
                    lw["dense_ws"], self.dense_up, HIDDEN, DENSE_INTER)))
                out.append((f"L{L}.down", lambda lw=lw, L=L: ops.launch_down_allreduce(
                    self.dense_up, lw["dense_down"], lw["dense_down_scale"], self.hid_a, self.resid[L + 1],
                    self.step_block, self.generation, 3, self.mailboxes, r, HIDDEN, DENSE_INTER)))
            else:
                out.append((f"L{L}.router", lambda lw=lw: router_launch(
                    self.hid_a, lw["ffn_gamma"], lw["router"], self.scores, self.x_mlp, HIDDEN, EXPERTS)))
                out.append((f"L{L}.expert_select", lambda lw=lw: select_launch(
                    self.x_mlp, self.scores, lw["router_bias"], lw["expert_table"], self.moe_up, self.probs,
                    self.indices)))
                out.append((f"L{L}.expert_down", lambda L=L: ops.launch_expert_down_rows_from_replay_domain(
                    self.domains[L], L, r, 1)))
        out.append(("head", lambda: head_launch(
            self.resid[w.layers], w.head_gamma, w.lm_head, self.logits, self.last_hidden, self.head_ws,
            HIDDEN, VOCAB_SHARD, self.inbox_ptrs, r)))
        if self.device_token:
            out.append(("argmax", lambda: ops.launch_tp8_argmax_rows(
                self.argmax_logits, self.token_out, self.top_p, self.inboxes, r)))

        def advance():
            self.pos += 1
            self.generation += 1
            self.epoch += 1
        out.append(("advance", advance))
        if mark:
            stamped = []
            for k, (name, fn) in enumerate(out):
                def run_and_stamp(fn=fn, k=k):
                    fn()
                    self.marker.fill_(k + 1)
                stamped.append((name, run_and_stamp))
            return stamped
        return out

    def step(self):
        """One decode step for this rank on its stream (eager or under capture)."""
        for _name, fn in self.phases():
            fn()

    def capture(self):
        with torch.cuda.device(self.dev):
            # keep_graph: leave the graph un-instantiated so its edges can be rewritten first.
            self.graph = torch.cuda.CUDAGraph(keep_graph=self.pdl)
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.step()
            if self.pdl:
                self.pdl_report = install_pdl_edges(self.graph, getattr(self, 'pdl_markers', None))
                self.graph.instantiate()

    def run_step(self):
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            if self.graph is not None:
                self.graph.replay()
            else:
                self.step()


# ------------------------------------------------------------ reference
class Reference:
    """Torch forward over the same per-rank weights with the kernels' rounding points."""

    def __init__(self, weights: list[RankWeights], layers: int, expert_down="bf16"):
        self.w, self.layers = weights, layers
        self.expert_down = expert_down
        self.k = [[[] for _ in range(RANKS)] for _ in range(layers)]   # per layer, per rank: list of k rows
        self.v = [[[] for _ in range(RANKS)] for _ in range(layers)]
        self.pos = 0

    @staticmethod
    def rmsnorm(x, gamma):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma).to(torch.bfloat16)

    @staticmethod
    def rope(x, table, pos):
        """Interleaved-pair rope on the first 64 dims of each 192-vector row (x [.., 192] f32)."""
        cs = table[pos].view(ROPE // 2, 2)
        pairs = x[..., :ROPE].reshape(*x.shape[:-1], ROPE // 2, 2)
        real = pairs[..., 0] * cs[:, 0] - pairs[..., 1] * cs[:, 1]
        imag = pairs[..., 0] * cs[:, 1] + pairs[..., 1] * cs[:, 0]
        out = x.clone()
        out[..., :ROPE] = torch.stack([real, imag], -1).reshape(*x.shape[:-1], ROPE)
        return out

    def step(self, token: int, *, trace=None, layer_inputs=None, attention_inputs=None):
        pos = self.pos
        dev0 = self.w[0].dev
        x = self.w[0].embedding[token].clone()   # bf16 [4096] on device 0
        for L in range(self.layers):
            if layer_inputs is not None and layer_inputs[L] is not None:
                x = layer_inputs[L].to(dev0).reshape(HIDDEN)
            detail = {}
            kind = self.w[0].layer[L]["kind"]
            partials = []
            for r in range(RANKS):
                w, lw = self.w[r], self.w[r].layer[L]
                xr = x.to(w.dev)
                xn = self.rmsnorm(xr, lw["attn_gamma"])
                table = w.rope_full if kind == "full_dsa" else w.rope_swa
                q = (xn.float() @ lw["wq"].float().T).to(torch.bfloat16).float().view(HEADS, HEAD_DIM)
                k = (xn.float() @ lw["wk"].float().T).to(torch.bfloat16).float()
                v = ((xn.float() @ lw["wv"].float().T) * lw.get("value_scale", 1.0)).to(torch.bfloat16)
                q = self.rope(q, table, pos).to(torch.bfloat16).float()
                k = self.rope(k, table, pos).to(torch.bfloat16)
                self.k[L][r].append(k)
                self.v[L][r].append(v)
                keys, vals = torch.stack(self.k[L][r]).float(), torch.stack(self.v[L][r]).float()
                if kind != "full_dsa":
                    keys, vals = keys[-WINDOW:], vals[-WINDOW:]
                scores = (q @ keys.T) * HEAD_DIM ** -0.5
                if kind != "full_dsa":
                    scores = torch.cat([scores, lw["sink"][:, None]], -1)
                    probs = torch.softmax(scores, -1)[:, :-1]
                else:
                    probs = torch.softmax(scores, -1)
                o = (probs @ vals).to(torch.bfloat16).float().reshape(K_UNPROJ)
                if lw["o_packed"].dtype == torch.bfloat16:
                    partials.append((lw["o_packed"].float() @ o).to(dev0))
                else:
                    partials.append((dequant(lw["o_q"], lw["o_s"]) @ o).to(torch.bfloat16).float().to(dev0))
            # The O-projection mailbox rounds its TP sum before adding residual.
            x_a = (x.float() + sum(partials).to(torch.bfloat16).float()).to(torch.bfloat16)
            if trace is not None:
                detail["attention_output"] = x_a.clone()
            if attention_inputs is not None:
                x_a = attention_inputs[L].to(dev0).reshape(HIDDEN)
            partials = []
            if L == 0:
                for r in range(RANKS):
                    w, lw = self.w[r], self.w[r].layer[L]
                    if "dense_gate" in lw:
                        xm = self.rmsnorm(x_a.to(w.dev), lw["ffn_gamma"]).float()
                        y = (torch.nn.functional.silu(lw["dense_gate"].float() @ xm) *
                             (lw["dense_up"].float() @ xm)).bfloat16().float()
                        partials.append((lw["dense_down"].float() @ y).to(dev0))
                        continue
                    xm = self.rmsnorm(x_a.to(w.dev), lw["ffn_gamma"]).to(torch.float16).float()
                    slots = DENSE_INTER // EXPERT_INTER
                    g = torch.cat([dequant(lw["dense_gate_q"][s], lw["dense_gate_s"][s]) @ xm for s in range(slots)])
                    u = torch.cat([dequant(lw["dense_up_q"][s], lw["dense_up_s"][s]) @ xm for s in range(slots)])
                    y = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
                    part = sum(dequant(lw["dense_down_q"][s], lw["dense_down_s"][s]) @ y[s * EXPERT_INTER:(s + 1) * EXPERT_INTER]
                               for s in range(slots))
                    partials.append(part.to(torch.bfloat16).float().to(dev0))
            else:
                lw0 = self.w[0].layer[L]
                xm0 = self.rmsnorm(x_a, lw0["ffn_gamma"])
                sc = xm0.float() @ lw0["router"].float().T
                sig = torch.sigmoid(sc)
                idx = torch.topk(sig + lw0["router_bias"], TOPK).indices
                pr = sig[idx] / (sig[idx].sum() + (1e-20 if self.expert_down == "w8a8" else 0)) * ROUTE_SCALE
                if trace is not None:
                    detail.update(scores=sc.clone(), indices=idx.clone(), probs=pr.clone())
                for r in range(RANKS):
                    w, lw = self.w[r], self.w[r].layer[L]
                    xm = self.rmsnorm(x_a.to(w.dev), lw["ffn_gamma"])
                    if self.expert_down == "w8a8":
                        from naivert.models.naive_n0p5.training_moe import quantize_down_activation_reference
                        xq, xs = quantize_down_activation_reference(xm.cpu())
                        x8 = (xq.float().reshape(-1, 128) * xs[:, None]).flatten().to(w.dev)
                    else:
                        x8 = xm.to(torch.float8_e4m3fn).float()
                    part = torch.zeros(HIDDEN, device=w.dev)
                    expert_parts = []
                    for e, p in zip(idx.tolist(), pr.tolist()):
                        gq, gs = w.expert(L, e, 0)
                        uq, us = w.expert(L, e, 1)
                        dq, ds = w.expert(L, e, 2)
                        y = torch.nn.functional.silu(w.dequant_expert(gq, gs) @ x8) * (w.dequant_expert(uq, us) @ x8)
                        if self.expert_down != "w8a8":
                            y = y.bfloat16().float()
                        if self.expert_down == "w8a8":
                            from naivert.models.naive_n0p5.training_moe import quantize_down_activation_reference
                            # The oracle is deliberately CPU-side. This torch
                            # build's NVRTC predates sm_103 and cannot JIT exp2.
                            yq, ys = quantize_down_activation_reference(y.cpu())
                            yq, ys = yq.to(w.dev), ys.to(w.dev)
                            expert_part = torch.zeros(HIDDEN, device=w.dev)
                            for kg in range(2):
                                dot = dq[:, kg*128:(kg+1)*128].float() @ yq[kg*128:(kg+1)*128].float()
                                expert_part += dot * (ds[:, kg].float().repeat_interleave(128) * ys[kg])
                            expert_parts.append(expert_part)
                        else:
                            part += p * (w.dequant_expert(dq, ds) @ y)
                    if self.expert_down == "w8a8":
                        partials.append(torch.stack(expert_parts).to(dev0))
                    else:
                        partials.append(part.to(torch.bfloat16).float().to(dev0))
            if L == 0:
                # Dense down uses the same separate TP-sum rounding as unproj.
                x = (x_a.float() + sum(partials).to(torch.bfloat16).float()).to(torch.bfloat16)
            elif self.expert_down == "w8a8":
                complete = sum(partials).to(torch.bfloat16).float()
                mixture = (complete * pr[:, None]).sum(0).to(torch.bfloat16)
                x = (mixture + x_a).to(torch.bfloat16)
            else:
                # Expert down starts with the residual and adds ranks 0..7 in FP32.
                total = x_a.float()
                for part in partials:
                    total = total + part
                x = total.to(torch.bfloat16)
            if trace is not None:
                detail["output"] = x.clone()
                trace.append(detail)
        xn = self.rmsnorm(x, self.w[0].head_gamma)
        logits = torch.cat([(xn.to(w.dev).float() @ w.lm_head_ref.float().T).to(dev0) for w in self.w])
        self.pos += 1
        return logits


# ------------------------------------------------------------------- driver

def top2_gap(logits):
    """Top-1 minus top-2 logit: the margin an argmax comparison rests on."""
    top = logits.float().flatten().topk(2).values
    return float(top[0] - top[1])


def agreement(logits, ref_logits):
    return {"finite": bool(torch.isfinite(logits).all() and torch.isfinite(ref_logits).all()),
            "logit_max_abs_diff": float((logits - ref_logits).abs().max()),
            "logit_scale": float(ref_logits.abs().max()),
            "argmax_same": int(logits.argmax()) == int(ref_logits.argmax()),
            "kernel_top1": int(logits.argmax()), "ref_top1": int(ref_logits.argmax()),
            "kernel_top2_gap": round(top2_gap(logits), 5), "ref_top2_gap": round(top2_gap(ref_logits), 5),
            # Decided: the reference's own margin exceeds twice the observed logit error
            # (the verify driver's rule); an argmax flip on an undecided step is a tie.
            "decided": top2_gap(ref_logits) > 2.0 * float((logits - ref_logits).abs().max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=len(LAYER_KINDS))
    ap.add_argument("--checkpoint", help="target FP8 checkpoint directory (or its parent); uses 16 index heads")
    ap.add_argument("--build-only", action="store_true", help="compile extensions without initializing CUDA")
    ap.add_argument("--correctness-only", action="store_true", help="write numerical checks and skip timing loops")
    ap.add_argument("--trace-reference", action="store_true",
                    help="diagnose first-token layer errors, including identical-input attention and FFN comparisons")
    ap.add_argument("--trace-tensors", help="save first-token operator inputs/outputs for isolated numerical diagnosis")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32, help="decode steps to time after the prompt")
    ap.add_argument("--check-steps", type=int, default=4, help="steps compared against the torch reference (0 = skip)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--debug-phases", action="store_true",
                    help="report the phase-lockstepped warm-up step's per-phase wall times")
    ap.add_argument("--sequential-debug", action="store_true",
                    help="issue the warm-up step rank by rank and watch the streams (hang diagnosis)")
    ap.add_argument("--out", default="")
    ap.add_argument("--experts", choices=("fp8", "mx"), default="fp8",
                    help="routed-expert weight format: fp8 block-128 or MX8 block-32 (separate model_ops build)")
    ap.add_argument("--expert-down", choices=("bf16", "w8a8"), default="bf16",
                    help="native W8A8 down with K128 activation scales and FP32 per-expert mailbox reduction")
    ap.add_argument("--pdl", action="store_true",
                    help="programmatic dependent launch: PDL kernel variants + programmatic graph edges")
    ap.add_argument("--fused-unproj", action="store_true",
                    help="o-projection + TP8 all-reduce inside the attention kernels; drops the unproj launch")
    ap.add_argument("--device-token", action="store_true",
                    help="TP8 argmax on device inside the graph feeds the next step's embedding")
    args = ap.parse_args()
    if args.expert_down == "w8a8" and (args.experts != "fp8" or args.sequential_debug):
        ap.error("W8A8 down requires fp8 block128 weights and lockstep warmup")
    if args.checkpoint and args.experts != "fp8":
        ap.error("the FP8 checkpoint uses the fp8 block-128 expert path")
    if args.trace_reference and (not args.check_steps or args.sequential_debug):
        ap.error("--trace-reference needs --check-steps > 0 and phase-lockstepped warmup")
    if args.trace_tensors and not args.trace_reference:
        ap.error("--trace-tensors requires --trace-reference")
    if args.build_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.check_steps and args.prompt_len + args.check_steps > 2048:
        ap.error("the decode torch reference covers dense attention only; use the DSA rows probe for sparse validation")
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    dsa_mod = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa.py")
    swa_mod = load_module(ROOT / "tests/gpu/attn_swa/run_attn_swa.py")
    t0 = time.time()
    ops = load_extension(weights=args.experts, expert_down=args.expert_down)
    print(f"experts: {args.experts}", flush=True)
    ext_dsa, ext_swa = dsa_mod.build(False, IHEADS), swa_mod.build()
    if args.build_only:
        return 0
    enable_torch_tp8_peers()
    print(f"extensions ready in {time.time() - t0:.0f} s; layers {args.layers} "
          f"({sum(k == 'full_dsa' for k in LAYER_KINDS[:args.layers])} full, "
          f"{sum(k != 'full_dsa' for k in LAYER_KINDS[:args.layers])} sliding-window)", flush=True)
    t0 = time.time()
    if args.checkpoint:
        from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, load_tp8_weights
        checkpoint = NaiveNCheckpoint(args.checkpoint)
        weights = load_tp8_weights(checkpoint, args.layers, args.max_seq_len,
                                   ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(), expert_down=args.expert_down)
    else:
        weights = [RankWeights(r, args.layers, args.seed, args.max_seq_len, ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(),
                               experts=args.experts, expert_down=args.expert_down) for r in range(RANKS)]
    print(f"weights prepared and packed in {time.time() - t0:.0f} s "
          f"({torch.cuda.memory_allocated(0) / 2**30:.1f} GiB on rank 0)", flush=True)
    mailboxes = [torch.zeros(mailbox_bytes(ops), dtype=torch.uint8, device=w.dev) for w in weights]
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=w.dev) for w in weights]
    runtimes = [RankRuntime(r, weights[r], ops, ext_dsa, ext_swa, args.max_seq_len, mailboxes, inboxes, pdl=args.pdl,
                            fused_unproj=args.fused_unproj, device_token=args.device_token)
                for r in range(RANKS)]
    reference = Reference(weights, args.layers, args.expert_down) if args.check_steps else None
    gen = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(0, VOCAB, (args.prompt_len,), generator=gen).tolist()

    def sync_all():
        for rt in runtimes:
            torch.cuda.synchronize(rt.dev)

    def set_token(token):
        # The graphs replay on the rank streams; the token write must be ordered on them too
        # (a default-stream write can trail reference work and be read one step late).
        for rt in runtimes:
            with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                rt.token.fill_(token)
                if rt.device_token:
                    rt.token_out[:1].fill_(token)

    def kernel_step(token):
        set_token(token)
        for rt in runtimes:
            rt.run_step()
        sync_all()
        logits = torch.cat([rt.logits[0].to(runtimes[0].dev) for rt in runtimes])
        return logits

    def sequential_debug_step(token, timeout_s=30.0):
        """Issue every rank's whole step in sequence (as a graph replay would), then watch the streams."""
        set_token(token)
        names = [n for n, _ in runtimes[0].phases(mark=True)]
        for rt in runtimes:
            with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                for name, fn in rt.phases(mark=True):
                    print(f"    issue rank {rt.rank} {name}", flush=True)
                    t_issue = time.time()
                    fn()
                    if time.time() - t_issue > 0.5:
                        print(f"      ^ the host spent {time.time() - t_issue:.1f} s inside this launch", flush=True)
        print("    all launches issued", flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            busy = [rt.rank for rt in runtimes if not rt.stream.query()]
            if not busy:
                sync_all()
                print("    sequential step completed", flush=True)
                return torch.cat([rt.logits[0].to(runtimes[0].dev) for rt in runtimes])
            time.sleep(1.0)
            done = {rt.rank: (names[int(rt.marker.item()) - 1] if rt.rank not in busy else "busy")
                    for rt in runtimes}
            print(f"    t={time.time() - t0:4.0f}s busy ranks {busy}; last completed phase per idle rank: "
                  f"{ {r: p for r, p in done.items() if r not in busy} }", flush=True)
        print("    HANG: ranks still busy after timeout:", busy, flush=True)
        os._exit(2)

    kernel_trace = [{} for _ in range(args.layers)]

    def lockstep_debug_step(token):
        """Every phase on all ranks, then a sync: a hang names the phase, and the syncs time it."""
        set_token(token)
        phase_lists = [rt.phases() for rt in runtimes]
        timings = []
        for k, (name, _) in enumerate(phase_lists[0]):
            print(f"    phase {name} ...", end="", flush=True)
            t0 = time.time()
            for rt, phases in zip(runtimes, phase_lists):
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    phases[k][1]()
            sync_all()
            if args.trace_reference and name.startswith("L"):
                layer, phase = name.split(".", 1)
                detail = kernel_trace[int(layer[1:])]
                if phase in ("unproj", "unproj_bf16") or phase.endswith("+unproj"):
                    detail["attention_output"] = runtimes[0].hid_a[0].clone()
                    if args.trace_tensors:
                        detail["proj_o"] = [rt.proj_o.cpu() for rt in runtimes]
                if phase == "router":
                    detail["scores"] = runtimes[0].scores[0].clone()
                    if args.trace_tensors:
                        detail["normalized"] = [rt.x_mlp.cpu() for rt in runtimes]
                if phase == "expert_select":
                    detail["indices"] = runtimes[0].indices[0].clone()
                    detail["probs"] = runtimes[0].probs[0].clone()
                    if args.trace_tensors:
                        detail["moe_up"] = [rt.moe_up.cpu() for rt in runtimes]
                if args.trace_tensors and phase in ("up_gate", "up_gate_bf16"):
                    detail["dense_up"] = [rt.dense_up.cpu() for rt in runtimes]
            dt = (time.time() - t0) * 1e3
            timings.append((name, round(dt, 3)))
            print(f" {dt:.2f} ms", flush=True)
        return torch.cat([rt.logits[0].to(runtimes[0].dev) for rt in runtimes]), timings

    report = {"layers": args.layers, "prompt_len": args.prompt_len, "graph": not args.no_graph,
              "checkpoint": str(checkpoint.root) if args.checkpoint else None,
              "index_heads": 16 if args.checkpoint else 64,
              "experts": args.experts, "expert_down": args.expert_down, "pdl": args.pdl, "fused_unproj": args.fused_unproj,
              "device_token": args.device_token}
    # Prompt (one token per step, eager for the first token so the graphs capture warmed-up code).
    t0 = time.time()
    if args.sequential_debug:
        logits = sequential_debug_step(prompt[0])
    else:
        # The warm-up step runs phase-lockstepped across ranks (an eager rank-sequential
        # issue order is not how production drives these kernels; see --sequential-debug).
        logits, timings = lockstep_debug_step(prompt[0])
        if args.debug_phases:
            report["first_step_phase_ms"] = timings
    if not args.no_graph:
        for rt in runtimes:
            rt.capture()
        if args.pdl:
            report["pdl"] = runtimes[0].pdl_report
            print(f"PDL: {report['pdl']['programmatic_edges']} programmatic edges over "
                  f"{report['pdl']['kernel_nodes']} kernel nodes", flush=True)
        print(f"captured {RANKS} graphs", flush=True)
    kernel_tokens, ref_tokens, checks = [], [], []
    reference_trace = [] if args.trace_reference else None
    ref_logits = reference.step(prompt[0], trace=reference_trace) if reference else None
    if args.trace_reference:
        matched_trace = []
        Reference(weights, args.layers, args.expert_down).step(
            prompt[0], trace=matched_trace,
            layer_inputs=[runtimes[0].resid[L][0] for L in range(args.layers)],
            attention_inputs=[t["attention_output"] for t in kernel_trace])

        def error(actual, expected):
            a, b = actual.float().flatten(), expected.to(actual.device).float().flatten()
            return dict(max_abs=float((a - b).abs().max()),
                        max_rel=float((a - b).abs().max() / b.abs().max().clamp_min(1e-9)),
                        rms_rel=float((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-9)))

        details = []
        for L, (kt, ref, matched) in enumerate(zip(kernel_trace, reference_trace, matched_trace)):
            actual = runtimes[0].resid[L + 1][0]
            entry = dict(layer=L, kind=weights[0].layer[L]["kind"],
                         cumulative_output=error(actual, ref["output"]),
                         matched_attention=error(kt["attention_output"], matched["attention_output"]),
                         matched_ffn=error(actual, matched["output"]),
                         rank_output_max_diff=max(float((actual.float() - rt.resid[L + 1][0].to(actual.device).float()).abs().max())
                                                  for rt in runtimes))
            if L:
                entry.update(router=error(kt["scores"], matched["scores"]),
                             kernel_indices=kt["indices"].flatten().tolist(),
                             matched_indices=matched["indices"].tolist(),
                             cumulative_indices=ref["indices"].tolist(),
                             kernel_probs=kt["probs"].flatten().tolist(),
                             matched_probs=matched["probs"].tolist())
            details.append(entry)
        report.update(trace=details, first_step=agreement(logits, ref_logits))
        if args.trace_tensors:
            def cpu_tree(value):
                if isinstance(value, torch.Tensor): return value.cpu()
                if isinstance(value, list): return [cpu_tree(v) for v in value]
                if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
                return value
            trace_path = Path(args.trace_tensors)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cpu_tree(dict(token=prompt[0], kernel=kernel_trace,
                residuals=runtimes[0].resid, reference=reference_trace, matched=matched_trace,
                logits=logits, reference_logits=ref_logits)), trace_path)
    if reference:
        checks.append({"step": 0, **agreement(logits, ref_logits)})
    for i, tok in enumerate(prompt[1:], start=1):
        logits = kernel_step(tok)
        if reference:
            ref_logits = reference.step(tok)   # the reference must see every prompt token
            if i < args.check_steps:
                checks.append({"step": i, **agreement(logits, ref_logits)})
    print(f"prompt of {args.prompt_len} tokens done in {time.time() - t0:.1f} s", flush=True)
    # Greedy decode: the kernel path's own tokens feed both paths (teacher forcing for the reference).
    token = int(logits.argmax())
    for i in range(args.check_steps):
        if not reference:
            break
        ref_logits = reference.step(token)
        logits = kernel_step(token)
        check = {"step": args.prompt_len + i, "token_in": token,
                 **agreement(logits, ref_logits)}
        if args.device_token:
            device_tokens = [int(rt.token_out[0].item()) for rt in runtimes]
            check["device_token"] = device_tokens[0]
            check["device_token_same"] = all(t == int(logits.argmax()) for t in device_tokens)
        checks.append(check)
        kernel_tokens.append(int(logits.argmax()))
        ref_tokens.append(int(ref_logits.argmax()))
        token = int(logits.argmax())
    report["checks"] = checks
    report["checked_steps_argmax_agree"] = f"{sum(c['argmax_same'] for c in checks)}/{len(checks)}"
    report["decided_steps_argmax_agree"] = (
        f"{sum(c['argmax_same'] for c in checks if c['decided'])}/{sum(c['decided'] for c in checks)}")
    report["checks_passed"] = all(c["finite"] and c["logit_max_abs_diff"] <= 0.01 * max(c["logit_scale"], 1e-6)
                                   and (not c["decided"] or c["argmax_same"]) for c in checks)
    if args.correctness_only:
        print(json.dumps(report, indent=1), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        return 0 if report["checks_passed"] else 1
    # Timing: graph replay in lockstep, and the full step including the argmax handoff.
    sync_all()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(runtimes[0].dev):
        start.record(runtimes[0].stream)
    for _ in range(args.steps):
        for rt in runtimes:
            rt.run_step()
    with torch.cuda.device(runtimes[0].dev):
        end.record(runtimes[0].stream)
    sync_all()
    report["replay_ms_per_step"] = round(start.elapsed_time(end) / args.steps, 4)
    t0 = time.time()
    if args.device_token:
        # One replay per token, no host involvement: the token travels head -> argmax -> embed on device.
        for _ in range(args.steps):
            for rt in runtimes:
                rt.run_step()
        sync_all()
        report["full_step_ms_with_argmax"] = round((time.time() - t0) / args.steps * 1e3, 3)
        report["final_device_tokens"] = [int(rt.token_out[0].item()) for rt in runtimes]
    else:
        for _ in range(args.steps):
            logits = kernel_step(token)
            token = int(logits.argmax())
        report["full_step_ms_with_argmax"] = round((time.time() - t0) / args.steps * 1e3, 3)
    report["final_pos"] = int(runtimes[0].pos.item())
    print(json.dumps(report, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
