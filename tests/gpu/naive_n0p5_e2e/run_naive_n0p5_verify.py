#!/usr/bin/env python3
"""Naive N verify-8: one target forward over eight query rows, end to end on 8 GPUs with
random weights, checked against eight single-token steps and benchmarked.

usage: run_naive_n0p5_verify.py [--layers 48] [--prompt-len 8] [--steps 32] [--seed 7] [--max-seq-len 4096]

A verify step feeds the eight draft tokens at positions p..p+7 as one batch: the attention
kernels run with eight query rows (attn_dsa / attn_swa built with NAIVERT_ATTN_QUERY_ROWS=8,
causal among the rows), the router, expert_select and expert_down as one eight-row launch each, and the layer-0 dense up-gate / down and the head too, so the step produces logits for all eight positions. Row q's argmax is compared with draft token
q+1 to count the accepted prefix; the last row yields the bonus token.

Correctness: the prompt is decoded token by token (the seq1 path), eight more greedy
seq1 steps record the per-position logits, then the positions are rewound and the same
eight tokens go through one verify step; its per-row logits must match the seq1 logits.
Speed: the verify step is captured per rank into a CUDA graph and replayed in lockstep.
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode = load_module(ROOT / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_decode.py")
RANKS, HIDDEN, HEADS, VDIM, K_UNPROJ = decode.RANKS, decode.HIDDEN, decode.HEADS, decode.VDIM, decode.K_UNPROJ
EXPERTS, TOPK, DENSE_INTER, VOCAB_SHARD, VOCAB = decode.EXPERTS, decode.TOPK, decode.DENSE_INTER, decode.VOCAB_SHARD, decode.VOCAB
LAYER_KINDS = decode.LAYER_KINDS
VROWS = 8
SLAB = 0x70000
MAILBOX_BYTES = VROWS * 4 * SLAB   # down rows8 needs active_rows x 4 slabs


def mailbox_bytes(ops, rows=None):
    """Bytes of one rank's mailbox for the verify runtimes: the attention-tail / dense-down region (MAILBOX_BYTES)
    followed by one expert_down region per query row (the rows are sliced from MAILBOX_BYTES on, so the two
    never alias). Every allocation that feeds VerifyRuntime must use this, not max() of the two."""
    return MAILBOX_BYTES + (VROWS if rows is None else rows) * int(ops.expert_down_mailbox_row_bytes())
UNIT = 8   # rows per kernel launch: the rows8 kernels; --rows 16 launches every one of them twice


def configure_rows(rows: int) -> None:
    """verify-N: N = 8 (the rows8 kernels) or 16 (two eight-row units per kernel)."""
    global VROWS, MAILBOX_BYTES
    assert rows in (8, 16), rows
    VROWS = rows
    MAILBOX_BYTES = VROWS * 4 * SLAB


class VerifyRuntime:
    """Eight-row verify step for one rank, sharing the weights and caches of the decode runtime."""

    def __init__(self, rt: "decode.RankRuntime", ext_dsa8, ext_swa8, ws_dsa8, ws_swa8, pdl=False, fused_unproj=False,
                 down="rows8", select="rows8", router="rows8", dense="rows8", head="rows8", prepared_gate_up=None):
        self.rt, self.w, self.ops, self.rank, self.dev = rt, rt.w, rt.ops, rt.rank, rt.dev
        self.stream = rt.stream
        self.pdl, self.pdl_report = pdl, None
        self.fused_unproj = fused_unproj
        self.down = down   # "rows8": one eight-row expert_down launch per layer
        self.select = select   # "rows8": one expert-major eight-row expert_select launch per layer
        self.router = router   # "rows8": one eight-row router launch per layer
        self.dense = dense   # "rows8": one eight-row up-gate and down launch for layer 0
        self.head = head   # "rows8": one eight-row head launch
        self.ext_dsa8, self.ext_swa8 = ext_dsa8, ext_swa8
        dev, layers = rt.dev, rt.w.layers
        self.units = VROWS // UNIT
        prepared_supported = (
            self.units == 1 and select == "rows8" and router == "rows8"
            and rt.moe_up.dtype == torch.float32)
        self.prepared_gate_up = prepared_supported if prepared_gate_up is None else bool(prepared_gate_up)
        if self.prepared_gate_up:
            assert prepared_supported, "prepared gate/up requires W8A8 verify-8 and rows8 router/select"
        self.prepared_mlp = (torch.empty(self.ops.prepared_mlp_input_bytes(), dtype=torch.uint8, device=dev)
                             if self.prepared_gate_up else None)
        if any(mode != "rows8" for mode in (down, select, router, dense, head)):
            raise ValueError("verify requires rows8 kernels")
        # Position of unit u's row 0 (unit 0 reads rt.pos itself; the others are set in-graph each step).
        self.pos_units = [rt.pos] + [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(self.units - 1)]
        self.tokens = torch.zeros(VROWS, dtype=torch.long, device=dev)
        self.resid = [torch.zeros(VROWS, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(layers + 1)]
        self.hid_a = torch.zeros(VROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.proj_o = torch.zeros(VROWS, HEADS, VDIM, dtype=torch.bfloat16, device=dev)
        self.scores = torch.zeros(VROWS, EXPERTS, dtype=torch.float32, device=dev)
        self.x_mlp = torch.zeros(VROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.moe_up = torch.zeros(VROWS, TOPK, 16, 16, dtype=rt.moe_up.dtype, device=dev)
        self.probs = torch.zeros(VROWS, TOPK, dtype=torch.float32, device=dev)
        self.indices = torch.zeros(VROWS, TOPK, dtype=torch.int32, device=dev)
        self.dense_up = torch.zeros(VROWS, DENSE_INTER, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(VROWS, VOCAB_SHARD, dtype=torch.float32, device=dev)
        self.last_hidden = torch.zeros(VROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.head_ws = torch.zeros(VROWS * 64, dtype=torch.uint8, device=dev)
        self.attn_ws = {"full_dsa": [torch.zeros(ws_dsa8, dtype=torch.uint8, device=dev) for _ in range(layers)],
                        "sliding_window": [torch.zeros(ws_swa8, dtype=torch.uint8, device=dev) for _ in range(layers)]}
        if hasattr(ext_dsa8, "prepare_maps"):
            with torch.cuda.device(dev):
                for L, lw in enumerate(self.w.layer):
                    if lw["kind"] == "full_dsa":
                        ext_dsa8.prepare_maps(self.attn_ws["full_dsa"][L], lw["wq"], lw["wk"], lw["wv"], lw["wik"])
        if hasattr(ext_swa8, "prepare_maps"):
            with torch.cuda.device(dev):
                for L, lw in enumerate(self.w.layer):
                    if lw["kind"] != "full_dsa":
                        ext_swa8.prepare_maps(self.attn_ws["sliding_window"][L], lw["wq"], lw["wk"], lw["wv"])
        domain_size = int(self.ops.expert_down_domain_size())
        self.domains = {}
        down_slab = int(self.ops.expert_down_mailbox_row_bytes())
        with torch.cuda.device(dev):
            for L in range(1, layers):
                lw = self.w.layer[L]
                dom = torch.zeros(VROWS * domain_size, dtype=torch.uint8, device=dev)
                if self.down == "rows8":
                    # One eight-row launch per unit: descriptor 48+L (the single-token path owns L) for
                    # unit 0, 96+L for unit 1; one mailbox slab per row (global row index), one tag per unit;
                    # row 0 of a unit prepared last (the stable table keeps row 0).
                    for g in reversed(range(VROWS)):
                        u = g // UNIT
                        # The expert_down region starts after the tail / dense slabs (no aliasing).
                        sl = [mb[MAILBOX_BYTES + g * down_slab:MAILBOX_BYTES + (g + 1) * down_slab] for mb in rt.mailboxes]
                        self.ops.prepare_expert_down_replay_domain_epoch(
                            dom[g * domain_size:(g + 1) * domain_size], self.moe_up[g], lw["expert_down"],
                            lw["expert_down_scale"], self.probs[g], self.indices[g], self.resid[L + 1][g],
                            self.hid_a[g], sl, 2 * L + 3 + 128 * u, 48 * (u + 1) + L, self.rank, rt.epoch)
                self.domains[L] = dom
        self.domain_size = domain_size
        self.graph = None

    def context_feature_for_rank(self):
        """Residual after this rank's selected target layer; no copy or launch."""
        from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS
        return self.resid[TARGET_LAYER_IDS[self.rank] + 1]

    def phases(self, *, head_inbox_ptrs=None, advance=True):
        rt, w, ops, r = self.rt, self.w, self.ops, self.rank
        head_inbox_ptrs = rt.inbox_ptrs if head_inbox_ptrs is None else head_inbox_ptrs
        # PDL consumers wait for their primary grid before reading activations.
        unproj_launch = ops.launch_unproj_allreduce_replay_pdl_consumer if self.pdl else ops.launch_unproj_allreduce_replay
        select8_launch = (ops.launch_expert_select_up_gate_silu_rows8_pdl_producer if self.pdl
                          else ops.launch_expert_select_up_gate_silu_rows8)
        if self.prepared_gate_up:
            prepared_select = (ops.launch_expert_select_up_gate_silu_rows8_prepared_pdl_producer if self.pdl
                               else ops.launch_expert_select_up_gate_silu_rows8_prepared)
            select8_launch = lambda *a: prepared_select(*a, self.prepared_mlp)
        down_launch = (ops.launch_expert_down_rows_from_replay_domain_pdl_consumer if self.pdl
                       else ops.launch_expert_down_rows_from_replay_domain)
        # One 8-CTA gather kernel instead of torch.index_select (6.6 us -> ~2.5 us in the round graph; same bytes).
        out = [("embed8", lambda: ops.gather_rows_bf16(w.embedding, self.tokens, self.resid[0]))]
        units = list(range(self.units))
        unit_slices = [slice(u * UNIT, (u + 1) * UNIT) for u in units]
        for u in units[1:]:
            # verify-16: unit u's rows start at pos + 8 u (an in-graph node; unit 0 reads rt.pos).
            out.append((f"pos_unit{u}", lambda u=u: torch.add(rt.pos, u * UNIT, out=self.pos_units[u])))
        for L in range(w.layers):
            lw = w.layer[L]
            for u in units:
                su = unit_slices[u]
                # With fused_unproj the attention kernel also computes hid_a = resid + allreduce(o_proj)
                # for its eight rows (one mailbox slab per row, one tag per unit).
                tail = (lw["o_packed"], lw["o_scale"], self.hid_a[su], rt.mailboxes, rt.epoch, r, 2 * L + 2 + 128 * u)
                unit_tag = f"{L}" if self.units == 1 else f"{L}.u{u}"
                if lw["kind"] == "full_dsa":
                    dsa_args = lambda lw=lw, L=L, su=su, u=u: (  # noqa: E731
                        self.resid[L][su], lw["attn_gamma"], lw["wq"], lw["wk"], lw["wv"], lw["wiq"], lw["wiq_scale"], lw["wik"],
                        lw["ik_w"], lw["ik_b"], lw["wis"], w.rope_full, lw["k_cache"], lw["v_cache"],
                        lw["ki_cache"], lw["ki_scale"], self.proj_o[su], self.attn_ws["full_dsa"][L], self.pos_units[u],
                        rt.max_seq_len, 0)
                    if self.fused_unproj:
                        out.append((f"L{unit_tag}.attn_dsa8+unproj", lambda a=dsa_args, t=tail, vs=lw.get("value_scale", 1.0): self.ext_dsa8.run_fused_scaled(*a(), *t, vs)))
                    else:
                        out.append((f"L{unit_tag}.attn_dsa8", lambda a=dsa_args, vs=lw.get("value_scale", 1.0): self.ext_dsa8.run_scaled(*a(), vs)))
                else:
                    swa_args = lambda lw=lw, L=L, su=su, u=u: (  # noqa: E731
                        self.resid[L][su], lw["attn_gamma"], lw["wq"], lw["wk"], lw["wv"], w.rope_swa, lw["sink"],
                        lw["k_cache"], lw["v_cache"], self.proj_o[su], self.attn_ws["sliding_window"][L], self.pos_units[u],
                        rt.max_seq_len, 0)
                    if self.fused_unproj:
                        out.append((f"L{unit_tag}.attn_swa8+unproj", lambda a=swa_args, t=tail, vs=lw.get("value_scale", 1.0): self.ext_swa8.run_fused_scaled(*a(), *t, vs)))
                    else:
                        out.append((f"L{unit_tag}.attn_swa8", lambda a=swa_args, vs=lw.get("value_scale", 1.0): self.ext_swa8.run_scaled(*a(), vs)))
            if not self.fused_unproj:
                # The unfused reference path launches one projection per row. Each row
                # owns a mailbox slice, so the layer tag can be reused without aliasing.
                for row in range(VROWS):
                    sr = slice(row, row + 1)
                    mb = [mailbox[row * SLAB:(row + 1) * SLAB] for mailbox in rt.mailboxes]
                    if lw["o_packed"].dtype == torch.bfloat16:
                        out.append((f"L{L}.unproj_bf16.r{row}", lambda lw=lw, L=L, sr=sr, mb=mb: ops.launch_checkpoint_bf16_linear(
                            self.proj_o[sr].reshape(1, K_UNPROJ), lw["o_packed"], self.resid[L][sr], self.hid_a[sr],
                            mb, rt.epoch, r, 2 * L + 2)))
                    else:
                        out.append((f"L{L}.unproj1.r{row}", lambda lw=lw, L=L, sr=sr, mb=mb: unproj_launch(
                            self.proj_o[sr].reshape(1, K_UNPROJ), lw["o_packed"], lw["o_scale"], self.resid[L][sr],
                            self.hid_a[sr], mb, r, 2 * L + 2, rt.epoch.view(torch.uint8))))
            if L == 0 and "dense_gate" in lw:
                slices = unit_slices
                for u, su in enumerate(slices):
                    out.append((f"L{L}.up_gate_bf16.u{u}", lambda lw=lw, su=su: ops.launch_checkpoint_bf16_swiglu(
                        self.hid_a[su], lw["ffn_gamma"], lw["dense_gate"], lw["dense_up"], self.dense_up[su])))
                    out.append((f"L{L}.down_bf16.u{u}", lambda lw=lw, L=L, su=su, u=u: ops.launch_checkpoint_bf16_linear(
                        self.dense_up[su], lw["dense_down"], self.hid_a[su], self.resid[L+1][su],
                        rt.mailboxes, rt.epoch, r, 3 + 128*u)))
            elif L == 0:
                if self.dense == "rows8":
                    # One eight-row up-gate and one eight-row down per unit (four mailbox slabs per row, tag 3 + 128 u).
                    up8_launch = ops.launch_rmsnorm_up_gate_pdl_consumer if self.pdl else ops.launch_rmsnorm_up_gate
                    down8_launch = ops.launch_down_allreduce_pdl_consumer if self.pdl else ops.launch_down_allreduce
                    for u in units:
                        su = unit_slices[u]
                        out.append((f"L{L}.up_gate8.u{u}", lambda lw=lw, f=up8_launch, su=su: f(
                            self.hid_a[su], lw["ffn_gamma"], rt.dummy8, rt.dummy16, rt.dummy8, rt.dummy16,
                            lw["dense_ws"], self.dense_up[su], HIDDEN, DENSE_INTER)))
                        out.append((f"L{L}.down8.u{u}", lambda lw=lw, L=L, f=down8_launch, su=su, u=u: f(
                            self.dense_up[su], lw["dense_down"], lw["dense_down_scale"], self.hid_a[su],
                            self.resid[L + 1][su], rt.step_block, rt.generation, 3 + 128 * u, rt.mailboxes, r,
                            HIDDEN, DENSE_INTER)))
            else:
                if self.router == "rows8":
                    router8_launch = (ops.launch_rmsnorm_expert_proj_pdl_producer if self.pdl
                                      else ops.launch_rmsnorm_expert_proj)   # Waits + triggers early
                    router_weight = lw["router"]
                    if self.prepared_gate_up:
                        packet_only = ("launch_rmsnorm_expert_proj_prepared_only_pdl_producer" if self.pdl
                                       else "launch_rmsnorm_expert_proj_prepared_only")
                        packed_only = packet_only.replace("_prepared", "_prepared_packed")
                        if "router_prepared" in lw:
                            router_weight = lw["router_prepared"]
                            packet_only = packed_only
                        prepared_router = getattr(ops, packet_only)
                        router8_launch = lambda *a, f=prepared_router: f(*a[:4], *a[5:], self.prepared_mlp)
                    for u in units:
                        su = unit_slices[u]
                        out.append((f"L{L}.router8.u{u}", lambda lw=lw, f=router8_launch, rw=router_weight, su=su: f(
                            self.hid_a[su], lw["ffn_gamma"], rw, self.scores[su], self.x_mlp[su], HIDDEN, EXPERTS)))
                if self.units > 1:
                    # verify-16: the eight-row expert_select and expert_down per unit (descriptor 48 (u + 1) + L).
                    for u in units:
                        su = unit_slices[u]
                        out.append((f"L{L}.expert_select8.u{u}", lambda lw=lw, su=su: select8_launch(
                            self.x_mlp[su], self.scores[su], lw["router_bias"], lw["expert_table"], self.moe_up[su],
                            self.probs[su], self.indices[su])))
                        out.append((f"L{L}.expert_down8.u{u}", lambda L=L, u=u: down_launch(
                            self.domains[L][u * UNIT * self.domain_size:(u + 1) * UNIT * self.domain_size],
                            48 * (u + 1) + L, r, UNIT)))
                if self.units == 1:
                    out.append((f"L{L}.expert_select8", lambda lw=lw: select8_launch(
                        self.x_mlp, self.scores, lw["router_bias"], lw["expert_table"], self.moe_up,
                        self.probs, self.indices)))
                    out.append((f"L{L}.expert_down8", lambda L=L: down_launch(self.domains[L], 48 + L, r, VROWS)))
        if self.head == "rows8":
            # One eight-row head launch per unit (48 KiB panes); under --pdl it waits for the last expert_down.
            head8_launch = (ops.launch_rmsnorm_head_proj_rows1_pdl_consumer if self.pdl
                            else ops.launch_rmsnorm_head_proj_rows1)
            for u in units:
                su = unit_slices[u]
                out.append((f"head8.u{u}", lambda f=head8_launch, su=su, u=u: f(
                    self.resid[w.layers][su], w.head_gamma, w.lm_head, self.logits[su], self.last_hidden[su],
                    self.head_ws[u * UNIT * 64:(u + 1) * UNIT * 64], HIDDEN, VOCAB_SHARD, head_inbox_ptrs, r)))

        def advance_state():
            rt.pos += VROWS
            rt.generation += 1
            rt.epoch += 1
        if advance:
            out.append(("advance8", advance_state))
        return out

    def capture(self):
        with torch.cuda.device(self.dev):
            self.graph = torch.cuda.CUDAGraph(keep_graph=self.pdl)
            with torch.cuda.graph(self.graph, stream=self.rt.stream):
                for _, fn in self.phases():
                    fn()
            if self.pdl:
                self.pdl_report = decode.install_pdl_edges(self.graph, getattr(self.rt, 'pdl_markers', None))
                self.graph.instantiate()

    def replay(self):
        with torch.cuda.device(self.dev), torch.cuda.stream(self.rt.stream):
            if self.graph is None:
                for _, fn in self.phases():
                    fn()
            else:
                self.graph.replay()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=len(LAYER_KINDS))
    ap.add_argument("--checkpoint", help="target FP8 checkpoint directory (or its parent)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-graph", action="store_true", help="run correctness without graph capture")
    ap.add_argument("--correctness-only", action="store_true", help="compare verify with sequential decode and skip timing")
    ap.add_argument("--trace-tensors", help="save sequential/verify intermediate tensors for diagnosis")
    ap.add_argument("--trace-row", type=int, default=0, choices=range(-1,8), help="which verification row to trace; -1 records all eight")
    ap.add_argument("--rows", type=int, choices=(8, 16), default=8,
                    help="verify rows per step: 8 (the rows8 kernels) or 16 (every rows8 kernel twice, per eight-row unit)")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--random-fill-pos", type=int, default=0,
                    help="skip prefill: fill caches with random bf16 and time verify-8/seq1 at this position")
    ap.add_argument("--steps", type=int, default=32, help="verify steps to time")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", default="")
    ap.add_argument("--timing-tokens", choices=("random", "continuation"), default="random",
                    help="tokens carried by the timed verify replays (the position advances 8 per replay either way): "
                         "eight seeded random tokens (default: diverse routing, comparable across prompt lengths) or "
                         "the model's greedy continuation (with random weights it degenerates to repeated tokens at "
                         "short prompts and under-reports the MoE cost by ~0.2 ms)")
    ap.add_argument("--experts", choices=("fp8", "mx"), default="fp8",
                    help="routed-expert weight format: fp8 block-128 or MX8 block-32 (separate model_ops build)")
    ap.add_argument("--pdl-markers", default=None,
                    help="Comma list of kernel-name markers that receive programmatic edges (default: the attention+expert_down set for --expert-down w8a8, the full set otherwise)")
    ap.add_argument("--pdl", action="store_true",
                    help="programmatic dependent launch in both graphs: PDL kernel variants + programmatic edges")
    ap.add_argument("--fused-unproj", action=argparse.BooleanOptionalAction, default=True,
                    help="o-projection + TP8 all-reduce inside the attention kernels, both graphs; "
                         "default on (--no-fused-unproj for per-row unproj launches)")
    ap.add_argument("--router", choices=("rows8",), default="rows8",
                    help="router as one eight-row launch per layer")
    ap.add_argument("--select", choices=("rows8",), default="rows8",
                    help="expert-major eight-row selector per unit")
    ap.add_argument("--down", choices=("rows8",), default="rows8",
                    help="expert_down as one eight-row launch per layer")
    ap.add_argument("--expert-down", choices=("bf16", "w8a8"), default="bf16",
                    help="W8A8 expert-down in both sequential decode and verification")
    ap.add_argument("--dense", choices=("rows8",), default="rows8",
                    help="layer-0 dense up-gate / down as one eight-row launch each")
    ap.add_argument("--head", choices=("rows8",), default="rows8",
                    help="LM head as one eight-row launch")
    args = ap.parse_args()
    if args.expert_down == "w8a8" and (args.experts != "fp8" or args.rows != 8 or args.down != "rows8"):
        ap.error("W8A8 down currently requires fp8 block128 weights, rows=8 and down=rows8")
    if args.random_fill_pos and (args.no_graph or args.correctness_only):
        ap.error("random-fill-pos is a separate graph timing mode")
    if args.checkpoint and args.experts != "fp8":
        ap.error("the FP8 checkpoint requires fp8 experts")
    if args.build_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure_rows(args.rows)
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    dsa_mod = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa.py")
    dsa_rows = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa_rows.py")
    swa_mod = load_module(ROOT / "tests/gpu/attn_swa/run_attn_swa.py")
    t0 = time.time()
    ops = load_extension(weights=args.experts, expert_down=args.expert_down)
    print(f"experts: {args.experts}", flush=True)
    index_heads = decode.IHEADS
    ext_dsa, ext_swa = dsa_mod.build(False, index_heads), swa_mod.build(1)
    ext_dsa8, ext_swa8 = dsa_rows.build(8, index_heads=index_heads), swa_mod.build(8)
    if args.build_only:
        return 0
    decode.enable_torch_tp8_peers()
    print(f"extensions ready in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    if args.checkpoint:
        from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, load_tp8_weights
        checkpoint = NaiveNCheckpoint(args.checkpoint)
        weights = load_tp8_weights(checkpoint, args.layers, args.max_seq_len,
                                   ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(), expert_down=args.expert_down)
    else:
        weights = [decode.RankWeights(r, args.layers, args.seed, args.max_seq_len, ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(),
                           experts=args.experts, expert_down=args.expert_down)
               for r in range(RANKS)]
    print(f"weights ready in {time.time() - t0:.0f} s ({torch.cuda.memory_allocated(0) / 2**30:.1f} GiB on rank 0)", flush=True)
    mailboxes = [torch.zeros(mailbox_bytes(ops), dtype=torch.uint8, device=w.dev) for w in weights]   # Disjoint expert_down region
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=w.dev) for w in weights]
    pdl_markers = (tuple(args.pdl_markers.split(',')) if args.pdl_markers else (decode.PDL_W8A8_MARKERS if args.expert_down == 'w8a8' else None))
    runtimes = [decode.RankRuntime(r, weights[r], ops, ext_dsa, ext_swa, args.max_seq_len, mailboxes, inboxes, pdl=args.pdl,
                                   fused_unproj=args.fused_unproj)
                for r in range(RANKS)]
    for rt in runtimes: rt.pdl_markers = pdl_markers
    verifies = [VerifyRuntime(rt, ext_dsa8, ext_swa8, ext_dsa8.workspace_bytes(), ext_swa8.workspace_bytes(), pdl=args.pdl,
                              fused_unproj=args.fused_unproj, down=args.down, select=args.select, router=args.router,
                              dense=args.dense, head=args.head)
                for rt in runtimes]
    gen = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(0, VOCAB, (args.prompt_len,), generator=gen).tolist()

    def sync_all():
        for rt in runtimes:
            torch.cuda.synchronize(rt.dev)

    def set_tokens(rt_list, attr, values):
        for rt in rt_list:
            with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                getattr(rt, attr).copy_(torch.as_tensor(values, dtype=torch.long, device=rt.dev))

    row_traces = {"sequential": {}, "verify": {}}

    def snapshot_row(name, rt_list, target, row):
        if name.startswith("head"):
            layer, phase, L = "head", "head", -1
        elif name.startswith("L"):
            layer, phase = name.split(".", 1)
            L = int(layer[1:])
        else:
            return
        fields = {}
        if phase == "head":
            fields = {"normalized": "last_hidden", "logits": "logits"}
        elif "attn_" in phase:
            fields["proj_o"] = "proj_o"
            if "+unproj" in phase:
                fields["attention"] = "hid_a"
            if "attn_swa" in phase and L in (20,21):
                for part in ("fused", "q_rot"):
                    values=[]
                    for rt in rt_list:
                        multi=isinstance(rt, VerifyRuntime)
                        ext=rt.ext_swa8 if multi else rt.ext_swa
                        ws=rt.attn_ws["sliding_window"][L] if multi else rt.w.layer[L]["attn_ws"]
                        offset,size=ext.layout()[part]
                        values.append(ws[offset:offset+size].view(torch.bfloat16).reshape(8 if multi else 1,-1)[row if multi else 0].cpu())
                    target[f"row{row}.{layer}.{part}"] = torch.stack(values)
        elif phase.startswith("unproj"):
            fields["attention"] = "hid_a"
        elif phase.startswith("router"):
            fields = {"normalized": "x_mlp", "scores": "scores"}
        elif phase.startswith("expert_select"):
            fields = {"moe_up": "moe_up", "probabilities": "probs", "indices": "indices"}
        elif phase.startswith("up_gate"):
            fields["dense_up"] = "dense_up"
        elif phase.startswith(("expert_down", "down")):
            target[f"row{row}.{layer}.residual"] = torch.stack([rt.resid[L+1][row if isinstance(rt, VerifyRuntime) else 0].cpu() for rt in rt_list])
        for key, attr in fields.items():
            target[f"row{row}.{layer}.{key}"] = torch.stack([getattr(rt, attr)[row if isinstance(rt, VerifyRuntime) else 0].cpu() for rt in rt_list])

    def lockstep(phase_lists, trace=None, rt_list=None, trace_index=None):
        for k in range(len(phase_lists[0])):
            for rt, phases in zip(runtimes, phase_lists):
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    phases[k][1]()
            sync_all()
            if trace is not None:
                rows = [trace_index] if trace_index is not None else (range(VROWS) if args.trace_row == -1 else [args.trace_row])
                for row in rows:
                    snapshot_row(phase_lists[0][k][0], rt_list, trace, row)

    def gather_logits(tensors):
        return torch.cat([t.to(runtimes[0].dev) for t in tensors], dim=-1)

    if args.random_fill_pos > 0:
        P = args.random_fill_pos
        for w in weights:
            with torch.cuda.device(w.dev):
                for L in range(w.layers):
                    lw = w.layer[L]
                    if lw["kind"] == "full_dsa":
                        lw["k_cache"][:P].normal_(); lw["v_cache"][:P].normal_()
                        rows = torch.randn(P, lw["ki_cache"].shape[1], device=w.dev)
                        scale = rows.abs().amax(dim=1).clamp_min(1e-30) / 448.0
                        lw["ki_cache"][:P] = (rows / scale[:, None]).to(torch.float8_e4m3fn)
                        lw["ki_scale"][:P] = scale
                    else:
                        lw["k_cache"].normal_(); lw["v_cache"].normal_()
        # Eight different tokens in the verify rows (the long-sequence runs found that eight copies of
        # token 1 route to ~8 distinct experts per layer, a degenerate regime that favours fine-grained
        # item lists); --timing-tokens continuation keeps the old fixed token for comparability.
        if args.timing_tokens == "random":
            tgen = torch.Generator().manual_seed(args.seed + 1000)
            fill_tokens = torch.randint(0, VOCAB, (VROWS,), generator=tgen).tolist()
        else:
            fill_tokens = [1] * VROWS
        set_tokens(runtimes, "token", fill_tokens[:1])
        set_tokens(verifies, "tokens", fill_tokens)
        def set_pos(v):
            for rt in runtimes:
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    rt.pos.fill_(v)
        set_pos(P); lockstep([rt.phases() for rt in runtimes])
        set_pos(P);
        for rt in runtimes: rt.capture()
        set_pos(P); lockstep([v.phases() for v in verifies])
        set_pos(P);
        for v in verifies: v.capture()
        sync_all()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        set_pos(P)
        with torch.cuda.device(runtimes[0].dev): start.record(runtimes[0].stream)
        for _ in range(args.steps):
            set_pos(P)
            for v in verifies: v.replay()
        with torch.cuda.device(runtimes[0].dev): end.record(runtimes[0].stream)
        sync_all()
        verify8 = round(start.elapsed_time(end) / args.steps, 4)
        set_pos(P)
        with torch.cuda.device(runtimes[0].dev): start.record(runtimes[0].stream)
        for _ in range(args.steps):
            set_pos(P)
            for rt in runtimes: rt.run_step()
        with torch.cuda.device(runtimes[0].dev): end.record(runtimes[0].stream)
        sync_all()
        seq1 = round(start.elapsed_time(end) / args.steps, 4)
        print(json.dumps({"verify_rows": VROWS, "random_fill_pos": P, "max_seq_len": args.max_seq_len, "timing_tokens": args.timing_tokens,
                          "expert_down": args.expert_down, "verify8_replay_ms_per_step": verify8, "seq1_replay_ms_per_step": seq1,
                          "verify8_over_seq1": round(verify8 / seq1, 3)}), flush=True)
        return 0
    # ---- prompt through the seq1 path (lockstep warm-up, then graphs) ----
    set_tokens(runtimes, "token", [prompt[0]])
    lockstep([rt.phases() for rt in runtimes])
    for rt in runtimes:
        if not args.no_graph:
            rt.capture()
    for tok in prompt[1:]:
        set_tokens(runtimes, "token", [tok])
        for rt in runtimes:
            rt.run_step()
        sync_all()
    logits = gather_logits([rt.logits[0] for rt in runtimes])
    base_pos = int(runtimes[0].pos.item())
    if args.trace_tensors:
        row_traces["prefix"] = {L: [{key: rt.w.layer[L][key].cpu() for key in ("k_cache", "v_cache")}
                                    for rt in runtimes] for L in (20,21) if L < args.layers}
    # ---- eight greedy seq1 steps: reference logits for positions base..base+7 ----
    seq_tokens, seq_logits = [int(logits.argmax())], []
    for q in range(VROWS):
        set_tokens(runtimes, "token", [seq_tokens[-1]])
        if args.trace_tensors and (args.trace_row == -1 or q == args.trace_row):
            lockstep([rt.phases() for rt in runtimes], row_traces["sequential"], runtimes, q)
        else:
            for rt in runtimes:
                rt.run_step()
        sync_all()
        lg = gather_logits([rt.logits[0] for rt in runtimes])
        seq_logits.append(lg.clone())
        seq_tokens.append(int(lg.argmax()))
    # ---- rewind and run the same eight tokens as one verify step ----
    for rt in runtimes:
        with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
            rt.pos.fill_(base_pos)
    set_tokens(verifies, "tokens", seq_tokens[:VROWS])
    lockstep([v.phases() for v in verifies], row_traces["verify"] if args.trace_tensors else None, verifies)
    vlogits = gather_logits([v.logits for v in verifies])            # [8, vocab]
    graph_matches_eager = None
    if not args.no_graph:
        eager_logits = vlogits.clone()
        for rt in runtimes:
            with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                rt.pos.fill_(base_pos)
        for v in verifies:
            v.capture()
        for v in verifies:
            v.replay()
        sync_all()
        vlogits = gather_logits([v.logits for v in verifies])
        graph_matches_eager = bool(torch.equal(vlogits, eager_logits))
    report = {"layers": args.layers, "verify_rows": VROWS, "base_pos": base_pos, "experts": args.experts, "pdl": args.pdl,
              "checkpoint": str(checkpoint.root) if args.checkpoint else None, "index_heads": index_heads,
              "graph": not args.no_graph, "graph_matches_eager": graph_matches_eager,
              "fused_unproj": args.fused_unproj, "down": args.down, "expert_down": args.expert_down, "rows": []}
    if args.trace_tensors:
        trace_path = Path(args.trace_tensors)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(row_traces, trace_path)
        differences = {}
        for name, a in row_traces["sequential"].items():
            b = row_traces["verify"][name]
            delta = b.double() - a.double()
            differences[name] = {"exact": torch.equal(a, b), "different_elements": int((a != b).sum()),
                "max_abs": float(delta.abs().max()),
                "relative_l2": float(delta.norm()/a.double().norm().clamp_min(1e-30))}
        report["trace_row"] = args.trace_row
        report["row_trace"] = differences
    for q in range(VROWS):
        diff = float((vlogits[q] - seq_logits[q]).abs().max())
        top2 = seq_logits[q].topk(2).values
        margin = float(top2[0] - top2[1])
        # Report raw argmax agreement as well as whether the sequential margin
        # exceeds twice the observed logit error. Both heads accumulate in FP32.
        scale = float(seq_logits[q].abs().max())
        report["rows"].append({"row": q, "logit_max_abs_diff": diff,
                               "logit_normalized_max_diff": diff / max(scale, 1e-30),
                               "logits_exact": bool(torch.equal(vlogits[q], seq_logits[q])),
                               "finite": bool(torch.isfinite(vlogits[q]).all() and torch.isfinite(seq_logits[q]).all()),
                               "logit_scale": scale,
                               "top2_margin": round(margin, 4), "decided": margin > 2 * diff,
                               "argmax_same": int(vlogits[q].argmax()) == int(seq_logits[q].argmax())})
    report["rows_argmax_agree"] = f"{sum(r['argmax_same'] for r in report['rows'])}/{VROWS}"
    decided = [r for r in report["rows"] if r["decided"]]
    report["decided_rows_argmax_agree"] = f"{sum(r['argmax_same'] for r in decided)}/{len(decided)}"
    # Acceptance against the draft (the seq1 continuation): row q predicts draft q+1.
    accepted = 0
    for q in range(VROWS - 1):
        if int(vlogits[q].argmax()) == seq_tokens[q + 1]:
            accepted += 1
        else:
            break
    report[f"accepted_prefix_of_{VROWS - 1}"] = accepted
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    for row in report["rows"]:
        print("  ", row, flush=True)
    report["checks_passed"] = all(r["finite"] and r["logit_max_abs_diff"] <= 0.01 * max(r["logit_scale"], 1e-6)
                                   and (not r["decided"] or r["argmax_same"]) for r in report["rows"])
    report["checks_passed"] = report["checks_passed"] and graph_matches_eager is not False
    if args.correctness_only:
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        return 0 if report["checks_passed"] else 1
    # ---- speed: verify-8 graph replay vs seq1 graph replay ----
    for v in verifies:
        if not args.no_graph and v.graph is None:
            v.capture()
    if args.timing_tokens == "random":
        tgen = torch.Generator().manual_seed(args.seed + 1000)
        set_tokens(verifies, "tokens", torch.randint(0, VOCAB, (VROWS,), generator=tgen).tolist())
        sync_all()
    if args.pdl:
        report["pdl_verify"] = verifies[0].pdl_report
        report["pdl_seq1"] = runtimes[0].pdl_report
        print(f"PDL: verify {report['pdl_verify']['programmatic_edges']} edges / {report['pdl_verify']['kernel_nodes']} nodes, "
              f"seq1 {report['pdl_seq1']['programmatic_edges']} / {report['pdl_seq1']['kernel_nodes']}", flush=True)
    sync_all()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # NVTX ranges delimit the two replay windows for an external profiler.
    torch.cuda.nvtx.range_push("verify8_replay")
    with torch.cuda.device(runtimes[0].dev):
        start.record(runtimes[0].stream)
    for _ in range(args.steps):
        for v in verifies:
            v.replay()
    with torch.cuda.device(runtimes[0].dev):
        end.record(runtimes[0].stream)
    sync_all()
    torch.cuda.nvtx.range_pop()
    report["verify8_replay_ms_per_step"] = round(start.elapsed_time(end) / args.steps, 4)
    torch.cuda.nvtx.range_push("seq1_replay")
    with torch.cuda.device(runtimes[0].dev):
        start.record(runtimes[0].stream)
    for _ in range(args.steps):
        for rt in runtimes:
            rt.run_step()
    with torch.cuda.device(runtimes[0].dev):
        end.record(runtimes[0].stream)
    sync_all()
    torch.cuda.nvtx.range_pop()
    report["seq1_replay_ms_per_step"] = round(start.elapsed_time(end) / args.steps, 4)
    report["verify8_ms_per_token_if_all_accepted"] = round(report["verify8_replay_ms_per_step"] / VROWS, 4)
    report["verify8_over_seq1"] = round(report["verify8_replay_ms_per_step"] / report["seq1_replay_ms_per_step"], 3)
    report["final_pos"] = int(runtimes[0].pos.item())
    report["timing_tokens"] = args.timing_tokens
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
