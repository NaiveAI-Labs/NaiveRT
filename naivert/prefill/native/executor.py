"""Native prefill over fixed 4,096-row chunks on eight GPU streams.

Attention uses cuBLAS projections, fused Q/K RoPE, FA4 sliding-window attention,
and the DSA indexer, stable top-k selection, and sparse attention bindings in
``naivert.kernels.naive_n0p5.prefill``. Index-key normalization defaults to
PyTorch LayerNorm; the decode-compatible spelling is selectable. DSA selection
covers the visible causal prefix, with query-row sharding beyond 65,536 keys.

Layer 0 uses a dense TP MLP. Later layers route each rank's 512 token rows to
experts, dispatch through DeepEP by default, and run compact grouped FP8
Gate/Up and Down GEMMs through DeepGEMM. Tagged-word peer mailboxes return
expert rows for FP32 weighting and combination. Target residual features are
copied to the per-rank draft buffers at the requested layer boundaries.

FlashInfer provides BF16 gather/reduce-scatter by default; NCCL handles
all-reduce and fallback collectives. NCCL dispatch is also selectable.
Whole-chunk CUDA graphs are keyed by chunk position and captured after an eager
warm-up. Segment capture and eager execution remain available. Device buffers
hold the valid-row count, so prompts can reuse a captured chunk shape.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill import adapters
from naivert.prefill.comms import TP8Comm
from naivert.prefill.execution import GraphSegments, PrefillError
from naivert.prefill.ops import DSA_ROPE_IMAG
from naivert.prefill.native.stable_dsa_topk import install_stable_epilogue

T = 4096
RANKS = 8
SLICE = T // RANKS                      # 512 rows a rank owns for the MoE
PREFIX = P.SWA_WINDOW[0]                # 127 ring rows a later chunk's window reaches back into
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

_COMBINE = None


def combine_extension(cuda_home: str | None = None):
    """Build the tagged-word expert-return extension with compact top-8 mailboxes."""
    global _COMBINE
    if _COMBINE is None:
        from naivert.prefill.native.combine_ext import load_combine_extension   # common builder, same flags
        _COMBINE = load_combine_extension(cuda_home)
    return _COMBINE


class _Rank:
    """One rank's workspaces."""

    def __init__(self, rank: int, dev: torch.device, dsa: bool, swa: bool, mailbox_bytes: int,
                 *, packed_dispatch=True):
        bf16 = dict(dtype=torch.bfloat16, device=dev)
        f32 = dict(dtype=torch.float32, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        i64 = dict(dtype=torch.int64, device=dev)
        self.rank, self.dev = rank, dev
        self.res = torch.zeros((T, P.HIDDEN), **bf16)          # replicated residual
        self.x = torch.zeros((T, P.HIDDEN), **bf16)            # normed rows / scratch
        self.qkv = torch.empty((T, P.QKV_ROWS), **bf16)
        self.v = torch.empty((T, P.KV_HEADS, P.V_HEAD_DIM), **bf16)
        self.partial = torch.zeros((T, P.HIDDEN), **bf16)      # O / dense partial for the all-reduce
        self.partial_slice = torch.zeros((SLICE, P.HIDDEN), **bf16)  # this rank's rows of the O sum (reduce-scatter)
        self.cu_q = torch.tensor([0, T], **i32)
        self.cu_k = torch.tensor([0, T], **i32)
        self.valid = torch.full((1,), T, **i64)                # device-side chunk length
        self.start = torch.zeros(1, **i64)                     # device-side chunk start
        self.ids = torch.zeros(T, **i64)
        self.arange = torch.arange(T, **i64)
        self.positions = torch.arange(T, **i64)
        self.local_src = torch.arange(T, **i64)                # chunk row a position takes its bytes from
        self.ring_src = torch.zeros(P.SWA_WINDOW[0] + 1, **i64)
        self.ring_rows = torch.zeros(P.SWA_WINDOW[0] + 1, **i64)
        self.route_w = torch.zeros((SLICE, P.TOPK), **f32)
        self.route_ids = torch.zeros((SLICE, P.TOPK), **i32)
        if dsa:
            self.qkw = torch.empty((T, P.INDEX_QKW_ROWS), **bf16)
            self.index_w = torch.empty((T, P.INDEX_HEADS), **bf16)
            self.ks = torch.zeros(T, **i32)
            self.ke = torch.ones(T, **i32)
            self.selected_ke = torch.ones(T, **i32)
            self.cu_dsa = torch.tensor([0, T], **i32)
        if swa:
            self.k_stage = torch.empty((PREFIX + T, P.KV_HEADS, P.HEAD_DIM), **bf16)
            self.v_stage = torch.empty((PREFIX + T, P.KV_HEADS, P.V_HEAD_DIM), **bf16)
            self.prefix_rows = torch.zeros(PREFIX, **i64)
            self.cu_k_prefix = torch.tensor([0, PREFIX + T], **i32)
        # dense (layer 0)
        self.gate_up = torch.empty((T, 2 * P.DENSE_INTERMEDIATE), **bf16)
        self.act = torch.empty((T, P.DENSE_INTERMEDIATE), **bf16)
        # MoE on the slice
        self.x_slice = torch.zeros((SLICE, P.HIDDEN), **bf16)
        self.router = torch.empty((SLICE, P.EXPERTS), **f32)
        self.q_rows = torch.empty((SLICE, P.HIDDEN), dtype=P.FP8_DTYPE, device=dev)
        self.s_rows = P.ue8m0_scale_buffer((), SLICE, P.HIDDEN, device=dev)
        if packed_dispatch:
            self.packed = torch.empty((SLICE, P.DISPATCH_ROW_BYTES), dtype=torch.uint8, device=dev)
            self.gathered = torch.empty((T, P.DISPATCH_ROW_BYTES), dtype=torch.uint8, device=dev)
        e = P.EXPERTS_PER_RANK
        self.expert_alignment = int(P.deep_gemm_module(dev).get_mk_alignment_for_contiguous_layout())
        if self.expert_alignment != 128:
            raise PrefillError('compact expert layout requires the validated 128-row DeepGEMM alignment')
        # All routed pairs plus strictly more than the maximum alignment
        # padding. The final row is unread trash for foreign scale writes.
        self.expert_rows = T * P.TOPK + e * self.expert_alignment
        self.q_buf = torch.empty((self.expert_rows, P.HIDDEN), dtype=P.FP8_DTYPE, device=dev)
        self.s_buf = P.ue8m0_scale_buffer((), self.expert_rows, P.HIDDEN, device=dev)
        self.up = torch.empty((self.expert_rows, 2 * P.EXPERT_INTERMEDIATE), **bf16)
        self.q_act = torch.empty((self.expert_rows, P.EXPERT_INTERMEDIATE), dtype=P.FP8_DTYPE, device=dev)
        self.s_act = P.ue8m0_scale_buffer((), self.expert_rows, P.EXPERT_INTERMEDIATE, device=dev)
        self.dn = torch.empty((self.expert_rows, P.HIDDEN), **bf16)
        self.moe_out = torch.empty((SLICE, P.HIDDEN), **bf16)
        # Rank r sends its own slice of this destination: an in-place gather.
        self.res_slices = self.res
        self.mailbox = torch.zeros(mailbox_bytes, dtype=torch.uint8, device=dev)
        self.epoch = torch.zeros(1, **i64)
        self.live = torch.zeros(1, **i32)
        self.owned_total = torch.zeros(1, dtype=torch.int64, device=dev)   # pairs this card served, all layers
        self.max_masked = torch.zeros(1, dtype=torch.int32, device=dev)      # largest expert row count seen
        self.error = torch.zeros(1, **i32)
        self.expected = P.expected_m(T)


class NativePrefill(GraphSegments):
    """The executor.  ``weights``: namespaces from ``naivert.prefill.operands.kernel_prefill_weights``.

    Whole-chunk capture is enabled by default. The segment fallback uses ``embed``,
    then per-layer attention, projection, residual/MLP preparation, expert execution,
    and residual gather. Collectives run between captured segments. ``valid`` is a
    device value; graph keys retain the chunk position and its visible-key extent.
    """

    def __init__(self, weights, streams, *, feature_layers, features, comm=None, graphs=True,
                 chunk_graphs=True, dispatch_backend=None, collective_backend=None):
        import os
        self.dispatch_backend = dispatch_backend or os.environ.get('NAIVERT_PREFILL_DISPATCH', 'deepep')
        if self.dispatch_backend not in ('deepep', 'nccl'):
            raise PrefillError(f'unknown prefill dispatch backend: {self.dispatch_backend}')
        self.dispatch = None
        self.collectives = None
        self.collective_backend = collective_backend or os.environ.get('NAIVERT_PREFILL_COLLECTIVES', 'flashinfer')
        if self.collective_backend not in ('flashinfer', 'nccl'):
            raise PrefillError(f'unknown prefill collective backend: {self.collective_backend}')
        self._closed = False
        self.weights, self.streams = list(weights), list(streams)
        if len(self.weights) != RANKS or len(self.streams) != RANKS:
            raise PrefillError("eight ranks and streams")
        self.ranks = RANKS
        self.chunk_tokens = T
        self.graphs = bool(graphs)
        self.devices = [torch.device(w.dev) for w in self.weights]
        self.layers = int(self.weights[0].layers)
        self.dsa = [self.weights[0].layer[L]["kind"] == "full_dsa" for L in range(self.layers)]
        self.feature_layers = [int(x) for x in feature_layers]
        self.features = list(features)
        self._wanted = {}
        for r, L in enumerate(self.feature_layers):
            self._wanted.setdefault(L, []).append(r)
        self.comm = comm or TP8Comm(self.devices, self.streams)
        self.versions = P.check_versions()
        for dev in self.devices:
            P.prepare_device(dev)
        from naivert.prefill.native.rope import load_extension as load_rope
        self.versions['prefill-qk-rope'] = load_rope()[1]
        install_stable_epilogue()                       # deterministic DSA tie order
        self.combine = combine_extension()
        self.comm.warmup()
        self.ws, self.pool = [], []
        self.index_rows, self.index_queries = [], []
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                self.ws.append(_Rank(r, dev, any(self.dsa), not all(self.dsa), self.combine.mailbox_bytes(),
                                     packed_dispatch=self.dispatch_backend == 'nccl'))
                w = self.weights[r]
                self.ws[-1].rope_tables = tuple(
                    torch.cat((table[:, 0::2], table[:, 1::2]), dim=-1).contiguous()
                    for table in (w.rope_swa, w.rope_full))
                self.pool.append(torch.cuda.graph_pool_handle())
        self.captured, self.eager_only, self._seen, self._stats = {}, {}, {}, {}
        self.chunk_graphs = bool(chunk_graphs)          # one graph per rank per chunk index, collectives inside
        self._chunk_seen, self._chunk_captured, self._chunk_pool = {}, {}, None
        self._launchers = ThreadPoolExecutor(max_workers=RANKS, thread_name_prefix="prefill-launch")
        self.threaded_replay = True
        import os as _os
        self.reference_index_norm = _os.environ.get("NAIVERT_PREFILL_REF_NORM", "1") == "1"   # nn.LayerNorm (inventory) vs the decode spelling
        self.reduce_scatter_o = _os.environ.get("NAIVERT_PREFILL_REDUCE_SCATTER", "1") == "1"
        if _os.environ.get("NAIVERT_PREFILL_CHUNK_GRAPHS", "1") != "1":
            self.chunk_graphs = False
        if self.dispatch_backend == 'deepep':
            from naivert.prefill.native.deepep import DeepEPDispatch
            try:
                self.dispatch = DeepEPDispatch(self.devices, self.streams, self.ws, self._launchers, tokens=T)
            except BaseException:
                self.close()
                raise
            self.versions['deep-ep'] = self.dispatch.version
        self.versions['prefill-dispatch'] = self.dispatch_backend
        if self.collective_backend == 'flashinfer':
            from naivert.prefill.native.collectives import FlashInferCollectives
            try:
                self.collectives = FlashInferCollectives(self.comm, reduce_scatter=True)
            except BaseException:
                self.close()
                raise
            self.comm = self.collectives
            self.versions['flashinfer-collectives'] = self.collectives.version
        self.versions['prefill-collectives'] = self.collective_backend
        self.versions['inplace-residual-gather'] = True
        self.versions['expert-layout'] = dict(kind='compact-psum', alignment=self.ws[0].expert_alignment,
                                            capacity=self.ws[0].expert_rows)
        self.versions['expert-preparation'] = 'sglang-atomic-scatter'
        # The long-span indexer partitions query rows; attention heads stay TP.
        # Static output/Q buffers bridge the gather and also support segment graphs.
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                self.index_rows.append(torch.empty((T, P.INDEX_TOPK), dtype=torch.int32, device=dev))
                self.index_queries.append(torch.empty((T, P.ATTN_HEADS, P.HEAD_DIM), dtype=torch.bfloat16, device=dev))
        self.versions['long-indexer-query-shards'] = RANKS
        self.sync()

    # ------------------------------------------------------------------ helpers
    def sync(self):
        for s in self.streams:
            s.synchronize()

    def close(self):
        """Drain graphs before freeing shared peer buffers; safe to call twice."""
        if self._closed:
            return
        self.sync()
        for graphs in self._chunk_captured.values():
            for graph in graphs:
                graph.reset()
        self._chunk_captured.clear()
        for captured in self.captured.values():
            for graph in captured.graph:
                graph.reset()
        self.captured.clear()
        if self.dispatch is not None:
            self.dispatch.close()
        if self.collectives is not None:
            self.collectives.close()
        self._launchers.shutdown(wait=True)
        self.ws.clear()
        self.index_rows.clear()
        self.index_queries.clear()
        self.pool.clear()
        self._chunk_pool = None
        self._closed = True

    def _ensure_open(self):
        if self._closed:
            raise PrefillError('native prefill executor is closed')

    def _each(self, fn):
        for r in range(RANKS):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                fn(r)

    @torch.inference_mode()
    def begin_request(self):
        """Fresh prompt at position 0: clear the DSA plain-key shadows and marks."""
        self._ensure_open()
        def go(r):
            for lw in self.weights[r].layer:
                if lw["kind"] == "full_dsa":
                    lw["ki_plain"].zero_(); lw["ki_plain_end"] = 0
        self._each(go)

    def warm_chunks(self, count, *, token=0):
        """Run chunk indices ``0..count-1`` twice on a dummy prompt: the first pass is the eager one
        (JIT, autotune), the second captures the chunk graphs, so no served request pays either.
        The caches hold dummy rows afterwards; ``begin_request`` and the next real prefill overwrite them."""
        self._ensure_open()
        count = int(count)
        began = time.perf_counter()
        for _ in range(2 if count > 0 else 0):
            self.begin_request()
            for c in range(count):
                self.run([int(token)] * T, start=c * T)
        self.begin_request()
        return dict(chunks=count, captured=sorted(self._chunk_captured),
                    seconds=round(time.perf_counter() - began, 3))

    @torch.inference_mode()
    def resume_from_decode(self, position):
        """Rebuild the native shadow from the current decode caches, including restored prefixes."""
        self._ensure_open()
        for r, w in enumerate(self.weights):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                for lw in w.layer:
                    if lw["kind"] == "full_dsa":
                        # Native operand dictionaries are shallow copies of the
                        # decode dictionaries: cache tensors alias, marks do not.
                        # A restored prefix may belong to another request, so no
                        # row of the old native shadow is known to be valid.
                        lw["ki_plain_end"] = 0
                        adapters.dsa_index_keys_rebuild(lw, position)

    # ------------------------------------------------------------------- stages
    def _set_prompt(self, tokens, start, valid):
        """Host -> device once per run, outside the graphs: token ids, ``start`` and ``valid``."""
        padded = torch.as_tensor(tokens + [tokens[-1]] * (T - valid), dtype=torch.long)
        def go(r):
            ws = self.ws[r]
            ws.ids.copy_(padded, non_blocking=False)
            ws.valid.fill_(valid)
            ws.start.fill_(start)
        self._each(go)

    def _embed(self, r):
        ws, w = self.ws[r], self.weights[r]
        last = ws.valid - 1
        torch.index_select(w.embedding, 0, ws.ids, out=ws.res)
        torch.minimum(ws.arange, last, out=ws.local_src)                          # padded rows repeat the last
        torch.add(ws.local_src, ws.start, out=ws.positions)                        # global positions
        if hasattr(ws, "ke"):
            ws.ke.copy_((torch.minimum(ws.arange + 1, ws.valid) + ws.start).to(torch.int32))
            torch.clamp(ws.ke, max=P.INDEX_TOPK, out=ws.selected_ke)
            ws.cu_dsa[1:].copy_((ws.valid + ws.start).to(torch.int32))
        win = P.SWA_WINDOW[0] + 1
        # the chunk's last min(valid, 128) rows -> ring rows; a short chunk repeats row 0
        p = torch.arange(win, device=ws.dev) + (ws.valid - win)
        torch.clamp(p, min=0, out=ws.ring_src)                                     # chunk rows
        torch.remainder(ws.ring_src + ws.start, adapters.RING_ROWS, out=ws.ring_rows)
        if hasattr(ws, "prefix_rows"):
            # the 127 positions before the chunk, as ring rows (read before this chunk publishes)
            torch.remainder(torch.arange(PREFIX, device=ws.dev) + (ws.start - PREFIX), adapters.RING_ROWS,
                            out=ws.prefix_rows)

    def _attention(self, r, L, c):
        dsa = self.dsa[L]
        span = (c + 1) * T                      # visible DSA keys [0, span)
        ws, w = self.ws[r], self.weights[r]
        lw = w.layer[L]
        P.rmsnorm(ws.res, lw["attn_gamma_bf16"], out=ws.x)
        P.linear(ws.x, lw["qkv"], out=ws.qkv)
        q, k, v = P.split_qkv(ws.qkv)
        q, k = P.rope_qk(q, k, ws.positions, ws.rope_tables[int(dsa)])
        P.scale_values(v, out=ws.v)
        if dsa:
            P.linear(ws.x, lw["qkw"], out=ws.qkw)
            iq, key, wts = P.split_indexer(ws.qkw)
            P.scale_index_weights(wts, out=ws.index_w)
            if self.reference_index_norm:
                key = torch.nn.functional.layer_norm(key, (P.INDEX_DIM,), lw["ik_w_bf16"], lw["ik_b_bf16"], P.NORM_EPS)
            else:
                key = P.index_key_norm(key, lw["ik_w"], lw["ik_b"])
            iq = P.rope(iq, ws.positions, w.rope_full, imag=DSA_ROPE_IMAG["iq"])
            key = P.rope(key.view(T, 1, P.INDEX_DIM), ws.positions, w.rope_full, imag=DSA_ROPE_IMAG["ik"])
            iq8, iq_scale = P.quantize_index_rows(iq)
            key8, key_scale = P.quantize_index_rows(key.view(T, P.INDEX_DIM))
            folded = P.fold_index_weights(ws.index_w, iq_scale)
            P.dsa_kv_write(lw["k_cache"], lw["v_cache"], lw["ki_cache"], lw["ki_scale"], k, ws.v,
                           key8, key_scale, ws.positions, ws.local_src, plain=lw["ki_plain"])
            keys, values = P.dsa_kv_view(lw["k_cache"], lw["v_cache"], span)
            ikeys, iscales = P.dsa_index_key_view(lw["ki_plain"], lw["ki_scale"], span)
            if c >= 16:
                lo, hi = r * SLICE, (r + 1) * SLICE
                chosen = P.indexer_select(iq8[lo:hi], folded[lo:hi], ikeys, iscales.unsqueeze(-1),
                                          ws.ks[lo:hi], ws.ke[lo:hi], max_seqlen=span)
                self.index_rows[r][lo:hi].copy_(chosen)
                # Static bridge also supports per-segment graph capture/fallback.
                self.index_queries[r].copy_(q)
                return
            indices = P.indexer_select(iq8, folded, ikeys, iscales.unsqueeze(-1), ws.ks, ws.ke, max_seqlen=span)
            o = P.sparse_attention(q, keys, values, indices, ws.cu_q, ws.cu_dsa,
                                   max_seqlen_q=T, max_seqlen_k=span,
                                   selected_starts=ws.ks, selected_ends=ws.selected_ke)
        else:
            if c:
                # the window reaches 127 positions into the previous chunk: stage them from the
                # ring first, then publish this chunk's tail (which may overwrite those rows)
                P.swa_kv_prefix_rows(lw["k_cache"], lw["v_cache"], ws.prefix_rows, ws.k_stage[:PREFIX], ws.v_stage[:PREFIX])
                ws.k_stage[PREFIX:].copy_(k); ws.v_stage[PREFIX:].copy_(ws.v)
                P.swa_kv_write(lw["k_cache"], lw["v_cache"], k, ws.v, ws.ring_rows, ws.ring_src)
                o = P.swa_attention(q, ws.k_stage, ws.v_stage, lw["sink_bf16"], ws.cu_q, ws.cu_k_prefix,
                                    max_seqlen_q=T, max_seqlen_k=PREFIX + T)
            else:
                P.swa_kv_write(lw["k_cache"], lw["v_cache"], k, ws.v, ws.ring_rows, ws.ring_src)
                ws.k_stage[:T].copy_(k); ws.v_stage[:T].copy_(ws.v)
                o = P.swa_attention(q, ws.k_stage[:T], ws.v_stage[:T], lw["sink_bf16"], ws.cu_q, ws.cu_k,
                                    max_seqlen_q=T, max_seqlen_k=T)
        P.linear(P.attn_output_rows(o), lw["o_bf16"], out=ws.partial)

    def _attention_finish(self, r, L, c):
        ws, lw = self.ws[r], self.weights[r].layer[L]
        q = self.index_queries[r]
        keys, values = P.dsa_kv_view(lw['k_cache'], lw['v_cache'], (c + 1) * T)
        o = P.sparse_attention(q, keys, values, self.index_rows[r], ws.cu_q, ws.cu_dsa,
                               max_seqlen_q=T, max_seqlen_k=(c + 1) * T,
                               selected_starts=ws.ks, selected_ends=ws.selected_ke)
        P.linear(P.attn_output_rows(o), lw['o_bf16'], out=ws.partial)

    def _post(self, r, L):
        """Residual add of the attention sum, then the MLP input side.

        Layer 0 (dense MLP on full rows) receives the all-reduced partial; MoE layers receive
        only this rank's 512-row slice of the sum (reduce-scatter, half the bytes): the MoE
        runs on the slice and the residual is all-gathered at the end of the layer anyway."""
        ws, lw = self.ws[r], self.weights[r].layer[L]
        if L == 0 or not self.reduce_scatter_o:
            ws.res.add_(ws.partial)
        if L == 0:
            P.rmsnorm(ws.res, lw["ffn_gamma_bf16"], out=ws.x)
            P.linear(ws.x, lw["dense_gate_up"], out=ws.gate_up)
            P.silu_and_mul(ws.gate_up, out=ws.act)
            P.linear(ws.act, lw["dense_down_bf16"], out=ws.partial)
            return
        lo = r * SLICE
        if self.reduce_scatter_o:
            ws.res[lo:lo + SLICE].add_(ws.partial_slice)
        P.rmsnorm(ws.res[lo:lo + SLICE], lw["ffn_gamma_bf16"], out=ws.x_slice)
        P.router_logits(ws.x_slice, lw["router"], out=ws.router)
        wts, ids = P.route(ws.router, lw["router_bias"])
        ws.route_w.copy_(wts); ws.route_ids.copy_(ids)         # static buffers: pool outputs are not safe to hold across a capture
        P.quantize_dispatch_rows(ws.x_slice, ws.q_rows, ws.s_rows)
        if self.dispatch is not None:
            self.dispatch.ids[r].copy_(ids)
        else:
            P.pack_dispatch_rows(ws.q_rows, ws.s_rows, ws.route_ids, ws.route_w, ws.packed)

    def _experts(self, r, L):
        ws, lw = self.ws[r], self.weights[r].layer[L]
        if self.dispatch is not None:
            codes, scales, ids = self.dispatch.received[r][:3]
            first_expert = 0
        else:
            fields = P.dispatch_rows(ws.gathered)
            codes, scales, ids = fields.codes, fields.scales, fields.ids
            first_expert = lw['ep_first_expert']
        counts, permutation, dest, ends, aligned = P.prepare_compact_experts(
            codes, scales, ids, ws.q_buf, ws.s_buf, first_expert=first_expert,
            alignment=ws.expert_alignment)
        (w13, w13_s), (w2, w2_s) = lw["ep_w13"], lw["ep_w2"]
        P.grouped_gemm_contiguous(ws.q_buf, ws.s_buf, w13, w13_s, ws.up, ends, ws.expected)
        P.act_quant_masked(ws.up.unsqueeze(0), ws.q_act.unsqueeze(0), ws.s_act.unsqueeze(0), aligned)
        P.grouped_gemm_contiguous(ws.q_act, ws.s_act, w2, w2_s, ws.dn, ends, ws.expected)
        # The grouping permutation already puts every local pair before foreign
        # pairs. Publication order is free: the receiver sums fixed top-k slots.
        # A flat pair ID is exactly token * TOPK + slot, so no second sort/map.
        dst = permutation
        src = dest.index_select(0, dst)
        if self.dispatch is not None:
            dst = self.dispatch.mailbox_pairs(r, permutation)
            route_ids = self.dispatch.ids[r]
        else:
            route_ids = ws.route_ids.to(torch.int64).contiguous()
        ws.live.copy_(counts.sum(dtype=torch.int32))
        ws.owned_total.add_(ws.live.to(torch.int64))
        torch.maximum(ws.max_masked, counts.max().reshape(1), out=ws.max_masked)
        ws.epoch.add_(1)
        self.combine.run(ws.dn.view(-1, P.HIDDEN), src, dst, [x.mailbox for x in self.ws], r,
                       route_ids, ws.route_w, ws.moe_out, ws.epoch, ws.error,
                       rows_dev=ws.live)
        lo = r * SLICE
        ws.res[lo:lo + SLICE].add_(ws.moe_out)
        ws.res_slice = ws.res[lo:lo + SLICE]

    def _gather_land(self, r, L, c):
        if r in self._wanted.get(L, []):
            self._feature_copy(r, c)

    def _feature_copy(self, r, c):
        lo = c * T
        rows = max(0, min(T, int(self.features[r].shape[0]) - lo))
        if rows:
            self.features[r][lo:lo + rows].copy_(self.ws[r].res[:rows])

    # ---------------------------------------------------------------------- run
    def _chunk_body(self, c, segment_graphs):
        """The chunk as segments (captured per segment when ``segment_graphs``) with eager
        collectives between them; also the body a whole-chunk capture records."""
        self._segment(("embed",), self._embed, graphs=segment_graphs)
        ws = self.ws
        for L in range(self.layers):
            self._segment(("attn", L, c), lambda r, L=L: self._attention(r, L, c), graphs=segment_graphs)
            if self.dsa[L] and c >= 16:
                # No reduction: the BF16 all-gather transports int32 payload bits.
                # Valid IDs [-1, 1048575] cannot equal its reserved 0x80008000 word.
                self.comm.all_gather_(
                    [v[r * SLICE:(r + 1) * SLICE].view(torch.bfloat16) for r, v in enumerate(self.index_rows)],
                    [v.view(torch.bfloat16) for v in self.index_rows])
                self._stats['collectives'] += 1
                self._segment(('attn_finish', L, c), lambda r, L=L, c=c: self._attention_finish(r, L, c),
                              graphs=segment_graphs)
            if L == 0 or not self.reduce_scatter_o:
                self.comm.all_reduce_([x.partial for x in ws])
            else:
                self.comm.reduce_scatter_([x.partial for x in ws], [x.partial_slice for x in ws])
            self._stats["collectives"] += 1
            self._segment(("post", L), lambda r, L=L: self._post(r, L), graphs=segment_graphs)
            if L == 0:
                self.comm.all_reduce_([x.partial for x in ws]); self._stats["collectives"] += 1
                self._segment(("dense_land",), lambda r: ws[r].res.add_(ws[r].partial), graphs=segment_graphs)
                if r_list := self._wanted.get(0, []):
                    self._segment(("feat0", c), lambda r: (self._feature_copy(r, c) if r in r_list else None), graphs=segment_graphs)
                continue
            if self.dispatch is not None:
                self.dispatch.run()
            else:
                self.comm.all_gather_([x.packed for x in ws], [x.gathered for x in ws])
            self._stats["collectives"] += 1
            # DeepEP allocates receive tensors per call. Whole-chunk capture
            # owns those allocations; a separately captured expert segment
            # would instead retain pointers from an earlier eager dispatch.
            self._segment(("experts", L), lambda r, L=L: self._experts(r, L),
                          graphs=segment_graphs and self.dispatch is None)
            self.comm.all_gather_([x.res[r * SLICE:(r + 1) * SLICE] for r, x in enumerate(ws)],
                                  [x.res_slices for x in ws]); self._stats["collectives"] += 1
            self._segment(("gather", L, c), lambda r, L=L: self._gather_land(r, L, c), graphs=segment_graphs)

    def _replay_one(self, graphs, r):
        began = time.perf_counter()
        with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
            graphs[r].replay()
        return time.perf_counter() - began

    def _replay_all(self, graphs):
        """Eight graph launches; a large graph costs the host milliseconds to launch, so the
        launches go out from eight threads (``replay`` releases the GIL) instead of in sequence,
        which would start rank 7 tens of milliseconds after rank 0 and leave rank 0 spinning."""
        began = time.perf_counter()
        if self.threaded_replay:
            costs = list(self._launchers.map(lambda r: self._replay_one(graphs, r), range(self.ranks)))
        else:
            costs = [self._replay_one(graphs, r) for r in range(self.ranks)]
        self._stats["launch_ms"] = dict(total=round((time.perf_counter() - began) * 1e3, 2),
                                        per_rank=[round(x * 1e3, 2) for x in costs], threaded=self.threaded_replay)

    def _run_chunk_graph(self, c):
        """Eager the first time a chunk index is seen (JIT, autotune), captured the second time as
        one graph per rank with the selected collectives and tagged-word return inside (all eight
        streams capture at once), replayed afterwards: eight launches a chunk."""
        done = self._chunk_captured.get(c)
        if done is not None:
            self._replay_all(done)
            self._stats["graph_replays"] += 1
            return
        seen = self._chunk_seen.get(c, 0)
        self._chunk_seen[c] = seen + 1
        if not seen:
            self._chunk_body(c, segment_graphs=False)
            self._stats["eager_segments"] += 1
            return
        if self._chunk_pool is None:
            self._chunk_pool = [torch.cuda.graph_pool_handle() for _ in self.devices]
        self.sync()
        graphs = []
        try:
            for r in range(self.ranks):
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    g = torch.cuda.CUDAGraph()
                    # relaxed: the in-process NCCL group launch makes a host call that the
                    # thread-local mode refuses; its kernels still record into each stream
                    g.capture_begin(pool=self._chunk_pool[r], capture_error_mode="relaxed")
                    graphs.append(g)
            self._chunk_body(c, segment_graphs=False)
            for r in range(self.ranks):
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    graphs[r].capture_end()
        except Exception as exc:                                       # noqa: BLE001
            for r in range(len(graphs)):
                try:
                    with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                        graphs[r].capture_end()
                except Exception:                                      # noqa: BLE001
                    pass
            for g in graphs:
                g.reset()
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            self._stats["chunk_capture_refused"] = reason
            self.chunk_capture_refused = reason
            print("CHUNK CAPTURE REFUSED", reason, flush=True)
            import traceback; traceback.print_exc()
            self.chunk_graphs = False
            self._chunk_body(c, segment_graphs=True)
            return
        self._chunk_captured[c] = graphs
        # a capture records and runs nothing: replay for the chunk under way
        self._replay_all(graphs)
        self._stats["graph_captures"] += 1
        self._stats["graph_replays"] += 1

    @torch.inference_mode()
    def run(self, tokens, start=0, *, graphs=None, record=None, tail=False):
        self._ensure_open()
        toks = [int(t) for t in tokens]
        valid = len(toks)
        if start % T or not 1 <= valid <= T:
            raise PrefillError(f"native prefill: chunks of at most {T} tokens starting at multiples of {T} "
                               f"(got start={start}, {valid})")
        c = start // T
        for w in self.weights:
            for lw in w.layer:
                if lw["kind"] == "full_dsa" and lw.get("ki_plain_end", 0) < start:
                    raise PrefillError(f"chunk {c}: the DSA index shadow covers {lw.get('ki_plain_end', 0)} < {start}")
                if lw["kind"] == "full_dsa" and int(lw["k_cache"].shape[0]) < start + T:
                    raise PrefillError("cache capacity below the chunk end")
        use_graphs = self.graphs if graphs is None else bool(graphs)
        began = time.perf_counter()
        self._stats = dict(tokens=valid, start=start, chunks=[dict(start=start, valid=valid, bucket=T)],
                           kernels="inventory", graphs=use_graphs, chunk_graphs=self.chunk_graphs and use_graphs,
                           graph_replays=0, graph_captures=0, eager_segments=0, collectives=0)
        self._set_prompt(toks, start, valid)
        for x in self.ws:
            x.owned_total.zero_(); x.max_masked.zero_()
        if self.chunk_graphs and use_graphs:
            self._run_chunk_graph(c)
        else:
            self._chunk_body(c, use_graphs)
        for w in self.weights:
            for lw in w.layer:
                if lw["kind"] == "full_dsa":
                    lw["ki_plain_end"] = start + valid
        self.sync()
        errors = [int(x.error.item()) for x in self.ws]
        if any(errors):
            raise PrefillError(f"tagged-word combine watchdog: {errors}")
        self._stats["owned_pairs_per_rank"] = [int(x.owned_total.item()) for x in self.ws]
        self._stats["max_expert_rows"] = [int(x.max_masked.item()) for x in self.ws]
        self._stats["expert_cap"] = T     # one token can select each expert at most once
        self._stats["expert_buffer_rows"] = self.ws[0].expert_rows
        if max(self._stats["max_expert_rows"]) > T:
            raise PrefillError(f"an expert received {max(self._stats['max_expert_rows'])} rows in a {T}-token chunk")
        self._stats["seconds"] = time.perf_counter() - began
        return self._stats


__all__ = ["NativePrefill", "combine_extension", "T"]
