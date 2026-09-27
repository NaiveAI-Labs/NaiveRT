"""Chunk-major Naive N0.5 prefill on SGLang's kernels: eight ranks, eight streams, one process.

This reference executor runs the stage bindings in
``naivert.kernels.naive_n0p5.prefill``.  Every chunk of the plan runs the whole model, E1 to F3,
before the next chunk starts, as SGLang's chunked prefill does, so a chunk's attention reads
what every earlier chunk published to the caches.

The scatter modes are the reference's TP8 + DeepEP layout, with ``T`` the chunk's bucket and
``T / 8`` rows a rank:

* layer 0 (DSA, dense MLP) is FULL: E2's all-reduced embedding is the residual, D1 norms it,
  D24 and M4 all-reduce;
* layer 1 takes those FULL rows (S1 over ``[T, 4096]``).  From its reduce-scatter on (S9, or
  D24' in a DSA layer) every MoE block is SCATTERED: rank ``r`` holds rows
  ``[r T / 8, (r + 1) T / 8)`` of both the hidden rows and the residual;
* layers 2 and up start SCATTERED: the layer before's MoE output, S1 over ``[T / 8, 4096]``,
  then S1b all-gathers the attention's input;
* after the last layer F1 adds the residual and all-gathers, F2 norms, and F3 keeps the last
  real row in ``last_hidden``. Head projection is outside this executor; the serving
  engine uses a final decode step for prompt logits.

The collectives are ``TP8Comm`` launches between per-rank segments
(``GraphSegments._segment``).  This reference path replaces DeepEP dispatch: X3 all-gathers
every rank's packed quantised rows, each card lays its 32 experts' rows out as
``[33, T, 4096]`` (row 32 is the trash expert) for DeepGEMM's masked grouped GEMM, and X7
sums the cards' fp32 partials with a reduce-scatter that leaves each rank its own rows.  The
next rank segment casts those rows to the inventory's bf16 ``[T / 8, 4096]`` (``X7(cast)``, at
the start of the next layer's attention input or of F1).  X8, SGLang's slice of DeepEP's
output back to the dispatched rows, launches nothing here: a rank holds exactly ``T / 8``.

A chunk is padded to its bucket.  Rows past ``valid`` repeat the last real token and
position (``ChunkMeta``), the caches take real rows only, and F3 reads row ``valid - 1``.

``graphs`` picks which form of a chunk's extents the stage bodies use, once, at construction,
so that a run's eager, captured and replayed passes all launch the same kernels:

* ``graphs=False`` passes ``start``, ``valid``, the SWA prefix and the DSA key
  end as host integers and calls the eager cache adapters;
* ``graphs=True`` calls the adapters' device forms over ``ChunkMeta``'s rows and
  positions, keeps the last row with ``ChunkMeta.last``, and leaves as host values only what a
  static shape already is -- the bucket, the SWA staging split at ``prefix``, and the DSA key
  end, which bound tensors rather than name rows.  Those last two are in the segment key, so a
  replay can never read a chunk's rows through another chunk's extent; the lengths the
  attention kernels mask by (``cu_swa`` / ``cu_dsa``) are device buffers rewritten per chunk.
  For the main cases (``start = 0``, one chunk) both collapse to the bucket, which is the
  per-bucket graph the plan asks for.

An optional ``observer`` sees every stage by its inventory ID (a call covering several stages
is named ``A+B``, a stage split around a collective ``A(part)``) and each stage's outputs.
Observers can inspect shapes, dtypes, and synchronization at stage boundaries.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.models.naive_n0p5.model import PRECISION_ID
from naivert.prefill.execution import GraphSegments, PrefillError
from naivert.prefill.ops import DSA_ROPE_IMAG
from naivert.prefill.metadata import ChunkMeta
from naivert.prefill.planning import Chunk, PrefillPlanConfig, resolve_plan_config

INDEX_KEY_REFUSAL = "prefill starts at {start}, but the contiguous DSA index prefix ends at {covered}"
#: rope's imaginary spelling at the SWA q/k sites, as at the DSA q/k sites
SWA_ROPE_IMAG = {"q": "cos", "k": "cos"}
_NO_SCOPE = nullcontext()

#: Required weight/cache fields prepared by ``naivert.prefill.operands``.
_LAYER_KEYS = {
    "all": ("qkv", "o_bf16", "attn_gamma_bf16", "ffn_gamma_bf16", "k_cache", "v_cache"),
    "full_dsa": ("qkw", "ik_w", "ik_b", "ki_cache", "ki_scale", "ki_plain"),
    "sliding_window": ("sink_bf16",),
    "dense": ("dense_gate_up", "dense_down_bf16"),
    "moe": ("router", "router_bias", "ep_w13", "ep_w2", "ep_first_expert"),
}


class StageObserver:
    """What ``PrefillExecutor`` reports about its stages; this base reports nothing.

    ``stage`` returns the context one stage's launches run in: per rank, or ``rank=None`` for
    a collective, which launches on every rank at once.  ``output`` then receives the stage's
    results as named tensors, outside that context.  ``layer`` is ``None`` for E1 - E2 and
    F1 - F3.
    """

    def stage(self, stage: str, layer: int | None, rank: int | None):
        return _NO_SCOPE

    def output(self, stage: str, layer: int | None, rank: int, tensors: dict) -> None:
        pass


class _Workspace:
    """One rank's buffers for one bucket: ``rows`` chunk rows, ``rows // 8`` MoE rows."""

    def __init__(self, rows: int, device: torch.device, *, dsa: bool, swa: bool) -> None:
        t, p, e, h = int(rows), int(rows) // P.RANKS, P.EXPERTS_PER_RANK, P.HIDDEN
        bf16 = dict(dtype=torch.bfloat16, device=device)
        fp32 = dict(dtype=torch.float32, device=device)
        fp8 = dict(dtype=P.FP8_DTYPE, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.rows, self.moe_rows = t, p
        # the residual stream, FULL and SCATTERED
        self.res_full, self.x_full, self.partial = (torch.zeros((t, h), **bf16) for _ in range(3))
        self.res_part, self.x_part = (torch.zeros((p, h), **bf16) for _ in range(2))
        # attention
        self.qkv = torch.empty((t, P.QKV_ROWS), **bf16)
        self.v = torch.empty((t, P.KV_HEADS, P.V_HEAD_DIM), **bf16)
        self.cu_q = torch.zeros(2, **i32)
        self.cu_q[1:].fill_(t)
        self.cu_swa, self.cu_dsa = torch.zeros(2, **i32), torch.zeros(2, **i32)
        if dsa:
            self.qkw = torch.empty((t, P.INDEX_QKW_ROWS), **bf16)
            self.index_w = torch.empty((t, P.INDEX_HEADS), **bf16)
        if swa:
            # FA4 reads the ring's prefix rows and the chunk's rows from one staging buffer
            prefix = P.SWA_WINDOW[0]
            self.k_stage = torch.empty((prefix + t, P.KV_HEADS, P.HEAD_DIM), **bf16)
            self.v_stage = torch.empty((prefix + t, P.KV_HEADS, P.V_HEAD_DIM), **bf16)
        # layer 0's dense MLP
        self.gate_up = torch.empty((t, 2 * P.DENSE_INTERMEDIATE), **bf16)
        self.act = torch.empty((t, P.DENSE_INTERMEDIATE), **bf16)
        # MoE: this rank's rows, the gathered rows, and the card's 32 experts
        self.router = torch.empty((p, P.EXPERTS), **fp32)
        self.q_rows = torch.empty((p, h), **fp8)
        self.s_rows = P.ue8m0_scale_buffer((), p, h, device=device)
        self.packed = torch.empty((p, P.DISPATCH_ROW_BYTES), dtype=torch.uint8, device=device)
        self.gathered = torch.empty((t, P.DISPATCH_ROW_BYTES), dtype=torch.uint8, device=device)
        self.q_buf = torch.empty((e + 1, t, h), **fp8)
        self.s_buf = P.ue8m0_scale_buffer((e + 1,), t, h, device=device)
        self.up = torch.empty((e, t, 2 * P.EXPERT_INTERMEDIATE), **bf16)
        self.q_act = torch.empty((e, t, P.EXPERT_INTERMEDIATE), **fp8)
        self.s_act = P.ue8m0_scale_buffer((e,), t, P.EXPERT_INTERMEDIATE, device=device)
        self.dn = torch.zeros((e + 1, t, h), **bf16)       # row e stays zero: the trash expert
        self.combine = torch.empty((t, h), **fp32)
        self.moe_out = torch.zeros((p, h), **fp32)
        self.expected = P.expected_m(t)


class PrefillExecutor(GraphSegments):
    """The SGLang-kernel prefill over eight ranks' weights, caches and streams."""

    def __init__(self, weights, streams, *, comm, max_seq_len, last_hidden=None,
                 index_capacity=None, buckets=None, plan_config=None, graphs=False):
        self.weights, self.streams = list(weights), list(streams)
        self.ranks = len(self.weights)
        if self.ranks != P.RANKS or len(self.streams) != P.RANKS:
            raise PrefillError("the kernel prefill requires eight rank weights and streams")
        if any(getattr(w, "precision", None) != PRECISION_ID for w in self.weights):
            raise PrefillError("the kernel prefill requires original Naive N0.5 operands on every rank")
        self.devices = [torch.device(w.dev) for w in self.weights]
        self.comm = comm
        self._plan_config = resolve_plan_config(buckets=buckets, config=plan_config)
        split = [b for b in self.buckets if b % P.RANKS]
        if split:
            raise PrefillError(f"buckets {split} do not split into {P.RANKS} scattered MoE shards")
        self.graphs = bool(graphs)
        self.observer: StageObserver | None = None
        self.layers = int(self.weights[0].layers)
        self.max_seq_len = int(max_seq_len)
        self.index_capacity = self.max_seq_len if index_capacity is None else int(index_capacity)
        self.max_positions = min(self.max_seq_len, self.index_capacity)
        self.dsa = [self.weights[0].layer[layer].get("kind") == "full_dsa"
                    for layer in range(self.layers)]
        self.dsa_layers = [layer for layer, full in enumerate(self.dsa) if full]
        self._check_operands()
        self.versions = P.check_versions()
        for dev in self.devices:
            P.prepare_device(dev)
        self.comm.warmup()
        self._allocate(last_hidden)

    def _check_operands(self) -> None:
        for rank, w in enumerate(self.weights):
            missing = [] if hasattr(w, "head_gamma_bf16") else ["head_gamma_bf16"]
            for layer, lw in enumerate(w.layer):
                want = (_LAYER_KEYS["all"] + _LAYER_KEYS[lw.get("kind")]
                        + _LAYER_KEYS["dense" if layer == 0 else "moe"])
                missing += [f"layer {layer}: {key}" for key in want if key not in lw]
                if layer and lw.get("ep_first_expert") != rank * P.EXPERTS_PER_RANK:
                    missing.append(f"layer {layer}: ep_first_expert {rank * P.EXPERTS_PER_RANK}")
            if missing:
                raise PrefillError(f"rank {rank} lacks the kernel prefill operands {missing}; load "
                                   "NaiveN0p5ModelResources(kernel_prefill=True)")
        if self.weights[0].layer[0].get("kind") != "full_dsa":
            raise PrefillError("the kernel prefill expects layer 0 to be the dense DSA layer")

    def _allocate(self, last_hidden) -> None:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dsa, swa = any(self.dsa), not all(self.dsa)
        self.pool, self._meta, self._ws = [], [], []
        self.last_hidden = [] if last_hidden is None else list(last_hidden)
        for rank, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[rank]):
                self.pool.append(torch.cuda.graph_pool_handle())
                self._meta.append({b: ChunkMeta(b, dev, max_positions=self.max_positions)
                                   for b in self.buckets})
                self._ws.append({b: _Workspace(b, dev, dsa=dsa, swa=swa) for b in self.buckets})
                if last_hidden is None:
                    self.last_hidden.append(torch.zeros((1, P.HIDDEN), dtype=torch.bfloat16, device=dev))
        self.captured, self.eager_only, self._seen, self._stats = {}, {}, {}, {}

    # ------------------------------------------------------------------------ reporting
    def _stage(self, stage: str, layer: int | None, rank: int | None):
        return _NO_SCOPE if self.observer is None else self.observer.stage(stage, layer, rank)

    def _output(self, stage: str, layer: int | None, rank: int, **tensors) -> None:
        if self.observer is not None:
            self.observer.output(stage, layer, rank, tensors)

    # ---------------------------------------------------------------------- collectives
    def _all_reduce(self, stage, layer, name, tensors) -> None:
        with self._stage(stage, layer, None):
            self.comm.all_reduce_(tensors)
        self._collected(stage, layer, name, tensors)

    def _reduce_scatter(self, stage, layer, name, inputs, outputs) -> None:
        with self._stage(stage, layer, None):
            self.comm.reduce_scatter_(inputs, outputs)
        self._collected(stage, layer, name, outputs)

    def _all_gather(self, stage, layer, name, inputs, outputs) -> None:
        with self._stage(stage, layer, None):
            self.comm.all_gather_(inputs, outputs)
        self._collected(stage, layer, name, outputs)

    def _collected(self, stage, layer, name, tensors) -> None:
        self._stats["collectives"] = self._stats.get("collectives", 0) + 1
        for rank, t in enumerate(tensors):
            self._output(stage, layer, rank, **{name: t})

    # ------------------------------------------------------------------- rank segments
    def _embed(self, rank: int, rows: int) -> None:
        w, ws, meta = self.weights[rank], self._ws[rank][rows], self._meta[rank][rows]
        first = rank * P.VOCAB_SHARD
        with self._stage("E1", None, rank):
            P.embed(meta.tokens, w.embedding[first:first + P.VOCAB_SHARD], first_row=first,
                    out=ws.res_full)
        self._output("E1", None, rank, hidden=ws.res_full)

    def _attention_input(self, rank: int, layer: int, rows: int) -> None:
        """Layers 2 and up: the previous MoE's rows in bf16, then S1 over them (before S1b)."""
        ws, lw = self._ws[rank][rows], self.weights[rank].layer[layer]
        with self._stage("X7(cast)", layer - 1, rank):
            ws.x_part.copy_(ws.moe_out)
        self._output("X7(cast)", layer - 1, rank, hidden=ws.x_part)
        with self._stage("S1", layer, rank):
            P.add_rmsnorm_(ws.x_part, ws.res_part, lw["attn_gamma_bf16"])
        self._output("S1", layer, rank, hidden=ws.x_part, residual=ws.res_part)

    def _attention(self, rank: int, layer: int, chunk: Chunk):
        """Layers 0 and 1 norm their FULL input here; then DSA or SWA attention into ``partial``."""
        ws, lw = self._ws[rank][chunk.bucket], self.weights[rank].layer[layer]
        if layer == 0:
            with self._stage("D1", layer, rank):
                P.rmsnorm(ws.res_full, lw["attn_gamma_bf16"], out=ws.x_full)
            self._output("D1", layer, rank, hidden=ws.x_full)
        elif layer == 1:
            with self._stage("S1", layer, rank):
                P.add_rmsnorm_(ws.x_full, ws.res_full, lw["attn_gamma_bf16"])
            self._output("S1", layer, rank, hidden=ws.x_full, residual=ws.res_full)
        if self.dsa[layer]:
            return self._dsa_attention(rank, layer, chunk)
        self._swa_attention(rank, layer, chunk)
        return None

    def _dsa_attention(self, rank: int, layer: int, chunk: Chunk) -> torch.Tensor:
        """D2 - D23 over the FULL rows; returns the selected key indices ``[T, 2048]``."""
        w, lw = self.weights[rank], self.weights[rank].layer[layer]
        rows, start, valid = chunk.bucket, chunk.start, chunk.valid
        ws, meta, end = self._ws[rank][rows], self._meta[rank][rows], start + valid
        with self._stage("D2", layer, rank):
            P.linear(ws.x_full, lw["qkv"], out=ws.qkv)
        q, k, v = P.split_qkv(ws.qkv)
        self._output("D2", layer, rank, qkv=ws.qkv)
        with self._stage("D3", layer, rank):
            q = P.rope(q, meta.positions, w.rope_full, imag=DSA_ROPE_IMAG["q"])
            k = P.rope(k, meta.positions, w.rope_full, imag=DSA_ROPE_IMAG["k"])
        self._output("D3", layer, rank, q=q, k=k)
        with self._stage("D4", layer, rank):
            P.scale_values(v, out=ws.v)
        self._output("D4", layer, rank, v=ws.v)
        with self._stage("D5", layer, rank):
            P.linear(ws.x_full, lw["qkw"], out=ws.qkw)
            iq, key, weights = P.split_indexer(ws.qkw)
        self._output("D5", layer, rank, qkw=ws.qkw, q=iq, key=key, weights=weights)
        with self._stage("D6", layer, rank):
            P.scale_index_weights(weights, out=ws.index_w)
        self._output("D6", layer, rank, weights=ws.index_w)
        with self._stage("D7", layer, rank):
            key = P.index_key_norm(key, lw["ik_w"], lw["ik_b"])
        self._output("D7", layer, rank, key=key)
        with self._stage("D8", layer, rank):
            iq = P.rope(iq, meta.positions, w.rope_full, imag=DSA_ROPE_IMAG["iq"])
            # [T, 1, 128]: rope_fma runs a [T, 128] key as this same one-head layout
            key = P.rope(key.view(rows, 1, P.INDEX_DIM), meta.positions, w.rope_full,
                         imag=DSA_ROPE_IMAG["ik"])
        self._output("D8", layer, rank, q=iq, key=key)
        with self._stage("D9/D10", layer, rank):
            iq8, iq_scale = P.quantize_index_rows(iq)
            key8, key_scale = P.quantize_index_rows(key.view(rows, P.INDEX_DIM))
        self._output("D9/D10", layer, rank, q=iq8, q_scale=iq_scale, key=key8, key_scale=key_scale)
        with self._stage("D11", layer, rank):
            folded = P.fold_index_weights(ws.index_w, iq_scale)
        self._output("D11", layer, rank, weights=folded)
        with self._stage("D12+D13", layer, rank):
            if self.graphs:
                P.dsa_kv_write(lw["k_cache"], lw["v_cache"], lw["ki_cache"], lw["ki_scale"], k,
                               ws.v, key8, key_scale, meta.positions, meta.src,
                               plain=lw["ki_plain"])
            else:
                P.dsa_kv_append(lw["k_cache"], lw["v_cache"], lw["ki_cache"], lw["ki_scale"], k,
                                ws.v, key8, key_scale, start=start, valid=valid,
                                plain=lw["ki_plain"])
        self._output("D12+D13", layer, rank, k=lw["k_cache"][start:end], v=lw["v_cache"][start:end],
                     key=lw["ki_plain"][start:end], swizzled=lw["ki_cache"][start:end],
                     scale=lw["ki_scale"][start:end])
        with self._stage("D14", layer, rank):
            keys, values = P.dsa_kv_view(lw["k_cache"], lw["v_cache"], end)
        self._output("D14", layer, rank, k=keys, v=values)
        with self._stage("D15", layer, rank):
            index_keys, index_scales = P.dsa_index_key_view(lw["ki_plain"], lw["ki_scale"], end)
            index_scales = index_scales.unsqueeze(-1)       # [end, 1]; the logits kernel flattens it
        self._output("D15", layer, rank, key=index_keys, scale=index_scales)
        with self._stage("D17+D18+D19", layer, rank):
            indices = P.indexer_select(iq8, folded, index_keys, index_scales, meta.ks, meta.ke,
                                       max_seqlen=end)
        self._output("D17+D18+D19", layer, rank, indices=indices)
        with self._stage("D21", layer, rank):
            o = P.sparse_attention(q, keys, values, indices, ws.cu_q, ws.cu_dsa,
                                   max_seqlen_q=rows, max_seqlen_k=end)
        self._output("D21", layer, rank, out=o)
        with self._stage("D22", layer, rank):
            o_rows = P.attn_output_rows(o)
        self._output("D22", layer, rank, rows=o_rows)
        with self._stage("D23", layer, rank):
            P.linear(o_rows, lw["o_bf16"], out=ws.partial)
        self._output("D23", layer, rank, partial=ws.partial)
        return indices

    def _swa_attention(self, rank: int, layer: int, chunk: Chunk) -> None:
        """S2 - S8 over the FULL rows."""
        w, lw = self.weights[rank], self.weights[rank].layer[layer]
        rows, start, valid, prefix = chunk.bucket, chunk.start, chunk.valid, chunk.prefix
        ws, meta, span = self._ws[rank][rows], self._meta[rank][rows], chunk.prefix + chunk.bucket
        with self._stage("S2", layer, rank):
            P.linear(ws.x_full, lw["qkv"], out=ws.qkv)
        self._output("S2", layer, rank, qkv=ws.qkv)
        with self._stage("S3", layer, rank):
            q, k, v = P.split_qkv(ws.qkv)
        self._output("S3", layer, rank, q=q, k=k, v=v)
        with self._stage("S4", layer, rank):
            q = P.rope(q, meta.positions, w.rope_swa, imag=SWA_ROPE_IMAG["q"])
            k = P.rope(k, meta.positions, w.rope_swa, imag=SWA_ROPE_IMAG["k"])
        self._output("S4", layer, rank, q=q, k=k)
        with self._stage("S5", layer, rank):
            P.scale_values(v, out=ws.v)
        self._output("S5", layer, rank, v=ws.v)
        with self._stage("S6", layer, rank):
            # The prefix leaves the ring before the chunk publishes: a chunk row can take the
            # ring row of a prefix position 136 earlier.
            if self.graphs:
                P.swa_kv_prefix_rows(lw["k_cache"], lw["v_cache"], meta.prefix_rows[:prefix],
                                     ws.k_stage[:prefix], ws.v_stage[:prefix])
            else:
                P.swa_kv_prefix(lw["k_cache"], lw["v_cache"], start, ws.k_stage[:prefix],
                                ws.v_stage[:prefix])
            ws.k_stage[prefix:span].copy_(k)
            ws.v_stage[prefix:span].copy_(ws.v)
            if self.graphs:
                P.swa_kv_write(lw["k_cache"], lw["v_cache"], k, ws.v, meta.publish_rows,
                               meta.publish_src)
            else:
                P.swa_kv_publish(lw["k_cache"], lw["v_cache"], k, ws.v, start=start, valid=valid)
        self._output("S6", layer, rank, k=ws.k_stage[:span], v=ws.v_stage[:span])
        with self._stage("S7", layer, rank):
            o = P.swa_attention(q, ws.k_stage[:span], ws.v_stage[:span], lw["sink_bf16"], ws.cu_q,
                                ws.cu_swa, max_seqlen_q=rows, max_seqlen_k=span)
        self._output("S7", layer, rank, out=o)
        with self._stage("S8", layer, rank):
            P.linear(P.attn_output_rows(o), lw["o_bf16"], out=ws.partial)
        self._output("S8", layer, rank, partial=ws.partial)

    def _dense(self, rank: int, rows: int) -> None:
        """Layer 0: D24's norm after its all-reduce, then M1 - M3 into ``x_full``."""
        ws, lw = self._ws[rank][rows], self.weights[rank].layer[0]
        with self._stage("D24(norm)", 0, rank):
            P.add_rmsnorm_(ws.partial, ws.res_full, lw["ffn_gamma_bf16"])
        self._output("D24(norm)", 0, rank, hidden=ws.partial, residual=ws.res_full)
        with self._stage("M1", 0, rank):
            P.linear(ws.partial, lw["dense_gate_up"], out=ws.gate_up)
        self._output("M1", 0, rank, gate_up=ws.gate_up)
        with self._stage("M2", 0, rank):
            P.silu_and_mul(ws.gate_up, out=ws.act)
        self._output("M2", 0, rank, act=ws.act)
        with self._stage("M3", 0, rank):
            P.linear(ws.act, lw["dense_down_bf16"], out=ws.x_full)
        self._output("M3", 0, rank, partial=ws.x_full)

    def _moe_input(self, rank: int, layer: int, rows: int) -> None:
        """The post norm on this rank's rows, X1 - X2, and X3's quantise-and-pack."""
        ws, lw = self._ws[rank][rows], self.weights[rank].layer[layer]
        norm = ("D24'" if self.dsa[layer] else "S9") + "(norm)"
        with self._stage(norm, layer, rank):
            if layer == 1:
                # FULL -> SCATTERED: this rank keeps its slice of the residual
                ws.res_part.copy_(ws.res_full[rank * ws.moe_rows:(rank + 1) * ws.moe_rows])
            P.add_rmsnorm_(ws.x_part, ws.res_part, lw["ffn_gamma_bf16"])
        self._output(norm, layer, rank, hidden=ws.x_part, residual=ws.res_part)
        with self._stage("X1", layer, rank):
            P.router_logits(ws.x_part, lw["router"], out=ws.router)
        self._output("X1", layer, rank, logits=ws.router)
        with self._stage("X2", layer, rank):
            weights, ids = P.route(ws.router, lw["router_bias"])
        self._output("X2", layer, rank, weights=weights, ids=ids)
        with self._stage("X3(pack)", layer, rank):
            P.quantize_dispatch_rows(ws.x_part, ws.q_rows, ws.s_rows)
            P.pack_dispatch_rows(ws.q_rows, ws.s_rows, ids, weights, ws.packed)
        self._output("X3(pack)", layer, rank, codes=ws.q_rows, scales=ws.s_rows, packed=ws.packed)

    def _experts(self, rank: int, layer: int, rows: int) -> torch.Tensor:
        """This card's 32 experts over every rank's gathered rows, and its fp32 X7 partial."""
        ws, lw, e = self._ws[rank][rows], self.weights[rank].layer[layer], P.EXPERTS_PER_RANK
        with self._stage("X3(layout)", layer, rank):
            fields = P.dispatch_rows(ws.gathered)
            slots = P.expert_slots(fields.ids, first_expert=lw["ep_first_expert"])
            P.scatter_to_experts(fields.codes, fields.scales, slots, ws.q_buf, ws.s_buf)
        self._output("X3(layout)", layer, rank, codes=ws.q_buf[:e], scales=ws.s_buf[:e],
                     masked_m=slots.masked_m)
        (w13, w13_scale), (w2, w2_scale) = lw["ep_w13"], lw["ep_w2"]
        with self._stage("X4", layer, rank):
            P.grouped_gemm(ws.q_buf[:e], ws.s_buf[:e], w13, w13_scale, ws.up, slots.masked_m,
                           ws.expected)
        self._output("X4", layer, rank, up=ws.up)
        with self._stage("X5", layer, rank):
            P.act_quant_masked(ws.up, ws.q_act, ws.s_act, slots.masked_m)
        self._output("X5", layer, rank, codes=ws.q_act, scales=ws.s_act)
        with self._stage("X6", layer, rank):
            P.grouped_gemm(ws.q_act, ws.s_act, w2, w2_scale, ws.dn[:e], slots.masked_m, ws.expected)
        self._output("X6", layer, rank, down=ws.dn[:e])
        with self._stage("X7(combine)", layer, rank):
            P.combine_partial(ws.dn, slots, fields.weights, ws.combine)
        self._output("X7(combine)", layer, rank, partial=ws.combine)
        return fields.ids

    def _final(self, rank: int, rows: int) -> None:
        """The last MoE's rows in bf16, then F1's residual add (before its all-gather)."""
        ws = self._ws[rank][rows]
        with self._stage("X7(cast)", self.layers - 1, rank):
            ws.x_part.copy_(ws.moe_out)
        self._output("X7(cast)", self.layers - 1, rank, hidden=ws.x_part)
        with self._stage("F1(add)", None, rank):
            P.add_residual_(ws.x_part, ws.res_part)
        self._output("F1(add)", None, rank, hidden=ws.x_part)

    def _head(self, rank: int, chunk: Chunk) -> None:
        ws, w = self._ws[rank][chunk.bucket], self.weights[rank]
        with self._stage("F2", None, rank):
            P.rmsnorm(ws.x_full, w.head_gamma_bf16, out=ws.partial)
        self._output("F2", None, rank, hidden=ws.partial)
        with self._stage("F3", None, rank):
            if self.graphs:
                P.last_row_into(ws.partial, self._meta[rank][chunk.bucket].last,
                                self.last_hidden[rank])
            else:
                self.last_hidden[rank].copy_(P.last_row(ws.partial, chunk.valid))
        self._output("F3", None, rank, last_hidden=self.last_hidden[rank])

    # ------------------------------------------------------------------------- a chunk
    def _set_chunk(self, chunk: Chunk, tokens) -> None:
        """Per-chunk metadata, outside every stage: ``ChunkMeta.set`` waits on its last copy."""
        span, end = chunk.prefix + chunk.bucket, chunk.start + chunk.valid
        for rank in range(self.ranks):
            with torch.cuda.device(self.devices[rank]), torch.cuda.stream(self.streams[rank]):
                self._meta[rank][chunk.bucket].set(chunk, tokens)
                ws = self._ws[rank][chunk.bucket]
                ws.cu_swa[1:].fill_(span)
                ws.cu_dsa[1:].fill_(end)

    def _record(self, record, number, kind, layer, rows) -> None:
        with torch.cuda.device(self.devices[0]), torch.cuda.stream(self.streams[0]):
            record(number, kind, layer, rows.clone())

    def _forward(self, number: int, chunk: Chunk, tokens, *, head: bool, record,
                 graphs: bool) -> None:
        rows, valid, end = chunk.bucket, chunk.valid, chunk.start + chunk.valid
        ws = [self._ws[rank][rows] for rank in range(self.ranks)]
        self._set_chunk(chunk, tokens)
        self._segment(("E1", rows), lambda r: self._embed(r, rows), graphs=graphs)
        self._all_reduce("E2", None, "hidden", [s.res_full for s in ws])
        for layer in range(self.layers):
            dsa = self.dsa[layer]
            if layer >= 2:
                self._segment(("attention_input", layer, rows),
                              lambda r, layer=layer: self._attention_input(r, layer, rows),
                              graphs=graphs)
                self._all_gather("S1b", layer, "hidden", [s.x_part for s in ws], [s.x_full for s in ws])
            # The one extent a capture of this layer's attention bakes in: a DSA layer bounds
            # the visible caches at ``end``, a SWA layer splits its staging at ``prefix``.
            selected = self._segment(("attention", layer, rows, end if dsa else chunk.prefix),
                                     lambda r, layer=layer: self._attention(r, layer, chunk),
                                     graphs=graphs)
            if dsa:
                for w in self.weights:
                    w.layer[layer]["ki_plain_end"] = end
                if record is not None:
                    self._record(record, number, "dsa", layer, selected[0][:valid])
            if layer == 0:
                self._all_reduce("D24(all_reduce)", layer, "partial", [s.partial for s in ws])
                self._segment(("dense", rows), lambda r: self._dense(r, rows), graphs=graphs)
                self._all_reduce("M4", layer, "hidden", [s.x_full for s in ws])
                continue
            self._reduce_scatter(("D24'" if dsa else "S9") + "(reduce_scatter)", layer, "hidden",
                                 [s.partial for s in ws], [s.x_part for s in ws])
            self._segment(("moe_input", layer, rows),
                          lambda r, layer=layer: self._moe_input(r, layer, rows), graphs=graphs)
            self._all_gather("X3(all_gather)", layer, "packed", [s.packed for s in ws],
                             [s.gathered for s in ws])
            ids = self._segment(("experts", layer, rows),
                                lambda r, layer=layer: self._experts(r, layer, rows), graphs=graphs)
            if record is not None:
                self._record(record, number, "moe", layer, ids[0][:valid])
            self._reduce_scatter("X7(reduce_scatter)", layer, "hidden", [s.combine for s in ws],
                                 [s.moe_out for s in ws])
        if head:
            self._segment(("final", rows), lambda r: self._final(r, rows), graphs=graphs)
            self._all_gather("F1(all_gather)", None, "hidden", [s.x_part for s in ws],
                             [s.x_full for s in ws])
            self._segment(("head", rows), lambda r: self._head(r, chunk), graphs=graphs)

    @torch.inference_mode()
    def run(self, tokens, start=0, *, graphs=None, record=None, tail=True):
        """Prefill ``tokens`` at ``start``; ``last_hidden`` holds the last row when ``tail``."""
        if torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32:
            raise PrefillError("the kernel prefill requires autocast and TF32 disabled")
        # Which form the stage bodies use was fixed at construction, so a run may only turn
        # capture off; turning it on here would capture the host integers of one chunk.
        use_graphs = self.graphs if graphs is None else bool(graphs)
        if use_graphs and not self.graphs:
            raise PrefillError("this executor was built with graphs=False; construct it with "
                               "graphs=True to capture the kernel prefill")
        toks = list(tokens)
        if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < P.VOCAB for t in toks):
            raise PrefillError("Invalid Naive N0.5 prompt token IDs")
        if start < 0 or start + len(toks) > self.max_positions:
            raise PrefillError("Naive N0.5 prefill exceeds cache capacity")
        self._stats = dict(tokens=len(toks), start=start, chunks=[], graphs=use_graphs,
                           plan_config=self.plan_config.as_dict(),
                           precision=self.weights[0].precision, kernels="sglang_m66",
                           swa_backend="fa4_cute", execution_order="chunk_major",
                           moe="ep8_masked_grouped_gemm", graph_replays=0, graph_captures=0,
                           eager_segments=0, collectives=0, rebuilt_index_keys=0,
                           eager_only=self.eager_only)
        for w in self.weights:
            for layer in self.dsa_layers:
                covered = w.layer[layer].get("ki_plain_end", 0)
                if covered < start:
                    raise PrefillError(INDEX_KEY_REFUSAL.format(start=start, covered=covered))
        chunks = self.plan_config.plan(len(toks), start=start)
        for number, chunk in enumerate(chunks):
            offset = chunk.start - start
            self._forward(number, chunk, toks[offset:offset + chunk.valid],
                          head=tail and number == len(chunks) - 1, record=record,
                          graphs=use_graphs)
        self._stats["chunks"] = [
            dict(start=chunk.start, valid=chunk.valid, bucket=chunk.bucket,
                 computed_rows=chunk.bucket, prefix=chunk.prefix, scope="chunk")
            for chunk in chunks]
        self._stats["eager_only"] = {repr(key): value for key, value in self.eager_only.items()}
        return self._stats

    @property
    def plan_config(self) -> PrefillPlanConfig:
        return self._plan_config

    @property
    def buckets(self) -> tuple[int, ...]:
        return self.plan_config.buckets


__all__ = ["INDEX_KEY_REFUSAL", "PrefillExecutor", "SWA_ROPE_IMAG", "StageObserver"]
