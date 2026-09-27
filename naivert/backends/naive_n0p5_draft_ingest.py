"""Batched draft context KV ingest: D1/D2 over thousands of prompt rows at once.

The draft's context K/V for a prompt position is a function of that position alone: the eight
target residual features (one target layer per rank) go through each rank's FC column shard,
are summed over the ranks and RMS-normalised, then projected into five K (head-normed, RoPE'd)
and five V rows. ``DraftRound.initialize`` runs that through the eight-row ``fc_ctx_kv`` kernel
one block at a time (about 0.065 ms per prompt token on eight B300s: 300 ms for a 4.5K prompt).
This class does the same arithmetic over up to ``rows`` positions per step with cuBLAS matmuls
and one NCCL all-reduce on the ranks' own streams, into the same ``kv_k`` / ``kv_v`` caches at
the same rows (cache row == position). Rounding points follow the kernel (BF16 partial, BF16
fold result, BF16 normalised context, BF16 K before the head norm and again before RoPE); the
accumulation order inside each matmul and the all-reduce differ, so the result is within BF16
rounding of the kernel's, not bitwise (``tests/gpu/naive_n0p5_draft/round/check_draft_ingest.py``).
"""
from __future__ import annotations

import os
import time

import torch

from naivert.models.naive_n0p5.dspark import RMS_EPSILON
from naivert.prefill.comms import TP8Comm

HIDDEN, HEAD, LAYERS, RANKS = 4096, 128, 5, 8


def unpack_fc(packed: torch.Tensor) -> torch.Tensor:
    """The row-major ``[4096 out, 4096 in]`` FC shard from ``dspark_context.pack_weights``' tiles."""
    if tuple(packed.shape) != (256, 128, 16, 32):
        raise ValueError(f"packed FC shard is {tuple(packed.shape)}, expected (256, 128, 16, 32)")
    return packed.permute(0, 2, 1, 3).reshape(HIDDEN, HIDDEN).contiguous()


class DraftContextIngest:
    """Write draft context K/V for a range of prompt positions from the target feature rows.

    Every intermediate lives in workspaces sized for ``rows`` positions and the step works with
    in-place and ``out=`` ops, so nothing is allocated per request (the caching allocator's
    fresh-size path costs tens of milliseconds). Each step also computes on a fixed ``rows`` rows
    whenever the feature buffer and the cache allow it, writing only the valid ones: a new row
    count otherwise pays 2-4 ms of cuBLAS algorithm selection and lazy kernel loading in the
    server. The constructor warms a full step on zero features; a step near the end of
    the feature buffer or cache can use fewer rows."""

    def __init__(self, core, streams, *, comm=None, rows=4096, min_rows=64, warm=True):
        self.core, self.buffers = core, tuple(core.buffers)
        self.streams = list(streams)
        self.rows, self.min_rows = int(rows), int(min_rows)
        if len(self.buffers) != RANKS or len(self.streams) != RANKS or self.rows < 1:
            raise ValueError("eight draft buffers, eight streams and a positive row count")
        self.devices = [b.device for b in self.buffers]
        for s, d in zip(self.streams, self.devices):
            if s.device != d:
                raise ValueError(f"stream {s} is not on {d}")
        self.capacity = int(self.buffers[0].capacity)
        self.comm = comm
        if self.comm is None:
            self.comm = TP8Comm(self.devices, self.streams)
            self.comm.warmup()
        self.fc, self.k, self.v, self.k_norm, self.gamma, self.ws = [], [], [], [], [], []
        R = self.rows
        for w, b in zip(core.weights, self.buffers):
            c = w.context
            if len(c["k"]) != LAYERS or any(tuple(t.shape) != (HEAD, HIDDEN) for t in (*c["k"], *c["v"])) or c["hidden_norm"].dtype != torch.float32:
                raise ValueError("unexpected draft context weight layout")
            with torch.cuda.device(b.device):
                self.fc.append(unpack_fc(c["fc"]))
                # the five layers' K (and V) projections as one [640, 4096] matmul each
                self.k.append(torch.cat(list(c["k"]), dim=0).contiguous())
                self.v.append(torch.cat(list(c["v"]), dim=0).contiguous())
                self.k_norm.append(torch.stack(list(c["k_norm"]), dim=0).float().contiguous())   # [5, 128]
                self.gamma.append(c["hidden_norm"])
                dev = b.device
                f32 = dict(dtype=torch.float32, device=dev)
                bf = dict(dtype=torch.bfloat16, device=dev)
                self.ws.append(dict(
                    p16=torch.empty(R, HIDDEN, **bf), p32=torch.empty(R, HIDDEN, **f32),
                    c16=torch.empty(R, HIDDEN, **bf), c32=torch.empty(R, HIDDEN, **f32), y16=torch.empty(R, HIDDEN, **bf),
                    k16=torch.empty(R, LAYERS * HEAD, **bf), k32=torch.empty(R, LAYERS, HEAD, **f32),
                    kn16=torch.empty(R, LAYERS, HEAD, **bf), rot=torch.empty(R, LAYERS, HEAD, **f32),
                    v16=torch.empty(R, LAYERS * HEAD, **bf), cos=torch.empty(R, HEAD, **f32), sin=torch.empty(R, HEAD, **f32),
                    pos=torch.arange(self.capacity, dtype=torch.int64, device=dev).clamp_(max=b.rope_cos.shape[0] - 1)))
        if warm:
            zeros = [torch.zeros(self.rows, HIDDEN, dtype=torch.bfloat16, device=d) for d in self.devices]
            self.ingest(zeros, 0, min(self.rows, self.capacity))
            for s in self.streams:
                s.synchronize()

    def _sync_all(self):
        for s in self.streams:
            s.synchronize()

    @torch.inference_mode()
    def ingest(self, features, start, end, timing=None):
        """Positions ``[start, end)``: ``features[r][start:end]`` on every rank -> ``kv_k`` / ``kv_v`` rows.

        ``timing`` (a list) receives per-step ms for the matmul launches, the all-reduce and the rest, each
        measured up to a full synchronisation, only when NAIVERT_INGEST_TIMING=1 (diagnostics: the syncs
        serialise the step)."""
        start, end = int(start), int(end)
        if not 0 <= start <= end <= self.capacity:
            raise ValueError(f"ingest range [{start}, {end}) outside the cache capacity {self.capacity}")
        fine = timing is not None and os.environ.get("NAIVERT_INGEST_TIMING") == "1"
        avail = min(int(f.shape[0]) for f in features)
        for s in range(start, end, self.rows):
            e = min(end, s + self.rows)
            n = e - s                                   # rows written this step
            m = min(self.rows, self.capacity - s, avail - s)   # rows computed: the full step when possible
            if m < n:
                raise ValueError(f"features hold {avail} rows, ingest needs [{s}, {e})")
            parts = []
            if fine:
                self._sync_all()
                t0 = time.perf_counter()
            for r in range(RANKS):
                w = self.ws[r]
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    x = features[r][s:s + m]
                    if x.dtype != torch.bfloat16 or x.shape != (m, HIDDEN):
                        raise ValueError(f"rank {r}: features[{s}:{s + m}] is {x.dtype} {tuple(x.shape)}")
                    torch.matmul(x, self.fc[r].t(), out=w["p16"][:m])       # BF16 out, FP32 accumulation: the rounded local partial
                    w["p32"][:m].copy_(w["p16"][:m])
                    parts.append(w["p32"][:m])
            if fine:
                self._sync_all()
                t1 = time.perf_counter()
            self.comm.all_reduce_(parts)                                    # the TP8 fold, summed in FP32
            if fine:
                self._sync_all()
                t2 = time.perf_counter()
            for r in range(RANKS):
                b, w = self.buffers[r], self.ws[r]
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    c16, c32, y16 = w["c16"][:m], w["c32"][:m], w["y16"][:m]
                    c16.copy_(parts[r])                                     # the fold result is rounded to BF16
                    c32.copy_(c16)
                    scale = torch.rsqrt(torch.linalg.vector_norm(c32, dim=-1, keepdim=True).square_().div_(HIDDEN).add_(RMS_EPSILON))
                    c32.mul_(scale).mul_(self.gamma[r])
                    y16.copy_(c32)                                          # the normalised context, BF16
                    pos = w["pos"][s:s + m]
                    cos, sin = w["cos"][:m], w["sin"][:m]
                    torch.index_select(b.rope_cos, 0, pos, out=cos)
                    torch.index_select(b.rope_sin, 0, pos, out=sin)
                    k16, k32, kn16, rot = w["k16"][:m], w["k32"][:m], w["kn16"][:m], w["rot"][:m]
                    torch.matmul(y16, self.k[r].t(), out=k16)              # BF16 raw K for the five layers
                    k32.copy_(k16.view(m, LAYERS, HEAD))
                    ks = torch.rsqrt(torch.linalg.vector_norm(k32, dim=-1, keepdim=True).square_().div_(HEAD).add_(RMS_EPSILON))
                    k32.mul_(ks).mul_(self.k_norm[r])
                    kn16.copy_(k32)                                         # rounded before RoPE
                    k32.copy_(kn16)
                    rot[..., :HEAD // 2].copy_(k32[..., HEAD // 2:]).neg_()
                    rot[..., HEAD // 2:].copy_(k32[..., :HEAD // 2])
                    k32.mul_(cos.unsqueeze(1)).addcmul_(rot, sin.unsqueeze(1))
                    b.kv_k[:, s:e, 0, :].copy_(k32[:n].transpose(0, 1))
                    v16 = w["v16"][:m]
                    torch.matmul(y16, self.v[r].t(), out=v16)
                    b.kv_v[:, s:e, 0, :].copy_(v16.view(m, LAYERS, HEAD)[:n].transpose(0, 1))
            if fine:
                self._sync_all()
                t3 = time.perf_counter()
                timing.append(dict(rows=n, computed=m, matmul_ms=round((t1 - t0) * 1e3, 2), allreduce_ms=round((t2 - t1) * 1e3, 2),
                                   rest_ms=round((t3 - t2) * 1e3, 2)))


__all__ = ["DraftContextIngest", "unpack_fc"]
