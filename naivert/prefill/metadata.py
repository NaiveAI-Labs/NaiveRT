"""Device metadata for external prefill: ragged offsets, positions and padding.

Built by naivert from ``(start, valid, bucket)``, never imported from sglang's scheduler
(no ``ForwardBatch``, no ``ReqToTokenPool``). ``planning.py`` owns chunk policy and buckets;
the default buckets are 256 / 1024 / 4096, static shapes, pad to the bucket and mask with
``valid``, so nothing a graph reads from here is a host value baked in at capture and these
buffers alone never tie a graph to one chunk.  What does tie it is the caller's segment key:
``executor.py``'s attention key carries the chunk extent (``end`` for a DSA layer,
``chunk.prefix`` for a SWA one), so a bucket driven at two different extents captures twice.

Building a chunk's tensors on the host from ``(start, valid)`` bakes that chunk into whatever
captures them, which is why this module has no such helper.  ``ChunkMeta`` is the form a
replayed graph can use: persistent buffers that one kernel rewrites from ``(start, valid)``
before each chunk (plan section 5a).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

# Re-export the established metadata API; planning itself has no GPU imports.
from naivert.prefill.planning import BUCKETS, WINDOW, Chunk, plan_chunks


RING_ROWS = 136


@triton.jit
def _chunk_meta(inp, tokens, positions, src, ke, prefix_rows, publish_rows, publish_src, last,
                rows, WINDOW: tl.constexpr, RING: tl.constexpr, BLOCK: tl.constexpr):
    # inp = [tokens[0:rows], start, valid], int64.  Row i past valid is padding: it reads the
    # last real row's token and position, so every per-row value below stays in range.
    pid = tl.program_id(0)
    start = tl.load(inp + rows)
    valid = tl.load(inp + rows + 1)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    inb = i < rows
    s = tl.minimum(i, valid - 1)
    tl.store(tokens + i, tl.load(inp + s, mask=inb, other=0), mask=inb)
    tl.store(src + i, s, mask=inb)
    tl.store(positions + i, start + s, mask=inb)
    tl.store(ke + i, (start + s + 1).to(tl.int32), mask=inb)
    if pid == 0:
        j = tl.arange(0, WINDOW)
        n = tl.minimum(start, WINDOW - 1)
        tl.store(prefix_rows + j, (start - n + j) % RING, mask=j < WINDOW - 1)
        p = tl.maximum(valid - WINDOW + j, 0)
        tl.store(publish_src + j, p)
        tl.store(publish_rows + j, (start + p) % RING)
        tl.store(last, valid - 1)


class ChunkMeta:
    """One bucket's per-chunk metadata, in buffers a captured graph reads by address.

    ``set`` copies ``[tokens, start, valid]`` into the device once and one kernel derives the
    rest there, on the current stream -- so nothing a graph reads from these buffers is a host
    value baked in at capture.  That is this class's guarantee; whether one graph then covers
    every chunk of its bucket is the caller's segment key, which in ``executor.py`` also carries
    the chunk extent:

    * ``tokens`` / ``positions`` ``[rows]`` int64: row ``i``'s token and absolute position.
    * ``src`` ``[rows]`` int64: ``min(i, valid - 1)``.  A padded row stands in for the last
      real one, so the positions of every row stay inside the tables and caches, and a
      scatter through ``src`` writes a padded row's slot with the bytes that slot gets anyway.
    * ``ks`` / ``ke`` ``[rows]`` int32: the DSA key range ``[0, start + i]`` -- a DSA layer is
      causal over the whole prefix, not a window -- with padded rows clamped to
      ``start + valid``.  An unclamped ``ke`` would point the indexer at cache rows this chunk
      never wrote: uninitialised memory, and on a fresh session a zero row that would still be
      *selected*.
    * ``prefix_rows`` ``[WINDOW - 1]``: the ring rows of positions ``[start - prefix, start)``,
      oldest first (``Chunk.prefix`` of them are live).
    * ``publish_rows`` / ``publish_src`` ``[WINDOW]``: ring row and chunk row of the chunk's
      last ``min(valid, WINDOW)`` positions; entries short of ``WINDOW`` repeat row 0, a
      duplicate slot written with identical bytes.
    * ``last`` ``[1]``: ``valid - 1``.
    """

    BLOCK = 1024

    def __init__(self, rows: int, device: torch.device, *, max_positions: int,
                 window: int = WINDOW, ring_rows: int = RING_ROWS) -> None:
        self.rows, self.device = int(rows), torch.device(device)
        self.max_positions, self.window, self.ring_rows = int(max_positions), int(window), int(ring_rows)
        i64 = dict(dtype=torch.int64, device=self.device)
        self._host = torch.zeros(self.rows + 2, dtype=torch.int64, pin_memory=True)
        self._host_np = self._host.numpy()
        self._inp = torch.zeros(self.rows + 2, **i64)
        self.tokens = torch.zeros(self.rows, **i64)
        self.positions = torch.zeros(self.rows, **i64)
        self.src = torch.zeros(self.rows, **i64)
        self.ks = torch.zeros(self.rows, dtype=torch.int32, device=self.device)
        self.ke = torch.ones(self.rows, dtype=torch.int32, device=self.device)
        self.prefix_rows = torch.zeros(self.window - 1, **i64)
        self.publish_rows = torch.zeros(self.window, **i64)
        self.publish_src = torch.zeros(self.window, **i64)
        self.last = torch.zeros(1, **i64)
        with torch.cuda.device(self.device):
            self._copied = torch.cuda.Event()
        self.chunk: Chunk | None = None

    def set(self, chunk: Chunk, tokens: Sequence[int] | None = None) -> None:
        """Point the buffers at ``chunk``; ``tokens`` are its ``valid`` real tokens.

        Without ``tokens`` a repeat of the current chunk is a no-op, which is what lets a
        per-layer caller run under a capture that its warm-up already set up.
        """
        if tokens is None and chunk == self.chunk:
            return
        if int(chunk.bucket) != self.rows:
            raise ValueError(f"ChunkMeta holds {self.rows} rows, the chunk is bucket {chunk.bucket}")
        if not 1 <= int(chunk.valid) <= self.rows or int(chunk.start) < 0:
            raise ValueError(f"chunk {chunk} has no real row in {self.rows}")
        if int(chunk.start) + int(chunk.valid) > self.max_positions:
            raise ValueError(f"chunk {chunk} ends past the {self.max_positions} positions the "
                             "caches and rope tables hold")
        with torch.cuda.device(self.device):
            # The previous copy out of the pinned block has to have happened before the block
            # is rewritten; one chunk later it long has.
            self._copied.synchronize()
            if tokens is not None:
                if len(tokens) != int(chunk.valid):
                    raise ValueError(f"{len(tokens)} tokens for a chunk of {chunk.valid}")
                self._host_np[:chunk.valid] = np.asarray(tokens, dtype=np.int64)
            self._host_np[self.rows] = int(chunk.start)
            self._host_np[self.rows + 1] = int(chunk.valid)
            self._inp.copy_(self._host, non_blocking=True)
            self._copied.record()
            _chunk_meta[(triton.cdiv(self.rows, self.BLOCK),)](
                self._inp, self.tokens, self.positions, self.src, self.ke, self.prefix_rows,
                self.publish_rows, self.publish_src, self.last, self.rows,
                WINDOW=self.window, RING=self.ring_rows, BLOCK=self.BLOCK)
        self.chunk = chunk
