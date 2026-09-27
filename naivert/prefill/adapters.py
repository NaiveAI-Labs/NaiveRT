"""Cache-publish glue for the external prefill path (plan section 4).

sglang has no kernel that writes naivert's caches: its ``set_kv_buffer`` scatters into
paged pools through ``req_to_token``, while naivert's SWA cache is a 136-row ring indexed
by ``position % 136`` and its DSA caches are position-indexed contiguous tensors.  These
adapters are the copies that land external-kernel outputs into that layout.  They are
torch index ops on purpose: they become CUDA only if a profile shows the glue is
significant, and then they move under ``naivert/body/prefill_adapters/``.

Nothing here computes numerics.  Everything published was produced upstream, in the byte
layout decode already uses: by the kernel prefill's projection, rope and value-scale stages
(``naivert.kernels.naive_n0p5.prefill``, driven by ``naivert.prefill.executor``) or by the
retained executors' ops.

Each publish comes twice.  ``swa_ring_publish`` / ``dsa_cache_append`` take the chunk as host
integers, which a captured graph would bake in; ``swa_ring_write`` / ``dsa_cache_write`` take
the rows and positions as device tensors (``metadata.ChunkMeta``), which is the form the
model's replayed graphs use.  They write the same bytes.
"""
from __future__ import annotations

import torch

# A SWA layer's ring: row = position % RING_ROWS, and a query at p attends [p-127, p].
# ``weights.allocate_caches`` sizes it; ``decode_modes.swa_ring_resumable_position``
# decides when a resumed session may reuse it.
RING_ROWS = 136
WINDOW = 128


class AdapterError(RuntimeError):
    """A publish that would corrupt a cache -- padded rows, a short source, a bad ring."""


def swa_ring_publish(lw: dict, k: torch.Tensor, v: torch.Tensor, start: int, valid: int,
                     *, ring_rows: int = RING_ROWS, window: int = WINDOW) -> int:
    """Publish the chunk's last ``min(valid, window)`` rows into the SWA ring.

    ``k``/``v`` hold the chunk's rows for positions ``start .. start + valid - 1`` in the
    byte layout decode writes (rope already applied, bf16).  Rows at or beyond ``valid``
    are padding and are never written: a padded row landing in the ring corrupts the
    window of every later query that reads it.

    Only the last ``window`` rows are published because only they are reachable: a query
    at ``p`` reads ``[p - window + 1, p]``, so once the chunk is over, every earlier row
    of it is already out of every future window.  Those rows are ``window`` consecutive
    positions, hence ``window`` distinct ring rows out of ``ring_rows`` -- no duplicate
    index, so the ``index_copy_`` is deterministic.

    Returns how many rows were published.
    """
    start, valid = int(start), int(valid)
    if valid < 0 or valid > int(k.shape[0]) or valid > int(v.shape[0]):
        raise AdapterError(f"valid={valid} is not covered by k {tuple(k.shape)} / v {tuple(v.shape)}")
    if window > ring_rows:
        raise AdapterError(f"window {window} exceeds the ring's {ring_rows} rows")
    keep = min(valid, window)
    if keep == 0:
        return 0
    first = start + valid - keep
    rows = (torch.arange(first, first + keep, device=k.device) % ring_rows).long()
    lw["k_cache"].index_copy_(0, rows, k[valid - keep:valid])
    lw["v_cache"].index_copy_(0, rows, v[valid - keep:valid])
    return keep


def swa_ring_write(lw: dict, k: torch.Tensor, v: torch.Tensor, rows: torch.Tensor,
                   src: torch.Tensor) -> None:
    """``swa_ring_publish`` with the ring rows and chunk rows on the device.

    ``rows`` / ``src`` are ``ChunkMeta.publish_rows`` / ``publish_src``: ``WINDOW`` entries,
    the last ``min(valid, WINDOW)`` real rows and, for a chunk shorter than the window,
    repeats of row 0.  A repeat lands in row 0's slot with row 0's bytes, so the duplicate
    index writes one value and the ``index_copy_`` stays deterministic.
    """
    lw["k_cache"].index_copy_(0, rows, k.index_select(0, src))
    lw["v_cache"].index_copy_(0, rows, v.index_select(0, src))


def swa_ring_rows(start: int, n: int, device: torch.device, *, ring_rows: int = RING_ROWS
                  ) -> torch.Tensor:
    """The ``n`` ring rows holding positions ``[start - n, start)``, oldest first."""
    return (torch.arange(start - n, start, device=device) % ring_rows).long()


def swa_ring_prefix_into(lw: dict, start: int, k_out: torch.Tensor, v_out: torch.Tensor, *,
                         ring_rows: int = RING_ROWS, window: int = WINDOW) -> int:
    """Gather the ring rows a chunk at ``start`` still attends into caller-owned staging.

    ``k_out`` / ``v_out`` are ``[n, 1, D]`` views of the kv buffer the attention call reads,
    so the gather lands where it is needed and nothing is concatenated afterwards.  ``n`` is
    ``min(start, window - 1)``: positions ``[start - n, start)``, the live part of a query's
    window that this chunk's own rows do not cover.
    """
    start = int(start)
    n = min(start, window - 1)
    if int(k_out.shape[0]) != n or int(v_out.shape[0]) != n:
        raise AdapterError(f"prefix staging holds {tuple(k_out.shape)} rows, the ring owes {n}")
    if n == 0:
        return 0
    rows = swa_ring_rows(start, n, k_out.device, ring_rows=ring_rows)
    torch.index_select(lw["k_cache"], 0, rows, out=k_out.view(n, -1))
    torch.index_select(lw["v_cache"], 0, rows, out=v_out.view(n, -1))
    return n


def swa_ring_prefix_rows(lw: dict, rows: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor) -> int:
    """``swa_ring_prefix_into`` with the ring rows on the device.

    ``rows`` is the live part of ``ChunkMeta.prefix_rows`` -- entry ``j`` is the ring row of
    position ``start - n + j`` -- so which rows are read is a device value and only *how many*
    is a shape.  ``swa_ring_prefix_into`` derives the same rows from a host ``start``, which a
    capture would bake in; these two gather the same bytes.
    """
    n = int(rows.shape[0])
    if int(k_out.shape[0]) != n or int(v_out.shape[0]) != n:
        raise AdapterError(f"prefix staging holds {tuple(k_out.shape)} rows, the ring owes {n}")
    if n == 0:
        return 0
    torch.index_select(lw["k_cache"], 0, rows, out=k_out.view(n, -1))
    torch.index_select(lw["v_cache"], 0, rows, out=v_out.view(n, -1))
    return n


# A DSA layer's caches are position-indexed and append-only: k [max_seq, 192] bf16 (roped),
# v [max_seq, 128] bf16, ki [max_seq, 128] e4m3 + one fp32 scale a row.  ``weights.
# allocate_caches`` sizes them; ``attn_dsa_body.cuh`` reads them in decode.
INDEX_DIM = 128
INDEX_CHUNK = 16                     # bytes of one ki row's UMMA chunk; 8 chunks a row


def dsa_swizzle_rows(codes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """decode's index-key cache image: the 16-byte chunk index XORed with ``position & 7``.

    ``attn_dsa_body.cuh:485`` (``publish_ki``) does not store the plain 128 e4m3 codes; it
    stores the UMMA K-major SW128 swizzle of them, so that the decode kernel's ``tcgen05``
    load of the cache needs no further permutation.  Prefill has to write the same bytes or
    the first decode step after it reads a scrambled key.

    The map is an involution -- applying it to a stored row gives the plain row back -- and
    it is the *only* thing standing between the cache and DeepGEMM's ``fp8_mqa_logits``,
    which wants plain rows.  Undoing it per chunk is not affordable (a gather over 4096 rows
    measured 63 us, and the indexer re-reads the whole visible prefix every chunk, so at 9
    DSA layers a 64K prefix would cost several ms of pure permutation a chunk), which is why
    ``dsa_cache_append`` keeps the plain codes in a second buffer instead: see
    ``dsa_index_keys``.
    """
    n = int(codes.shape[0])
    if int(codes.shape[-1]) != INDEX_DIM:
        raise AdapterError(f"index keys must be {INDEX_DIM} wide, got {tuple(codes.shape)}")
    if int(positions.shape[0]) != n:
        raise AdapterError(f"{n} rows but {tuple(positions.shape)} positions")
    chunks = INDEX_DIM // INDEX_CHUNK                                  # 8
    j = torch.arange(chunks, device=codes.device)[None, :] ^ (positions.to(torch.long)[:, None] % chunks)
    u8 = codes.view(torch.uint8).reshape(n, chunks, INDEX_CHUNK)
    return u8.gather(1, j[..., None].expand(n, chunks, INDEX_CHUNK)).reshape(n, INDEX_DIM).view(codes.dtype)


def dsa_index_keys(lw: dict, capacity: int, *, dim: int = INDEX_DIM) -> torch.Tensor:
    """The plain-e4m3 shadow of ``ki_cache`` that the prefill indexer reads, allocated once.

    ``ki_cache`` holds the swizzled image decode needs; ``fp8_mqa_logits`` wants the plain
    rows.  Rather than permute the whole visible prefix back on every chunk of every layer,
    prefill keeps the plain codes it has just computed -- 128 B a token a layer, 72 MiB per
    rank for a 64K prompt over the nine DSA layers, against the 8.4 GiB the three real
    caches already take at that length.

    It shadows the cache only over ``[0, lw["ki_plain_end"])``.  Decode appends to
    ``ki_cache`` alone, so the mark is what separates the rows prefill wrote from the ones
    it did not: ``dsa_index_keys_clamp`` pulls it back to a request's start (decode is about
    to overwrite everything past that), and ``dsa_index_keys_rebuild`` fills the rows decode
    wrote in between from ``ki_cache`` itself.  A new buffer covers nothing, so allocating
    one resets the mark.
    """
    buf = lw.get("ki_plain")
    if buf is None or int(buf.shape[0]) < int(capacity):
        buf = torch.zeros((int(capacity), int(dim)), dtype=torch.float8_e4m3fn,
                          device=lw["ki_cache"].device)
        lw["ki_plain"] = buf
        lw["ki_plain_end"] = 0
    return buf


def dsa_cache_append(lw: dict, k: torch.Tensor, v: torch.Tensor, ki: torch.Tensor,
                     ki_scale: torch.Tensor, start: int, valid: int, *,
                     plain: torch.Tensor | None = None) -> int:
    """Append this chunk's real rows to a DSA layer's position-indexed caches.

    ``k`` ``[rows, 192]`` and ``v`` ``[rows, 128]`` are bf16 in decode's byte layout (rope
    applied to k; v already multiplied by the 0.707 value scale, ``prefill.scale_values``
    D4 / S5); ``ki`` ``[rows, 128]`` is
    the **plain** e4m3 quantisation of the index key and ``ki_scale`` ``[rows]`` its fp32
    per-row scale (row amax / 448).  Rows at or beyond ``valid`` are padding and are never
    written: unlike the SWA ring they would not even be overwritten later, since the next
    chunk starts at ``start + valid``.

    The ki bytes land swizzled, which is what ``publish_ki`` writes.  ``plain`` is the
    unswizzled shadow the indexer reads (``dsa_index_keys``); it is passed in rather than
    allocated here so nothing about how big it is, or whether prefill keeps one at all, is
    decided inside a copy.  Returns the new high-water mark ``start + valid``.
    """
    start, valid = int(start), int(valid)
    rows = int(k.shape[0])
    if valid < 0 or valid > rows or valid > int(v.shape[0]) or valid > int(ki.shape[0]):
        raise AdapterError(
            f"valid={valid} is not covered by k {tuple(k.shape)} / v {tuple(v.shape)} / ki {tuple(ki.shape)}")
    if int(ki_scale.shape[0]) < valid:
        raise AdapterError(f"ki_scale {tuple(ki_scale.shape)} is shorter than valid={valid}")
    end = start + valid
    if end > int(lw["k_cache"].shape[0]):
        raise AdapterError(f"chunk ends at {end}, the cache holds {int(lw['k_cache'].shape[0])} positions")
    if valid == 0:
        return end
    lw["k_cache"][start:end].copy_(k[:valid])
    lw["v_cache"][start:end].copy_(v[:valid])
    lw["ki_scale"][start:end].copy_(ki_scale[:valid])
    pos = torch.arange(start, end, device=ki.device)
    lw["ki_cache"][start:end].copy_(dsa_swizzle_rows(ki[:valid], pos))
    if plain is not None:
        if int(plain.shape[0]) < end:
            raise AdapterError(f"the plain index-key shadow holds {int(plain.shape[0])} rows, "
                               f"this chunk ends at {end}")
        plain[start:end].copy_(ki[:valid])
    return end


def dsa_cache_write(lw: dict, k: torch.Tensor, v: torch.Tensor, ki: torch.Tensor,
                    ki_scale: torch.Tensor, positions: torch.Tensor, src: torch.Tensor, *,
                    plain: torch.Tensor | None = None) -> None:
    """``dsa_cache_append`` with the positions on the device (``ChunkMeta.positions`` / ``src``).

    Every row is written: row ``i`` lands at ``positions[i]`` with the bytes of chunk row
    ``src[i]``.  For a real row that is itself; a padded row stands in for the last real one,
    so it rewrites that row's position with that row's bytes -- a duplicate index carrying
    one value, never a padded row in the cache.  The e4m3 tensors move as bytes: the index
    ops are dtype-agnostic there and the cache wants the codes unchanged.
    """
    lw["k_cache"].index_copy_(0, positions, k.index_select(0, src))
    lw["v_cache"].index_copy_(0, positions, v.index_select(0, src))
    lw["ki_scale"].index_copy_(0, positions, ki_scale.index_select(0, src))
    codes = ki.view(torch.uint8).index_select(0, src).view(ki.dtype)
    lw["ki_cache"].view(torch.uint8).index_copy_(
        0, positions, dsa_swizzle_rows(codes, positions).view(torch.uint8))
    if plain is not None:
        plain.view(torch.uint8).index_copy_(0, positions, codes.view(torch.uint8))


def dsa_index_keys_clamp(lw: dict, start: int) -> None:
    """Pull the plain copy's mark back to ``start``: every position from there on is about to
    be rewritten -- by this request's prefill, or by its decode, which never touches the copy."""
    lw["ki_plain_end"] = min(int(lw.get("ki_plain_end", 0)), int(start))


def dsa_index_keys_rebuild(lw: dict, end: int) -> int:
    """Extend the plain copy to ``end`` from ``ki_cache``, whose rows past the mark decode wrote.

    The swizzle is an involution, so un-swizzling a stored row is applying it again.  Returns
    how many rows were rebuilt.  The caller has clamped the mark for this request; rows below
    it are the ones prefill wrote, byte-identical to what this would recompute.
    """
    plain = lw.get("ki_plain")
    if plain is None:
        raise AdapterError("dsa_index_keys_rebuild before dsa_index_keys allocated the copy")
    mark, end = int(lw.get("ki_plain_end", 0)), int(end)
    if end > int(plain.shape[0]):
        raise AdapterError(f"the plain copy holds {int(plain.shape[0])} rows, asked to cover {end}")
    if end <= mark:
        return 0
    pos = torch.arange(mark, end, device=plain.device)
    plain[mark:end].copy_(dsa_swizzle_rows(lw["ki_cache"][mark:end], pos))
    lw["ki_plain_end"] = end
    return end - mark


def publish_native_cache_rows(layer, positions):
    """Update existing shadow buffers from device positions; no host state mutation."""
    if layer["kind"] == "full_dsa":
        encoded = layer["ki_cache"].view(torch.uint8).index_select(0, positions)
        plain = dsa_swizzle_rows(encoded, positions)
        layer["ki_plain"].view(torch.uint8).index_copy_(0, positions, plain)
    else:
        ring = positions % RING_ROWS
        for native, shadow in (("k_cache", "swa_full_k"), ("v_cache", "swa_full_v")):
            values = layer[native].index_select(0, ring).reshape(len(positions), -1)
            layer[shadow].index_copy_(0, positions, values)


def publish_native_cache_row(layer, position):
    """Publish one native-written row to the existing prefill shadow buffers."""
    if layer["kind"] == "full_dsa":
        if layer["ki_plain_end"] != position:
            raise RuntimeError("Native DSA publication requires a contiguous prefill prefix")
    positions = torch.tensor([position], device=layer["k_cache"].device)
    publish_native_cache_rows(layer, positions)
    if layer["kind"] == "full_dsa":
        layer["ki_plain_end"] = position+1
