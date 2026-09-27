"""Deterministic 4K DSA epilogue for the existing library radix selector.

The radix result supplies the exact cutoff score. Select strictly greater scores
and the lowest key IDs needed among ties, emitting ascending IDs before TP gather.
Valid scores must not be NaN. Scores and attention arithmetic are unchanged.
"""
import torch
import triton
import triton.language as tl


SINGLE_BLOCK_MAX_KEYS = 65536
KEY_TILE = 16384
MAX_VISIBLE_KEYS = 1048576


@triton.jit
def _ordered_key(score):
    bits = score.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)


@triton.jit
def _stable_epilogue(logits, raw_indices, lengths, starts, values_out, indices_out,
                     QUERY_START: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                     GLOBAL: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    start = tl.load(starts + row) if GLOBAL else 0
    slot = tl.arange(0, K)
    out_base = (row + QUERY_START) * K
    if length <= K:
        dense_valid = slot < length
        dense_score = tl.load(logits + row * N + slot, dense_valid, other=-float('inf'))
        tl.store(values_out + out_base + slot, dense_score)
        tl.store(indices_out + out_base + slot, tl.where(dense_valid, slot + start, -1))
    else:
        selected = tl.load(raw_indices + row * K + slot)
        selected_score = tl.load(logits + row * N + selected)
        cutoff = tl.min(selected_score, 0)
        col = tl.arange(0, BLOCK_N)
        valid = (col < length) & (col < N)
        score = tl.load(logits + row * N + col, valid, other=-float('inf'))
        key = _ordered_key(score)
        threshold = _ordered_key(cutoff)
        # Native ordered_float_key distinguishes +0 from -0. The library's
        # floating cutoff does not promise which zero sign it will return.
        if cutoff == 0.0:
            at_least_positive_zero = tl.sum((valid & (key >= 0x80000000)).to(tl.int32), 0)
            threshold = tl.where(at_least_positive_zero >= K, 0x80000000, 0x7fffffff).to(tl.uint32)
        higher = valid & (key > threshold)
        equal = valid & (key == threshold)
        remaining = K - tl.sum(higher.to(tl.int32), 0)
        tie_number = tl.cumsum(equal.to(tl.int32), 0)
        keep = higher | (equal & (tie_number <= remaining))
        offset = tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(values_out + out_base + offset, score, keep)
        tl.store(indices_out + out_base + offset, col + start, keep)


class StableTopKEpilogue:
    """Match the library epilogue launch API; the native executor's chunk spans are
    multiples of 4096 (``max_seqlen`` = visible keys), so the block covers the span."""

    returns_sorted_indices = True

    def __getitem__(self, grid):
        def launch(logits, indices, lengths, starts, values_out, indices_out, *,
                   query_start, num_rows, topk, max_seqlen, total_rows,
                   return_global_indices, BLOCK_K):
            assert topk == 2048 and 1 <= max_seqlen <= MAX_VISIBLE_KEYS, max_seqlen
            assert logits.is_contiguous() and indices.is_contiguous()
            assert query_start + num_rows <= total_rows
            if max_seqlen > SINGLE_BLOCK_MAX_KEYS:
                values, selected = tiled_stable_topk(logits, lengths, starts,
                                                     return_global_indices=return_global_indices)
                values_out[query_start:query_start + num_rows].copy_(values)
                indices_out[query_start:query_start + num_rows].copy_(selected)
                return
            block = max(triton.next_power_of_2(int(max_seqlen)), 2048)
            _stable_epilogue[(num_rows,)](logits, indices, lengths, starts,
                values_out, indices_out, query_start, topk, max_seqlen,
                return_global_indices, block, num_warps=4 if block <= 4096 else 8)
        return launch


stable_epilogue = StableTopKEpilogue()

def tiled_stable_topk(logits, lengths, starts, *, return_global_indices=True):
    """Exact top-2048 using the library radix selector and existing stable kernel on bounded tiles.

    A global winner must be in its tile's top-2048 under the same ordering.
    Tiles emit ascending key IDs, so concatenating candidates preserves the
    key-ID tie order for the second selection. A valid row occupies a prefix
    of the key span; consequently its valid tile candidates also form a prefix.
    This host composition needs no host reads of GPU data. The vendored radix
    padding fix also supports valid negative infinity in candidate rows.
    """
    from naivert.kernels.naive_n0p5 import prefill as P
    rows, width = logits.shape
    topk = 2048
    if not 1 <= width <= MAX_VISIBLE_KEYS:
        raise ValueError(f"stable DSA selection supports at most {MAX_VISIBLE_KEYS} visible keys")
    tiles = triton.cdiv(width, KEY_TILE)
    padded_width = tiles * KEY_TILE
    if padded_width == width:
        tiled = logits.view(rows * tiles, KEY_TILE)
    else:
        padded = torch.empty((rows, padded_width), device=logits.device, dtype=logits.dtype)
        padded[:, :width].copy_(logits)
        # Invalid tails are never read by either existing selection kernel.
        tiled = padded.view(rows * tiles, KEY_TILE)
    offsets = torch.arange(tiles, device=logits.device, dtype=torch.int32) * KEY_TILE
    tile_lengths = (lengths[:, None] - offsets[None, :]).clamp(min=0, max=KEY_TILE).flatten()
    tile_starts = offsets[None, :].expand(rows, tiles)
    if return_global_indices:
        tile_starts = tile_starts + starts[:, None]
    tile_starts = tile_starts.contiguous().flatten()
    radix = P._module(P._TOPK).invoke_indexer_topk_prefill
    raw = radix(tiled, torch.zeros_like(tile_lengths), tile_lengths, topk)
    values = torch.empty((rows * tiles, topk), device=logits.device, dtype=torch.float32)
    indices = torch.empty((rows * tiles, topk), device=logits.device, dtype=torch.int32)
    stable_epilogue[(rows * tiles,)](tiled, raw, tile_lengths, tile_starts, values, indices,
        query_start=0, num_rows=rows * tiles, topk=topk, max_seqlen=KEY_TILE,
        total_rows=rows * tiles, return_global_indices=True, BLOCK_K=256)
    candidates = values.view(rows, tiles * topk)
    candidate_indices = indices.view(rows, tiles * topk)
    candidate_lengths = tile_lengths.view(rows, tiles).clamp(max=topk).sum(dim=1, dtype=torch.int32)
    candidate_starts = torch.zeros_like(candidate_lengths)
    if candidates.shape[1] > KEY_TILE:
        result, slots = tiled_stable_topk(candidates, candidate_lengths, candidate_starts, return_global_indices=False)
    else:
        raw = radix(candidates, candidate_starts, candidate_lengths, topk)
        result = torch.empty((rows, topk), device=logits.device, dtype=torch.float32)
        slots = torch.empty((rows, topk), device=logits.device, dtype=torch.int32)
        stable_epilogue[(rows,)](candidates, raw, candidate_lengths, candidate_starts, result, slots,
            query_start=0, num_rows=rows, topk=topk, max_seqlen=tiles * topk,
            total_rows=rows, return_global_indices=False, BLOCK_K=256)
    selected = candidate_indices.gather(1, slots.clamp(min=0).long())
    selected.masked_fill_(slots < 0, -1)
    return result, selected


def tiled_indexer_select(q, weights, k, k_scale, k_start, k_end, *, max_seqlen, chunk=None, topk=2048):
    """Expose query/key-range work items to the unchanged persistent logits kernel.

    Group replicas by key tile so each eight-query CTA scans one short range.
    Query batching bounds logits memory; no arithmetic or device code changes.
    """
    from naivert.kernels.naive_n0p5 import prefill as P
    if topk != 2048 or not 1 <= max_seqlen <= MAX_VISIBLE_KEYS:
        raise ValueError(f"invalid tiled DSA shape: topk={topk}, width={max_seqlen}")
    rows = q.shape[0]
    tile = KEY_TILE
    tiles = (max_seqlen + tile - 1) // tile
    if chunk is None:
        # Eight query rows make one persistent query work item. Keep tile
        # boundaries aligned, and include padded key columns in the budget.
        chunk = max(8, P.indexer_chunk_rows(rows, tiles * tile) // 8 * 8)
    if chunk <= 0:
        raise ValueError("DSA query chunk must be positive")
    module = P._module(P._TOPK)
    output = torch.empty((rows, topk), device=q.device, dtype=torch.int32)
    offsets = torch.arange(tiles, device=q.device, dtype=torch.int32)[:, None] * tile
    keys = k.view(-1, 1, P.INDEX_DIM)
    scales = k_scale.contiguous()
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        count = end - begin
        starts = (k_start[None, begin:end] + offsets).flatten()
        ends = torch.maximum(starts, torch.minimum(
            k_end[None, begin:end].expand(tiles, count).flatten(), starts + tile))
        logits = torch.empty((tiles * count, tile), device=q.device, dtype=torch.float32)
        module.indexer_logits_prefill_fwd(
            q[begin:end].repeat(tiles, 1, 1), keys, weights[begin:end].repeat(tiles, 1),
            starts, ends, max_seqlen=tile, output=logits, k_scales=scales,
            is_compressed_logits=True, is_indexer_topk=True)
        joined = logits.view(tiles, count, tile).transpose(0, 1).reshape(count, tiles * tile)
        _, indices = tiled_stable_topk(joined, k_end[begin:end] - k_start[begin:end], k_start[begin:end])
        output[begin:end].copy_(indices)
    return output


def install_stable_epilogue():
    """Patch the vendored selector actually called by the prefill bindings."""
    from naivert.kernels.naive_n0p5 import prefill as P
    topk_module = P._module(P._TOPK)
    if not isinstance(topk_module._indexer_topk_prefill_epilogue_kernel, StableTopKEpilogue):
        topk_module._indexer_topk_prefill_epilogue_kernel = stable_epilogue
    return topk_module
