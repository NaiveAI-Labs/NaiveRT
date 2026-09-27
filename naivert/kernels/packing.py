"""Kernel-facing weight layouts, parametrized by the model's dimensions.

These functions encode what the expert_select / expert_down / unproj_o / head
kernels expect in memory (page order, QMMA swizzles, packed scales). They know
nothing about a particular model: callers pass hidden width, tile widths and
shard sizes. ``quantize_fp8_block`` produces the fp8 e4m3 + per-block scale
form the expert and dense kernels consume from bf16 weights.
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
def pack_router_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return [original, half-swapped] FP32 images; preserve every payload bit.

    The first image remains row-major for decode and legacy callers. The second
    swaps the two float4 halves in alternating groups of four eight-value chunks,
    matching the explicit prepared-packed router bindings. Pack before capture.
    """
    if weight.dtype != torch.float32 or weight.ndim != 2 or weight.shape[1] % 64:
        raise ValueError("router weight must be FP32 [experts, hidden], hidden divisible by 64")
    source = weight.contiguous().view(torch.int32)
    result = torch.empty((2, *weight.shape), dtype=torch.float32, device=weight.device)
    bits = result.view(torch.int32)
    k = torch.arange(weight.shape[1], device=weight.device)
    physical = k ^ ((k >> 3) & 4)
    bits[0].copy_(source)
    bits[1].index_copy_(1, physical, source)
    return result

def swizzle_qmma_16x32(value: torch.Tensor) -> torch.Tensor:
    prefix = value.shape[:-2]
    value = value.reshape(*prefix, 2, 8, 2, 4, 4)
    value = value.transpose(-4, -3).transpose(-5, -4)
    return value.reshape(*prefix, 4, 32, 4).transpose(-3, -2).contiguous()


def swizzle_qmma_8x32(value: torch.Tensor) -> torch.Tensor:
    prefix = value.shape[:-2]
    return value.reshape(*prefix, 8, 2, 4, 4).transpose(-2, -3).contiguous()


def pack_gate_up(
    gate: torch.Tensor,
    gate_scale: torch.Tensor,
    up: torch.Tensor,
    up_scale: torch.Tensor,
    *,
    hidden: int,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused gate/up expert pages: [experts, 16, hidden/1024, 32*1024 + 128 scale bytes].

    Expert intermediate per rank is 256 rows (16 x 16); hidden is paged by 1024.
    """
    experts = int(gate.shape[0])
    pages = hidden // 1024
    if pages * 1024 != hidden or int(gate.shape[1]) != 256:
        raise ValueError("pack_gate_up expects hidden % 1024 == 0 and 256 expert rows per rank")

    def pack_weight(value: torch.Tensor) -> torch.Tensor:
        value = (
            value.reshape(experts, 16, 16, pages, 1024)
            .transpose(2, 3)
            .reshape(experts, 16, pages, 16, 32, 32)
            .transpose(3, 4)
        )
        return swizzle_qmma_16x32(value).reshape(experts, 16, pages, 16, 1024)

    def pack_scale(value: torch.Tensor) -> torch.Tensor:
        return (
            value.reshape(experts, 2, 1, hidden // 128)
            .repeat(1, 1, 8, 1)
            .reshape(experts, 16, 1, pages, 8)
            .transpose(2, 3)
        )

    weights = torch.cat((pack_weight(gate), pack_weight(up)), dim=3).reshape(experts, 16, pages, 32 * 1024)
    scales = (
        torch.cat((pack_scale(gate_scale), pack_scale(up_scale)), dim=3)
        .reshape(experts, 16, pages, 16)
        .to(scale_dtype)
        .view(torch.float8_e4m3fn)
    )
    packed = torch.zeros((experts, 16, pages, 32 * 1024 + 128), dtype=torch.float8_e4m3fn, device=weights.device)
    packed[..., : 32 * 1024].copy_(weights)
    packed[..., 32 * 1024 : 32 * 1024 + scales.shape[-1]].copy_(scales)
    return packed.contiguous()


def pack_down(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    hidden: int,
    outputs_per_tile: int,
    inter: int = 256,
    layout: str = "qmma",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert down projection in output tiles: [experts, hidden, inter] swizzled per 16-row chunk.

    A tile of ``outputs_per_tile`` rows is packed as 16-row QMMA chunks plus one
    8-row chunk when the width leaves a remainder of 8 (56 = 3 x 16 + 8; 64 = 4 x 16).
    ``umma_sw128`` instead stores two K128 SW128 panels for native W8A8 down.
    It permutes the original E4M3 bytes without changing their block128 scales.
    """
    experts = int(weight.shape[0])
    tiles = hidden // outputs_per_tile
    if tiles * outputs_per_tile != hidden:
        raise ValueError("outputs_per_tile must divide hidden")
    value = weight.reshape(experts, tiles, outputs_per_tile, inter)
    if layout == "umma_sw128":
        if outputs_per_tile != 64 or inter != 256 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("umma_sw128 down requires E4M3, 64 output rows, and K256")
        # Physical 16-byte chunk = logical chunk XOR (row % 8). XOR is its
        # own inverse, so gather logical chunks in physical-address order.
        panels = value.view(torch.uint8).reshape(experts, tiles, 64, 2, 8, 16).permute(0, 1, 3, 2, 4, 5)
        rows = torch.arange(64, device=weight.device)
        chunks = torch.arange(8, device=weight.device)
        index = ((rows[:, None] & 7) ^ chunks[None, :])[None, None, None, :, :, None]
        packed_weight = panels.gather(4, index.expand_as(panels)).reshape(experts, hidden, inter).view(weight.dtype)
    elif layout == "qmma":
        chunks = []
        start = 0
        while outputs_per_tile - start >= 16:
            chunk = value[:, :, start : start + 16].reshape(experts, tiles, 16, inter // 32, 32).transpose(2, 3)
            chunks.append(swizzle_qmma_16x32(chunk).reshape(experts, tiles, -1))
            start += 16
        if outputs_per_tile - start == 8:
            tail = value[:, :, start : start + 8].reshape(experts, tiles, 8, inter // 32, 32).transpose(2, 3)
            chunks.append(swizzle_qmma_8x32(tail).reshape(experts, tiles, -1))
        elif outputs_per_tile != start:
            raise ValueError("outputs_per_tile must be a multiple of 16, or 16k + 8")
        packed_weight = torch.cat(chunks, dim=2).reshape(experts, hidden, inter)
    else:
        raise ValueError(f"unknown down weight layout: {layout}")
    # One scale per (128-row hidden block, 128-wide inter block), spread over the tiles it covers
    # (16 entries per tile after padding). For a 7168 hidden size, hidden/128 == 56 == the tile width.
    scale_blocks = inter // 128
    packed_scale = (
        scale.reshape(experts, hidden // 128, 1, scale_blocks)
        .repeat(1, 1, 16, 1)
        .reshape(experts, tiles, -1)
    )
    padding = torch.zeros((experts, tiles, 16 - packed_scale.shape[-1]), dtype=scale.dtype, device=scale.device)
    packed_scale = torch.cat((packed_scale, padding), dim=2).reshape(experts, tiles * 8, 2).to(scale_dtype)
    return packed_weight.contiguous(), packed_scale.contiguous()


def pack_unproj(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    hidden: int,
    outputs_per_tile: int = 64,
    input_slots: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output projection in output tiles: [tiles, slots, outputs_per_tile * inter] swizzled."""
    tiles = hidden // outputs_per_tile
    projection_k = int(weight.shape[1])
    inter = projection_k // input_slots
    if tiles * outputs_per_tile != hidden or inter * input_slots != projection_k or inter % 32:
        raise ValueError("pack_unproj dimensions do not tile")
    value = (
        weight.reshape(tiles, outputs_per_tile, input_slots, inter)
        .transpose(1, 2)
        .reshape(tiles, input_slots, outputs_per_tile // 16, 16, inter // 32, 32)
        .transpose(-2, -3)
    )
    value = swizzle_qmma_16x32(value).reshape(tiles, input_slots, -1)
    return value.contiguous(), scale.to(torch.bfloat16).contiguous()


def pack_head(weight: torch.Tensor, *, hidden: int) -> torch.Tensor:
    """LM head shard [vocab_shard, hidden] as [vocab_shard/16 * hidden/1024, 16, 1024] pages."""
    vocab_shard = int(weight.shape[0])
    pages = hidden // 1024
    if vocab_shard % 16 or pages * 1024 != hidden:
        raise ValueError("pack_head expects vocab_shard % 16 == 0 and hidden % 1024 == 0")
    return (
        weight.reshape(vocab_shard // 16, 16, pages, 1024)
        .transpose(1, 2)
        .reshape(vocab_shard // 16 * pages, 16, 1024)
        .contiguous()
    )


def up_gate_fragment_order(i0: int, k0: int) -> tuple[int, int]:
    """(row, k) within a 16 x 32 chunk that rmsnorm_up_gate reads from packed position (i0, k0).

    Measured with one-hot weights (tests/gpu/naive_n0p5_ops/probe_up_gate.py): the kernel's f16
    m16n8k16 fragment reader interleaves the chunk differently from the fp8 m16n8k32 order
    pack_gate_up's swizzle targets.
    """
    b = lambda v, n: (v >> n) & 1  # noqa: E731
    row = b(k0, 3) | b(i0, 0) << 1 | b(i0, 1) << 2 | b(k0, 1) << 3
    k = b(k0, 0) | b(k0, 4) << 1 | b(k0, 2) << 2 | b(i0, 3) << 3 | b(i0, 2) << 4
    return row, k


def permute_for_up_gate(weight: torch.Tensor) -> torch.Tensor:
    """Reorder a natural [rows, hidden] dense gate/up weight so that pack_gate_up's pages
    read back naturally through rmsnorm_up_gate (the converted checkpoint already
    stores its dense weights this way; Naive N's HF weights need it).

    The permutation stays inside each 16-row x 32-column chunk, so 128 x 128 block scales
    are unaffected and it commutes with quantize_fp8_block.
    """
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if rows % 16 or cols % 32:
        raise ValueError("permute_for_up_gate expects rows % 16 == 0 and columns % 32 == 0")
    src_row = torch.empty((16, 32), dtype=torch.long)
    src_col = torch.empty((16, 32), dtype=torch.long)
    for i0 in range(16):
        for k0 in range(32):
            src_row[i0, k0], src_col[i0, k0] = up_gate_fragment_order(i0, k0)
    chunks = weight.reshape(rows // 16, 16, cols // 32, 32).permute(0, 2, 1, 3)   # [G, C, r, k]
    packed = chunks[:, :, src_row.to(weight.device), src_col.to(weight.device)]   # [G, C, i0, k0]
    return packed.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()


def quantize_fp8_block(weight: torch.Tensor, *, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [rows, cols] -> (fp8 e4m3 [rows, cols], fp32 scale [rows/block, cols/block]).

    Dequantization is ``q * scale`` per block, the convention the fp8 kernels and the
    block-scaled FP8 checkpoints (``weight_scale_inv``) use. Rows and cols must be multiples of ``block``.
    """
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if rows % block or cols % block:
        raise ValueError("quantize_fp8_block needs dimensions that are multiples of the block")
    blocks = weight.float().reshape(rows // block, block, cols // block, block)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(rows, cols).contiguous(), scale.reshape(rows // block, cols // block).contiguous()


__all__ = [
    "FP8_MAX",
    "pack_down",
    "pack_gate_up",
    "pack_head",
    "pack_unproj",
    "quantize_fp8_block",
    "swizzle_qmma_16x32",
    "swizzle_qmma_8x32",
]


# --------------------------------------------------------------------- mxfp8
MX_BLOCK = 32
MX_GATE_UP_PAGE_BYTES = 32 * 1024 + 2 * 32 * 16   # 32 KiB of e4m3 + ue8m0 scales [branch][K block][row]


def quantize_mxfp8_block32(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [rows, cols] -> (e4m3 [rows, cols], ue8m0 scale codes uint8 [rows, cols/32]).

    OCP MX layout: per (row, 32-wide K block) one shared power-of-two scale 2^E stored as the ue8m0
    byte E + 127; elements are w / 2^E rounded to e4m3. E = ceil(log2(amax / 448)) keeps every
    element in range (no saturation). An all-zero block carries byte 127 (scale 1). Dequantisation
    is ``q * 2^(byte - 127)``, exactly what the mx kernels compute.
    """
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if cols % MX_BLOCK:
        raise ValueError(f"quantize_mxfp8_block32 needs cols % {MX_BLOCK} == 0, got {cols}")
    w = weight.float().reshape(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = w.abs().amax(dim=-1, keepdim=True)
    # E = ceil(log2(amax / 448)): the largest power of two under which the block's amax still fits e4m3
    # (amax / 2^E in (224, 448]), so no element saturates; the OCP encoder's floor(log2(amax)) - 8 would
    # clip the top of the block to 448. With amax = m * 2^e from frexp (m in [0.5, 1)): E = e - 8 - [m <= 0.875].
    # (frexp and shifts only: this torch build JIT-compiles log2 / exp2 through NVRTC, which rejects sm_103.)
    mantissa, frexp_exponent = torch.frexp(amax)
    exponent = frexp_exponent.to(torch.int32) - 8 - (mantissa <= 0.875).to(torch.int32)
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(exponent))
    exponent = exponent.clamp(-126, 127)
    scale = ((exponent + 127) << 23).view(torch.float32)      # exact 2^E as the ue8m0 bit pattern
    q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    codes = (exponent + 127).to(torch.uint8)
    return q.reshape(rows, cols).contiguous(), codes.reshape(rows, cols // MX_BLOCK).contiguous()


def dequantize_mxfp8_block32(q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_mxfp8_block32 (fp32 result), for any leading batch dims."""
    scale = (codes.to(torch.int32) << 23).view(torch.float32).repeat_interleave(MX_BLOCK, dim=-1)   # 2^(code - 127)
    return q.float() * scale


def _gate_up_page_order(value: torch.Tensor, *, experts: int, pages: int) -> torch.Tensor:
    """[experts, 256, hidden] -> [experts, 16 groups, pages, 16 rows x 1024 K] in the kernel's page order."""
    value = (
        value.reshape(experts, 16, 16, pages, 1024)
        .transpose(2, 3)
        .reshape(experts, 16, pages, 16, 32, 32)
        .transpose(3, 4)
    )
    return swizzle_qmma_16x32(value).reshape(experts, 16, pages, 16, 1024)

def _down_tile_order(value: torch.Tensor, *, experts: int, tiles: int, outputs_per_tile: int, inter: int) -> torch.Tensor:
    """[experts, hidden, inter] -> [experts, tiles, outputs_per_tile * inter] in pack_down's chunk order."""
    value = value.reshape(experts, tiles, outputs_per_tile, inter)
    chunks = []
    start = 0
    while outputs_per_tile - start >= 16:
        chunk = value[:, :, start : start + 16].reshape(experts, tiles, 16, inter // 32, 32).transpose(2, 3)
        chunks.append(swizzle_qmma_16x32(chunk).reshape(experts, tiles, -1))
        start += 16
    if outputs_per_tile - start == 8:
        tail = value[:, :, start : start + 8].reshape(experts, tiles, 8, inter // 32, 32).transpose(2, 3)
        chunks.append(swizzle_qmma_8x32(tail).reshape(experts, tiles, -1))
    elif outputs_per_tile != start:
        raise ValueError("outputs_per_tile must be a multiple of 16, or 16k + 8")
    return torch.cat(chunks, dim=2)

def pack_gate_up_mx(
    gate_q: torch.Tensor,
    gate_codes: torch.Tensor,
    up_q: torch.Tensor,
    up_codes: torch.Tensor,
    *,
    hidden: int,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused gate/up expert pages of e4m3 with MX scales: uint8 [experts, 16, hidden/1024, 33792].

    Page = the fp8 page's 32 x 1024 byte order (pack_gate_up), then ue8m0 scales
    [branch 2][K block 32][row 16] (1024 B). Inputs from quantize_mxfp8_block32.
    """
    experts = int(gate_q.shape[0])
    pages = hidden // 1024
    if pages * 1024 != hidden or int(gate_q.shape[1]) != 256:
        raise ValueError("pack_gate_up_mx expects hidden % 1024 == 0 and 256 expert rows per rank")
    if gate_q.dtype != torch.float8_e4m3fn or up_q.dtype != torch.float8_e4m3fn:
        raise ValueError("pack_gate_up_mx expects e4m3 weights")
    if gate_codes.dtype != torch.uint8 or up_codes.dtype != torch.uint8:
        raise ValueError("pack_gate_up_mx expects uint8 ue8m0 scale codes")
    weights = torch.cat(
        (_gate_up_page_order(gate_q.view(torch.uint8), experts=experts, pages=pages),
         _gate_up_page_order(up_q.view(torch.uint8), experts=experts, pages=pages)), dim=3
    ).reshape(experts, 16, pages, 32 * 1024)

    def page_scales(codes: torch.Tensor) -> torch.Tensor:
        # [experts, 256 rows, hidden/32 blocks] -> [experts, group, page, block, row]
        return codes.reshape(experts, 16, 16, pages, 32).permute(0, 1, 3, 4, 2)

    scales = (
        torch.stack((page_scales(gate_codes), page_scales(up_codes)), dim=3)
        .reshape(experts, 16, pages, 2 * 32 * 16)
        .contiguous()
    )
    packed = torch.cat((weights, scales), dim=-1).contiguous()
    assert packed.shape[-1] == MX_GATE_UP_PAGE_BYTES
    return packed


def mx_down_tile_bytes(*, outputs_per_tile: int, inter: int = 256) -> int:
    return outputs_per_tile * inter + outputs_per_tile * (inter // MX_BLOCK)


def pack_down_mx(
    q: torch.Tensor,
    codes: torch.Tensor,
    *,
    hidden: int,
    outputs_per_tile: int,
    inter: int = 256,
) -> torch.Tensor:
    """Expert down projection tiles of e4m3 with MX scales: uint8 [experts, tiles, tile bytes].

    Tile = pack_down's chunk order (outputs_per_tile * inter bytes), then ue8m0 scales
    [row outputs_per_tile][K block inter/32]. Inputs from quantize_mxfp8_block32 on the
    [hidden, inter] matrix of each expert.
    """
    experts = int(q.shape[0])
    tiles = hidden // outputs_per_tile
    if tiles * outputs_per_tile != hidden:
        raise ValueError("outputs_per_tile must divide hidden")
    if q.dtype != torch.float8_e4m3fn or codes.dtype != torch.uint8:
        raise ValueError("pack_down_mx expects e4m3 weights and uint8 scale codes")
    weights = _down_tile_order(q.view(torch.uint8), experts=experts, tiles=tiles, outputs_per_tile=outputs_per_tile, inter=inter)
    scales = codes.reshape(experts, tiles, outputs_per_tile * (inter // MX_BLOCK)).contiguous()
    packed = torch.cat((weights, scales), dim=-1).contiguous()
    assert packed.shape[-1] == mx_down_tile_bytes(outputs_per_tile=outputs_per_tile, inter=inter)
    return packed
