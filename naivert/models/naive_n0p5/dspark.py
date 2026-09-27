"""Naive N0.5 draft weights and fixed storage for graph composition.

This is the SWA1024, seven-query-row checkpoint contract. The target/verify
owner supplies block geometry, masked target features and the input residual;
the compute chain does not infer accepted lengths or move a host cursor.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import math
import os

import torch

from naivert.kernels.naive_n0p5 import (
    dspark_attention, dspark_context, dspark_head_markov, dspark_mlp,
)

TARGET_LAYER_IDS = (1, 7, 14, 20, 26, 32, 39, 45)
RMS_EPSILON = 1e-5
WINDOW_SIZE = 1024
FIRST_LOGIT_ROW = 0
MASK_TOKEN_ID = 151675
VOCAB = 152576
VOCAB_SHARD = VOCAB // 8


def validate_draft_config(config):
    """Refuse a checkpoint with different math or query/feature semantics."""
    expected = dict(
        hidden_size=4096, intermediate_size=4096, num_hidden_layers=5,
        num_attention_heads=32, num_key_value_heads=4, head_dim=128,
        vocab_size=VOCAB, markov_rank=256, block_size=7, rms_norm_eps=RMS_EPSILON,
        sliding_window=WINDOW_SIZE, use_sliding_window=True,
        layer_types=["sliding_attention"] * 5, hidden_act="silu",
        markov_head_type="vanilla", enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"unsupported Naive N0.5 draft {name}: {config.get(name)!r}")
    flash = config.get("dflash_config", {})
    rope = config.get("rope_parameters", {})
    if (tuple(flash.get("target_layer_ids", ())) != TARGET_LAYER_IDS
            or flash.get("mask_token_id") != MASK_TOKEN_ID
            or flash.get("use_mask_embedding") is not True
            or rope.get("rope_theta") != 10000.0 or rope.get("rope_type") != "default"
            or any(config.get(n, False) for n in
                   ("attention_bias", "use_target_kv", "use_target_kv_inject", "use_target_kv_fuse"))):
        raise ValueError("unsupported Naive N0.5 draft feature, embedding or RoPE contract")


@dataclass(frozen=True)
class DraftRankWeights:
    context: Mapping
    attention: tuple[Mapping, ...]
    mlp: tuple[Mapping, ...]
    head: Mapping
    bias_table: torch.Tensor
    mask_embedding: torch.Tensor


def prepare_rank_weights(raw, head_shard, *, rank, device, kernels):
    """Pack caller-converted checkpoint tensors, then prepare the Markov table.

    ``raw`` uses checkpoint names without the optional ``model.`` prefix.
    BF16 conversion belongs to the checkpoint loader. In particular, supplied
    FP32 gamma is preserved by the packers. All raw tensors and head_shard
    share a source device; only packed tensors are copied to the selected GPU.
    This allocation-heavy function must complete before graph capture.
    """
    device = torch.device(device)
    if not 0 <= rank < 8 or device != torch.device("cuda", kernels.devices[rank]):
        raise ValueError("draft rank/device does not match the prepared kernel set")
    layers = []
    for i in range(5):
        prefix = f"layers.{i}."
        layers.append(dict(
            **{n: raw[prefix + f"self_attn.{n}_proj.weight"] for n in ("q", "k", "v", "o")},
            **{n: raw[prefix + f"self_attn.{n}.weight"] for n in ("q_norm", "k_norm")},
            input_ln=raw[prefix + "input_layernorm.weight"],
            post_ln=raw[prefix + "post_attention_layernorm.weight"],
            **{n: raw[prefix + f"mlp.{n}_proj.weight"] for n in ("gate", "up", "down")},
        ))
    def transfer(values):
        return MappingProxyType({n: (tuple(t.to(device) for t in x) if isinstance(x, list)
                                     else x.to(device)) for n, x in values.items()})
    context = transfer(dspark_context.pack_weights(
        raw["fc.weight"], raw["hidden_norm.weight"], layers, rank=rank))
    attention = tuple(transfer(dspark_attention.pack_weights(
        *(lw[n] for n in ("q", "k", "v", "o", "input_ln", "q_norm", "k_norm")), rank=rank))
        for lw in layers)
    mlp = tuple(transfer(dspark_mlp.pack_weights(
        *(lw[n] for n in ("gate", "up", "down", "post_ln")), rank=rank)) for lw in layers)
    head = transfer(dspark_head_markov.pack_head_weights(head_shard, raw["norm.weight"]))
    w1, w2 = (raw[f"markov_head.markov_w{i}.weight"] for i in (1, 2))
    mask = raw["mask_embedding"]
    if (w1.shape != (VOCAB, 256) or w2.shape != (VOCAB, 256)
            or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
            or mask.numel() != 4096 or mask.dtype != torch.bfloat16):
        raise ValueError("invalid draft Markov matrices or mask embedding")
    with torch.cuda.device(device):
        bias = torch.empty((VOCAB, VOCAB_SHARD), dtype=torch.bfloat16, device=device)
        kernels.head_markov.prepare_markov_bias(
            w1.contiguous().to(device),
            w2[rank * VOCAB_SHARD:(rank + 1) * VOCAB_SHARD].contiguous().to(device), bias)
    return DraftRankWeights(context, attention, mlp, head, bias, mask.to(device))


def default_inv_temperature():
    """Default 1/T for the request parameter block: NAIVERT_DSPARK_INV_TEMPERATURE or 1."""
    value = os.environ.get("NAIVERT_DSPARK_INV_TEMPERATURE")
    return float(value) if value else 1.0


# The nucleus search needs a mass it can reach: the float probabilities of a peaky distribution sum
# to slightly less than one, so top_p = 1 would find no nucleus at all (all-zero p, garbage tokens).
# Map 1.0 to 1 - 2^-20 to leave a margin below unit probability mass.
TOP_P_MAX = 1.0 - 2.0 ** -20


def sampling_params_tensor(*, repetition=1.0, frequency=0.0, presence=0.0, inv_temperature=None, top_p=0.95):
    """The per-request sampling parameter block the sampler and the draft head read: float32[8] =
    {repetition theta, frequency alpha, presence beta, penalty active, 1/T, reserved, top-p as one
    float64 in [6:8]}. Defaults (penalty off, 1/T from the environment or 1, top-p 0.95) reproduce
    the previous compile-time behaviour bitwise."""
    repetition, frequency, presence = float(repetition), float(frequency), float(presence)
    inv = default_inv_temperature() if inv_temperature is None else float(inv_temperature)
    if not (repetition > 0.0 and math.isfinite(repetition) and math.isfinite(frequency) and math.isfinite(presence)):
        raise ValueError("repeat penalty must be finite with a positive repetition factor")
    if not (0.0 < inv <= 1000.0):
        raise ValueError("inverse temperature out of range")
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must lie in (0, 1]")
    top_p = min(float(top_p), TOP_P_MAX)
    active = repetition != 1.0 or frequency != 0.0 or presence != 0.0
    t = torch.zeros(8, dtype=torch.float32)
    t[:5] = torch.tensor([repetition, frequency, presence, 1.0 if active else 0.0, inv], dtype=torch.float32)
    t[6:8].view(torch.float64)[0] = float(top_p)
    return t


class DraftBuffers:
    """One rank's persistent inputs, outputs, scratch and communication state.

    Initialize every input before the first launch. Thereafter the round's
    device tail owns positions, masks, input residual and sampling counters.
    Do not replace storage, clear epochs or replay concurrently on these buffers.
    ``sampling`` changes the capture contract, never a per-round host branch.
    """
    def __init__(self, kernels, *, rank, capacity, sampling=False):
        # The row bound is the range the draft kernels are validated at (rows index with 32-bit ints; the
        # SWA D3 binding pins the same number), raised from 65536 to 1M with the capacity-independence check; the target KV, not the draft, bounds the rows a box can hold.
        if not 0 <= rank < 8 or type(capacity) is not int or capacity % 4 or not 8 <= capacity <= 1048576:
            raise ValueError("draft requires rank 0..7 and capacity a multiple of four in [8,1048576]")
        self.rank, self.capacity, self.sampling = rank, capacity, bool(sampling)
        self.device = torch.device("cuda", kernels.devices[rank])
        def z(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        self.empty_bf16 = z(0)
        self.empty_float = z(0, dtype=torch.float32)
        self.empty_int = z(0, dtype=torch.int32)
        self.empty_long = z(0, dtype=torch.int64)
        self.target_hidden = z(8, 4096)
        self.ctx_valid = z(8)
        self.ctx_pos = z(8, dtype=torch.int64)
        self.kv_dst = z(8, dtype=torch.int64)
        self.blk_pos = z(8, dtype=torch.int64)
        self.key_add = z(capacity + 8, dtype=torch.float32)
        self.ctx_sum = z(8, 4096)
        self.ssq = z(8, 128, dtype=torch.float32)
        self.kv_partials = z(kernels.context.partial_elements(), dtype=torch.float32)
        self.kv_k = z(5, capacity + 1, 1, 128)
        self.kv_v = z(5, capacity + 1, 1, 128)
        inv = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32, device=self.device) / 128))
        angles = torch.outer(torch.arange(capacity + 8, dtype=torch.float32, device=self.device), inv)
        angles = torch.cat((angles, angles), dim=-1)
        self.rope_cos, self.rope_sin = angles.cos().contiguous(), angles.sin().contiguous()
        self.resid = tuple(z(8, 4096) for _ in range(6))
        self.hid_a = z(8, 4096)
        self.qk_raw = z(8, 640)
        self.v_blk = z(8, 128)
        self.attn_out = z(8, 512)
        self.split_scratch = z(kernels.attention.split_elems(), dtype=torch.float32)
        self.activation = z(8, 512)
        self.block_ids = z(8, dtype=torch.int64)
        self.draft_tokens = z(7, dtype=torch.int64)
        self.prev_token = z(1, dtype=torch.int64)
        self.logit_trace = z(7, VOCAB_SHARD, dtype=torch.float32) if sampling else self.empty_float
        self.sampling_state = z(2, dtype=torch.int64)
        # Per-request sampling: token counts over the committed context (full vocabulary, each rank
        # reads its shard; the tail increments) and the parameter block (see sampling_params_tensor).
        self.rep_count = z(VOCAB, dtype=torch.int32)
        self.sampling_params = sampling_params_tensor().to(self.device)
        self.control = z(kernels.head_markov.head_markov_control_elements(), dtype=torch.int64)
        self.mailbox = z(kernels.context.mailbox_bytes(), dtype=torch.uint8)
        self.barrier = z(2, dtype=torch.int32)
        self.epoch = z(1, dtype=torch.int32)
        self.wdog = z(1, dtype=torch.int32)


class StagedDraftContext:
    """Prepare every verified row; the tail commits only its retained prefix.

    ``features`` is the target's unmasked eight-row export for this rank. The
    current block's absolute positions remain unchanged until the tail. This
    storage is private to one draft buffer set and one graph execution domain.
    """
    def __init__(self, buffers, features):
        if (features.shape != (8, 4096) or features.dtype != torch.bfloat16
                or features.device != buffers.device or not features.is_contiguous()):
            raise ValueError("staged context requires the rank's contiguous BF16 [8,4096] target export")
        self.buffers, self.features = buffers, features
        self.valid = torch.ones(8, dtype=torch.bfloat16, device=buffers.device)
        self.destinations = torch.arange(8, dtype=torch.int64, device=buffers.device)
        self.key = torch.empty((5, 8, 1, 128), dtype=torch.bfloat16, device=buffers.device)
        self.value = torch.empty_like(self.key)
