"""FP8 target checkpoint operands for the active TP8 Naive N kernels.

Expert E4M3 codes are preserved. The W8A8 path also preserves FP32 expert scales and
router weights, retains BF16 non-expert weights, and scales projected V.
The default BF16 expert-down build retains its original packed weight formats.
"""
from __future__ import annotations

import hashlib
import os as _os   # `os` is a local name in attention()
from naivert.models import registry as _registry
import json
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from threading import Lock

import torch
from safetensors import safe_open

from naivert.checkpoint import _tensor_metadata
from naivert.kernels import packing
from naivert.model import ModelPackageError
from .model import NAIVE_N0P5_HYBRID

INDEX_HEADS = 16
DEFAULT_ROOT = Path(_os.environ.get("NAIVERT_TARGET_CHECKPOINT")
                    or f"/local_models/{_registry.MODELS[_registry.DEFAULT_MODEL].target}")
INDEX_SHA256 = "4ab9ee82a323b14889552ffff46a22328f23802fa8ab90575ce6d4851c672cb2"
CONFIG_SHA256 = "32e17e24aaa95f09bfc5be8e1091441eb18737dc4bd32186b076268dad21a5fe"
# Known config.json digests. The release checkpoint has the same index, tensor names,
# dtypes, shapes and shard layout as the preview checkpoint; only config.json differs (its
# quantization ignore list names q/k/v/o per projection instead of self_attn).
CONFIG_SHA256_BY_VARIANT = {"n0.5-preview": CONFIG_SHA256, "n0.5": "d7c4a2043f2e71ad9cee250232003be66c7198f55694d5225093b6f668a448a4"}
METADATA_SHA256 = "742e2fabc73fb9fea91c41e71b52f1b6b755f31258688723e6af4334ad3e359d"


def interleave_rope(x: torch.Tensor, head_dim: int, *, projection: bool = False) -> torch.Tensor:
    """HF rotate_half rows -> adjacent pairs; only the first 64 dims move."""
    y = x.T if projection else x
    shape = y.shape
    y = y.reshape(-1, head_dim).clone()
    perm = torch.stack((torch.arange(32, device=x.device), torch.arange(32, 64, device=x.device)), 1).flatten()
    y[:, :64] = y[:, perm]
    y = y.reshape(shape)
    return (y.T if projection else y).contiguous()


def shard_fp8_block(weight, scale, rank: int, *, axis: int, ranks: int = 8):
    """Shard codes and their 128x128 dequantization scales along the same axis.

    Scales remain FP32 here. Only packing / the matched kernel reference rounds
    them to BF16, so callers can also measure that conversion error explicitly.
    """
    if axis not in (0, 1) or not 0 <= rank < ranks:
        raise ValueError("invalid TP rank or shard axis")
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise ValueError("expected E4M3 matrix and FP32 block scales")
    if any(n % 128 for n in weight.shape) or tuple(scale.shape) != tuple(n // 128 for n in weight.shape):
        raise ValueError("scale shape must describe 128x128 weight blocks")
    width = weight.shape[axis] // ranks
    if weight.shape[axis] % ranks or width % 128:
        raise ValueError("TP shards must end on 128-element block boundaries")
    return (weight.narrow(axis, rank * width, width).contiguous(),
            scale.narrow(axis, rank * (width // 128), width // 128).contiguous())


def dequant_block(q, s):
    m, n = q.shape
    return (q.float().reshape(m // 128, 128, n // 128, 128) * s.float()[:, None, :, None]).reshape(m, n)


class NaiveNCheckpoint:
    """Identity-checked reader. Header validation reads no weight payloads."""
    index_heads = INDEX_HEADS

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self._reader_lock = Lock()
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "config.json").is_file():          # a checkpoint root holding step_* directories
            steps = sorted(c for c in self.root.glob("step_*") if c.is_dir())
            if steps:
                self.root = steps[-1]
        config_digest = hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest()
        self.variant = next((k for k, v in CONFIG_SHA256_BY_VARIANT.items() if v == config_digest), None)
        if self.variant is None:
            raise ModelPackageError("checkpoint identity mismatch: config.json")
        if hashlib.sha256((self.root / "model.safetensors.index.json").read_bytes()).hexdigest() != INDEX_SHA256:
            raise ModelPackageError("checkpoint identity mismatch: model.safetensors.index.json")
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        count, digest, dtypes = _tensor_metadata(self.root, tuple(sorted(self.weight_map.items())))
        if count != 72664 or digest != METADATA_SHA256 or dict(dtypes) != {"BF16": 378, "F32": 36190, "F8_E4M3": 36096}:
            raise ModelPackageError("checkpoint tensor metadata mismatch")

    @cached_property
    def _file_contexts(self):
        return ExitStack()

    @cached_property
    def _readers(self):
        return {}

    def close(self):
        with self._reader_lock:
            self._file_contexts.close()
            self._readers.clear()

    def read(self, key, device="cpu", *, rows=None, cols=None):
        # Keep at most 49 mappings open. Reopening and parsing a shard for
        # each of 72k tensors is particularly expensive on the AFS mount.
        shard = self.weight_map[key]
        with self._reader_lock:
            if shard not in self._readers:
                self._readers[shard] = self._file_contexts.enter_context(
                    safe_open(self.root / shard, framework="pt", device="cpu"))
            f = self._readers[shard]
        if rows is None and cols is None:
            value = f.get_tensor(key)
        else:
            value = f.get_slice(key)[rows if rows is not None else slice(None),
                                     cols if cols is not None else slice(None)]
        return value.contiguous().to(device)

    def attention_hf(self, layer: int, rank: int, device="cpu"):
        if not 0 <= rank < 8 or not 0 <= layer < 48:
            raise ValueError("expected layer 0..47 and TP rank 0..7")
        layout = NAIVE_N0P5_HYBRID.layout_of(layer)
        kv = layout.kv_head_for_rank(rank, 8)
        p = f"model.layers.{layer}."
        a = p + "self_attn."
        read = lambda key, **kw: self.read(key, device, **kw)
        w = dict(kind=NAIVE_N0P5_HYBRID.layer_kinds[layer],
                 attn_gamma=read(p + "input_layernorm.weight").float(),
                 ffn_gamma=read(p + "post_attention_layernorm.weight").float(),
                 wq=read(a + "q_proj.weight", rows=slice(rank * 1536, (rank + 1) * 1536)),
                 wk=read(a + "k_proj.weight", rows=slice(kv * 192, (kv + 1) * 192)),
                 wv=read(a + "v_proj.weight", rows=slice(kv * 128, (kv + 1) * 128)),
                 value_scale=0.707)
        if w["kind"] == "full_dsa":
            for dest, source in (("wiq", "wq.weight"), ("wik", "wk.weight"), ("wis", "weights_proj.weight"),
                                 ("ik_w", "k_norm.weight"), ("ik_b", "k_norm.bias")):
                w[dest] = read(a + "indexer." + source)
            w["ik_w"], w["ik_b"] = w["ik_w"].float(), w["ik_b"].float()
        else:
            w["sink"] = read(a + "attention_sink_bias")[rank * 8:(rank + 1) * 8].float().contiguous()
        return w

    def attention(self, layer: int, rank: int, device="cpu", *, training_graph=False):
        w = self.attention_hf(layer, rank, device)
        for name in ("wq", "wk"):
            w[name] = interleave_rope(w[name], 192, projection=True)
        if w["kind"] == "full_dsa":
            w["wiq"] = interleave_rope(w["wiq"], 128, projection=True)
            if training_graph:
                w["wiq_scale"] = torch.ones((INDEX_HEADS, 32), dtype=torch.float32, device=device)
            else:
                w["wiq"], w["wiq_scale"] = packing.quantize_fp8_block(w["wiq"])
            w["wik"] = interleave_rope(w["wik"], 128, projection=True)
            w["ik_w"], w["ik_b"] = (interleave_rope(w[n], 128) for n in ("ik_w", "ik_b"))
        if training_graph and _os.environ.get("NAIVERT_O_PROJ_BF16") == "1":
            # bf16 O projection through Bf16LinearAllreduce (A/B form)
            w["o_packed"] = self.read(f"model.layers.{layer}.self_attn.o_proj.weight", device,
                                      cols=slice(rank * 1024, (rank + 1) * 1024))
            w["o_scale"] = torch.ones((32, 8), dtype=torch.bfloat16, device=device)
            return w
        # Fp8 e4m3 + bf16 128x128 block scales, the fused staged tail (half the O bytes per layer)
        oq, os = packing.quantize_fp8_block(self.read(f"model.layers.{layer}.self_attn.o_proj.weight", device,
                                                     cols=slice(rank * 1024, (rank + 1) * 1024)))
        w["o_q"], w["o_s"] = oq, os.to(torch.bfloat16)
        w["o_packed"], w["o_scale"] = packing.pack_unproj(oq, os, hidden=4096, outputs_per_tile=64, input_slots=4)
        return w

    def expert(self, layer: int, expert: int, which: int, rank: int, device="cpu", *, kernel_scales=True):
        if not 1 <= layer < 48 or not 0 <= expert < 256 or which not in (0, 1, 2):
            raise ValueError("invalid routed expert")
        if not 0 <= rank < 8:
            raise ValueError("expected TP rank 0..7")
        p = f"model.layers.{layer}.mlp.experts.{expert}.{('gate', 'up', 'down')[which]}_proj."
        # Slice before transferring: a rank owns two of the 16 intermediate blocks.
        axis = "cols" if which == 2 else "rows"
        q = self.read(p + "weight", device, **{axis: slice(rank * 256, (rank + 1) * 256)})
        s = self.read(p + "weight_scale_inv", device, **{axis: slice(rank * 2, (rank + 1) * 2)})
        return q, s.to(torch.bfloat16) if kernel_scales else s

    def mlp(self, layer: int, rank: int, device="cpu", *, expert_down="bf16", select="rows8"):
        if expert_down not in ("bf16", "w8a8"):
            raise ValueError("expert_down must be bf16 or w8a8")
        if select != "rows8":
            raise ValueError("select must be rows8")
        p = f"model.layers.{layer}.mlp."
        if layer == 0:
            gate, up = (self.read(p + name + "_proj.weight", device,
                                  rows=slice(rank * 2048, (rank + 1) * 2048)) for name in ("gate", "up"))
            if expert_down == "w8a8":
                return dict(dense_gate=gate, dense_up=up,
                            dense_down=self.read(p + "down_proj.weight", device,
                                                 cols=slice(rank * 2048, (rank + 1) * 2048)))
            def parts(g, u):
                pairs = [packing.quantize_fp8_block(t) for t in torch.cat((g, u)).reshape(16, 256, 4096)]
                q, s = (torch.stack(t) for t in zip(*pairs))
                return q[:8], s[:8].to(torch.bfloat16), q[8:], s[8:].to(torch.bfloat16)
            gq, gs, uq, us = parts(gate, up)
            native = packing.pack_gate_up(*parts(packing.permute_for_up_gate(gate), packing.permute_for_up_gate(up)), hidden=4096)
            down = self.read(p + "down_proj.weight", device, cols=slice(rank * 2048, (rank + 1) * 2048))
            dq, ds = (torch.stack(t) for t in zip(*(packing.quantize_fp8_block(c) for c in down.reshape(4096, 8, 256).transpose(0, 1))))
            ds = ds.to(torch.bfloat16)
            packed, scale = packing.pack_down(dq, ds, hidden=4096, outputs_per_tile=64, inter=256)
            return dict(dense_gate_q=gq, dense_gate_s=gs, dense_up_q=uq, dense_up_s=us,
                        dense_ws=native.view(torch.uint8).reshape(128, 4, 32896)[:, :, :32800].contiguous(),
                        dense_down_q=dq, dense_down_s=ds, dense_down=packed, dense_down_scale=scale)
        scale_dtype = torch.float32 if expert_down == "w8a8" else torch.bfloat16
        pairs = []
        for which in range(3):
            pairs.append(tuple(torch.stack(t) for t in zip(*(self.expert(layer, e, which, rank, device, kernel_scales=expert_down != "w8a8") for e in range(256)))))
        (gq, gs), (uq, us), (dq, ds) = pairs
        down, scale = packing.pack_down(dq, ds, hidden=4096, outputs_per_tile=64, inter=256,
                                        layout="umma_sw128" if expert_down == "w8a8" else "qmma", scale_dtype=scale_dtype)
        # The W8A8 ABI preserves original FP32 router weights and expert scales.
        out = dict(router=self.read(p + "gate.weight", device).to(scale_dtype),
                   router_bias=self.read(p + "gate.e_score_correction_bias", device).float(),
                   expert_table=packing.pack_gate_up(gq, gs, uq, us, hidden=4096, scale_dtype=scale_dtype),
                   expert_down=down, expert_down_scale=scale)
        return out


class NaiveNRankWeights:
    """Checkpoint-backed weights and caches consumed by the Naive N decode harness."""

    def __init__(self, checkpoint: NaiveNCheckpoint, rank: int, layers: int, max_seq_len: int,
                 dsa_workspace_bytes: int, swa_workspace_bytes: int, *, expert_down="bf16", select="rows8", ops=None):
        from .layouts import rope_table
        if not 0 <= rank < 8 or not 1 <= layers <= 48 or not 1 <= max_seq_len <= 1048576:
            raise ValueError("invalid TP rank, layer count, or sequence capacity")
        self.checkpoint = checkpoint
        self.expert_down = expert_down
        self.rank, self.layers, self.dev = rank, layers, torch.device(f"cuda:{rank}")
        self.experts = "fp8"
        self.embedding = checkpoint.read("model.embed_tokens.weight", self.dev)
        self.head_gamma = checkpoint.read("model.norm.weight", self.dev).float()
        self.lm_head_ref = checkpoint.read("lm_head.weight", self.dev, rows=slice(rank * 19072, (rank + 1) * 19072))
        self.lm_head = packing.pack_head(self.lm_head_ref, hidden=4096)
        self.rope_full = rope_table(1e7, 64, max_seq_len).to(self.dev)
        self.rope_swa = rope_table(1e4, 64, max_seq_len).to(self.dev)
        self.layer = []
        for layer in range(layers):
            w = checkpoint.attention(layer, rank, self.dev, training_graph=expert_down == "w8a8")
            w.update(checkpoint.mlp(layer, rank, self.dev, expert_down=expert_down, select=select))
            if expert_down == "w8a8" and "router" in w:
                w["router_prepared"] = packing.pack_router_weight(w["router"])
                w["router"] = w["router_prepared"][0]
            full = w["kind"] == "full_dsa"
            capacity = max_seq_len if full else 136
            w["k_cache"] = torch.zeros(capacity, 192, dtype=torch.bfloat16, device=self.dev)
            w["v_cache"] = torch.zeros(capacity, 128, dtype=torch.bfloat16, device=self.dev)
            if full:
                w["ki_cache"] = torch.zeros(capacity, 128, dtype=torch.float8_e4m3fn, device=self.dev)
                w["ki_scale"] = torch.zeros(capacity, dtype=torch.float32, device=self.dev)
            w["attn_ws"] = torch.zeros(dsa_workspace_bytes if full else swa_workspace_bytes,
                                       dtype=torch.uint8, device=self.dev)
            self.layer.append(w)

    def expert(self, layer, expert, which):
        return self.checkpoint.expert(layer, expert, which, self.rank, self.dev,
                                      kernel_scales=self.expert_down != "w8a8")

    @staticmethod
    def dequant_expert(q, s):
        return dequant_block(q, s)


def load_tp8_weights(checkpoint: NaiveNCheckpoint, layers: int, max_seq_len: int,
                     dsa_workspace_bytes: int, swa_workspace_bytes: int, *, workers: int = 8,
                     expert_down="bf16", select="rows8", ops=None):
    """Load independent rank shards concurrently; results remain in rank order."""
    if not 1 <= workers <= 8:
        raise ValueError("checkpoint loader workers must be between 1 and 8")

    def load_rank(rank):
        with torch.cuda.device(rank):
            w = NaiveNRankWeights(checkpoint, rank, layers, max_seq_len,
                               dsa_workspace_bytes, swa_workspace_bytes, expert_down=expert_down, select=select, ops=ops)
            torch.cuda.synchronize(rank)
            return w

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(load_rank, range(8)))
