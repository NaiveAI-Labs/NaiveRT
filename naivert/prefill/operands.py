"""The kernel prefill's operands beside this tree's W8A8 decode weights.

``kernel_prefill_weights`` returns, per rank, the namespace ``naivert.prefill.executor``
expects (``precision``, ``embedding``, ``rope_full``/``rope_swa``, ``head_gamma_bf16``,
``layer[...]``).  Each layer dict is a *shallow copy* of the decode layer dict plus the
prefill keys, so ``k_cache`` / ``v_cache`` / ``ki_cache`` / ``ki_scale`` are the decode
tensors themselves (the executor prefills what decode reads) while keys decode packs
differently (``router``) are overridden only in the copy.

Operand sources (``weights.add_kernel_prefill_layer`` in the source tree):
  qkv [1856, 4096] = cat(wq, wk, wv) bf16 with the rope rows interleaved (as loaded);
  o_bf16 [4096, 1024] this rank's O columns; attn/ffn gamma bf16;
  DSA: qkw [2192, 4096] = cat(wiq, wik, wis) bf16, ik_w/ik_b fp32, ki_plain shadow;
  SWA: sink_bf16; layer 0: dense_gate_up [4096, 4096], dense_down_bf16 [4096, 2048];
  MoE: router fp32 [256, 4096], router_bias fp32, EP8 experts (this rank's 32 whole experts,
  UE8M0-requantised for DeepGEMM's masked grouped GEMM), ep_first_expert.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.models.naive_n0p5.model import PRECISION_ID
from naivert.prefill import adapters


def load_expert_rank(checkpoint, layer: int, rank: int, device) -> dict:
    """EP8: rank ``r``'s whole experts ``[32r, 32r + 32)`` as SGLang's DeepGEMM runner holds them.

    ``w13`` stacks each expert's gate and up rows, e4m3 ``[32, 4096, 4096]``; ``w2`` is down,
    e4m3 ``[32, 4096, 2048]``.  The checkpoint's block scales are re-quantised to UE8M0 block
    scales and packed to the int32 words the masked grouped GEMM reads.
    """
    per, group = P.EXPERTS_PER_RANK, P.QUANT_GROUP
    inter, hidden, fp8 = P.EXPERT_INTERMEDIATE, P.HIDDEN, torch.float8_e4m3fn
    w13 = torch.empty((per, 2 * inter, hidden), dtype=fp8, device=device)
    w13_s = torch.empty((per, 2 * inter // group, hidden // group), dtype=torch.float32, device=device)
    w2 = torch.empty((per, hidden, inter), dtype=fp8, device=device)
    w2_s = torch.empty((per, hidden // group, inter // group), dtype=torch.float32, device=device)
    for i in range(per):
        stem = f"model.layers.{layer}.mlp.experts.{rank * per + i}."
        codes = lambda name: checkpoint.read(stem + name + "_proj.weight", device).to(fp8)
        scales = lambda name: checkpoint.read(stem + name + "_proj.weight_scale_inv", device).to(torch.float32)
        w13[i, :inter], w13[i, inter:] = codes("gate"), codes("up")
        w13_s[i, :inter // group], w13_s[i, inter // group:] = scales("gate"), scales("up")
        w2[i], w2_s[i] = codes("down"), scales("down")
    packed = {}
    for name, q, s in (("w13", w13, w13_s), ("w2", w2, w2_s)):
        q, s = P.requantize_expert_ue8m0(q, s)
        packed[name] = (q, P.pack_expert_weight_scales(s, rows=int(q.shape[1])))
    return packed


def _bf16(t: torch.Tensor, what: str) -> torch.Tensor:
    if t.dtype is not torch.bfloat16:
        raise ValueError(f"{what} is {t.dtype}; the kernel prefill wants the bf16 operand "
                         "(load the W8A8 decode weights: NaiveNRankWeights(expert_down='w8a8'))")
    return t.contiguous()


def kernel_prefill_layer(lw: dict, checkpoint, layer: int, rank: int, device, *, capacity: int) -> dict:
    w = dict(lw)
    p = f"model.layers.{layer}."
    w["qkv"] = torch.cat([_bf16(w[n], n) for n in ("wq", "wk", "wv")]).contiguous()
    if tuple(w["qkv"].shape) != (P.QKV_ROWS, P.HIDDEN):
        raise ValueError(f"layer {layer} qkv {tuple(w['qkv'].shape)}")
    w["o_bf16"] = _bf16(checkpoint.read(p + "self_attn.o_proj.weight", device,
                                        cols=slice(rank * 1024, (rank + 1) * 1024)), "o_proj")
    w["attn_gamma_bf16"] = w["attn_gamma"].to(torch.bfloat16).contiguous()
    w["ffn_gamma_bf16"] = w["ffn_gamma"].to(torch.bfloat16).contiguous()
    if w["kind"] == "full_dsa":
        w["qkw"] = torch.cat([_bf16(w[n], n) for n in ("wiq", "wik", "wis")]).contiguous()
        if tuple(w["qkw"].shape) != (P.INDEX_QKW_ROWS, P.HIDDEN):
            raise ValueError(f"layer {layer} qkw {tuple(w['qkw'].shape)}")
        for n in ("ik_w", "ik_b"):
            w[n] = w[n].float().contiguous()
        # the reference recipe's index-key norm is nn.LayerNorm in bf16 (one launch); the
        # decode-exact 20-launch spelling stays available through prefill.ops
        w["ik_w_bf16"], w["ik_b_bf16"] = w["ik_w"].to(torch.bfloat16).contiguous(), w["ik_b"].to(torch.bfloat16).contiguous()
        adapters.dsa_index_keys(w, capacity)
    else:
        w["sink_bf16"] = w["sink"].to(torch.bfloat16).contiguous()
    if layer == 0:
        w["dense_gate_up"] = torch.cat([_bf16(w[n], n) for n in ("dense_gate", "dense_up")]).contiguous()
        w["dense_down_bf16"] = _bf16(w["dense_down"], "dense_down")
    else:
        w["router"] = checkpoint.read(p + "mlp.gate.weight", device).float().contiguous()
        w["router_bias"] = w["router_bias"].float().contiguous()
        experts = load_expert_rank(checkpoint, layer, rank, device)
        w["ep_w13"], w["ep_w2"] = experts["w13"], experts["w2"]
        w["ep_first_expert"] = rank * P.EXPERTS_PER_RANK
    return w


def kernel_prefill_rank(rw, checkpoint, *, capacity: int):
    """One rank's executor namespace over ``NaiveNRankWeights`` ``rw`` (caches shared)."""
    device = rw.dev
    with torch.inference_mode(), torch.cuda.device(device):
        layers = [kernel_prefill_layer(lw, checkpoint, layer, rw.rank, device, capacity=capacity)
                  for layer, lw in enumerate(rw.layer)]
        ns = SimpleNamespace(rank=rw.rank, dev=device, layers=len(layers), layer=layers,
                             precision=PRECISION_ID, embedding=_bf16(rw.embedding, "embedding"),
                             rope_full=rw.rope_full, rope_swa=rw.rope_swa,
                             head_gamma_bf16=rw.head_gamma.to(torch.bfloat16).contiguous(),
                             lm_head_ref=rw.lm_head_ref)
        torch.cuda.synchronize(device)
    return ns


def kernel_prefill_weights(weights, checkpoint, *, capacity: int, workers: int = 8):
    """The eight executor namespaces, loaded concurrently (EP experts are the bulk)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda rw: kernel_prefill_rank(rw, checkpoint, capacity=capacity), weights))


__all__ = ["kernel_prefill_layer", "kernel_prefill_rank", "kernel_prefill_weights", "load_expert_rank"]
