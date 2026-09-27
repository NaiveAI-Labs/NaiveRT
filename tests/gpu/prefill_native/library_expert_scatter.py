"""Experimental composition of unchanged SGLang EP scatter kernels.

No device body is modified. Atomic assignment may permute rows within each
expert; callers use the returned pair map and retain fixed top-k combination.
"""
import torch
from naivert.kernels.naive_n0p5 import prefill as P
from naivert.kernels.naive_n0p5.prefill_sglang.scatter import _fwd_kernel_ep_scatter_2


def prepare(codes, scales, ids, q_buf, s_buf, *, first_expert=0, library_offsets=True):
    if library_offsets:
        return P.prepare_compact_experts(codes, scales, ids, q_buf, s_buf,
                                         first_expert=first_expert, alignment=128)
    experts, alignment = P.EXPERTS_PER_RANK, 128
    local = ids.to(torch.int64) - first_expert
    owned = (local >= 0) & (local < experts)
    local_ids = torch.where(owned, local, -1)
    keys = torch.where(owned, local, experts).reshape(-1).to(torch.uint8)
    keys, permutation = torch.sort(keys, stable=True)
    boundaries = torch.searchsorted(keys, torch.arange(experts+1, device=ids.device, dtype=torch.uint8))
    counts = (boundaries[1:] - boundaries[:-1]).to(torch.int32)
    aligned = (counts + (alignment-1)) // alignment * alignment
    total = aligned.sum(dtype=torch.int32).reshape(1)
    ends = aligned.cumsum(0, dtype=torch.int32) - aligned
    dest = torch.full_like(ids, -1, dtype=torch.int64)
    n, k = ids.shape
    # Copy code bytes so FP8 encodings are preserved. The scale path is
    # selected explicitly, with eight packed int32 UE8M0 words per H4096 row.
    _fwd_kernel_ep_scatter_2[(min(n, 8192),)](
        n, ends, codes.view(torch.uint8), codes.stride(0), codes.stride(1),
        scales, scales.stride(0), scales.stride(1),
        local_ids, local_ids.stride(0), local_ids.stride(1),
        q_buf.view(torch.uint8), q_buf.stride(0), q_buf.stride(1),
        s_buf, s_buf.stride(0), s_buf.stride(1),
        dest, dest.stride(0), dest.stride(1),
        topk_num=k, HIDDEN_SIZE=P.HIDDEN, HIDDEN_SIZE_PAD=P.HIDDEN,
        SCALE_HIDDEN_SIZE=P.DISPATCH_WORDS, SCALE_HIDDEN_SIZE_PAD=P.DISPATCH_WORDS,
        ATOMIC_ADD_SEM=None, IS_FP8=True, num_warps=8)
    # Atomic increments leave actual ends, the existing DeepGEMM psum ABI.
    return counts, permutation, dest.reshape(-1), ends, total
