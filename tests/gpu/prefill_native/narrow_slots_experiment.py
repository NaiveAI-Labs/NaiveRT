"""Use a stable narrow expert-key sort instead of a 64-bit unique pair key.

Existing ATen sort/search/scatter primitives only. Expert and row slots remain
int64; this changes the temporary sort key, not the public layout or ordering.
"""
import os

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native.executor import NativePrefill as Baseline


def narrow_slots(topk_ids, *, first_expert, experts=P.EXPERTS_PER_RANK, key_dtype=torch.uint8):
    P._on_current_device(topk_ids)
    assert 0 < experts <= torch.iinfo(key_dtype).max
    n, k = topk_ids.shape
    local = topk_ids.to(torch.int64) - first_expert
    owned = (local >= 0) & (local < experts)
    expert = torch.where(owned, local, experts)
    flat = expert.reshape(-1)
    keys, perm = torch.sort(flat.to(key_dtype), stable=True)
    boundaries = torch.arange(experts + 1, device=topk_ids.device, dtype=key_dtype)
    starts = torch.searchsorted(keys, boundaries)
    order = torch.arange(n * k, device=topk_ids.device, dtype=torch.int64)
    rank = order - starts.index_select(0, flat.index_select(0, perm))
    slot = torch.empty_like(flat).scatter_(0, perm, rank)
    slot = torch.where(owned.reshape(-1), slot, 0).view(n, k)
    return P.ExpertSlots(expert, slot, (starts[1:] - starts[:-1]).to(torch.int32), perm)


class NativePrefill(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_dtype = getattr(torch, os.environ.get('NAIVERT_SLOT_KEY_DTYPE', 'uint8'))
        self.versions['expert-slot-key-dtype-experiment'] = str(self.key_dtype)

    def _experts(self, r, L):
        original = P.expert_slots
        P.expert_slots = lambda ids, **kw: narrow_slots(ids, key_dtype=self.key_dtype, **kw)
        try:
            return super()._experts(r, L)
        finally:
            P.expert_slots = original
