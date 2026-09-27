"""Bind the existing Naive N verify-8 provider to draft-owned tokens and tail state."""
import ctypes

import torch

from naivert.backends.naive_n0p5_draft_tail import DraftVerifyState
from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS, VOCAB_SHARD
from naivert.runtime.mtp_contract import _DecodeControlLayout


class DraftTarget:
    """An uncaptured verify-8 provider per rank; kernels and caches stay owned there.

    Providers expose ``rt``, ``resid``, ``tokens``, ``logits`` and
    ``phases(head_inbox_ptrs=..., advance=False)``. The phase list must end after
    its head and must not advance target state: DraftTail performs that update.
    Construction aliases the provider's token input to D56's block_ids. Never
    rebind a provider that already has a captured graph.
    """
    def __init__(self, core, verifies):
        self.core, self.verifies = core, tuple(verifies)
        if len(self.verifies) != 8:
            raise ValueError("draft target requires eight verify providers")
        for r, (b, v) in enumerate(zip(core.buffers, self.verifies)):
            if (v.rank != r or v.dev != b.device or v.rt.max_seq_len != b.capacity
                    or v.graph is not None or len(v.resid) != 49
                    or v.tokens.shape != (8,) or v.tokens.dtype != torch.int64
                    or v.logits.shape != (8, VOCAB_SHARD) or v.logits.dtype != torch.float32
                    or v.logits.device != b.device or not v.logits.is_contiguous()):
                raise ValueError(f"rank {r}: incompatible or already captured verify provider")
            for t in v.resid:
                if (t.shape != (8, 4096) or t.dtype != torch.bfloat16 or
                        t.device != b.device or not t.is_contiguous()):
                    raise ValueError(f"rank {r}: target residuals must be BF16 [8,4096]")
            for t, dtype in ((v.rt.pos, torch.int32), (v.rt.generation, torch.int32),
                             (v.rt.epoch, torch.int64)):
                if t.numel()!=1 or t.dtype!=dtype or t.device!=b.device or not t.is_contiguous():
                    raise ValueError(f"rank {r}: incompatible target control scalar")
        # Keep verify's head protocol separate from seq1/prefill's head inbox.
        self.inboxes = tuple(torch.zeros(4096, dtype=torch.uint8, device=b.device) for b in core.buffers)
        self.inbox_ptrs = tuple(t.data_ptr() for t in self.inboxes)
        self.states, self.phases = [], []
        self.streams = tuple(v.rt.stream for v in self.verifies)
        self.logits = tuple(v.logits for v in self.verifies)
        for r, (b, v) in enumerate(zip(core.buffers, self.verifies)):
            v.tokens = b.block_ids
            state = DraftVerifyState(v.resid[TARGET_LAYER_IDS[r]+1], v.rt.pos,
                v.rt.generation, v.rt.epoch, torch.zeros(8,dtype=torch.int32,device=b.device),
                torch.zeros(ctypes.sizeof(_DecodeControlLayout),dtype=torch.uint8,device=b.device),
                torch.zeros(1,dtype=torch.int64,device=b.device))
            phases = tuple(v.phases(head_inbox_ptrs=self.inbox_ptrs, advance=False))
            if not phases or not phases[-1][0].startswith('head') or any(n=='advance8' for n,_ in phases):
                raise ValueError("verify provider must end after head without advancing state")
            self.states.append(state)
            self.phases.append(phases)
        if len({tuple(n for n,_ in phases) for phases in self.phases})!=1:
            raise ValueError("verify phase order differs across ranks")
