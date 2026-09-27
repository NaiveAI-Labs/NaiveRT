"""Stable bindings between draft compute, target sampling and the device tail."""
from dataclasses import dataclass

import torch

from naivert.kernels.naive_n0p5.dspark_tail import load_tail
from naivert.models.naive_n0p5.dspark import MASK_TOKEN_ID


@dataclass(frozen=True)
class DraftVerifyState:
    """Target-owned state; every tensor must outlive the captured round."""
    features: torch.Tensor
    position: torch.Tensor
    generation: torch.Tensor
    epoch: torch.Tensor
    tokens: torch.Tensor
    decode_control: torch.Tensor
    previous_complete_ns: torch.Tensor


class DraftTail:
    """T=1 accept/geometry/output tail with an optional staged-KV commit.

    Construct before capture, after the core and sampler. The sampler must use
    ``defer_consumed=True`` and precede this tail. Staged D12 must also precede
    it. ``chain=True`` requires a device-launch graph; ordinary eager/captured
    calls use ``chain=False``. All eight ranks execute the same graph topology.

    The owner initializes ``fields[r]['cache_len']`` / ``['ctx_rows']``, target
    state and the decode control before the first round. Subsequent updates are
    device-only. Host output buffers, cancellation and every bound tensor remain
    strongly referenced here. No tensor is allocated in the launch methods.
    """
    def __init__(self, core, targets, sampler, embeddings, *, host_ring, host_flags,
                 cancel, ring_capacity, stages=None, chain=True):
        self.core, self.targets, self.sampler = core, tuple(targets), sampler
        self.embeddings = tuple(embeddings)
        self.stages = None if stages is None else tuple(stages)
        if (len(self.targets) != 8 or len(self.embeddings) != 8
                or sampler.anchor or len(sampler.buffers) != 8
                or any(not b.sampling or sampler.buffers[r] is not b
                       for r, b in enumerate(core.buffers))
                or (self.stages is not None and len(self.stages) != 8)):
            raise ValueError("draft tail requires the core's TP8 T=1 sampler and rank state")
        if type(ring_capacity) is not int or ring_capacity <= 0:
            raise ValueError("output ring capacity must be a positive integer")
        self.host_ring, self.host_flags, self.cancel = host_ring, host_flags, cancel
        self.ring_capacity, self.chain = ring_capacity, bool(chain)
        # The persistent binding omits READY. Its deferred CONSUMED tail owns
        # the generation increment normally performed by the READY phase.
        self.extension = load_tail(no_ready=sampler.persistent)
        self.fields, self.geometry_values, self.accept_values, self.states = [], [], [], []
        dtype = dict(Int=torch.int32, Long=torch.int64, Float=torch.float32,
                     Bool=torch.bool, BFloat16=torch.bfloat16)
        for rank, (b, w, target, embed) in enumerate(zip(
                core.buffers, core.weights, self.targets, self.embeddings)):
            if (sampler.outputs[rank] is not target.tokens or
                    target.features.shape != (8, 4096) or
                    target.features.dtype != torch.bfloat16 or
                    target.features.device != b.device or not target.features.is_contiguous()):
                raise ValueError(f"rank {rank}: target features/tokens do not match the sampler")
            if self.stages is not None and (self.stages[rank].buffers is not b
                                            or self.stages[rank].features is not target.features):
                raise ValueError(f"rank {rank}: staged context must use these target features")
            fields = dict(target_hidden=b.target_hidden, ctx_valid=b.ctx_valid,
                          ctx_pos=b.ctx_pos, kv_dst=b.kv_dst, blk_pos=b.blk_pos,
                          key_add=b.key_add, draft_tokens=b.draft_tokens,
                          block_ids=b.block_ids, resid0=b.resid[0], embed=embed,
                          mask_embedding=w.mask_embedding)
            geometry_layout, accept_layout = self.extension.layouts(b.capacity)
            for name, kind, count in (*geometry_layout, *accept_layout):
                if name not in fields:
                    if count < 0:
                        raise ValueError(f"rank {rank}: missing externally owned {name}")
                    fields[name] = torch.zeros(count, dtype=dtype[kind], device=b.device)
                tensor = fields[name]
                if (tensor.device != b.device or tensor.dtype != dtype[kind]
                        or not tensor.is_contiguous()
                        or (tensor.numel() != count if count >= 0 else
                            tensor.ndim != 2 or tensor.shape[1] != 4096 or tensor.shape[0] < 1)):
                    raise ValueError(f"rank {rank}: invalid tail field {name}")
            fields['row_pos'].copy_(torch.arange(8, device=b.device))
            fields['trash_row'].fill_(b.capacity)
            fields['kv_pos'].copy_(torch.arange(b.capacity + 8, device=b.device))
            fields['kv_pos'][b.capacity:] = -1  # The eight draft rows are always visible.
            arrivals = torch.zeros(1, dtype=torch.int32, device=b.device)
            counts = torch.zeros(10, dtype=torch.int64, device=b.device)
            state = [target.features, target.position, target.generation, target.epoch,
                     arrivals, target.tokens, b.sampling_state, counts,
                     target.decode_control, target.previous_complete_ns, b.wdog]
            if self.stages is not None:
                stage = self.stages[rank]
                state.extend((stage.key, stage.value, b.kv_k, b.kv_v))
            self.fields.append(fields)
            self.geometry_values.append(tuple(fields[n] for n in self.extension.geometry_fields))
            self.accept_values.append(tuple(fields[n] for n in self.extension.accept_fields))
            self.states.append(tuple(state))
        self.arguments = tuple((self.geometry_values[r], self.accept_values[r], b.capacity,
                                MASK_TOKEN_ID, self.states[r], sampler.flag_ptrs,
                                host_ring, host_flags, cancel, ring_capacity, r, self.chain,
                                self.stages is not None)
                               for r, b in enumerate(core.buffers))

    def geometry(self, rank):
        """Prepare initial geometry after the owner sets cache_len / ctx_rows."""
        self.extension.geometry(self.geometry_values[rank], self.core.buffers[rank].capacity)

    def run(self, rank):
        # The repeat-penalty counts go by keyword: the pinned reference tail has the old signature.
        self.extension.epilogue(*self.arguments[rank], rep_count=self.core.buffers[rank].rep_count)
