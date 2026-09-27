"""CPU storage checks for target export/token/control ownership, without CUDA."""
from types import SimpleNamespace

import pytest
import torch

from naivert.backends.naive_n0p5_draft_target import DraftTarget
from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS


def inputs():
    buffers=[];verifies=[]
    for rank in range(8):
        b=SimpleNamespace(device=torch.device('cpu'),capacity=2304,
                          block_ids=torch.zeros(8,dtype=torch.int64))
        rt=SimpleNamespace(max_seq_len=2304,stream=None,pos=torch.zeros(1,dtype=torch.int32),
                           generation=torch.zeros(1,dtype=torch.int32),epoch=torch.zeros(1,dtype=torch.int64))
        v=SimpleNamespace(rank=rank,dev=b.device,rt=rt,graph=None,
            resid=[torch.empty(8,4096,dtype=torch.bfloat16) for _ in range(49)],
            tokens=torch.empty(8,dtype=torch.int64),logits=torch.empty(8,19072))
        def phases(*,head_inbox_ptrs,advance,v=v):
            v.received=(head_inbox_ptrs,advance)
            return [('embed8',lambda:None),('head8.u0',lambda:None)]
        v.phases=phases
        buffers.append(b);verifies.append(v)
    return SimpleNamespace(buffers=buffers),verifies


def test_bind_existing_storage_without_export_copy():
    core,verifies=inputs();target=DraftTarget(core,verifies)
    for r,(b,v) in enumerate(zip(core.buffers,verifies)):
        assert v.tokens is b.block_ids
        assert target.states[r].features is v.resid[TARGET_LAYER_IDS[r]+1]
        assert target.states[r].position is v.rt.pos
        assert target.states[r].epoch is v.rt.epoch
        assert target.logits[r] is v.logits
        assert v.received==(target.inbox_ptrs,False)


def test_reject_captured_provider_before_rebinding():
    core,verifies=inputs();tokens=[v.tokens for v in verifies]
    verifies[7].graph=object()
    with pytest.raises(ValueError,match='captured'):DraftTarget(core,verifies)
    assert all(v.tokens is token for v,token in zip(verifies,tokens))
