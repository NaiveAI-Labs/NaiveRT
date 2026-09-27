"""Reject incompatible round ownership before compilation or TP waits."""
from types import SimpleNamespace

import pytest
import torch

from naivert.backends.naive_n0p5_draft import DraftCore
from naivert.backends.naive_n0p5_draft_tail import DraftTail
from naivert.models.naive_n0p5.dspark import StagedDraftContext


def test_staging_rejects_foreign_shape_and_device():
    b = SimpleNamespace(device=torch.device('cuda', 0))
    for features in (torch.zeros(8, 4096), torch.zeros(7, 4096, dtype=torch.bfloat16),
                     torch.zeros(8, 4096, dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match='target export'):
            StagedDraftContext(b, features)


def test_stage_cannot_borrow_another_cores_buffers():
    b = [SimpleNamespace(rank=r, device=torch.device('cuda',r), capacity=2304,
                         sampling=True, mailbox=None) for r in range(8)]
    core = DraftCore(SimpleNamespace(devices=tuple(range(8))),
                     [SimpleNamespace(attention=(None,)*5, mlp=(None,)*5)]*8, b)
    with pytest.raises(ValueError, match='another draft buffer'):
        core.stage_context(0, SimpleNamespace(buffers=b[1]))


def test_tail_rejects_wrong_sampler_before_build():
    buffers = [SimpleNamespace(sampling=True) for _ in range(8)]
    core = SimpleNamespace(buffers=buffers)
    sampler = SimpleNamespace(anchor=True, buffers=buffers)
    kwargs = dict(host_ring=None,host_flags=None,cancel=None,ring_capacity=64)
    with pytest.raises(ValueError, match='TP8 T=1'):
        DraftTail(core, [None]*8, sampler, [None]*8, **kwargs)
    sampler.anchor = False
    sampler.buffers = list(reversed(buffers))
    with pytest.raises(ValueError, match='TP8 T=1'):
        DraftTail(core, [None]*8, sampler, [None]*8, **kwargs)
    sampler.buffers = buffers
    buffers[7].sampling = False
    with pytest.raises(ValueError, match='TP8 T=1'):
        DraftTail(core, [None]*8, sampler, [None]*8, **kwargs)
