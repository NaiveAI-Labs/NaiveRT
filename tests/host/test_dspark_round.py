"""Invalid request bounds must fail before ingestion or any CUDA operation."""
from types import SimpleNamespace

import pytest

from naivert.backends.naive_n0p5_draft_round import DraftRound


@pytest.mark.parametrize('change',[
    {'position':0},{'position':2297},{'position':200.0},
    {'anchor':-1},{'anchor':152576},{'seed':1<<63},{'seed':-(1<<63)-1},
    {'max_tokens':0},{'max_tokens':65},{'eos':-1},{'eos':152576},
])
def test_request_bounds_before_cuda(change):
    rd=DraftRound.__new__(DraftRound)
    rd.core=SimpleNamespace(buffers=[SimpleNamespace(capacity=2304)])
    rd.capacity=64
    values=dict(position=200,anchor=9,seed=1234,max_tokens=64)
    values.update(change)
    with pytest.raises(ValueError,match='request'):
        rd.initialize([None]*8,**values)


def test_persistent_rejects_unvalidated_staged_schedule_before_cuda():
    with pytest.raises(ValueError,match='serial D12'):
        DraftRound(None,None,None,output_capacity=64,staged=True,persistent_sampler=True)
