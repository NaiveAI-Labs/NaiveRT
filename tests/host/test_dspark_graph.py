"""Check that the D12 fork preserves target PDL and fails before invalid edits."""
from types import SimpleNamespace

import pytest

from naivert.runtime.cuda_backend import CudaGraphEdgeData, CudaGraphSnapshot
from naivert.runtime import dspark_graph


NAMES=('draft_head','target_embed','rmsnorm_head_proj','naivert_dspark_fc_ctx_kv_kernel',
       'rendezvous','normalize','broadcast_tokens','naivert_dspark_tail_chained_entry')


class Graph:
    def __init__(self):
        self.library=object();self.capabilities=SimpleNamespace(runtime_major=13)
        self.handles=(108,101,106,102,104,105,103,107)  # Deliberately not linear enumeration.
        pdl=CudaGraphEdgeData();pdl.from_port=1;pdl.type=1
        self.edges={(h,h+1):bytes(pdl if h==102 else CudaGraphEdgeData()) for h in range(101,108)}
    def snapshot_graph(self,graph):
        index={h:i for i,h in enumerate(self.handles)}
        return CudaGraphSnapshot(self.handles,(0,)*8,
            tuple((index[a],index[b]) for a,b in self.edges),tuple(self.edges.values()))
    def configure(self,*args):
        def add(graph,source,destination,data,count):
            self.edges[source[0],destination[0]]=bytes(data[0]);return 0
        def remove(graph,source,destination,data,count):
            del self.edges[source[0],destination[0]];return 0
        return remove,add


def test_fork_preserves_other_edges(monkeypatch):
    graph=Graph();before=dict(graph.edges)
    monkeypatch.setattr(dspark_graph,'_configure',graph.configure)
    report=dspark_graph.fork_staged_context(graph,123,NAMES)
    assert report['nodes']==8 and report['edges']==8 and report['d12_position']==3
    assert graph.edges[102,103]==before[102,103]  # Target's PDL bytes unchanged.
    assert (104,105) not in graph.edges
    assert graph.edges[103,105]==graph.edges[104,108]==bytes(CudaGraphEdgeData())
    assert all(graph.edges[k]==v for k,v in before.items() if k!=(104,105))


@pytest.mark.parametrize('names', [NAMES[:2]+('other',)+NAMES[3:],
                                 NAMES[:4]+('other',)+NAMES[5:],
                                 NAMES[:-1]+('other',),
                                 NAMES[:2]+('naivert_dspark_fc_ctx_kv_kernel',)+NAMES[3:]])
def test_invalid_schedule_does_not_mutate(monkeypatch,names):
    graph=Graph();before=dict(graph.edges)
    monkeypatch.setattr(dspark_graph,'_configure',graph.configure)
    with pytest.raises(ValueError):dspark_graph.fork_staged_context(graph,123,names)
    assert graph.edges==before
