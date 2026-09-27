"""Exact D12/sampler fork after target head, preserving every other graph edge."""
import ctypes as C

from naivert.runtime.cuda_backend import CudaGraphEdgeData
from naivert.runtime.pdl_graph_edges import _check, _configure


def fork_staged_context(api, graph, names):
    """Apply after PDL installation, before graph instantiation.

    Capture starts linear: target head -> D12 -> sampler -> tail. The result
    forks D12 and the sampler after the head, joining both at the tail. No
    target-forward overlap is introduced. A failed audit must never launch.
    """
    before = api.snapshot_graph(graph)
    order = before.linear_order(kernel_only=True)
    handles = [before.node_handles[i] for i in order]
    if len(names)!=len(handles):
        raise ValueError('kernel-name inventory differs from graph')
    def unique(marker):
        positions = [i for i,n in enumerate(names) if marker in n]
        if len(positions)!=1:
            raise ValueError(f'expected one {marker} kernel, found {positions}')
        return positions[0]
    d12, tail = unique('naivert_dspark_fc_ctx_kv_kernel'), unique('naivert_dspark_tail_chained_entry')
    if (d12<1 or d12+1>=tail or tail!=len(handles)-1
            or 'head_proj' not in names[d12-1] or 'rendezvous' not in names[d12+1]):
        raise ValueError('staged D12 must follow target head and precede sampler/tail')
    edges = {(before.node_handles[a],before.node_handles[b]):bytes(data)
             for (a,b),data in zip(before.edges,before.edge_data)}
    default = CudaGraphEdgeData()
    for a,b in ((d12-1,d12),(d12,d12+1),(tail-1,tail)):
        if edges[handles[a],handles[b]]!=bytes(default):
            raise ValueError('staging fork/join requires ordinary completion edges')
    remove, add = _configure(api.library, int(api.capabilities.runtime_major))
    def edit(fn,a,b):
        _check(fn(C.c_void_p(graph),(C.c_void_p*1)(handles[a]),(C.c_void_p*1)(handles[b]),
                  (CudaGraphEdgeData*1)(default),1),'D12 graph dependency')
    edit(add,d12-1,d12+1)
    edit(add,d12,tail)
    edit(remove,d12,d12+1)
    expected = dict(edges)
    del expected[handles[d12],handles[d12+1]]
    expected[handles[d12-1],handles[d12+1]] = bytes(default)
    expected[handles[d12],handles[tail]] = bytes(default)
    after = api.snapshot_graph(graph)
    actual = {(after.node_handles[a],after.node_handles[b]):bytes(data)
              for (a,b),data in zip(after.edges,after.edge_data)}
    if set(after.node_handles)!=set(handles) or any(after.node_types) or actual!=expected:
        raise RuntimeError('staging mutation changed unrelated nodes or dependencies')
    # The only new edges advance in the original topological order. Preserve
    # all PDL bytes verbatim, and prove a single input and final tail sink.
    degree = {h:0 for h in handles}; successors = {h:[] for h in handles}
    for source,destination in actual:
        degree[destination]+=1; successors[source].append(destination)
    roots = [h for h in handles if degree[h]==0]
    pending = roots.copy(); visited = []
    while pending:
        node=pending.pop(); visited.append(node)
        for child in successors[node]:
            degree[child]-=1
            if degree[child]==0: pending.append(child)
    if roots!=[handles[0]] or len(visited)!=len(handles) or [h for h in handles if not successors[h]]!=[handles[-1]]:
        raise RuntimeError('staged graph is cyclic or has an unexpected root/sink')
    return dict(nodes=len(handles),edges=len(actual),d12_position=d12,
                sampler_position=d12+1,tail_position=tail,preserves_other_edges=True)
