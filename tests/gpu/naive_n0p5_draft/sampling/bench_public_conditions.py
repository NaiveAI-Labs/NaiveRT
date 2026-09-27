"""Public CDF sampler on the sampler branch's fixed logits, varying reuse.

No production changes or per-warp instrumentation. Consumed remains in the
sampler, matching the standalone benchmark. Every timed replay checks tokens,
p/q and external epochs outside the event interval.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import subprocess
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def survivor_counts(sampler):
    """Read existing tagged mailbox state after replay, without device probes."""
    import numpy as np
    u64=C.c_uint64;f32=C.c_float;f64=C.c_double;i32=C.c_int32
    class Inbox(C.Structure):
        _fields_=[('maximum',u64*(15*8)),('sum',u64*(15*8*2)),
            ('histogram',u64*(4*8*8*256*2)),('proposal_p',u64*7),('proposal_q',u64*7),
            ('mass',u64*(8*8*2)),('winner',u64*(8*8)),('epoch',u64),
            ('gumbel',u64*(8*8*2)),('cand_count',u64*(8*8)),('cand',u64*(8*8*4096))]
    class Mailbox(C.Structure):
        _fields_=[('flags',u64*(8*3)),('maximum',f32*15),('sum',f64*15),
            ('histogram',f64*(4*8*256)),('proposal_p',f32*7),('proposal_q',f32*7),
            ('mass',f64*8),('winner',i32*8),('inbox',Inbox)]
    assert sampler.ext.mailbox_size()==C.sizeof(Mailbox)
    result=[]
    for mailbox in sampler.mailboxes:
        raw=mailbox.cpu().numpy()
        words=np.frombuffer(raw,dtype=np.uint64,count=64,
            offset=Mailbox.inbox.offset+Inbox.cand_count.offset).reshape(8,8)
        epoch=int(np.frombuffer(raw,dtype=np.uint64,count=1,offset=Mailbox.inbox.offset+Inbox.epoch.offset)[0])
        assert np.all(words>>np.uint64(32)==epoch)
        result.append((words & np.uint64(0xffffffff)).tolist())
    assert all(x==result[0] for x in result)
    counts=result[0]
    return [dict(row=i,by_source_rank=row,total=sum(row),
        path='global_radix' if max(row)>4096 or sum(row)>8192 else
             ('direct' if sum(row)<=256 else 'local_radix')) for i,row in enumerate(counts)]


def main(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(a.repo))
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler, SHARD
    from naivert.runtime.cuda_backend import CudaRuntimeGraphApi, CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    def xids():
        return [s for s in subprocess.check_output(['dmesg'], text=True).splitlines() if 'NVRM: Xid' in s]
    initial_xids = xids()
    api = CudaRuntimeGraphApi.load('/opt/cuda132-min/lib64/libcudart.so')
    api.library.cudaDeviceEnablePeerAccess.argtypes = [C.c_int, C.c_uint]
    for rank in range(8):
        with torch.cuda.device(rank):
            for peer in range(8):
                if peer == rank: continue
                rc = api.library.cudaDeviceEnablePeerAccess(peer, 0)
                assert rc in (0, 704), rc
                if rc == 704: api.library.cudaGetLastError()
    spec = importlib.util.spec_from_file_location('naivert_dspark_benchmark_gate', a.gate)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    raw = {n: np.fromfile(a.inputs / (n+'.bin'), dtype=np.int64 if n=='proposals' else np.float32)
           for n in ('target', 'draft', 'proposals')}
    p = torch.from_numpy(raw['target'].reshape(8, 8*SHARD))
    q = torch.from_numpy(raw['draft'].reshape(7, 8*SHARD))
    logits=[]; buffers=[]; outputs=[]; evictions=[]; streams=[]
    for rank in range(8):
        with torch.cuda.device(rank):
            streams.append(torch.cuda.Stream(device=rank))
            logits.append(p[:,rank*SHARD:(rank+1)*SHARD].contiguous().cuda(rank))
            buffers.append(SimpleNamespace(device=torch.device('cuda',rank),
                logit_trace=q[:,rank*SHARD:(rank+1)*SHARD].contiguous().cuda(rank),
                draft_tokens=torch.from_numpy(raw['proposals'].copy()).cuda(rank),
                sampling_state=torch.tensor([1234,0],dtype=torch.int64,device=rank),
                wdog=torch.zeros(1,dtype=torch.int32,device=rank)))
            outputs.append(torch.empty(8,dtype=torch.int32,device=rank))
            evictions.append(torch.zeros(128*1024*1024,dtype=torch.uint8,device=rank))
    samplers={n:TargetSampler(logits,buffers,outputs,persistent=n=='persistent')
              for n in ('chain','persistent')}
    binaries={n:dict(path=s.ext.__file__,sha256=sha(s.ext.__file__)) for n,s in samplers.items()}
    binaries['gate']=dict(path=str(a.gate),sha256=sha(a.gate))
    def sync():
        for stream in streams: stream.synchronize()
        assert all(b.wdog.item()==0 for b in buffers)
    for rank in range(8): torch.cuda.synchronize(rank)
    graphs={}; owned=[]
    for name,batch in (('chain',1),('persistent',1),('persistent',16)):
        execs=[]
        for rank in range(8):
            with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
                g=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(g,stream=streams[rank]):
                    for _ in range(batch):samplers[name].run(rank)
                audit=api.inspect_graph(g.raw_cuda_graph())
                assert audit['node_count']==batch*(33 if name=='chain' else 1)
                executable=api.graph_instantiate(g.raw_cuda_graph(),flags=CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH)
                api.graph_upload(executable,streams[rank].cuda_stream)
                execs.append(executable);owned.append(g)
        graphs[name,batch]=execs
    sync()
    flag=api.host_alloc(4,flags=3);word=C.c_uint32.from_address(flag.value);word.value=0
    mapped=[];events=[];ticket=0;fields=0
    for rank in range(8):
        with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
            mapped.append(gate.mapped_pointer(flag.value))
            gate.wait_start(mapped[-1],0,buffers[rank].wdog)
            pair=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for event in pair:event.record()
            events.append(pair)
    sync()
    def values(name):
        s=samplers[name]
        return [dict(tokens=s.outputs[r],p=s.full[r],q=s.draft_full[r]) for r in range(8)]
    def call(name,batch,evict,reference=None):
        nonlocal ticket,fields
        s=samplers[name]
        before=[f.cpu().tolist() for f in s.flags]
        if evict:
            for rank in range(8):
                with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):evictions[rank].add_(1)
            sync()
        ticket+=1
        for rank in range(8):
            with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
                gate.wait_start(mapped[rank],ticket,buffers[rank].wdog)
                events[rank][0].record()
                api.graph_launch(graphs[name,batch][rank],streams[rank].cuda_stream)
                events[rank][1].record()
        word.value=ticket;sync()
        times=[start.elapsed_time(end)*1000/batch for start,end in events]
        for rank,f in enumerate(s.flags):
            gen=before[rank][2]+batch
            assert f.tolist()==[0 if s.persistent else gen,gen,gen]
        if reference is not None:
            for rank,row in enumerate(values(name)):
                for key,x in row.items():
                    assert torch.equal(x.view(torch.uint8),reference[rank][key].view(torch.uint8)),(name,batch,evict,rank,key)
                    fields+=1
        return times
    call('chain',1,False)
    reference=[{k:x.clone() for k,x in row.items()} for row in values('chain')]
    for rank in range(8):torch.cuda.synchronize(rank)
    records=[]
    cases=[(1,False),(16,False),(1,True),(16,True)]
    for cycle in range(a.cycles+4):
        order=cases if cycle%2==0 else cases[::-1]
        for batch,evict in order:
            times=call('persistent',batch,evict,reference)
            if cycle>=4:records.append(dict(cycle=cycle-4,batch=batch,evict=evict,per_rank_us=times))
        if cycle%10==3:print('cycles',cycle+1,flush=True)
    summary=[]
    for batch,evict in cases:
        samples=[max(r['per_rank_us']) for r in records if (r['batch'],r['evict'])==(batch,evict)]
        summary.append(dict(batch=batch,evict=evict,median_us=st.median(samples),min_us=min(samples),max_us=max(samples)))
    profiled=[]
    if a.profile:
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for batch,evict in cases:
                label=f'sampler_conditions.b{batch}.e{int(evict)}'
                torch.cuda.nvtx.range_push(label)
                try:times=call('persistent',batch,evict,reference)
                finally:torch.cuda.nvtx.range_pop()
                profiled.append(dict(label=label,batch=batch,evict=evict,per_rank_us=times))
        finally:torch.cuda.cudart().cudaProfilerStop()
    assert initial_xids==xids()
    assert all(sha(v['path'])==v['sha256'] for v in binaries.values())
    survivors=survivor_counts(samplers['persistent'])
    result=dict(pass_checks=True,summary=summary,samples=records,profiled=profiled,fields=fields,binaries=binaries,
        survivor_counts=survivors,
        inputs={n:sha(a.inputs/(n+'.bin')) for n in raw},harness_sha256=sha(__file__),no_new_xids=True,
        scope=__doc__,timer='Max-rank CUDA events / batch, mapped-host gate, device-launchable graph; four warmup cycles')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    sync();api.host_free(flag)
    for execs in graphs.values():
        for rank,executable in enumerate(execs):
            with torch.cuda.device(rank):api.graph_exec_destroy(executable)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','inputs','gate','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cycles',type=int,default=40)
    p.add_argument('--profile',action='store_true')
    main(p.parse_args())
