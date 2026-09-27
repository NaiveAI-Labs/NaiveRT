"""Test an existing DeepEP combine template instantiated for FP32.

No expert GEMMs are timed. Already-weighted rank partials are supplied by the
test, so these measurements bound transport only, not full MoE speedup. Run via
gpu_profile_queue. The module is built by prepare_deepep_fp32_combine.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def run(a, report):
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    spec = importlib.util.spec_from_file_location('naivert_deep_ep_fp32_probe', a.library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    devices = [torch.device('cuda', r) for r in range(8)]
    streams = [torch.cuda.Stream(device=d) for d in devices]
    # Upstream Buffer::total_bytes is int. Keep each channel data region below
    # 2 GiB; 128 SMs x 256 queue rows x H4096 FP32 hits that signed limit.
    channel_data_bytes = (a.sms // 2) * 8 * a.recv_tokens * a.hidden * 4
    assert channel_data_bytes < 2**31, 'DeepEP Buffer byte count would overflow int32'
    config = module.Config(a.sms, a.send_tokens, a.recv_tokens, 6, 128)
    nvl_bytes = config.get_nvl_buffer_size_hint(a.hidden*4, 8)
    buffers = []
    report.update(library=str(a.library), library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                  scope='already-weighted FP32 rank partials; local expert combination excluded',
                  sms=a.sms, send_tokens=a.send_tokens, recv_tokens=a.recv_tokens,
                  channel_data_bytes=channel_data_bytes,
                  hidden=a.hidden, nvl_bytes_per_gpu=nvl_bytes, cases=[])

    def mark(stage):
        report['stage'] = stage
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        print(stage, flush=True)

    def sync():
        for stream in streams:
            stream.synchronize()

    def each(fn, r):
        with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
            return fn(r)

    def parallel(pool, fn):
        return list(pool.map(lambda r: each(fn, r), range(8)))

    def capture(fn):
        graphs,events=[],[]
        for r in range(8):
            with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
                g=torch.cuda.CUDAGraph()
                start,end=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
                g.capture_begin(capture_error_mode='relaxed'); start.record(streams[r])
                graphs.append(g); events.append((start,end))
        for _ in range(a.inner):
            for r in range(8): each(fn,r)
        for r in range(8):
            with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
                events[r][1].record(streams[r]); graphs[r].capture_end()
        return graphs,events

    try:
        for r in range(8):
            buffers.append(each(lambda i: module.Buffer(i,8,nvl_bytes,0,False,True),r))
        for r,b in enumerate(buffers): each(lambda i:b.sync_local(buffers),r)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for case in ('uniform','one_owner'):
                mark(case+'_dispatch')
                rng=torch.Generator().manual_seed(1437)
                ids_cpu=torch.rand(4096,256,generator=rng).topk(8,dim=-1).indices
                if case=='one_owner': ids_cpu.fill_(0); ids_cpu[:]=torch.arange(8)
                weights_cpu=torch.full((4096,8),1/8,dtype=torch.float32)
                ids=[each(lambda i:ids_cpu[i*512:(i+1)*512].to(devices[i]),r) for r in range(8)]
                weights=[each(lambda i:weights_cpu[i*512:(i+1)*512].to(devices[i]),r) for r in range(8)]
                x=[each(lambda i:torch.zeros((512,16),device=devices[i],dtype=torch.bfloat16),r) for r in range(8)]
                def dispatch(r):
                    counts,_,experts,in_rank,_=buffers[r].get_dispatch_layout(ids[r],256,None,False,False)
                    return buffers[r].intranode_dispatch(x[r],None,ids[r],weights[r],counts,in_rank,experts,
                                                        0,None,None,1,4096,config,None,False,False)
                received=parallel(pool,dispatch); sync()
                heads=[each(lambda r:received[r][9].clone(),r) for r in range(8)]
                # Source rank and local row come directly from the dispatch handle.
                maps=[]
                for r,rec in enumerate(received):
                    prefix,src=rec[5],rec[8]
                    def mapping(i):
                        owner=torch.searchsorted(prefix[:,i].contiguous(),torch.arange(4096,device=devices[i]),right=True)
                        return (src.long()+owner.long()*512).clamp(0,4095)
                    maps.append(each(mapping,r))
                partials=[]
                for r in range(8):
                    partials.append(each(lambda i:torch.full((4096,a.hidden),float(i+1),device=devices[i]),r))
                outputs=[None]*8
                def combine(r):
                    rec=received[r]
                    rec[9].copy_(heads[r])  # combine updates its handle; restore it for repeated probes
                    # Combine needs recv_channel_prefix_matrix (tuple 7), not
                    # the dispatch sender's channel prefix (tuple 6).
                    outputs[r]=buffers[r].intranode_combine(partials[r],None,None,None,rec[8],rec[5],rec[7],
                                                            rec[9],config,None,False,False)[0]
                mark(case+'_eager_integer')
                parallel(pool,combine); sync()
                owner_mask=torch.stack([((ids_cpu//32)==r).any(-1) for r in range(8)])
                expected=(owner_mask*torch.arange(1,9)[:,None]).sum(0).float()
                for r in range(8):
                    assert torch.equal(outputs[r].cpu(),expected[r*512:(r+1)*512,None].expand(512,a.hidden))
                mark(case+'_capture')
                graphs,events=capture(combine)
                def replay(): parallel(pool,lambda r:graphs[r].replay()); sync()
                replay()
                # Change the payload after capture. Compare all output elements to FP64.
                tables=[torch.randn(4096,a.hidden,generator=torch.Generator().manual_seed(971+r)) for r in range(8)]
                for r in range(8):
                    each(lambda i:partials[i].copy_(tables[i].to(devices[i]).index_select(0,maps[i])),r)
                sync(); replay()
                checks=[]
                for r in range(8):
                    lo,hi=r*512,(r+1)*512
                    oracle=sum(tables[i][lo:hi].double()*owner_mask[i,lo:hi,None] for i in range(8))
                    got=outputs[r].cpu().double(); error=got-oracle
                    assert torch.isfinite(got).all()
                    assert float(error.abs().max()) < 1e-5
                    checks.append(dict(rank=r,rms=float(error.square().mean().sqrt()),max_abs=float(error.abs().max())))
                saved=[each(lambda i:outputs[i].clone(),r) for r in range(8)]
                sync()
                replay()
                assert all(torch.equal(left,right) for left,right in zip(saved,outputs))
                samples=[]
                mark(case+'_timing')
                for rep in range(a.samples+5):
                    start=time.perf_counter(); replay()
                    if rep>=5:samples.append(dict(wall_us=(time.perf_counter()-start)*1e6/a.inner,
                                                 rank_us=[s.elapsed_time(e)*1000/a.inner for s,e in events]))
                report['cases'].append(dict(name=case,checks=checks,replay_exact=True,samples_us=samples,
                                            inner=a.inner,
                                            median_wall_us=statistics.median(v['wall_us'] for v in samples),
                                            median_slowest_rank_us=statistics.median(max(v['rank_us']) for v in samples)))
                print(case,report['cases'][-1]['median_wall_us'],'us',flush=True)
                for g in graphs:g.reset()
                del graphs,outputs,partials,tables,saved,received,heads,maps
        report.update(stage='complete',success=True)
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            for d in devices:torch.cuda.synchronize(d)
            for b in buffers:b.destroy(True)
            report['cleanup_completed']=True
        except BaseException as exc:
            report['cleanup_error']=repr(exc)
            if not failed:raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sms',type=int,default=128)
    p.add_argument('--hidden',type=int,default=4096)
    p.add_argument('--recv-tokens',type=int,default=128)
    p.add_argument('--send-tokens',type=int,default=6)
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--inner',type=int,default=16)
    a=p.parse_args()
    report=dict(success=False,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:run(a,report)
    except BaseException as exc:report.update(success=False,error=repr(exc));raise
    finally:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
