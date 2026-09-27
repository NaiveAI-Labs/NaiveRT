"""TP8 query-sharded indexer versus replicated work; shared queue only.

The BF16 gather copies payload bits, except its reserved 0x80008000 readiness
word. DSA IDs in [-1, 1048575] never equal that word. Test the complete ID
range, including BF16 NaN/Inf encodings, before benchmarking real selections.
"""
import argparse
import json
import sys
from pathlib import Path
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import stable_dsa_topk as S
from naivert.prefill.comms import TP8Comm
from naivert.prefill.native.collectives import FlashInferCollectives

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--widths', default='69632,131072,1048576')
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    devices = [torch.device('cuda', r) for r in range(8)]
    streams = [torch.cuda.Stream(device=d) for d in devices]
    pool = ThreadPoolExecutor(max_workers=8)
    comm = FlashInferCollectives(TP8Comm(devices, streams))
    S.install_stable_epilogue()
    for d in devices:
        P.prepare_device(d)
    outputs = [torch.empty((4096, 2048), device=d, dtype=torch.int32) for d in devices]
    def sync():
        for s in streams:
            s.synchronize()
    def each(fn):
        for r, d in enumerate(devices):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                fn(r)
    def gather():
        comm.all_gather_([o[r*512:(r+1)*512].view(torch.bfloat16) for r,o in enumerate(outputs)],
                         [o.view(torch.bfloat16) for o in outputs])
    def capture(fn):
        sync(); graphs=[]
        for r,d in enumerate(devices):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                g=torch.cuda.CUDAGraph(); g.capture_begin(capture_error_mode='relaxed'); graphs.append(g)
        fn()
        for r,d in enumerate(devices):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                graphs[r].capture_end()
        return graphs
    def replay(graphs):
        def one(r):
            with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
                graphs[r].replay()
        list(pool.map(one, range(8)))
    def check(expected):
        sync()
        for r,o in enumerate(outputs):
            assert torch.equal(o.cpu(), expected), r
    report={'transport':[], 'indexer':[], 'collective':comm.version}
    out=args.out
    # All possible IDs, including -1, and changed payloads on the captured path.
    graphs=None
    for iteration in range(3):
        expected=((torch.arange(4096*2048,dtype=torch.int32)+iteration*32768) % 1048577 - 1).reshape(4096,2048)
        each(lambda r: outputs[r][r*512:(r+1)*512].copy_(expected[r*512:(r+1)*512]))
        if graphs is None:
            gather(); check(expected); graphs=capture(gather)
        replay(graphs); check(expected)
        report['transport'].append({'iteration':iteration,'exact_all_ranks':True})
    for g in graphs: g.reset()
    print('TRANSPORT_OK',flush=True)
    torch.manual_seed(20260922)
    for width in map(int, args.widths.split(',')):
        q0=(torch.randn(4096,16,128)*.2).to(P.FP8_DTYPE)
        k0=(torch.randn(width,128)*.2).to(P.FP8_DTYPE)
        w0=torch.randn(4096,16)*.1
        qs=[q0.to(d) for d in devices]; ks=[k0.to(d) for d in devices]; ws=[w0.to(d) for d in devices]
        scales=[torch.ones(width,1,device=d) for d in devices]
        starts=[torch.zeros(4096,device=d,dtype=torch.int32) for d in devices]
        ends=[torch.arange(width-4095,width+1,device=d,dtype=torch.int32) for d in devices]
        for d in devices: torch.cuda.synchronize(d)
        def local(r):
            lo,hi=r*512,(r+1)*512
            outputs[r][lo:hi].copy_(P.indexer_select(qs[r][lo:hi],ws[r][lo:hi],ks[r],scales[r],
                                    starts[r][lo:hi],ends[r][lo:hi],max_seqlen=width))
        def run(): each(local); gather()
        with torch.cuda.device(0), torch.cuda.stream(streams[0]):
            expected=P.indexer_select(qs[0],ws[0],ks[0],scales[0],starts[0],ends[0],max_seqlen=width)
        sync(); expected=expected.cpu()
        def full():
            each(lambda r: outputs[r].copy_(P.indexer_select(qs[r],ws[r],ks[r],scales[r],starts[r],ends[r],max_seqlen=width)))
        full(); check(expected); full_graphs=capture(full)
        full_samples=[]
        for rep in range(9):
            sync(); began=time.perf_counter(); replay(full_graphs); sync()
            if rep>=3: full_samples.append((time.perf_counter()-began)*1000)
        check(expected)
        for g in full_graphs: g.reset()
        del full_graphs
        run(); check(expected); graphs=capture(run)
        samples=[]
        for rep in range(9):
            sync(); began=time.perf_counter(); replay(graphs); sync()
            if rep>=3: samples.append((time.perf_counter()-began)*1000)
        check(expected)
        each(lambda r: qs[r].zero_())
        replay(graphs)
        check(torch.arange(2048,dtype=torch.int32).expand(4096,2048))
        row=dict(width=width,exact_all_ranks=True,captured_changed_input_exact=True,
                 wall_median_ms=statistics.median(samples),wall_samples_ms=samples,
                 replicated_wall_median_ms=statistics.median(full_samples),replicated_wall_samples_ms=full_samples,
                 speedup=statistics.median(full_samples)/statistics.median(samples))
        report['indexer'].append(row); out.write_text(json.dumps(report,indent=2)+'\n')
        print('SHARDED',json.dumps(row),flush=True)
        for g in graphs: g.reset()
        del qs,ks,ws,scales,starts,ends
        torch.cuda.empty_cache()
    comm.close(); pool.shutdown()
    report['success']=True; out.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
