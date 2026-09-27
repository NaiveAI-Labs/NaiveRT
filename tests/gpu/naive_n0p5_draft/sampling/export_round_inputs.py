"""Export sampler inputs from the existing fixed-round real-model setup.

Uses the pinned round harness's request checks and a replacement measurement
callback. Only disarms output publication/recursion, as the timing harness does.
No runtime source changes. The exported files stay outside Git.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def measure(a,*,core,target,rounds,anchor_sampler,api,features,restore,sync):
    import torch
    from naivert.runtime.mtp_contract import _DecodeControlLayout
    active_offset=type(_DecodeControlLayout().value).active_rows.offset
    records=[];rd=rounds['persistent'];streams=target.streams
    repeated=None
    if a.repeat_sampler:
        from copy import copy
        # Reuse every buffer, function closure and kernel argument. Only the
        # executable topology differs; disarmed tails never recurse into it.
        repeated=copy(rd)
        repeated.graphs=[];repeated.execs=[];repeated.audits=[]
        repeated.phases=tuple(tuple(item for name,fn in phases
            for item in ([(name,fn)]*(16 if name=='sample' else 1))) for phases in repeated.phases)
        repeated.capture()
        for audit in repeated.audits:
            assert len(audit['kernel_names'])==222
            assert sum('naivert_dspark_persistent' in n for n in audit['kernel_names'])==16
    for position in (200,2200):
        restore(position)
        for r,b in enumerate(core.buffers):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                b.sampling_state.copy_(torch.tensor([1234,0],dtype=torch.int64))
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]):anchor_sampler.run(r)
        sync();anchors=[v.rt.token_out[0].item() for v in target.verifies]
        assert len(set(anchors))==1
        def setup(round_):
            restore(position)
            round_.initialize([f[:position] for f in features],position=position,anchor=anchors[0],seed=1234,max_tokens=64)
            for r in range(8):
                with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                    core.context(r)
                    target.states[r].decode_control[active_offset:active_offset+4].zero_()
            sync()
        setup(rd)
        if repeated:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f'full_model_sampler.{position}.b1')
        rd.launch();sync()
        if repeated:
            torch.cuda.nvtx.range_pop();torch.cuda.cudart().cudaProfilerStop()
        proposals=[b.draft_tokens.cpu() for b in core.buffers]
        assert all(torch.equal(x,proposals[0]) for x in proposals)
        root=a.out.parent/str(position);root.mkdir(exist_ok=False)
        values=dict(target=torch.cat([v.logits.cpu() for v in target.verifies],dim=1),
                    draft=torch.cat([b.logit_trace.cpu() for b in core.buffers],dim=1),
                    proposals=proposals[0])
        for name,x in values.items():x.contiguous().numpy().tofile(root/(name+'.bin'))
        records.append(dict(context=position,anchor=anchors[0],seed=1234,counter=0,
            inputs={n:dict(sha256=sha(root/(n+'.bin')),shape=list(x.shape)) for n,x in values.items()}))
        if repeated:
            expected=[dict(p=rd.sampler.full[r].cpu(),q=rd.sampler.draft_full[r].cpu(),
                           tokens=rd.sampler.outputs[r].cpu()) for r in range(8)]
            setup(repeated)
            before=[f.cpu().tolist() for f in repeated.sampler.flags]
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f'full_model_sampler.{position}.b16')
            repeated.launch();sync()
            torch.cuda.nvtx.range_pop();torch.cuda.cudart().cudaProfilerStop()
            for r,row in enumerate(expected):
                for key,x in dict(p=repeated.sampler.full[r],q=repeated.sampler.draft_full[r],
                                  tokens=repeated.sampler.outputs[r]).items():
                    assert torch.equal(x.cpu().view(torch.uint8),row[key].view(torch.uint8)),(position,r,key)
                gen=before[r][2]+1
                assert repeated.sampler.flags[r].tolist()==[0,gen,gen]
            for name,x in dict(target=torch.cat([v.logits.cpu() for v in target.verifies],dim=1),
                               draft=torch.cat([b.logit_trace.cpu() for b in core.buffers],dim=1),
                               proposals=core.buffers[0].draft_tokens.cpu()).items():
                assert torch.equal(x.view(torch.uint8),values[name].view(torch.uint8)),(position,name)
            records[-1]['repeat16_exact']=True
            records[-1]['repeat16_same_sampler_buffers']=repeated.sampler is rd.sampler
    if repeated:repeated.close()
    return dict(scope=__doc__,records=records,harness_sha256=sha(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','target','checkpoint','native-tail','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--repeat-sampler',action='store_true')
    args=p.parse_args()
    callback=ModuleType('benchmark');callback.measure=measure;sys.modules['benchmark']=callback
    path=args.repo/'tests/gpu/naive_n0p5_draft/round/run.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('pinned_round_inputs',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=SimpleNamespace(**vars(args),cycles=1,persistent=True,profile_fixed=False,profile_device_chain=False,
                      head_candidate=None,attention_candidate=None)
    module.work(a)
