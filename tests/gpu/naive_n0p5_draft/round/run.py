"""Real Naive N0.5 target + draft requests: host-serial reference versus device rounds.

Uses a repeated technical-text prompt at fixed lengths, not the archived 44
requests. Correctness checks validate target exports and round composition;
optional fixed-round timing and device-chain profiling use separate run modes.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


STAMP_NAMES=['0 epoch synced','1 tile max + arrive','2 lead: maxima pushed','3 peaks polled','4 exp done + arrive',
             '5 lead: sums pushed','6 inverses polled','7 normalized','8 pass24 state','9 pass16 state','10 pass8 state',
             '11 pass0 state','12 nucleus + arrive','13 lead: token + consumed-arrival','14 end']


def stamp_table(st,rows=8,tiles=10):
    """Per point: max over the row CTAs of (stamp - earliest point-0 stamp), us; rank 0 and the max rank."""
    out={}
    for kind in ('in_round','alone'):
        per_rank=[]
        for r in range(8):
            v=st[kind][r];ctas=rows*tiles;stride=st['ctas']
            t0=min(v[c] for c in range(ctas))
            per_rank.append([max((v[i*stride+c]-t0)/1e3 for c in range(ctas)) for i in range(15)])
        out[kind]=dict(rank0=[round(x,2) for x in per_rank[0]],
                       max_rank=[round(max(pr[i] for pr in per_rank),2) for i in range(15)])
    return out


def _work(a, resources):
    import torch
    import protocol
    from safetensors import safe_open
    from transformers import AutoTokenizer
    sys.path.insert(0,str(a.repo))
    from naivert.backends.naive_n0p5_draft import DraftCore
    from naivert.backends.naive_n0p5_draft_target import DraftTarget
    from naivert.backends.naive_n0p5_draft_round import DraftRound
    from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    from naivert.models.naive_n0p5.dspark import DraftBuffers,prepare_rank_weights,validate_draft_config,TARGET_LAYER_IDS
    from prefill import prefill_tokens
    from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint,load_tp8_weights
    from naivert.runtime.cuda_backend import CudaRuntimeGraphApi
    from naivert.runtime.mtp_contract import _DecodeControlLayout as Control
    from naivert.runtime.sequencer import _SequencerOutputRecordLayout as Record
    from naivert.runtime.pdl_graph_edges import install_programmatic_edges
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    def xids():
        try:out=subprocess.run(['dmesg'],text=True,capture_output=True).stdout   # unprivileged pods cannot read the kernel buffer
        except FileNotFoundError:out=''
        return [s for s in out.splitlines() if 'NVRM: Xid' in s]
    before_xids=xids()
    inventory={str(p.relative_to(a.repo)):sha(p) for p in (a.repo/'naivert').rglob('*')
               if p.is_file() and p.suffix in ('.py','.cu','.cuh','.cpp','.hpp','.h')}
    verify=load('run_naive_n0p5_verify',a.repo/'tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py');decode=verify.decode
    dsa=load('run_attn_dsa',a.repo/'tests/gpu/attn_dsa/run_attn_dsa.py')
    dsa8=load('run_attn_dsa_rows',a.repo/'tests/gpu/attn_dsa/run_attn_dsa_rows.py')
    swa=load('run_attn_swa',a.repo/'tests/gpu/attn_swa/run_attn_swa.py')
    ops=load_extension(weights='fp8',expert_down='w8a8')
    ext_dsa,ext_dsa8,ext_swa,ext_swa8=dsa.build(False,16),dsa8.build(8,index_heads=16),swa.build(1),swa.build(8)
    decode.enable_torch_tp8_peers()
    kernels=prepare_draft(range(8))
    print('extensions ready',flush=True)
    weights=load_tp8_weights(NaiveNCheckpoint(a.target),48,2304,ext_dsa.workspace_bytes(),ext_swa.workspace_bytes(),
                            expert_down='w8a8',ops=ops)
    print('target weights ready',flush=True)
    validate_draft_config(json.loads((a.checkpoint/'config.json').read_text()))
    with safe_open(a.checkpoint/'model.safetensors',framework='pt',device='cpu') as f:
        raw={k.removeprefix('model.'):f.get_tensor(k).bfloat16().contiguous() for k in f.keys()}
    draft_weights=[]
    for r in range(8):
        draft_weights.append(prepare_rank_weights(raw,weights[r].lm_head_ref.cpu(),rank=r,
                                                 device=f'cuda:{r}',kernels=kernels))
        torch.cuda.synchronize(r)
    del raw
    mailboxes=[torch.zeros(verify.mailbox_bytes(ops),dtype=torch.uint8,device=r) for r in range(8)]
    inboxes=[torch.zeros(4096,dtype=torch.uint8,device=r) for r in range(8)]
    ranks=[decode.RankRuntime(r,weights[r],ops,ext_dsa,ext_swa,2304,mailboxes,inboxes,pdl=True,fused_unproj=True) for r in range(8)]
    for rt in ranks:rt.pdl_markers=decode.PDL_W8A8_MARKERS
    verifies=[verify.VerifyRuntime(rt,ext_dsa8,ext_swa8,ext_dsa8.workspace_bytes(),ext_swa8.workspace_bytes(),
                                  pdl=True,fused_unproj=True) for rt in ranks]
    buffers=[DraftBuffers(kernels,rank=r,capacity=2304,sampling=True) for r in range(8)]
    core=DraftCore(kernels,draft_weights,buffers)
    target=DraftTarget(core,verifies)
    api=CudaRuntimeGraphApi.load(decode.torch_cudart_path())
    dpdl=bool(getattr(a,'draft_pdl',False))
    rounds={n:DraftRound(core,target,api,output_capacity=64,staged=n=='staged',draft_pdl=dpdl) for n in ('serial','staged')}
    if a.persistent:
        rounds['persistent']=DraftRound(core,target,api,output_capacity=64,staged=False,persistent_sampler=True,draft_pdl=dpdl)
    noise_build=None;noise_module=None
    if getattr(a,'head_candidate',None):
        noise_build=json.loads((a.head_candidate.parent.parent/'build.json').read_text())
        assert noise_build['pass_checks'] and noise_build['preserved_instruction_identity']
        assert sha(a.head_candidate)==noise_build['binary_sha256']
        assert all(sha(a.repo/n)==h for n,h in noise_build['source_inventory'].items())
        assert sha(Path(kernels.head_markov.__file__))==noise_build['control_sha256']
        noise_module=load('naivert_dspark_head_markov_noise',a.head_candidate)
        for r in range(8):noise_module.configure(r)
        rounds['noise']=DraftRound(core,target,api,output_capacity=64,staged=False,persistent_sampler=True,draft_pdl=dpdl)
    anchor_sampler=TargetSampler([rt.logits for rt in ranks],buffers,[rt.token_out[:1] for rt in ranks],anchor=True)
    anchor_persistent=(TargetSampler([rt.logits for rt in ranks],buffers,[rt.token_out[:1] for rt in ranks],
                                    anchor=True,persistent=True) if a.persistent else None)
    old_tail=load('naivert_mtp8_control',a.native_tail) if a.native_tail else None
    modules={'target':ops,'dsa':ext_dsa,'dsa8':ext_dsa8,'swa':ext_swa,'swa8':ext_swa8,
             **({'native_tail':old_tail} if old_tail else {}),'tail':rounds['serial'].tail.extension,'sampler':anchor_sampler.ext,
             **{n:getattr(kernels,n) for n in ('context','attention','mlp','head_markov')}}
    binaries={n:dict(path=m.__file__,sha256=sha(m.__file__)) for n,m in modules.items()}
    if noise_module:
        binaries['noise_head']=dict(path=noise_module.__file__,sha256=sha(noise_module.__file__))
    if anchor_persistent:
        binaries['persistent_sampler']=dict(path=anchor_persistent.ext.__file__,sha256=sha(anchor_persistent.ext.__file__))
        persistent_tail=rounds['persistent'].tail.extension
        binaries['persistent_tail']=dict(path=persistent_tail.__file__,sha256=sha(persistent_tail.__file__))
    def sync():
        for rt in ranks:rt.stream.synchronize()
        assert all(b.wdog.item()==0 and b.barrier[0].item()==0 for b in buffers)
    for r in range(8):torch.cuda.synchronize(r)
    for rt in ranks:rt.capture()
    # Select only the captured D56 function. Runtime source/defaults and the
    # shared buffers stay unchanged; restore the normal kernel set immediately.
    from dataclasses import replace
    for name,rd in rounds.items():
        try:
            if name=='noise':core.kernels=replace(kernels,head_markov=noise_module)
            rd.capture()
        finally:core.kernels=kernels
        if noise_module:
            for audit in rd.audits:
                count=sum('dspark_head_markov_noise_kernel' in n for n in audit['kernel_names'])
                assert count==int(name=='noise'),(name,count)
    if noise_module:
        for original,changed in zip(rounds['persistent'].audits,rounds['noise'].audits):
            # Mangled names encode identifier length, so compare
            # the unchanged prefix/suffix instead of a raw substring replace.
            left=original['kernel_names'];right=changed['kernel_names']
            assert len(left)==len(right)==207
            differences=[(x,y) for x,y in zip(left,right) if x!=y]
            assert len(differences)==1
            x,y=differences[0]
            assert 'naivert_dspark_head_markov_kernel' in x and 'dspark_head_markov_noise_kernel' in y
    reference=[]
    serial=rounds['serial']
    for r,rt in (enumerate(ranks) if old_tail else ()):
        with torch.cuda.device(r),torch.cuda.stream(rt.stream):
            g=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g,stream=rt.stream):
                for _,fn in serial.phases[r][:-1]:fn()
                old_tail.epilogue(*serial.tail.arguments[r][:-2],False)
            names=api.kernel_function_names(g.raw_cuda_graph())
            install_programmatic_edges(api,g.raw_cuda_graph(),from_positions=[i-1 for i,n in enumerate(names)
                if i and any(marker in n for marker in rt.pdl_markers)])
            g.instantiate();reference.append(g)
    sync()
    print('captured real target/draft graphs',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(a.target,local_files_only=True,trust_remote_code=False)
    text='Explain how memory bandwidth, cache reuse, and tensor parallel communication affect GPU inference latency. '
    part=tokenizer.encode(text,add_special_tokens=False)
    prompt=(part*((2200+len(part)-1)//len(part)))[:2200]
    features=[torch.empty(2200,4096,dtype=torch.bfloat16,device=r) for r in range(8)]
    snapshots={}
    cache_names=('k_cache','v_cache','ki_cache','ki_scale')
    prefill_mode=getattr(a,'prefill','verify8')
    kprefill=None
    if prefill_mode in ('kernel','native'):
        from naivert.prefill.comms import TP8Comm
        from naivert.prefill.draft_features import DraftFeaturePrefillExecutor
        from naivert.prefill.planning import PrefillPlanConfig
        from naivert.prefill.operands import kernel_prefill_weights
        kweights=kernel_prefill_weights(weights,NaiveNCheckpoint(a.target),capacity=2304)
        streams=[rt.stream for rt in ranks]
        if prefill_mode=='native':
            from naivert.prefill.native.executor import NativePrefill
            kprefill=NativePrefill(kweights,streams,feature_layers=TARGET_LAYER_IDS,features=features)
            resources.callback(kprefill.close)
        else:
            kprefill=DraftFeaturePrefillExecutor(kweights,streams,comm=TP8Comm([w.dev for w in kweights],streams),
                max_seq_len=2304,graphs=True,plan_config=PrefillPlanConfig(strategy='budget'),feature_layers=TARGET_LAYER_IDS,features=features)
        print('kernel prefill executor ready',kprefill.versions,flush=True)
    for start,end in ((0,200),(200,2200)):
        # Each segment ends in seq1 steps, so the snapshot's logits/resid are the seq1 step's.
        prefill_tokens(ranks,verifies,prompt[:end],features,TARGET_LAYER_IDS,start=start,mode=prefill_mode,executor=kprefill)
        sync()
        snapshots[end]=dict(caches=[[{n:lw[n].cpu().clone() for n in cache_names if n in lw}
                                  for lw in w.layer] for w in weights],
                            logits=[rt.logits.cpu().clone() for rt in ranks])
        print('prefilled real prompt',end,prefill_mode,flush=True)
    if kprefill is not None and hasattr(kprefill, 'close'):
        kprefill.close()
    def restore(position):
        for r,rt in enumerate(ranks):
            with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                for lw,stored in zip(weights[r].layer,snapshots[position]['caches'][r]):
                    for n,x in stored.items():lw[n].copy_(x)
                rt.logits.copy_(snapshots[position]['logits'][r]);rt.pos.fill_(position)
        sync()
    def control(rd,r):return Control.from_buffer_copy(bytes(target.states[r].decode_control.cpu().tolist())).value
    def read_tokens(rd,count):
        return [Record.from_buffer_copy(bytes(rd.host_ring[i*C.sizeof(Record):(i+1)*C.sizeof(Record)].tolist())).token
                for i in range(count)]
    fields=byte_count=0;records=[];anchor_protocol=[]
    for position in (200,2200):
        for budget in (16,64):
            seed=1234 if budget==16 else -1
            restore(position)
            for r,b in enumerate(buffers):
                with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):
                    b.sampling_state.copy_(torch.tensor([seed,0],dtype=torch.int64))
            for r in range(8):
                with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):anchor_sampler.run(r)
            sync();anchors=[rt.token_out[0].item() for rt in ranks];assert len(set(anchors))==1
            if anchor_persistent:
                before=protocol.snapshot(anchor_persistent)
                for r in range(8):
                    with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):anchor_persistent.run(r)
                sync()
                anchor_protocol.append(protocol.check(anchor_persistent,before,1))
                assert [rt.token_out[0].item() for rt in ranks]==anchors
                assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8))
                           for x,y in zip(anchor_sampler.full,anchor_persistent.full))
            expected=None
            for mode in (*(('reference',) if old_tail else ()),*rounds):
                rd=rounds['serial' if mode=='reference' else mode]
                restore(position)
                rd.initialize([f[:position] for f in features],position=position,anchor=anchors[0],
                              seed=seed,max_tokens=budget)
                sync()
                before=protocol.snapshot(rd.sampler)
                if mode=='reference':
                    while True:
                        for r in range(8):
                            with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):reference[r].replay()
                        sync()
                        if control(rd,0).flags&1:break
                else:
                    flags=rd.host_flags.numpy()
                    trace=a.profile_device_chain and mode=='persistent' and budget==64
                    if trace:
                        torch.cuda.cudart().cudaProfilerStart()
                        torch.cuda.nvtx.range_push(f'device_chain.{position}.{mode}')
                    try:
                        rd.launch();deadline=time.monotonic()+90
                        if trace:torch.cuda.nvtx.mark(f'submitted.{position}.{mode}')
                        while not (flags&1).any():
                            if time.monotonic()>deadline:raise TimeoutError('device round did not publish stop')
                            time.sleep(.001)
                        if trace:torch.cuda.nvtx.mark(f'observed_stop.{position}.{mode}')
                        sync()
                    finally:
                        if trace:
                            torch.cuda.nvtx.range_pop()
                            torch.cuda.cudart().cudaProfilerStop()
                ctrls=[control(rd,r) for r in range(8)]
                assert all(c.output_count==budget and c.flags&1 for c in ctrls)
                assert len({c.iteration for c in ctrls})==1
                sampler_protocol=protocol.check(rd.sampler,before,ctrls[0].iteration)
                sampler_stamps=None
                if mode=='persistent' and hasattr(rd.sampler.ext,'stamps'):
                    # Stamps build: the last round's sampler phases in situ, then the same kernel launched
                    # alone on the same buffers (4 back-to-back invocations per rank) as the standalone reference.
                    in_round=[rd.sampler.ext.stamps(r).tolist() for r in range(8)]
                    for _ in range(4):
                        for r in range(8):
                            with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):rd.sampler.run(r)
                        sync()
                    alone=[rd.sampler.ext.stamps(r).tolist() for r in range(8)]
                    sampler_stamps=dict(points=int(rd.sampler.ext.stamp_points()),ctas=int(rd.sampler.ext.stamp_ctas()),
                                        in_round=in_round,alone=alone)
                    print('STAMPS',position,budget,json.dumps(stamp_table(sampler_stamps)),flush=True)
                tokens=read_tokens(rd,budget)
                # Flush the serial path's pending final accepted context only
                # after its last draft and target outputs have been computed.
                if not rd.staged:
                    for r in range(8):
                        with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):core.context(r)
                    sync()
                result=[]
                for r,(b,v) in enumerate(zip(buffers,verifies)):
                    result.append({**{f'export{layer}':v.resid[layer+1].cpu().clone() for layer in TARGET_LAYER_IDS},
                        'target_logits':v.logits.cpu().clone(),'draft_logits':b.logit_trace.cpu().clone(),
                        'draft_tokens':b.draft_tokens.cpu().clone(),'p':rd.sampler.full[r].cpu().clone(),
                        'q':rd.sampler.draft_full[r].cpu().clone(),'counts':rd.tail.states[r][7].cpu().clone(),
                        'control':target.states[r].decode_control.cpu().clone(),
                        'draft_k':b.kv_k[:,:b.capacity].contiguous().cpu().clone(),
                        'draft_v':b.kv_v[:,:b.capacity].contiguous().cpu().clone(),
                        **{f'cache{layer}.{n}':lw[n].cpu().clone() for layer,lw in enumerate(weights[r].layer)
                           for n in cache_names if n in lw}})
                if expected is None:expected=(tokens,result)
                else:
                    assert tokens==expected[0],(position,budget,mode,'tokens')
                    for r,row in enumerate(result):
                        for n,x in row.items():
                            if not torch.equal(x.view(torch.uint8),expected[1][r][n].view(torch.uint8)):
                                torch.save(dict(position=position,budget=budget,mode=mode,rank=r,field=n,
                                                actual=x,expected=expected[1][r][n]),a.out.with_suffix('.failure.pt'))
                                raise AssertionError((position,budget,mode,r,n))
                            fields+=1;byte_count+=x.numel()*x.element_size()
                records.append(dict(position=position,budget=budget,seed=seed,mode=mode,anchor=anchors[0],
                    tokens=tokens,iterations=ctrls[0].iteration,counts=result[0]['counts'].tolist(),exact=True,
                    sampler_protocol=sampler_protocol,sampler_stamps=sampler_stamps))
                print('PASS request',position,budget,mode,'rounds',ctrls[0].iteration,flush=True)
    timing=None
    if a.cycles:
        from benchmark import measure
        timing=measure(a,core=core,target=target,rounds=rounds,anchor_sampler=anchor_sampler,
                       api=api,features=features,restore=restore,sync=sync)
    for rd in rounds.values():rd.close()
    assert xids()==before_xids
    assert all(sha(a.repo/n)==h for n,h in inventory.items())
    assert all(sha(x['path'])==x['sha256'] for x in binaries.values())
    if noise_build:
        generated=a.head_candidate.parent.parent
        assert all(sha(generated/n)==h for n,h in noise_build['generated_sha256'].items())
    a.out.write_text(json.dumps(dict(pass_checks=True,scope=__doc__,records=records,fields=fields,bytes_checked=byte_count,
        graphs={n:rd.audits for n,rd in rounds.items()},source_inventory=inventory,binaries=binaries,
        harness_sha256=sha(__file__),verify_driver_sha256=sha(a.repo/'tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py'),
        prompt_tokens=prompt,no_new_xids=True,timing=timing,anchor_protocol=anchor_protocol,
        profile_device_chain=a.profile_device_chain,head_candidate_build=noise_build),indent=2)+'\n')
    print('PASS complete requests',fields,byte_count,flush=True)


def work(a):
    with ExitStack() as resources:
        _work(a, resources)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('repo','target','checkpoint','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--native-tail',type=Path,help='pinned reference tail DSO: adds the host-replayed reference mode; required by --cycles')
    p.add_argument('--cycles',type=int,default=0)
    p.add_argument('--profile-fixed',action='store_true')
    p.add_argument('--persistent',action='store_true')
    p.add_argument('--profile-device-chain',action='store_true')
    p.add_argument('--draft-pdl',action='store_true',help='programmatic edges into D3/D4 inside the draft chain')
    p.add_argument('--prefill',choices=('verify8','seq1','kernel','native'),default='verify8',help='prompt ingestion: verify8 replay (default), seq1 replay, bucketed kernel reference, or native 4096-row chunks; each finishes with a seq1 tail')
    p.add_argument('--head-candidate',type=Path,help='Test-only public D56 early-noise extension')
    a=p.parse_args()
    if a.cycles and not a.native_tail:
        p.error('--cycles needs --native-tail (the benchmark gate DSO is looked up beside it)')
    if a.profile_device_chain and (not a.persistent or a.profile_fixed or a.cycles):
        p.error('--profile-device-chain requires --persistent and no fixed-round timing')
    if a.head_candidate and (not a.persistent or a.profile_fixed or a.profile_device_chain):
        p.error('--head-candidate requires --persistent and an unprofiled run')
    work(a)
