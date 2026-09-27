"""Full request gate for public staged and persistent rounds.

Archive mode requires the historical tokens. Current-target mode uses pinned
native draft/tail kernels with the same target math and records archive drift
separately. Both check tokens, acceptance, state and caches, not serving speed.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack

from run import load, sha
import protocol


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def _work(a, resources):
    SEED=int(os.environ.get('NAIVERT_REQUEST_SEED','1234'))   # acceptance-only spread runs; the archive gate is 1234
    candidate_path=getattr(a,'native_candidate',None)
    candidate_source=None if candidate_path is None else dict(path=str(candidate_path),sha256=sha(candidate_path))
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    sys.path.insert(0,str(a.repo))
    capacity=65536
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
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    def xids():return [s for s in subprocess.check_output(['dmesg'],text=True).splitlines() if 'NVRM: Xid' in s]
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
    weights=load_tp8_weights(NaiveNCheckpoint(a.target),48,capacity,ext_dsa.workspace_bytes(),ext_swa.workspace_bytes(),
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
    mailboxes=[torch.zeros(verify.mailbox_bytes(ops),dtype=torch.uint8,device=r) for r in range(8)]
    inboxes=[torch.zeros(4096,dtype=torch.uint8,device=r) for r in range(8)]
    ranks=[decode.RankRuntime(r,weights[r],ops,ext_dsa,ext_swa,capacity,mailboxes,inboxes,pdl=True,fused_unproj=True) for r in range(8)]
    for rt in ranks:rt.pdl_markers=decode.PDL_W8A8_MARKERS
    verifies=[verify.VerifyRuntime(rt,ext_dsa8,ext_swa8,ext_dsa8.workspace_bytes(),ext_swa8.workspace_bytes(),
                                  pdl=True,fused_unproj=True) for rt in ranks]
    buffers=[DraftBuffers(kernels,rank=r,capacity=capacity,sampling=True) for r in range(8)]
    kprefill=None
    if a.prefill in ('kernel','native'):
        from naivert.prefill.comms import TP8Comm
        from naivert.prefill.draft_features import DraftFeaturePrefillExecutor
        from naivert.prefill.planning import PrefillPlanConfig
        from naivert.prefill.operands import kernel_prefill_weights
        t0=time.monotonic()
        kweights=kernel_prefill_weights(weights,NaiveNCheckpoint(a.target),capacity=capacity)
        print('kernel prefill operands ready',round(time.monotonic()-t0,1),'s',flush=True)
    core=DraftCore(kernels,draft_weights,buffers)
    native_core=None
    if a.diagnose_first or a.current_target_reference:
        from native_reference import NativeDraftReference
        native_core=NativeDraftReference(core,raw,a.native_core)
        native_tail=load('naivert_mtp8_control',a.native_tail)
    del raw
    target=DraftTarget(core,verifies)
    api=CudaRuntimeGraphApi.load(decode.torch_cudart_path())
    dpdl=bool(getattr(a,'draft_pdl',False))
    rounds={
        'staged':DraftRound(core,target,api,output_capacity=24576,staged=True,draft_pdl=dpdl),
        'persistent':DraftRound(core,target,api,output_capacity=24576,staged=False,persistent_sampler=True,draft_pdl=dpdl),
    }
    if native_core is not None:
        rounds={'native':DraftRound(core,target,api,output_capacity=24576,staged=False),**rounds}
    if a.acceptance_only:rounds={'persistent':rounds['persistent']}
    anchors={name:TargetSampler([rt.logits for rt in ranks],buffers,[rt.token_out[:1] for rt in ranks],
                              anchor=True,persistent=name=='persistent') for name in rounds}
    modules={'target':ops,'dsa':ext_dsa,'dsa8':ext_dsa8,'swa':ext_swa,'swa8':ext_swa8,
             **{n:getattr(kernels,n) for n in ('context','attention','mlp','head_markov')},
             **{n+'.sampler':s.ext for n,s in anchors.items()},
             **{n+'.tail':rd.tail.extension for n,rd in rounds.items()}}
    if native_core is not None:
        modules.update(native_core=native_core.native,native_tail=native_tail)
    binaries={n:dict(path=m.__file__,sha256=sha(m.__file__)) for n,m in modules.items()}
    def sync():
        for rt in ranks:rt.stream.synchronize()
        assert all(b.wdog.item()==0 and b.barrier[0].item()==0 for b in buffers)
        assert all(s[4].item()==0 for rd in rounds.values() for s in rd.tail.states)
    for r in range(8):torch.cuda.synchronize(r)
    for rt in ranks:rt.capture()
    reference=[]
    for name,rd in rounds.items():
        if name!='native':
            rd.capture()
            continue
        from naivert.runtime.pdl_graph_edges import install_programmatic_edges
        for r,rt in enumerate(ranks):
            with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                g=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(g,stream=rt.stream):
                    native_core.run(r)
                    for _,fn in target.phases[r]:fn()
                    rd.sampler.run(r,defer_consumed=True)
                    native_tail.epilogue(*rd.tail.arguments[r][:-2],False)
                names=api.kernel_function_names(g.raw_cuda_graph())
                pdl=install_programmatic_edges(api,g.raw_cuda_graph(),from_positions=[i-1 for i,n in enumerate(names)
                    if i and any(marker in n for marker in rt.pdl_markers)])
                rd.audits.append(dict(rank=r,pdl=pdl,kernel_names=names))
                g.instantiate();reference.append(g)
        # Initialization must also ingest prompt KV through the pinned D12.
        rd.core=native_core
    sync()
    old=json.loads((a.baseline/'manifest.json').read_text())
    refs=[json.loads(line) for line in (a.baseline/'completed.jsonl').read_text().splitlines()]
    old_summary=json.loads((a.baseline/'summary.json').read_text())
    assert old_summary['pass_'] and len(refs)==44
    assert (old['rows'],old['tp'],old['temperature'],old['top_p'],old['seed'],old['context_len'],old['max_new_tokens'])==(44,8,1,.95,1234,65536,24576)
    assert sha(a.checkpoint/'model.safetensors')==old['draft_sha256']
    assert sha(a.target/'model.safetensors.index.json')==old['target_spec']['checkpoint']['index_sha256']
    dataset=Path(old['dataset']);assert sha(dataset)==old['dataset_sha256']
    for name,digest in old['tokenizer_sha256'].items():assert sha(a.target/name)==digest,name
    tokenizer=AutoTokenizer.from_pretrained(a.target,local_files_only=True,trust_remote_code=False)
    inputs=[]
    if a.dataset:
        assert a.acceptance_only,'--dataset requires --acceptance-only'
        dataset=a.dataset
        rows=[json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        refs=[dict(id=r['id'],prompt_tokens=None,generated_token_ids=[],finish_reason=None,accepted_drafts=None,proposed_drafts=None,verify_calls=None,accepted_drafts_histogram=None) for r in rows]
        for row in rows:
            tokens=tokenizer.encode(row['prompt_text'],add_special_tokens=False)
            if row.get('ctx_tokens') is not None and len(tokens)!=row['ctx_tokens']:
                print('WARNING tokenized length',len(tokens),'differs from ctx_tokens',row['ctx_tokens'],row['id'],flush=True)
            assert len(tokens)+24576+8<capacity
            inputs.append(tokens)
    else:
        rows=[json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        assert [r['id'] for r in rows]==[r['id'] for r in refs]
        for row,ref in zip(rows,refs):
            tokens=tokenizer.encode(row['prompt_text'],add_special_tokens=False)
            assert len(tokens)==ref['prompt_tokens'] and token_digest(tokens)==ref['prompt_tokens_sha256']
            assert len(tokens)+24576+8<capacity
            inputs.append(tokens)
    if a.first:
        assert a.acceptance_only,'--first requires --acceptance-only'
        inputs,refs=inputs[:a.first],refs[:a.first]
    features=[torch.empty(max(map(len,inputs)),4096,dtype=torch.bfloat16,device=r) for r in range(8)]
    if a.prefill in ('kernel','native'):
        streams=[rt.stream for rt in ranks]
        if a.prefill=='native':
            from naivert.prefill.native.executor import NativePrefill
            candidate_type=load('native_prefill_request_candidate',candidate_path).NativePrefill if candidate_path else NativePrefill
            kprefill=candidate_type(kweights,streams,feature_layers=TARGET_LAYER_IDS,features=features)
            resources.callback(kprefill.close)
            warm=min(a.prefill_warm_chunks,capacity//4096)
            if warm>0:print('native prefill warm chunks',kprefill.warm_chunks(warm),flush=True)
        else:
            kprefill=DraftFeaturePrefillExecutor(kweights,streams,comm=TP8Comm([w.dev for w in kweights],streams),
                max_seq_len=capacity,graphs=True,plan_config=PrefillPlanConfig(strategy='budget'),feature_layers=TARGET_LAYER_IDS,features=features)
        print('kernel prefill executor ready',kprefill.versions,flush=True)
    manifest=dict(scope=__doc__,source_inventory=inventory,binaries=binaries,prefill=a.prefill,
        kernel_prefill=None if kprefill is None else dict(buckets=list(getattr(kprefill,'buckets',(4096,))),versions=kprefill.versions,
            feature_layers=list(TARGET_LAYER_IDS),tail='last prompt token by seq1'),
        harness_sha256=sha(__file__),baseline=str(a.baseline),native_candidate=candidate_source,
        native_reference_sha256=sha(Path(__file__).with_name('native_reference.py')) if native_core else None,
        diagnosis_only=a.diagnose_first,target_math='current W8A8; archived target is older math',
        reference_kind='pinned native draft/tail with current target' if native_core else 'archived output',
        baseline_sha256={n:sha(a.baseline/n) for n in ('manifest.json','summary.json','completed.jsonl')},
        dataset=str(dataset),dataset_sha256=sha(dataset),draft_sha256=old['draft_sha256'],capacity=capacity,
        max_new_tokens=24576,seed=SEED,temperature=1/float(os.environ.get('NAIVERT_DSPARK_INV_TEMPERATURE','1')),top_p=.95,
        acceptance_only=a.acceptance_only,
        expected_requests=1 if a.diagnose_first else len(inputs),
        graphs={n:rd.audits for n,rd in rounds.items()})
    write(a.out/'manifest.json',manifest)
    cache_names=('k_cache','v_cache','ki_cache','ki_scale')
    records=[];checks=[]
    def state_digest(rd):
        output=[]
        for r,(b,t,v) in enumerate(zip(buffers,target.states,verifies)):
            end=int(t.position.item())
            values=dict(target_logits=v.logits,draft_logits=b.logit_trace,draft_tokens=b.draft_tokens,
                p=rd.sampler.full[r],q=rd.sampler.draft_full[r],counts=rd.tail.states[r][7],
                control=t.decode_control,sampling_state=b.sampling_state,
                draft_k=b.kv_k[:,:end],draft_v=b.kv_v[:,:end],
                **{f'export{layer}':v.resid[layer+1] for layer in TARGET_LAYER_IDS},
                **{f'cache{layer}.{n}':lw[n][:min(lw[n].shape[0],end+8)]
                   for layer,lw in enumerate(weights[r].layer) for n in cache_names if n in lw})
            output.append({n:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                           for n,x in values.items()})
        return output
    try:
        cases=list(zip(inputs,refs))[:1] if a.diagnose_first else zip(inputs,refs)
        for index,(tokens,ref) in enumerate(cases):
            print('REQUEST',index+1,'/',len(inputs),ref['id'],'prompt',len(tokens),flush=True)
            reset_start=time.monotonic()       # harness only: a server keeps its caches
            for r,rt in enumerate(ranks):
                with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                    rt.pos.zero_()
                    for lw in weights[r].layer:
                        for name in cache_names:
                            if name in lw:lw[name].zero_()
            sync();reset_seconds=time.monotonic()-reset_start
            prefill_start=time.monotonic()     # prefill proper: executor request reset, prompt ingestion, seq1 tail
            if kprefill is not None:kprefill.begin_request()
            # Protocol epochs remain live across prefill, modes and requests.
            prefill_phases={}
            prefill_tokens(ranks,verifies,tokens,features,TARGET_LAYER_IDS,mode=a.prefill,executor=kprefill,
                progress=lambda done:print('PREFILL',index+1,done,flush=True) if done%2048<8 else None,timing=prefill_phases)
            sync();prefill_seconds=time.monotonic()-prefill_start
            snapshot_start=time.monotonic();saved=[]   # harness only: the state the mode comparison restores
            for r,rt in enumerate(ranks):
                with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                    saved.append(dict(logits=rt.logits.clone(),caches=[
                        {n:lw[n][:min(lw[n].shape[0],len(tokens))].clone() for n in cache_names if n in lw}
                        for lw in weights[r].layer]))
            sync();expected=None;snapshot_seconds=time.monotonic()-snapshot_start
            prefill_stats=None if kprefill is None else {k:v for k,v in kprefill._stats.items() if k in ('tokens','chunks','graph_replays','graph_captures','eager_segments','collectives','seconds','owned_pairs')}
            for name,rd in rounds.items():
                for r,(rt,b) in enumerate(zip(ranks,buffers)):
                    with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                        for lw,stored in zip(weights[r].layer,saved[r]['caches']):
                            for n,x in stored.items():lw[n].zero_();lw[n][:x.shape[0]].copy_(x)
                        rt.logits.copy_(saved[r]['logits']);rt.pos.fill_(len(tokens))
                        b.kv_k.zero_();b.kv_v.zero_();b.sampling_state.copy_(torch.tensor([SEED,0],dtype=torch.int64))
                sync();before_anchor=protocol.snapshot(anchors[name])
                for r in range(8):
                    with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):anchors[name].run(r)
                sync();anchor_epochs=protocol.check(anchors[name],before_anchor,1)
                selected=[rt.token_out[0].item() for rt in ranks]
                assert len(set(selected))==1
                anchor=selected[0];assert anchor!=151645,'unexpected immediate EOS needs a separate zero-round case'
                rd.initialize([f[:len(tokens)] for f in features],position=len(tokens),anchor=anchor,
                              seed=SEED,max_tokens=24575,eos=151645)
                sync();before=protocol.snapshot(rd.sampler)
                flags=rd.host_flags.numpy();start=time.monotonic();last=start
                ev=[]
                for r,rt in enumerate(ranks):
                    with torch.cuda.device(r):
                        a_=torch.cuda.Event(enable_timing=True);b_=torch.cuda.Event(enable_timing=True);a_.record(rt.stream);ev.append((a_,b_))
                if name!='native':rd.launch()
                for r,rt in enumerate(ranks):
                    with torch.cuda.device(r):ev[r][1].record(rt.stream)   # fires when stream work incl. tail-launched graphs completes
                while not (flags&1).any():
                    if name=='native':
                        for r,rt in enumerate(ranks):
                            with torch.cuda.device(r),torch.cuda.stream(rt.stream):reference[r].replay()
                        sync()
                    now=time.monotonic()
                    if now-start>300:raise TimeoutError((index,name,'request did not stop within 300 s'))
                    if now-last>30:
                        print('RUNNING',index+1,name,round(now-start,1),'seconds',flush=True);last=now
                    time.sleep(.001)
                stop_seen=time.monotonic()   # wall time of the request: launch -> host sees the stop flag (before any post-processing)
                sync()
                chain_event_ms=[ev[r][0].elapsed_time(ev[r][1]) for r in range(8)]
                controls=[Control.from_buffer_copy(bytes(t.decode_control.cpu().tolist())).value for t in target.states]
                count=controls[0].output_count
                assert all(c.output_count==count and c.iteration==controls[0].iteration and c.flags&1
                           and not c.flags&(1<<31) and c.committed_pos==len(tokens)+count for c in controls)
                epoch_check=protocol.check(rd.sampler,before,controls[0].iteration)
                generated=[anchor];round_ns=[]
                for j in range(count):
                    rec=Record.from_buffer_copy(bytes(rd.host_ring[j*C.sizeof(Record):(j+1)*C.sizeof(Record)].tolist()))
                    assert (rec.generation,rec.output_count,rec.cur_pos)==(j+1,j+1,len(tokens)+j+1)
                    generated.append(rec.token)
                    # one entry per round: (previous round complete, this round complete) from the device clock
                    if not round_ns or round_ns[-1][1]!=rec.parent_complete_ns:
                        round_ns.append((rec.parent_start_ns,rec.parent_complete_ns,rec.cur_pos))
                reason='stop' if generated[-1]==151645 else 'length'
                assert reason=='stop' or len(generated)==24576
                counts=[s[7].cpu().tolist() for s in rd.tail.states];assert all(v==counts[0] for v in counts)
                accepted,verifies_count,*hist=counts[0];assert verifies_count==controls[0].iteration and sum(hist)==verifies_count
                if not rd.staged:
                    for r in range(8):
                        with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):rd.core.context(r)
                    sync()
                state=state_digest(rd)
                record=dict(id=ref['id'],mode=name,prompt_tokens=len(tokens),output_tokens=len(generated),
                    generated_token_ids=generated,generated_sha256=token_digest(generated),finish_reason=reason,
                    accepted_drafts=accepted,proposed_drafts=7*verifies_count,verify_calls=verifies_count,
                    accepted_drafts_histogram={str(i):v for i,v in enumerate(hist)},
                    state_digest=state,sampler_protocol=epoch_check,anchor_protocol=anchor_epochs,
                    request_seconds=stop_seen-start,postprocess_seconds=time.monotonic()-stop_seen,prefill_seconds=prefill_seconds,prefill_stats=prefill_stats,prefill_phases=prefill_phases,reset_seconds=reset_seconds,snapshot_seconds=snapshot_seconds,
                    round_device_ns=[(int(b-a) if a else None, int(pos)) for a,b,pos in round_ns],
                    round_stamps_ns=[(int(a),int(b)) for a,b,_ in round_ns],chain_event_ms=chain_event_ms)
                keyset=('generated_token_ids','finish_reason','accepted_drafts','proposed_drafts','verify_calls','accepted_drafts_histogram')
                archived={k:record[k]==ref[k] for k in keyset}
                first_difference=next((i for i,(x,y) in enumerate(zip(generated,ref['generated_token_ids'])) if x!=y),None)
                if first_difference is None and len(generated)!=len(ref['generated_token_ids']):
                    first_difference=min(len(generated),len(ref['generated_token_ids']))
                check=dict(id=ref['id'],mode=name,archived=archived,first_archived_difference=first_difference,
                           current_modes_equal=expected is None or all(record[k]==expected[k] for k in (*keyset,'state_digest')))
                records.append(record);checks.append(check)
                with (a.out/'completed.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                write(a.out/'checks.json',checks)
                print('COMPLETE',index+1,name,len(generated),'tokens',verifies_count,'rounds',
                      'archive',all(archived.values()),'current modes',check['current_modes_equal'],flush=True)
                assert check['current_modes_equal'],check
                if expected is None:expected=record
            # Archive mode remains strict. The separate current-target gate
            # compares to the pinned native control and reports archive drift.
            if a.diagnose_first:
                diagnostic=dict(diagnosis_only=True,requests=1,
                    native_current_target_exact=all(c['current_modes_equal'] for c in checks),
                    archived_exact=all(all(c['archived'].values()) for c in checks),
                    no_new_xids=xids()==before_xids,
                    sources_unchanged=all(sha(a.repo/n)==h for n,h in inventory.items()) and
                        (candidate_source is None or sha(candidate_path)==candidate_source['sha256']),
                    binaries_unchanged=all(sha(v['path'])==v['sha256'] for v in binaries.values()))
                write(a.out/'diagnostic.json',diagnostic)
                assert all(diagnostic[k] for k in ('native_current_target_exact','no_new_xids','sources_unchanged','binaries_unchanged'))
                return
            if not a.current_target_reference and not a.acceptance_only:
                assert all(all(c['archived'].values()) for c in checks[-2:]),checks[-2:]
            del saved,expected,state
        assert len(records)==len(inputs)*len(rounds)
        assert all(c['current_modes_equal'] for c in checks)
        assert xids()==before_xids
        assert all(sha(a.repo/n)==h for n,h in inventory.items())
        assert candidate_source is None or sha(candidate_path)==candidate_source['sha256']
        assert all(sha(v['path'])==v['sha256'] for v in binaries.values())
        persistent=[r for r in records if r['mode']=='persistent']
        accepted=sum(r['accepted_drafts'] for r in persistent);proposed=sum(r['proposed_drafts'] for r in persistent)
        summary=dict(pass_checks=True,requests=len(persistent),tokens=sum(r['output_tokens'] for r in persistent),
            temperature=manifest['temperature'],acceptance_only=a.acceptance_only,
            per_request=[dict(id=r['id'],prompt_tokens=r['prompt_tokens'],output_tokens=r['output_tokens'],accepted=r['accepted_drafts'],proposed=r['proposed_drafts'],rounds=r['verify_calls'],finish=r['finish_reason'],seconds=round(r['request_seconds'],2)) for r in persistent],
            verify_calls=sum(r['verify_calls'] for r in persistent),accepted_drafts=accepted,proposed_drafts=proposed,
            acceptance=accepted/proposed,
            historical_acceptance=old_summary['draft_accept_rate'],
            historical_acceptance_delta_pp=100*(accepted/proposed-old_summary['draft_accept_rate']),
            reference_kind=manifest['reference_kind'],
            native_current_target_exact=True if native_core else None,
            archived_exact=all(all(c['archived'].values()) for c in checks),
            archived_matched_requests=sum(all(c['archived'].values()) for c in checks if c['mode']=='persistent'),
            current_modes_exact=True,no_new_xids=True,
            sources_unchanged=True,binaries_unchanged=True)
        write(a.out/'summary.json',summary);print('PASS',json.dumps(summary),flush=True)
    except BaseException as exc:
        write(a.out/'failure.json',dict(completed_modes=len(records),error=repr(exc)));raise
    finally:
        if kprefill is not None and hasattr(kprefill, 'close'):
            kprefill.close()
        for rd in rounds.values():rd.close()


def work(a):
    with ExitStack() as resources:
        _work(a, resources)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','target','checkpoint','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--diagnose-first',action='store_true',help='Compare pinned old draft/tail against the current target on request 1; does not pass the 44-request gate')
    p.add_argument('--current-target-reference',action='store_true',help='Check all 44 requests against pinned native draft/tail kernels with the current target; report archive drift separately')
    p.add_argument('--acceptance-only',action='store_true',help='Run only the persistent round; report per-request acceptance without archive token assertions (e.g. at a sampling temperature other than the archive)')
    p.add_argument('--first',type=int,help='acceptance-only: run only the first N archive requests')
    p.add_argument('--prefill-warm-chunks',type=int,default=3,help='native: at start-up run chunk indices 0..N-1 twice on a dummy prompt (eager pass, then the capture), capped by the cache capacity; 0 disables')
    p.add_argument('--dataset',type=Path,help='acceptance-only: run this prompt_text jsonl instead of the archived 44 (no archive comparison; --baseline still supplies the run parameters)')
    p.add_argument('--native-core',type=Path)
    p.add_argument('--native-tail',type=Path)
    p.add_argument('--prefill',choices=('verify8','seq1','kernel','native'),default='verify8',help='prompt ingestion: verify8 replay (default), seq1 replay, bucketed kernel reference, or native 4096-row chunks; each finishes with a seq1 tail')
    p.add_argument('--native-candidate',type=Path,help='isolated experiment exporting NativePrefill; requires --prefill native')
    p.add_argument('--draft-pdl',action='store_true',help='programmatic edges into D3/D4/D56 inside the draft chain (public rounds only; the native control is unchanged)')
    a=p.parse_args()
    if a.native_candidate is not None and a.prefill!='native':
        p.error('--native-candidate requires --prefill native')
    if (a.diagnose_first or a.current_target_reference) and (a.native_core is None or a.native_tail is None):
        p.error('native reference requires --native-core and --native-tail')
    a.out.mkdir(parents=True,exist_ok=False);work(a)
