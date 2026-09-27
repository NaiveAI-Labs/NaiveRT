"""Fixed-position public round timing; restore and equality checks outside events."""
import ctypes as C
import hashlib
import importlib.util
import json
import random
import statistics as st
import protocol


def measure(a,*,core,target,rounds,anchor_sampler,api,features,restore,sync):
    import torch
    from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS
    from naivert.runtime.mtp_contract import _DecodeControlLayout
    assert a.cycles>=4
    buffers=core.buffers;streams=target.streams
    control_mode='staged' if a.persistent else 'serial'
    candidate_mode='persistent' if a.persistent else 'staged'
    if getattr(a,'head_candidate',None):
        control_mode,candidate_mode='persistent','noise'
    abba=(control_mode,candidate_mode,candidate_mode,control_mode)
    gate_path=a.native_tail.parent.parent/'dspark_benchmark_gate/naivert_dspark_benchmark_gate.so'
    spec=importlib.util.spec_from_file_location('naivert_dspark_benchmark_gate',gate_path)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    gate_hash=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    flag=api.host_alloc(4,flags=3);word=C.c_uint32.from_address(flag.value);word.value=0
    mapped=[];events=[];ticket=0;fields=byte_count=0;results=[];protocol_checks=0
    cache_names=('k_cache','v_cache','ki_cache','ki_scale')
    active_offset=type(_DecodeControlLayout().value).active_rows.offset
    try:
        for r,b in enumerate(buffers):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                mapped.append(gate.mapped_pointer(flag.value))
                gate.wait_start(mapped[r],0,b.wdog)
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                for event in pair:event.record()
                events.append(pair)
        sync()
        for position in (200,2200):
            restore(position)
            for r,b in enumerate(buffers):
                with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                    b.sampling_state.copy_(torch.tensor([1234,0],dtype=torch.int64))
            for r in range(8):
                with torch.cuda.device(r),torch.cuda.stream(streams[r]):anchor_sampler.run(r)
            sync();anchors=[v.rt.token_out[0].item() for v in target.verifies]
            assert len(set(anchors))==1
            saved={}
            for name,rd in rounds.items():
                restore(position)
                rd.initialize([f[:position] for f in features],position=position,anchor=anchors[0],
                              seed=1234,max_tokens=64)
                # Match complete prompt KV. Serial still performs its ordinary
                # D12 inside the timed round; staged D12 runs after the head.
                if not rd.staged:
                    for r in range(8):
                        with torch.cuda.device(r),torch.cuda.stream(streams[r]):core.context(r)
                sync();saved[name]=[]
                for r,(b,t,v) in enumerate(zip(buffers,target.states,target.verifies)):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        # Disarm only output publication and recursive launch.
                        # Acceptance, geometry and staged KV commit still run.
                        t.decode_control[active_offset:active_offset+4].zero_()
                        values={k:x for k,x in rd.tail.fields[r].items() if k not in ('embed','mask_embedding')}
                        values.update(kv_k=b.kv_k,kv_v=b.kv_v,sampling_state=b.sampling_state,
                                      counts=rd.tail.states[r][7],control=t.decode_control)
                        saved[name].append(dict(values={k:(x,x.clone()) for k,x in values.items()},
                            caches=[{k:(lw[k],lw[k].clone()) for k in cache_names if k in lw} for lw in v.w.layer]))
                sync()

            def reset(name):
                for r,(b,t,saved_rank) in enumerate(zip(buffers,target.states,saved[name])):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        for live,value in saved_rank['values'].values():live.copy_(value)
                        for layer in saved_rank['caches']:
                            for live,value in layer.values():live.copy_(value)
                        t.position.fill_(position);t.previous_complete_ns.zero_()
                # Never restore target/draft epochs or sampler peer flags.
                sync()

            def outputs(name):
                rd=rounds[name]
                sync()
                if not rd.staged:
                    for r in range(8):
                        with torch.cuda.device(r),torch.cuda.stream(streams[r]):core.context(r)
                    sync()
                rows=[]
                for r,(b,t,v) in enumerate(zip(buffers,target.states,target.verifies)):
                    rows.append({**{f'export{layer}':v.resid[layer+1] for layer in TARGET_LAYER_IDS},
                        'target_logits':v.logits,'draft_logits':b.logit_trace,'draft_tokens':b.draft_tokens,
                        'p':rd.sampler.full[r],'q':rd.sampler.draft_full[r],
                        'counts':rd.tail.states[r][7],'control':t.decode_control,'position':t.position,
                        **{k:rd.tail.fields[r][k] for k in ('cache_len','ctx_rows','block_ids','resid0')},
                        'sampling_state':b.sampling_state,'target_hidden':b.target_hidden,
                        'draft_k':b.kv_k[:,:position+8].contiguous(),
                        'draft_v':b.kv_v[:,:position+8].contiguous(),
                        **{f'cache{layer}.{k}':lw[k] for layer,lw in enumerate(v.w.layer)
                           for k in cache_names if k in lw}})
                return rows

            reset('serial');rounds['serial'].launch();initial=outputs('serial')
            expected=[]
            for r,row in enumerate(initial):
                with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                    expected.append({k:x.clone() for k,x in row.items()})
            sync()
            def call(name,profile_label=None):
                nonlocal ticket,fields,byte_count,protocol_checks
                reset(name);ticket+=1
                before=protocol.snapshot(rounds[name].sampler)
                if profile_label:torch.cuda.nvtx.range_push(profile_label)
                for r,b in enumerate(buffers):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        gate.wait_start(mapped[r],ticket,b.wdog)
                        events[r][0].record()
                        api.graph_launch(rounds[name].execs[r],streams[r].cuda_stream)
                        events[r][1].record()
                word.value=ticket
                if profile_label:
                    sync();torch.cuda.nvtx.range_pop()
                actual=outputs(name)
                protocol.check(rounds[name].sampler,before,1);protocol_checks+=1
                times=[start.elapsed_time(end)*1000 for start,end in events]
                for r,row in enumerate(actual):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        checks=[]
                        for key,x in row.items():
                            checks.append(torch.all(x.contiguous().view(torch.uint8)==expected[r][key].contiguous().view(torch.uint8)))
                            fields+=1;byte_count+=x.numel()*x.element_size()
                        passed=torch.stack(checks).all()
                    streams[r].synchronize()
                    if not passed.item():
                        for key,x in row.items():
                            assert torch.equal(x.contiguous().view(torch.uint8),expected[r][key].contiguous().view(torch.uint8)),(position,name,r,key)
                return times

            records=[]
            for cycle in range(a.cycles+4):
                trial=[call(name) for name in abba]
                if cycle>=4:records.append(trial)
                if (cycle+1)%8==0:print('timing progress',position,cycle+1,'cycles',flush=True)
            maximum=[[max(rank_times) for rank_times in trial] for trial in records]
            delta=[(v[1]+v[2]-v[0]-v[3])/2 for v in maximum]
            rng=random.Random(20260918)
            boot=sorted(st.median(rng.choices(delta,k=len(delta))) for _ in range(5000))
            result=dict(context=position,anchor=anchors[0],cycles=a.cycles,per_rank_abba_us=records,
                control_mode=control_mode,candidate_mode=candidate_mode,
                paired_median_delta_us=st.median(delta),bootstrap95_delta_us=[boot[124],boot[4874]])
            result[control_mode+'_median_us']=st.median(v[i] for v in maximum for i in (0,3))
            result[candidate_mode+'_median_us']=st.median(v[i] for v in maximum for i in (1,2))
            if a.profile_fixed:
                torch.cuda.cudart().cudaProfilerStart()
                try:
                    result['profiled_abba_us']=[call(name,f'fixed.{position}.{name}.{i}')
                        for i,name in enumerate(abba)]
                finally:torch.cuda.cudart().cudaProfilerStop()
            results.append(result)
            print('TIMING',json.dumps({k:v for k,v in result.items() if k!='per_rank_abba_us'}),flush=True)
            del saved,expected,initial
    finally:
        sync();api.host_free(flag)
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest()==gate_hash
    return dict(scope=__doc__,timer='Max-rank CUDA events, mapped-host start gate; four warmup ABBA cycles',
        commit='active_rows=0; acceptance and KV commit execute, output publication and next launch disabled',
        fields=fields,bytes_checked=byte_count,timing=results,gate_sha256=gate_hash,
        sampler_protocol_checks=protocol_checks,
        harness_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
