"""Sequential multi-session decode with parked prefixes (persistent round).

Every dataset row is a session (uuid). Turn 1 decodes the row's prompt from an
empty cache; turn 2 restores that session's parked prefix and prefills only the
follow-up user turn. Sessions run interleaved in pairs (A1 B1 A2 B2 ...) so each
turn 2 follows another session's decode. ``--fresh-check N`` additionally
re-decodes turn 2 of the first N sessions from a from-scratch prefill of the
whole conversation and records the first token difference.
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

from run import load, sha
import protocol

FOLLOW_UP = ("Now make a second, different version of the same piece: switch to a dark color theme "
             "and add at least one animated element. Return the complete code again inside a single "
             "fenced code block.")
EOS = 151645


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def work(a):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    sys.path.insert(0,str(a.repo))
    capacity=65536
    from naivert.backends.naive_n0p5_draft import DraftCore
    from naivert.backends.naive_n0p5_draft_target import DraftTarget
    from naivert.backends.naive_n0p5_draft_round import DraftRound
    from naivert.backends.naive_n0p5_prefix_store import PrefixStore,CACHE_NAMES
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
    del raw
    mailboxes=[torch.zeros(verify.mailbox_bytes(ops),dtype=torch.uint8,device=r) for r in range(8)]
    inboxes=[torch.zeros(4096,dtype=torch.uint8,device=r) for r in range(8)]
    ranks=[decode.RankRuntime(r,weights[r],ops,ext_dsa,ext_swa,capacity,mailboxes,inboxes,pdl=True,fused_unproj=True) for r in range(8)]
    for rt in ranks:rt.pdl_markers=decode.PDL_W8A8_MARKERS
    verifies=[verify.VerifyRuntime(rt,ext_dsa8,ext_swa8,ext_dsa8.workspace_bytes(),ext_swa8.workspace_bytes(),
                                  pdl=True,fused_unproj=True) for rt in ranks]
    buffers=[DraftBuffers(kernels,rank=r,capacity=capacity,sampling=True) for r in range(8)]
    core=DraftCore(kernels,draft_weights,buffers)
    target=DraftTarget(core,verifies)
    api=CudaRuntimeGraphApi.load(decode.torch_cudart_path())
    rd=DraftRound(core,target,api,output_capacity=24576,staged=False,persistent_sampler=True,draft_pdl=True)
    anchor_sampler=TargetSampler([rt.logits for rt in ranks],buffers,[rt.token_out[:1] for rt in ranks],anchor=True,persistent=True)
    modules={'target':ops,'dsa':ext_dsa,'dsa8':ext_dsa8,'swa':ext_swa,'swa8':ext_swa8,
             **{n:getattr(kernels,n) for n in ('context','attention','mlp','head_markov')},
             'persistent.sampler':anchor_sampler.ext,'persistent.tail':rd.tail.extension}
    binaries={n:dict(path=m.__file__,sha256=sha(m.__file__)) for n,m in modules.items()}
    def sync():
        for rt in ranks:rt.stream.synchronize()
        assert all(b.wdog.item()==0 and b.barrier[0].item()==0 for b in buffers)
        assert all(s[4].item()==0 for s in rd.tail.states)
    for r in range(8):torch.cuda.synchronize(r)
    for rt in ranks:rt.capture()
    rd.capture();sync()
    tokenizer=AutoTokenizer.from_pretrained(a.target,local_files_only=True,trust_remote_code=False)
    rows=[json.loads(line) for line in a.dataset.read_text().splitlines() if line.strip()]
    if a.limit:rows=rows[:a.limit]
    suffix=tokenizer.encode('<|im_start|>user\n'+a.follow_up+'<|im_end|><|im_start|>assistant\n<think></think>',add_special_tokens=False)
    sessions=[]
    for row in rows:
        tokens=tokenizer.encode(row['prompt_text'],add_special_tokens=False)
        assert len(tokens)+2*24576+len(suffix)+16<capacity,row['id']
        sessions.append(dict(uuid=row['id'],turn1=tokens,task=row.get('task')))
    features=[torch.empty(capacity,4096,dtype=torch.bfloat16,device=r) for r in range(8)]
    store=PrefixStore(weights,buffers,host=a.host_store)
    order=[]
    for i in range(0,len(sessions),2):
        pair=sessions[i:i+2]
        for turn in (1,2):
            for s in pair:order.append((s,turn))
    manifest=dict(scope=__doc__,source_inventory=inventory,binaries=binaries,prefill='verify8',
        harness_sha256=sha(__file__),dataset=str(a.dataset),dataset_sha256=sha(a.dataset),
        follow_up=a.follow_up,follow_up_tokens=len(suffix),capacity=capacity,max_new_tokens=24576,seed=1234,
        temperature=1/float(os.environ.get('NAIVERT_DSPARK_INV_TEMPERATURE','1')),top_p=.95,
        sessions=len(sessions),order=[(s['uuid'],t) for s,t in order],fresh_check=a.fresh_check,host_store=a.host_store,
        graphs=rd.audits)
    write(a.out/'manifest.json',manifest)
    records=[]
    def decode_once(uuid,turn,tokens,start,label):
        """Prefill tokens[start:] on top of the working caches, decode to stop, park the result."""
        prefill_start=time.monotonic()
        prefill_tokens(ranks,verifies,tokens,features,TARGET_LAYER_IDS,start=start,mode='verify8',
            progress=lambda done:print('PREFILL',label,done,flush=True) if done%2048<8 else None)
        sync();prefill_seconds=time.monotonic()-prefill_start
        for r,(rt,b) in enumerate(zip(ranks,buffers)):
            with torch.cuda.device(r),torch.cuda.stream(rt.stream):
                rt.pos.fill_(len(tokens));b.sampling_state.copy_(torch.tensor([1234,0],dtype=torch.int64))
        sync();before_anchor=protocol.snapshot(anchor_sampler)
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):anchor_sampler.run(r)
        sync();anchor_epochs=protocol.check(anchor_sampler,before_anchor,1)
        selected=[rt.token_out[0].item() for rt in ranks];assert len(set(selected))==1
        anchor=selected[0];assert anchor!=EOS,'unexpected immediate EOS needs a separate zero-round case'
        rd.initialize([f[:len(tokens)] for f in features],position=len(tokens),anchor=anchor,
                      seed=1234,max_tokens=24575,eos=EOS,ingest_from=start)
        sync();before=protocol.snapshot(rd.sampler)
        flags=rd.host_flags.numpy();t0=time.monotonic();last=t0
        ev=[]
        for r,rt in enumerate(ranks):
            with torch.cuda.device(r):
                a_=torch.cuda.Event(enable_timing=True);b_=torch.cuda.Event(enable_timing=True);a_.record(rt.stream);ev.append((a_,b_))
        rd.launch()
        for r,rt in enumerate(ranks):
            with torch.cuda.device(r):ev[r][1].record(rt.stream)
        while not (flags&1).any():
            now=time.monotonic()
            if now-t0>300:raise TimeoutError((label,'request did not stop within 300 s'))
            if now-last>30:print('RUNNING',label,round(now-t0,1),'seconds',flush=True);last=now
            time.sleep(.001)
        stop_seen=time.monotonic();sync()
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
            if not round_ns or round_ns[-1][1]!=rec.parent_complete_ns:
                round_ns.append((rec.parent_start_ns,rec.parent_complete_ns,rec.cur_pos))
        reason='stop' if generated[-1]==EOS else 'length'
        assert reason=='stop' or len(generated)==24576
        counts=[s[7].cpu().tolist() for s in rd.tail.states];assert all(v==counts[0] for v in counts)
        accepted,verifies_count,*hist=counts[0];assert verifies_count==controls[0].iteration and sum(hist)==verifies_count
        # The serial round leaves the last committed block's draft KV to the next D12; complete it here.
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(ranks[r].stream):rd.core.context(r)
        sync()
        position=len(tokens)+count
        save_seconds=store.save(uuid,ranks,position=position,tokens=tokens+generated)
        return dict(id=uuid,turn=turn,label=label,prompt_tokens=len(tokens),new_prompt_tokens=len(tokens)-start,prefill_start=start,
            output_tokens=len(generated),generated_token_ids=generated,generated_sha256=token_digest(generated),finish_reason=reason,
            accepted_drafts=accepted,proposed_drafts=7*verifies_count,verify_calls=verifies_count,
            accepted_drafts_histogram={str(i):v for i,v in enumerate(hist)},
            sampler_protocol=epoch_check,anchor_protocol=anchor_epochs,
            request_seconds=stop_seen-t0,postprocess_seconds=time.monotonic()-stop_seen,prefill_seconds=prefill_seconds,
            save_seconds=save_seconds,parked_bytes_per_rank=store.bytes(uuid)//8,parked_position=position,
            round_device_ns=[(int(b-a_) if a_ else None,int(pos)) for a_,b,pos in round_ns],
            round_stamps_ns=[(int(a_),int(b)) for a_,b,_ in round_ns],chain_event_ms=chain_event_ms)
    try:
        done=set()
        for index,(s,turn) in enumerate(order):
            uuid=s['uuid'];label=f'{uuid}/t{turn}'
            print('SESSION',index+1,'/',len(order),label,flush=True)
            if turn==1:
                store.clear(ranks);sync()
                tokens,start,restore_seconds=s['turn1'],0,0.0
            else:
                position,prev,restore_seconds=store.restore(uuid,ranks);sync()
                assert position==len(prev)-1
                tokens,start=prev+suffix,position
            record=decode_once(uuid,turn,tokens,start,label)
            record.update(kind='parked' if turn==2 else 'fresh',restore_seconds=restore_seconds,parked_store_bytes_per_rank=store.bytes()//8)
            records.append(record)
            with (a.out/'completed.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('COMPLETE',label,record['output_tokens'],'tokens',record['verify_calls'],'rounds',record['finish_reason'],
                  'prefill',round(record['prefill_seconds'],3),'restore',round(restore_seconds,4),'save',round(record['save_seconds'],4),flush=True)
            if turn==2:
                s['turn2_tokens']=tokens;s['turn2_generated']=record['generated_token_ids'];done.add(uuid)
            if turn==2 and sessions.index(s)<a.fresh_check:
                # Same conversation, prefilled from scratch: is the parked continuation exact?
                store.clear(ranks);sync()
                fresh=decode_once(uuid,2,tokens,0,label+'/fresh')
                parked=record['generated_token_ids'];fg=fresh['generated_token_ids']
                first=next((i for i,(x,y) in enumerate(zip(parked,fg)) if x!=y),None)
                if first is None and len(parked)!=len(fg):first=min(len(parked),len(fg))
                fresh.update(kind='fresh_check',first_difference=first,exact=first is None,
                             parked_output_tokens=len(parked),parked_verify_calls=record['verify_calls'])
                records.append(fresh)
                with (a.out/'completed.jsonl').open('a') as f:f.write(json.dumps(fresh)+'\n')
                print('FRESHCHECK',label,'exact' if first is None else f'first difference at {first}',
                      'parked',len(parked),'fresh',len(fg),'tokens',flush=True)
                store.evict(uuid)
    finally:
        after=xids()
        def agg(kind):
            rs=[r for r in records if r['kind']==kind]
            if not rs:return None
            dev=[(sum(d for d,_ in r['round_device_ns'] if d),len([d for d,_ in r['round_device_ns'] if d])) for r in rs]
            return dict(n=len(rs),tokens=sum(r['output_tokens'] for r in rs),rounds=sum(r['verify_calls'] for r in rs),
                acceptance=sum(r['accepted_drafts'] for r in rs)/max(1,sum(r['proposed_drafts'] for r in rs)),
                tokens_per_round=sum(r['output_tokens'] for r in rs)/max(1,sum(r['verify_calls'] for r in rs)),
                wall_tok_s=sum(r['output_tokens'] for r in rs)/max(1e-9,sum(r['request_seconds'] for r in rs)),
                device_tok_s=sum(r['output_tokens']-1 for r in rs)/max(1e-9,sum(d for d,_ in dev)/1e9),
                mean_round_us=sum(d for d,_ in dev)/max(1,sum(n for _,n in dev))/1e3,
                prefill_seconds=sum(r['prefill_seconds'] for r in rs),new_prompt_tokens=sum(r['new_prompt_tokens'] for r in rs),
                restore_seconds=sum(r.get('restore_seconds',0) for r in rs),save_seconds=sum(r['save_seconds'] for r in rs),
                stops=sum(r['finish_reason']=='stop' for r in rs))
        summary=dict(sessions=len(sessions),completed=len(records),turn1=agg('fresh'),turn2_parked=agg('parked'),
            fresh_check=[dict(id=r['id'],exact=r['exact'],first_difference=r['first_difference'],parked_tokens=r['parked_output_tokens'],
                              fresh_tokens=r['output_tokens']) for r in records if r['kind']=='fresh_check'],
            xids=after[len(before_xids):],pass_=len(records)>=len(order))
        write(a.out/'summary.json',summary)
        print('SUMMARY',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--follow-up',default=FOLLOW_UP)
    p.add_argument('--fresh-check',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--host-store',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True);work(a)


if __name__=='__main__':main()
