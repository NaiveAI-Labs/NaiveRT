"""CPU-only kernel identity and sampler timeline comparison; no stall attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics as st
import subprocess


MARKER='naivert_dspark_persistent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(db,rank,lo,hi):
    return db.execute('''SELECT s.value,k.start,k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.mangledName
        WHERE k.deviceId=? AND k.start>=? AND k.end<=? ORDER BY k.start,k.end''',
        (rank,lo,hi)).fetchall()


def windows(db,pattern):
    return db.execute('''SELECT n.start,n.end,COALESCE(n.text,s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text,s.value) LIKE ? AND n.end IS NOT NULL ORDER BY n.start''',
        (pattern,)).fetchall()


def connect(path):
    return sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)


def main(a):
    code=[]
    for path in a.binary:
        sass=subprocess.check_output([str(a.cuobjdump),'--dump-sass',str(path)],text=True)
        sections=[s for s in sass.split('Function : ') if s.startswith('_ZN25naivert_dspark_persistent7sampler')]
        assert len(sections)==1
        words=re.findall(r'/\* (0x[0-9a-fA-F]+) \*/',sections[0])
        assert words
        code.append(dict(path=str(path),sha256=sha(path),instruction_words=len(words),
            word_hash=hashlib.sha256(' '.join(words).encode()).hexdigest()))
    assert len(code)==2 and code[0]['word_hash']==code[1]['word_hash']
    integrated=[]
    for path in a.integrated:
        with connect(path) as db:
            for lo,hi,label in windows(db,'fixed.%persistent.%'):
                ranks=[]
                for rank in range(8):
                    kernels=rows(db,rank,lo,hi)
                    sampler=[x for x in kernels if MARKER in x[0]]
                    assert len(sampler)==1
                    sample=sampler[0];index=kernels.index(sample)
                    assert 'dspark_tail' in kernels[index+1][0]
                    ranks.append(dict(rank=rank,start_ns=sample[1],end_ns=sample[2],
                        target_end_ns=kernels[index-1][2],duration_us=(sample[2]-sample[1])/1000))
                first=min(r['start_ns'] for r in ranks);last=max(r['start_ns'] for r in ranks)
                end=max(r['end_ns'] for r in ranks)
                integrated.append(dict(label=label,database=str(path),ranks=ranks,
                    start_spread_us=(last-first)/1000,last_start_to_last_end_us=(end-last)/1000))
    assert len(integrated)==4
    isolated=[]
    with connect(a.profile) as db:
        for lo,hi,label in windows(db,'sampler_conditions.%'):
            batch=int(label.split('.')[1][1:]);ranks=[]
            for rank in range(8):
                samples=[dict(start_ns=s,end_ns=e,duration_us=(e-s)/1000)
                    for n,s,e in rows(db,rank,lo,hi) if MARKER in n]
                assert len(samples)==batch,(label,rank,len(samples))
                ranks.append(dict(rank=rank,samples=samples,first_us=samples[0]['duration_us'],
                    subsequent_mean_us=st.mean(x['duration_us'] for x in samples[1:]) if batch>1 else None,
                    kernel_span_us=(samples[-1]['end_ns']-samples[0]['start_ns'])/1000))
            isolated.append(dict(label=label,ranks=ranks))
    assert len(isolated)==4
    repeated=[]
    for path in a.full_model:
        with connect(path) as db:
            for lo,hi,label in windows(db,'full_model_sampler.%'):
                batch=int(label.split('.')[-1][1:]);ranks=[]
                for rank in range(8):
                    kernels=rows(db,rank,lo,hi)
                    assert len(kernels)==206+batch,(label,rank,len(kernels))
                    samples=[dict(start_ns=s,end_ns=e,duration_us=(e-s)/1000)
                        for n,s,e in kernels if MARKER in n]
                    assert len(samples)==batch
                    ranks.append(dict(rank=rank,samples=samples,first_us=samples[0]['duration_us'],
                        subsequent_mean_us=st.mean(x['duration_us'] for x in samples[1:]) if batch>1 else None))
                repeated.append(dict(label=label,database=str(path),ranks=ranks))
    if a.full_model:assert len(repeated)==4
    out=dict(pass_checks=True,code=code,integrated=integrated,isolated=isolated,
             full_model_repeated=repeated,scope=__doc__,harness_sha256=sha(__file__))
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(integrated=[{k:v for k,v in x.items() if k!='ranks'} for x in integrated],
        isolated=[dict(label=x['label'],rank0={k:v for k,v in x['ranks'][0].items() if k!='samples'},
                       max_first_us=max(r['first_us'] for r in x['ranks'])) for x in isolated],
        full_model_repeated=[dict(label=x['label'],rank0={k:v for k,v in x['ranks'][0].items() if k!='samples'})
                             for x in repeated]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,action='append',required=True)
    p.add_argument('--integrated',type=Path,action='append',required=True)
    p.add_argument('--full-model',type=Path,action='append',default=[])
    for name in ('profile','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cuobjdump',type=Path,default=Path('/opt/cuda132-min/bin/cuobjdump'))
    main(p.parse_args())
