"""Read fixed-round Nsight node durations; no per-warp or resource attribution."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics as st


def category(name):
    for marker,label in (('naivert_dspark_fc_ctx_kv_kernel','d12'),('naivert_dspark_attn_rows8_swa_kernel','d3'),
                         ('naivert_dspark_mlp_rows8_kernel','d4'),('naivert_dspark_head_markov_kernel','d56'),
                         ('naivert_dspark_sharded','sampler'),('naivert_dspark_persistent','sampler'),
                         ('naivert_dspark_tail_chained_entry','tail')):
        if marker in name:return label
    return 'target'


def analyze(result,databases):
    data=json.loads(result.read_text());assert data['pass_checks']
    samples=[];ranges=[]
    for path in databases:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            windows=db.execute('''SELECT n.start,n.end,COALESCE(n.text,s.value)
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
                WHERE COALESCE(n.text,s.value) LIKE 'fixed.%' AND n.end IS NOT NULL
                ORDER BY n.start''').fetchall()
            assert len(windows)==4,(path,len(windows))
            for lo,hi,label in windows:
                _,position,mode,trial=label.split('.');position=int(position)
                ranges.append(dict(label=label,start=lo,end=hi,database=path.name))
                for rank in range(8):
                    expected=data['graphs'][mode][rank]['kernel_names']
                    rows=[dict(name=n,start=s,end=e,node=g) for n,s,e,g in db.execute('''
                        SELECT s.value,k.start,k.end,k.graphNodeId
                        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.mangledName
                        WHERE k.deviceId=? AND k.start>=? AND k.end<=? ORDER BY k.start,k.end''',(rank,lo,hi))]
                    kernels=[x for x in rows if x['name'] in expected]
                    other=[x for x in rows if x['name'] not in expected]
                    assert all('dspark_benchmark_start_gate' in x['name'] for x in other),other
                    assert len(kernels)==len(expected) and len(expected) in (207,238),(label,rank,len(kernels))
                    assert Counter(x['name'] for x in kernels)==Counter(expected),(label,rank,'inventory')
                    # Only D12 has a graph branch. Target PDL overlaps lifetimes
                    # but retains launch order in this schedule.
                    ordinary=[x['name'] for x in kernels if category(x['name'])!='d12']
                    assert ordinary==[n for n in expected if category(n)!='d12'],(label,rank,'order')
                    groups=defaultdict(list)
                    for x in kernels:groups[category(x['name'])].append(x)
                    def span(xs):return (max(x['end'] for x in xs)-min(x['start'] for x in xs))/1000
                    metrics={}
                    for group,xs in groups.items():
                        metrics[group+'_activity_us']=sum(x['end']-x['start'] for x in xs)/1000
                        metrics[group+'_span_us']=span(xs)
                    draft=[x for x in kernels if category(x['name']) in ('d3','d4','d56') or
                           (mode!='staged' and category(x['name'])=='d12')]
                    metrics['draft_prefix_span_us']=span(draft)
                    metrics['round_kernel_span_us']=span(kernels)
                    metrics['post_target_span_us']=(max(x['end'] for x in groups['tail'])-
                        max(x['end'] for x in groups['target']))/1000
                    samples.append(dict(context=position,mode=mode,trial=int(trial),rank=rank,
                        metrics=metrics,counts={k:len(v) for k,v in groups.items()},
                        nodes=[dict(category=category(x['name']),duration_us=(x['end']-x['start'])/1000)
                               for x in kernels]))
    assert len(ranges)==8 and len(samples)==64
    summaries=[]
    for position in (200,2200):
        for mode in sorted({x['mode'] for x in samples if x['context']==position}):
            rows=[x for x in samples if x['context']==position and x['mode']==mode]
            assert len(rows)==16
            summaries.append(dict(context=position,mode=mode,
                rank0_mean={key:st.mean(x['metrics'][key] for x in rows if x['rank']==0)
                            for key in rows[0]['metrics']},
                rank_max_mean={key:st.mean(max(x['metrics'][key] for x in rows if x['trial']==trial)
                                          for trial in sorted({x['trial'] for x in rows}))
                               for key in rows[0]['metrics']},counts=rows[0]['counts']))
    perturbation=[]
    for row in data['timing']['timing']:
        for mode,indices in ((row.get('control_mode','serial'),(0,3)),
                             (row.get('candidate_mode','staged'),(1,2))):
            prof=st.mean(max(row['profiled_abba_us'][i]) for i in indices)
            baseline=row[mode+'_median_us']
            perturbation.append(dict(context=row['context'],mode=mode,profiled_event_mean_us=prof,
                unprofiled_event_median_us=baseline,relative_percent=(prof/baseline-1)*100))
    return dict(pass_checks=True,scope=__doc__,summary=summaries,samples=samples,ranges=ranges,
        perturbation=perturbation,result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
        note='Two profiled rounds per mode/context. Activity sums include overlapping PDL lifetimes; spans include gaps. Do not sum stage maxima or treat activity as useful compute.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    p.add_argument('--database',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=analyze(a.result,a.database)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('samples','ranges')},indent=2))
