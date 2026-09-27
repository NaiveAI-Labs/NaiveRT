"""Verify complete device-launched rounds and zero intervening host CUDA calls.

Whole-graph execution evidence only: no per-warp, critical-path or speed claim.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def analyze(result, databases):
    data=json.loads(result.read_text())
    assert data['pass_checks'] and data['profile_device_chain']
    cases=[]
    for path in databases:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables
            windows=db.execute('''SELECT n.start,n.end,COALESCE(n.text,s.value)
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
                WHERE COALESCE(n.text,s.value) LIKE 'device_chain.%' AND n.end IS NOT NULL
                ORDER BY n.start''').fetchall()
            assert len(windows)==1,(path,len(windows))
            for lo,hi,label in windows:
                _,position,mode=label.split('.');position=int(position)
                assert mode=='persistent'
                record=next(r for r in data['records'] if
                    (r['position'],r['budget'],r['mode'])==(position,64,mode))
                iterations=record['iterations'];assert iterations>=4
                marker_times=[]
                for prefix in ('submitted','observed_stop'):
                    markers=db.execute('''SELECT n.start FROM NVTX_EVENTS n
                        LEFT JOIN StringIds s ON s.id=n.textId
                        WHERE COALESCE(n.text,s.value)=? AND n.start>=? AND n.start<=?''',
                        (f'{prefix}.{position}.{mode}',lo,hi)).fetchall()
                    assert len(markers)==1,(label,prefix,markers)
                    marker_times.append(markers[0][0])
                ranks=[];first_ends=[];last_starts=[]
                for rank in range(8):
                    expected=data['graphs'][mode][rank]['kernel_names']
                    assert len(expected)==207
                    rows=db.execute('''SELECT s.value,k.start,k.end,k.graphNodeId
                        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.mangledName
                        WHERE k.deviceId=? AND k.start>=? AND k.end<=? ORDER BY k.start,k.end''',
                        (rank,lo,hi)).fetchall()
                    assert len(rows)==len(expected)*iterations,(label,rank,len(rows),iterations)
                    chunks=[rows[i:i+len(expected)] for i in range(0,len(rows),len(expected))]
                    for step,chunk in enumerate(chunks):
                        names=[r[0] for r in chunk]
                        assert Counter(names)==Counter(expected),(label,rank,step,'inventory')
                        assert names==expected,(label,rank,step,'order')
                        assert 'naivert_dspark_tail_chained_entry' in names[-1]
                        if step:
                            assert chunk[0][1]>=max(r[2] for r in chunks[step-1]),(label,rank,step,'round overlap')
                    first_ends.append(max(r[2] for r in chunks[0]))
                    last_starts.append(min(r[1] for r in chunks[-1]))
                    ranks.append(dict(rank=rank,rounds=iterations,kernels=len(rows),
                        first_start_ns=rows[0][1],last_end_ns=max(r[2] for r in rows),
                        boundaries=[dict(start_ns=min(r[1] for r in c),end_ns=max(r[2] for r in c))
                                    for c in chunks]))
                # Exclude startup submission and final drain on every rank.
                # The interval includes complete interior rounds and their gaps.
                start=max(first_ends);end=min(last_starts)
                assert lo<marker_times[0]<start<end<marker_times[1]<hi,(label,marker_times,start,end)
                intervals={'host_wait':marker_times,'steady_rounds':[start,end]}
                api_checks={}
                for name,(a,b) in intervals.items():
                    api_checks[name]={}
                    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
                        if table not in tables:
                            api_checks[name][table]=dict(table_present=False,count=0)
                            continue
                        calls=db.execute(f'''SELECT s.value,k.start,k.end FROM {table} k
                            LEFT JOIN StringIds s ON s.id=k.nameId
                            WHERE k.start<? AND k.end>? ORDER BY k.start''',(b,a)).fetchall()
                        assert not calls,(label,name,table,calls[:20])
                        api_checks[name][table]=dict(table_present=True,count=len(calls))
                transfers={}
                for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
                    count=(db.execute(f'SELECT COUNT(*) FROM {table} WHERE start<? AND end>?',
                                      (end,start)).fetchone()[0] if table in tables else 0)
                    assert count==0,(label,table,count)
                    transfers[table]=count
                # Ensure the CUDA API trace is populated; an empty API capture
                # must not masquerade as evidence of zero calls between rounds.
                runtime_count=db.execute('''SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE start>=? AND end<=?''',(lo,hi)).fetchone()[0]
                assert runtime_count>=8,(label,'missing startup/drain CUDA API coverage',runtime_count)
                cases.append(dict(context=position,mode=mode,database=path.name,
                    iterations=iterations,kernel_count=sum(r['kernels'] for r in ranks),
                    runtime_calls_in_whole_range=runtime_count,api_checks=api_checks,
                    steady_start_ns=start,steady_end_ns=end,transfers=transfers,ranks=ranks))
    assert sorted(c['context'] for c in cases)==[200,2200]
    return dict(pass_checks=True,scope=__doc__,cases=cases,
        result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
        note='Collection retains recursive launches and publication. Correctness is checked by the GPU harness; this analyzer verifies topology and API absence, not latency or resource bottlenecks.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--database',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=analyze(a.result,a.database)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'cases':[{k:v for k,v in c.items() if k!='ranks'} for c in result['cases']]},indent=2))
