"""Expand validated persistent-round node traces into named runtime parts.

Kernel lifetimes include waits and overlap under PDL. Their sums are activity,
not additive wall time or a claim about useful compute.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st


def target_part(name):
    for marker, part in (('indexSelectSmallIndex', 'embedding'),
                         ('attn_dsa', 'dsa_attention_and_output_projection'),
                         ('attn_swa', 'swa_attention_and_output_projection'),
                         ('checkpoint_bf1613swiglu', 'dense_gate_up_swiglu'),
                         ('checkpoint_bf1613linear', 'dense_down_and_tp'),
                         ('prepared_packed_router', 'moe_router'),
                         ('prepared_select', 'moe_gate_up_swiglu'),
                         ('expert_down_allreduce', 'moe_down_combine_and_tp'),
                         ('rmsnorm_head_proj', 'norm_and_lm_head')):
        if marker in name: return part
    raise ValueError('unmapped target kernel: '+name)


def describe(analysis, result):
    data = json.loads(analysis.read_text()); run = json.loads(result.read_text())
    assert data['pass_checks'] and run['pass_checks']
    assert hashlib.sha256(result.read_bytes()).hexdigest() == data['result_sha256']
    samples = []
    for row in data['samples']:
        if row['mode'] != 'persistent': continue
        names = run['graphs']['persistent'][row['rank']]['kernel_names']
        assert len(names) == len(row['nodes']) == 207
        # The persistent schedule is serial outside target PDL. Its validated
        # chronological order equals this graph's recorded launch order.
        target = defaultdict(list)
        draft = defaultdict(list)
        for name, node in zip(names, row['nodes']):
            if node['category']=='target': target[target_part(name)].append(node['duration_us'])
            elif node['category'] in ('d12','d3','d4','d56'):
                draft[node['category']].append(node['duration_us'])
        assert sum(map(len, target.values())) == 193
        samples.append(dict(context=row['context'], rank=row['rank'], trial=row['trial'],
            target={k:dict(count=len(v), activity_us=sum(v), per_call_us=v) for k,v in target.items()},
            draft=draft, metrics=row['metrics']))
    summary = []
    for context in (200, 2200):
        rows = [r for r in samples if r['context']==context and r['rank']==0]
        assert len(rows)==2
        target = {k:dict(count=rows[0]['target'][k]['count'],
                        activity_us=st.mean(r['target'][k]['activity_us'] for r in rows),
                        mean_call_us=st.mean(v for r in rows for v in r['target'][k]['per_call_us']))
                  for k in rows[0]['target']}
        metrics = {k:st.mean(r['metrics'][k] for r in rows) for k in rows[0]['metrics']}
        metrics['between_stage_gap_us'] = metrics['round_kernel_span_us']-sum(metrics[k] for k in
            ('draft_prefix_span_us','target_span_us','sampler_span_us','tail_span_us'))
        metrics['within_draft_gap_us'] = metrics['draft_prefix_span_us']-sum(metrics[k] for k in
            ('d12_activity_us','d3_activity_us','d4_activity_us','d56_activity_us'))
        summary.append(dict(context=context, rank=0, rounds=2, target=target, metrics=metrics,
            draft_layer_us={k:[st.mean(r['draft'][k][i] for r in rows) for i in range(len(rows[0]['draft'][k]))]
                            for k in rows[0]['draft']}))
    return dict(scope=__doc__, pass_checks=True, summary=summary,
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (analysis,result)},
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('analysis','result','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); value=describe(a.analysis,a.result)
    a.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value['summary'],indent=2))
