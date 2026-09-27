"""Verify completed current-target request evidence and retain a compact report."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    checks=json.loads((root/'checks.json').read_text())
    records=[json.loads(x) for x in (root/'completed.jsonl').read_text().splitlines()]
    assert summary['pass_checks'] and summary['native_current_target_exact']
    assert summary['reference_kind']=='pinned native draft/tail with current target'
    assert not manifest['diagnosis_only'] and manifest['expected_requests']==44
    assert all(summary[k] for k in ('current_modes_exact','no_new_xids','sources_unchanged','binaries_unchanged'))
    assert len(records)==len(checks)==132 and all(c['current_modes_equal'] for c in checks)
    groups=defaultdict(dict)
    for record,check in zip(records,checks):
        assert (record['id'],record['mode'])==(check['id'],check['mode'])
        assert record['mode'] not in groups[record['id']]
        groups[record['id']][record['mode']]=(record,check)
    assert len(groups)==44
    fields=('generated_token_ids','finish_reason','accepted_drafts','proposed_drafts',
            'verify_calls','accepted_drafts_histogram','state_digest')
    cases=[];all_tokens=all_accepted=all_proposed=all_verifies=state_fields=0
    for identity,modes in groups.items():
        assert set(modes)=={'native','staged','persistent'}
        ref=modes['native'][0]
        for name,(record,check) in modes.items():
            assert all(record[k]==ref[k] for k in fields),(identity,name)
            tokens=record['generated_token_ids']
            assert len(tokens)==record['output_tokens']
            assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==record['generated_sha256']
            assert sum(record['accepted_drafts_histogram'].values())==record['verify_calls']
            assert sum(int(k)*v for k,v in record['accepted_drafts_histogram'].items())==record['accepted_drafts']
            assert record['proposed_drafts']==7*record['verify_calls']
            assert len(record['state_digest'])==8 and all(record['state_digest'])
            for key,steps in [('sampler_protocol',record['verify_calls']),('anchor_protocol',1)]:
                protocol=record[key]
                assert protocol['steps']==steps and len(protocol['before'])==len(protocol['after'])==8
                for before,after in zip(protocol['before'],protocol['after']):
                    assert after[1]-before[1]==after[2]-before[2]==steps
                    assert after[0]-before[0]==(0 if name=='persistent' else steps)
        count=sum(map(len,ref['state_digest']))
        state_fields+=2*count
        all_tokens+=ref['output_tokens'];all_accepted+=ref['accepted_drafts']
        all_proposed+=ref['proposed_drafts'];all_verifies+=ref['verify_calls']
        cases.append(dict(id=identity,prompt_tokens=ref['prompt_tokens'],output_tokens=ref['output_tokens'],
            generated_sha256=ref['generated_sha256'],finish_reason=ref['finish_reason'],
            accepted_drafts=ref['accepted_drafts'],proposed_drafts=ref['proposed_drafts'],
            verify_calls=ref['verify_calls'],accepted_drafts_histogram=ref['accepted_drafts_histogram'],
            state_fields_per_mode=count,native_staged_persistent_exact=True,
            archived_checks=modes['persistent'][1]['archived'],
            first_archived_difference=modes['persistent'][1]['first_archived_difference']))
    assert (summary['tokens'],summary['accepted_drafts'],summary['proposed_drafts'],summary['verify_calls'])==(
        all_tokens,all_accepted,all_proposed,all_verifies)
    assert summary['acceptance']==all_accepted/all_proposed
    return dict(summary=summary,cases=cases,state_field_comparisons=state_fields,
        scope='Same current target/prefill; independently pinned draft/tail entries. Final state hashes and outputs rechecked by this collector. Not an independent target-math oracle or latency benchmark.',
        reference_kind=manifest['reference_kind'],binaries=manifest['binaries'],
        source_inventory_sha256=hashlib.sha256(json.dumps(manifest['source_inventory'],sort_keys=True).encode()).hexdigest(),
        dataset_sha256=manifest['dataset_sha256'],draft_sha256=manifest['draft_sha256'],
        harness_sha256=manifest['harness_sha256'],native_reference_sha256=manifest['native_reference_sha256'],
        raw_directory=str(root),raw_sha256={name:sha(root/name) for name in
            ('summary.json','manifest.json','checks.json','completed.jsonl')},collector_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=collect(a.root)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=result['summary'],state_field_comparisons=result['state_field_comparisons']),indent=2))
