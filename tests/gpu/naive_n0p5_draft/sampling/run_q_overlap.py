"""Queue-only wrapper for the optimistic split-q sampler screen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics as st
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=32)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    files = [a.binary, a.binary.parent/'split_q.cuh', a.binary.parent/'source.json']
    files += [a.inputs/n for n in ('target.bin', 'draft.bin', 'proposals.bin')]
    inventory = {str(f): sha(f) for f in files}
    before = subprocess.check_output(['dmesg'], text=True)
    (a.out/'dmesg_before.log').write_text(before)
    cmd = [str(a.binary), str(a.inputs), str(a.repeats)]
    try:
        with (a.out/'stdout.json').open('w') as out, (a.out/'stderr.log').open('w') as err:
            subprocess.run(cmd, stdout=out, stderr=err, check=True, timeout=600)
        r = json.loads((a.out/'stdout.json').read_text())
        assert r['pass_checks'] and r['bitwise'] and r['cases'] == 44
        for t in r['timing']:
            cycles = t['max_rank_abba_us']
            assert len(cycles) == a.repeats
            assert t['baseline_nodes'] == 33
            assert t['nodes_per_call'] == {'prepared_q_tail': 33, 'split_sequential': 43, 'q_prepare_only': 10}[t['variant']]
            delta = [(c[1]+c[2]-c[0]-c[3])/2 for c in cycles]
            ratios = [(c[1]+c[2])/(c[0]+c[3]) for c in cycles]
            rng = random.Random(20260918)
            boot = sorted(st.median(rng.choices(delta, k=len(delta))) for _ in range(5000))
            t.update(baseline_median_us=st.median(c[i] for c in cycles for i in (0, 3)),
                     candidate_median_us=st.median(c[i] for c in cycles for i in (1, 2)),
                     paired_median_delta_us=st.median(delta), paired_median_ratio=st.median(ratios),
                     paired_delta_bootstrap95_us=[boot[124], boot[4874]])
            print(json.dumps({k: v for k, v in t.items() if k != 'max_rank_abba_us'}), flush=True)
        assert all(sha(Path(f)) == h for f, h in inventory.items())
        r['provenance'] = dict(command=cmd, sha256=inventory, files_unchanged=True,
            scope='Captured real logits, no target forward. prepared_q_tail excludes all q preparation cost.',
            timer='Max-rank CUDA events, 16 invocations per graph, ABBA, four warmup cycles')
        (a.out/'result.json').write_text(json.dumps(r, indent=2)+'\n')
    finally:
        after = subprocess.check_output(['dmesg'], text=True)
        (a.out/'dmesg_after.log').write_text(after)
        xids = lambda s: Counter(x for x in s.splitlines() if 'NVRM: Xid' in x)
        delta_xids = list((xids(after)-xids(before)).elements())
        (a.out/'health.json').write_text(json.dumps(dict(new_xids=delta_xids), indent=2)+'\n')
    assert not delta_xids
