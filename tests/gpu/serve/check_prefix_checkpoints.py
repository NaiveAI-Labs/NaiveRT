"""Real target/draft checkpoint gate: retry, ring rollback and an edited history."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from transformers import AutoTokenizer
from naivert.backends.naive_n0p5_prefix_store import CACHE_NAMES
from naivert.backends.naive_n0p5_session_engine import SessionEngine


def equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8).cpu(), b.contiguous().view(torch.uint8).cpu())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True, type=Path)
    ap.add_argument('--draft', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.target, local_files_only=True, trust_remote_code=False)
    engine = SessionEngine.build(repo=Path(__file__).resolve().parents[3], target=a.target,
        checkpoint=a.draft, max_decode_tokens=64, capacity=16384, output_capacity=256,
        tokenizer=tokenizer, prefill='native', prefill_warm_chunks=2, resume_policy='replay',
        default_temperature=0.01, default_top_p=1.0, log_fn=lambda x: print(x, flush=True))
    engine.store.host = True
    native = engine.native
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    oracles = {}
    checkpoint = engine.store.checkpoint

    def capture(ranks, *, position):
        cp = checkpoint(ranks, position=position)
        # Independent full target snapshot taken at the actual boundary, before
        # later prefill/decode overwrites any ring slots. Not used by production.
        per_rank = []
        for w in engine.weights:
            per_rank.append([{n: (lw[n] if lw['kind'] == 'sliding_window' else lw[n][:position]).cpu().clone()
                              for n in CACHE_NAMES if n in lw} for lw in w.layer])
        oracles[position] = per_rank
        return cp
    engine.store.checkpoint = capture

    def run(tokens, sid):
        s = engine.open_session(tokens, session_id=sid, max_decode_tokens=64,
                                sampling=dict(seed=1234, temperature=0.01, top_p=1.0))
        out = []
        try:
            for _ in range(64):
                t = s.step(); out.append(t)
                if t == 151645: break
        finally:
            s.close()
        return dict(tokens=out, prefill=s.runtime.prepared_prefill.as_dict())

    def verify_restore(key, position, oracle):
        entry = engine.store.entries[key]
        engine.store.restore(key, engine.ranks, position=position)
        target_ok, draft_ok = [], []
        for r, (w, b) in enumerate(zip(engine.weights, engine.buffers)):
            target_ok.append(all(equal(lw[n] if lw['kind'] == 'sliding_window' else lw[n][:position], expected[n])
                                 for lw, expected in zip(w.layer, oracle[r]) for n in expected))
            dk, dv = entry['ranks'][r][1]
            draft_ok.append(equal(b.kv_k[:, :position], dk[:, :position]) and equal(b.kv_v[:, :position], dv[:, :position]))
        assert all(target_ok) and all(draft_ok), (target_ok, draft_ok)
        return dict(position=position, all_rank_target_exact=target_ok, all_rank_draft_exact=draft_ok)

    try:
        base = tokenizer.encode('Explain the design trade-offs in a runtime that reuses cached prefixes. ', add_special_tokens=False)
        for mode, n in [('verify8', 1025), ('native', 1025), ('native', 4501)]:
            engine.native = native if mode == 'native' else None
            engine.store.entries.clear(); oracles.clear()
            tokens = (base * (n // len(base) + 1))[:n]
            first = run(tokens, 'parent')
            entry = engine.store.entries['parent']
            oracle = oracles[n-1]
            proof = verify_restore('parent', n-1, oracle)
            retry = run(tokens, 'parent')
            assert retry['prefill']['start_position'] == n-1, retry['prefill']
            row = dict(mode=mode, prompt_tokens=n, restore=proof,
                       retry_tokens_equal=first['tokens'] == retry['tokens'],
                       first=first, retry=retry)
            report['cases'].append(row)
            (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            assert row['retry_tokens_equal'], 'retry changed generated token IDs'
            # Extend a real parked sequence, preserving its older prompt checkpoint.
            latest = max(engine.store.entries, key=lambda k: engine.store.entries[k]['saved_at'])
            continuation = list(engine.store.entries[latest]['tokens']) + base * 40
            extended = run(continuation, 'parent')
            latest = max(engine.store.entries, key=lambda k: engine.store.entries[k]['saved_at'])
            row['inherited_restore'] = verify_restore(latest, n-1, oracle)
            # Rewrite inside the prior answer, before the newest prompt checkpoint.
            edited = continuation[:]
            edited[n+3] = (edited[n+3] + 1) % 151000
            rewritten = run(edited, 'parent')
            assert rewritten['prefill']['start_position'] == n-1, rewritten['prefill']
            row.update(continuation=extended, history_edit=rewritten)
            print('CHECKPOINT_CASE_OK', mode, n, flush=True)
            (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        report['success'] = True
    finally:
        engine.native = native
        engine.close()
        report['closed'] = engine._closed and native._closed
        (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PREFIX_CHECKPOINTS_OK', flush=True)


if __name__ == '__main__':
    main()
