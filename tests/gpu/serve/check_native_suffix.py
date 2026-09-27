"""Same-prefix native append: exact composition, target/draft handoff and latency."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from transformers import AutoTokenizer
from naivert.backends.naive_n0p5_session_engine import SessionEngine
from naivert.backends.naive_n0p5_prefix_store import CACHE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(a.target, local_files_only=True, trust_remote_code=False)
    engine = SessionEngine.build(repo=Path(__file__).resolve().parents[3], target=a.target,
        checkpoint=a.draft, max_decode_tokens=64, capacity=98304, output_capacity=256,
        tokenizer=tokenizer, prefill='native', prefill_warm_chunks=2, resume_policy='cost',
        default_temperature=0.01, default_top_p=1., log_fn=lambda s: print(s, flush=True))
    engine.store.host = True
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[],
                  timing='same restored prefix; restore excluded; three alternating samples after warmup; target prefill includes bridge and seq1 tail, excludes draft ingest')

    def save():
        (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def generate(tokens, key):
        s = engine.open_session(tokens, session_id=key, max_decode_tokens=64,
                                sampling=dict(seed=1234, temperature=0.01, top_p=1.))
        output = []
        try:
            for _ in range(64):
                t = s.step(); output.append(t)
                if t == 151645: break
        finally:
            s.close()
        return dict(tokens=output, prefill=s.runtime.prepared_prefill.as_dict())

    def snapshot(n, start):
        engine._sync()
        out = []
        for r, w in enumerate(engine.weights):
            caches = [{k: (lw[k] if lw['kind'] == 'sliding_window' else lw[k][:n]).cpu().clone()
                       for k in CACHE_NAMES if k in lw} for lw in w.layer]
            out.append((caches, engine.features[r][start:n].cpu().clone(), engine.ranks[r].logits.cpu().clone()))
        return out

    def same(left, right):
        return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))

    def compare(left, right):
        return [dict(caches=all(same(v, b[0][L][k]) for L, layer in enumerate(a[0]) for k, v in layer.items()),
                     features=same(a[1], b[1]), logits=same(a[2], b[2])) for a, b in zip(left, right)]

    try:
        base = tokenizer.encode('Explain how a runtime can reuse a cached prefix without changing the result. ', add_special_tokens=False)
        for start in (8063, 69631):
            engine.store.entries.clear()
            original = (base * ((start + 1) // len(base) + 1))[:start + 1]
            generate(original, 'parent')
            key = max(engine.store.entries, key=lambda k: engine.store.entries[k]['saved_at'])
            assert start in engine.store.entries[key]['checkpoints']
            suffix = (base * (8193 // len(base) + 1))[:8193]
            suffix[0] = (original[start] + 1) % 151000
            tokens = original[:start] + suffix
            n = len(tokens); aligned = (start + 4095) // 4096 * 4096
            def restore():
                engine.store.restore(key, engine.ranks, position=start)
                engine._sync()
            def target(mode, position=start, supplied=tokens):
                engine.prefill_tokens(engine.ranks, engine.verifies, supplied, engine.features,
                                      engine.layer_ids, start=position, mode=mode,
                                      executor=engine.native if mode == 'native' else None)
                engine._sync()
            # Explicit bridge and aligned native calls form an independent schedule oracle.
            restore(); began = time.perf_counter()
            target('verify8', supplied=tokens[:aligned]); target('native', position=aligned)
            first_use_ms = (time.perf_counter()-began)*1000
            expected = snapshot(n, start)
            # A restore may follow an unrelated request. Poison the native shadow
            # to prove restore invalidates it and resume rebuilds from parked KV.
            for r, w in enumerate(engine.native.weights):
                with torch.inference_mode(), torch.cuda.device(r), torch.cuda.stream(engine.ranks[r].stream):
                    for lw in w.layer:
                        if 'ki_plain' in lw:
                            lw['ki_plain'].zero_(); lw['ki_plain_end'] = n
            engine._sync()
            restore(); began = time.perf_counter(); target('native')
            capture_ms = (time.perf_counter()-began)*1000
            actual = snapshot(n, start)
            exact = compare(expected, actual)
            assert all(all(r.values()) for r in exact), exact
            native_logits = actual[0][2].double().flatten()
            del expected
            restore(); target('verify8')
            reference_logits = engine.ranks[0].logits.detach().cpu().double().flatten()
            old_logp = reference_logits.log_softmax(-1); new_logp = native_logits.log_softmax(-1)
            kl = float((old_logp.exp() * (old_logp - new_logp)).sum())
            top = reference_logits.topk(2).values
            same_top = int(reference_logits.argmax()) == int(native_logits.argmax())
            accuracy = dict(last_token_kl=kl, top1_equal=same_top, reference_margin=float(top[0]-top[1]),
                            relative_rms=float((native_logits-reference_logits).square().mean().sqrt() /
                                               reference_logits.square().mean().sqrt().clamp_min(1e-30)))
            # Record independent baselines before making an accuracy verdict.
            restore(); target('verify8')
            repeat_logits = engine.ranks[0].logits.detach().cpu().double().flatten()
            accuracy['verify8_repeat_exact'] = torch.equal(reference_logits, repeat_logits)
            engine.store.clear(engine.ranks); engine._sync(); engine.native.begin_request()
            target('native', position=0)
            full_logits = engine.ranks[0].logits.detach().cpu().double().flatten()
            full_logp = full_logits.log_softmax(-1)
            accuracy['full_native_vs_verify8_kl'] = float((old_logp.exp() * (old_logp-full_logp)).sum())
            accuracy['append_vs_full_native_kl'] = float((full_logp.exp() * (full_logp-new_logp)).sum())
            accuracy['full_native_top1_equal'] = int(full_logits.argmax()) == int(reference_logits.argmax())
            # Use a shared 64-token continuation to distinguish a one-position
            # native/replay rounding gap from a systematic append regression.
            forced, reference_rows = [], []
            restore(); target('verify8')
            for i in range(64):
                logits = engine.ranks[0].logits.detach().cpu().double().flatten()
                if i == 0:
                    accuracy['verify8_after_full_native_exact'] = torch.equal(reference_logits, logits)
                reference_rows.append(logits); forced.append(int(logits.argmax()))
                if i < 63: target('seq1', position=n+i, supplied=tokens+forced)
            def forced_metrics():
                kls, agreement, confident, confident_agree = [], 0, 0, 0
                for i, ref in enumerate(reference_rows):
                    logits = engine.ranks[0].logits.detach().cpu().double().flatten()
                    lp = ref.log_softmax(-1)
                    kls.append(float((lp.exp() * (lp-logits.log_softmax(-1))).sum()))
                    match = int(ref.argmax()) == int(logits.argmax())
                    margin = ref.topk(2).values
                    decided = float(margin[0]-margin[1]) >= 1.
                    agreement += match; confident += decided; confident_agree += match and decided
                    if i < 63: target('seq1', position=n+i, supplied=tokens+forced[:i+1])
                return dict(mean_kl=statistics.mean(kls), max_kl=max(kls), agreement=agreement,
                            confident=confident, confident_agreement=confident_agree, tokens=64)
            restore(); target('native')
            repeated = snapshot(n, start)
            accuracy['native_after_full_native_exact'] = compare(actual, repeated)
            del actual, repeated
            accuracy['append_forced'] = forced_metrics()
            engine.store.clear(engine.ranks); engine._sync(); engine.native.begin_request()
            target('native', position=0)
            accuracy['full_native_forced'] = forced_metrics()
            accuracy['single_token_gate_pass'] = bool(torch.isfinite(native_logits).all() and kl < .05 and
                                                     (same_top or accuracy['reference_margin'] < 1.))
            forced_result = accuracy['append_forced']
            accuracy['gate_pass'] = bool(torch.isfinite(native_logits).all() and
                accuracy['verify8_repeat_exact'] and accuracy['verify8_after_full_native_exact'] and
                all(all(r.values()) for r in accuracy['native_after_full_native_exact']) and
                forced_result['mean_kl'] < .05 and
                forced_result['confident_agreement'] == forced_result['confident'])
            samples = {'verify8': [], 'native': []}
            for rep in range(3):
                for mode in (('verify8', 'native') if rep % 2 == 0 else ('native', 'verify8')):
                    restore(); began = time.perf_counter(); target(mode)
                    samples[mode].append((time.perf_counter()-began)*1000)
            row = dict(start=start, prompt_tokens=n, suffix_tokens=n-start, bridge_tokens=aligned-start,
                       composition_exact=exact, accuracy=accuracy, first_use_ms=first_use_ms,
                       capture_ms=capture_ms, samples_ms=samples,
                       median_ms={k: statistics.median(v) for k, v in samples.items()})
            report['cases'].append(row); save()
            served = generate(tokens, 'parent')
            assert served['prefill']['prefill_path'] == 'native_suffix', served['prefill']
            assert served['prefill']['start_position'] == start, served['prefill']
            assert served['prefill']['ingest_rows'] == n-1-start
            row['target_draft_session'] = served
            save(); print('NATIVE_SUFFIX_CASE', start, row['median_ms'], accuracy, flush=True)
        from prefix_http_probe import check
        report['http'] = check(engine, tokenizer)
        report['success'] = all(c['accuracy']['gate_pass'] for c in report['cases'])
        assert report['success'], 'suffix accuracy gate failed; see per-case baselines'
    finally:
        engine.close(); report['closed'] = engine._closed; save()


if __name__ == '__main__':
    main()
