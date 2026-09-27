"""Saved ring states permit exact rollback; arbitrary common prefixes do not."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from naivert.backends.naive_n0p5_prefix_store import PrefixStore
from naivert.backends.naive_n0p5_session_engine import SessionEngine, select_resume


def test_selects_only_recorded_rollback_positions():
    entries = {'parent': (9, list(range(10))), 'parent#2': (7, [90] * 7)}
    points = {'parent': {3: None, 6: None, 8: None}}
    assert select_resume(entries, list(range(6)) + [99], 'parent', checkpoints=points) == ('parent', 6)
    assert select_resume(entries, list(range(5)) + [99], 'parent', checkpoints=points) == ('parent', 3)
    assert select_resume(entries, list(range(6)) + [99], 'parent') is None
    assert select_resume(entries, list(range(9)), 'parent', checkpoints=points) == ('parent', 8)
    assert select_resume(entries, [99] * 10, 'parent', checkpoints=points) is None
    assert select_resume(entries, list(range(11)), 'parent', checkpoints=points) == ('parent', 9)


class SplitTokenizer:
    """The sampler's compound token has the same text as two canonical tokens."""
    def decode(self, tokens, **kwargs):
        return ''.join('ab' if t == 500 else chr(t) for t in tokens)

    def encode(self, text, **kwargs):
        return list(map(ord, text))


def fake_engine(entries):
    engine = SessionEngine.__new__(SessionEngine)
    engine.tokenizer = SplitTokenizer()
    engine.store = PrefixStore([None] * 8, [None] * 8)
    engine.store.entries = {k: dict(position=len(v), tokens=v) for k, v in entries.items()}
    return engine


def test_alignment_finds_sibling_and_anonymous_predecessors():
    e = fake_engine({'parent': list(map(ord, 'other')), 'parent#2': [500], 'anon-7': [500, ord('c')]})
    text = 'abcd'
    aligned = e.align_prompt(list(map(ord, text)), text=text, session_id='parent')
    assert aligned == [500, ord('c'), ord('d')]
    assert select_resume(e._entries(), aligned, 'parent') == ('anon-7', 2)
    assert e.align_prompt(list(map(ord, text)), text=text, session_id='missing') == aligned
    different = list(map(ord, 'axbcd'))
    assert e.align_prompt(different, text='axbcd', session_id='parent') == different


def test_alignment_preserves_exact_tokens_and_prefers_key_on_text_ties():
    e = fake_engine({'parent#2': [500], 'parent': [500]})
    assert e.align_prompt([500, 99], text='abc', session_id='parent') == [500, 99]
    assert e.align_prompt([97, 98, 99], text='abc', session_id='parent') == [500, 99]
    # No token after the parked position: keep the prompt for checkpoint selection.
    assert e.align_prompt([97, 98], text='ab', session_id='parent') == [97, 98]


def test_checkpoint_alignment_preserves_sampled_ids_on_retry_and_rewrite():
    e = fake_engine({'parent': list(map(ord, 'old branch')), 'parent#2': [500, 99, 100, 101]})
    cp = {1: {'position': 1}, 2: {'position': 2}}
    e.store.entries['parent#2']['checkpoints'] = cp
    # The parked answer is longer than this retry; only its earlier checkpoint matches.
    assert e.align_prompt([97, 98, 99], text='abc', session_id='parent') == [500, 99]
    aligned = e.align_prompt([97, 98, 99, 88], text='abcX', session_id='parent')
    assert aligned == [500, 99, 88]
    assert select_resume(e._entries(), aligned, 'parent', checkpoints={'parent#2': cp}) == ('parent#2', 2)
    assert cp[2]['text'] == 'abc'


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
    weights, buffers, ranks = [], [], []
    for r in range(8):
        weights.append(SimpleNamespace(layer=[
            dict(kind='full_dsa', **{k: torch.zeros(12, 2, dtype=torch.int32)
                 for k in ('k_cache', 'v_cache', 'ki_cache', 'ki_scale')}),
            dict(kind='sliding_window', k_cache=torch.zeros(4, 2, dtype=torch.int32),
                 v_cache=torch.zeros(4, 2, dtype=torch.int32))]))
        buffers.append(SimpleNamespace(kv_k=torch.zeros(2, 12, 2), kv_v=torch.zeros(2, 12, 2)))
        ranks.append(SimpleNamespace(stream=SimpleNamespace(synchronize=lambda: None), pos=torch.zeros(1, dtype=torch.int32)))
    return PrefixStore(weights, buffers), ranks


def append_rows(store, start, end):
    for r, w in enumerate(store.weights):
        for p in range(start, end):
            for L, lw in enumerate(w.layer):
                for n, t in lw.items():
                    if n != 'kind': t[p if L == 0 else p % 4].fill_(r * 100 + p)
            store.buffers[r].kv_k[:, p].fill_(r * 100 + p)
            store.buffers[r].kv_v[:, p].fill_(r * 100 + p + 1)


def test_restore_checkpoint_truncates_append_only_rows_and_recovers_wrapped_ring(store):
    s, ranks = store
    append_rows(s, 0, 5)
    cp = s.checkpoint(ranks, position=5)
    append_rows(s, 5, 9)  # overwrite every ring slot
    final_ring = s.weights[0].layer[1]['k_cache'].clone()
    s.save('a', ranks, position=9, tokens=list(range(10)), checkpoints={5: cp})
    base_bytes = sum(x.numel() * x.element_size() for layers, draft in s.entries['a']['ranks']
                     for x in [*(t for layer in layers for t in layer.values()), *draft])
    assert s.bytes('a') == base_bytes + cp['bytes']
    position, tokens, _ = s.restore('a', ranks, position=5)
    assert position == 5 and tokens == list(range(10))
    for r, (w, b, rt) in enumerate(zip(s.weights, s.buffers, ranks)):
        assert rt.pos.item() == 5
        for t in (v for k, v in w.layer[0].items() if k != 'kind'):
            assert torch.equal(t[:5, 0], torch.arange(5) + r * 100)
            assert not t[5:].any()
        assert torch.equal(w.layer[1]['k_cache'], cp['rings'][r][1]['k_cache'])
        assert not b.kv_k[:, 5:].any() and not b.kv_v[:, 5:].any()
    s.restore('a', ranks)
    assert torch.equal(s.weights[0].layer[1]['k_cache'], final_ring)
    assert ranks[0].pos.item() == 9
    before = s.weights[0].layer[0]['k_cache'].clone()
    with pytest.raises(ValueError, match='without.*checkpoint'):
        s.restore('a', ranks, position=6)
    assert torch.equal(before, s.weights[0].layer[0]['k_cache'])


def test_checkpoint_survives_superseded_entry_eviction(store):
    s, ranks = store
    append_rows(s, 0, 5)
    cp = s.checkpoint(ranks, position=5)
    s.save('a', ranks, position=5, tokens=list(range(6)), checkpoints={5: cp})
    inherited = dict(s.entries['a']['checkpoints'])
    s.evict('a')
    append_rows(s, 5, 9)
    s.save('b', ranks, position=9, tokens=list(range(10)), checkpoints=inherited)
    s.restore('b', ranks, position=5)
    assert torch.equal(s.weights[0].layer[1]['k_cache'], cp['rings'][0][1]['k_cache'])
