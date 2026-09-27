"""The native append bridge must process every new position exactly once."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location('prefill_schedule', Path(__file__).resolve().parents[1] / 'gpu/naive_n0p5_draft/round/prefill.py')
prefill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefill)


@pytest.mark.parametrize('start,n', [(0, 1), (1, 2), (1, 4096), (4095, 4097),
                                    (4095, 4098), (4096, 8200), (4097, 8200), (8191, 12290)])
def test_bridge_native_chunks_and_seq1_tail(monkeypatch, start, n):
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
    seen, rebuilds, chunks, progress = [], [], [], []
    rt = SimpleNamespace(stream=SimpleNamespace(synchronize=lambda: None),
                         pos=torch.tensor([start]), token=torch.zeros(1, dtype=torch.long),
                         resid=torch.zeros(2, 1, 1))
    def step():
        p = rt.pos.item(); seen.append(p)
        assert rt.token.item() == p
        rt.resid[1, 0, 0] = p
        rt.pos.add_(1)
    rt.run_step = step
    verify = SimpleNamespace(rt=rt, graph=True, tokens=torch.zeros(8, dtype=torch.long), resid=torch.zeros(2, 8, 1))
    def replay():
        p = rt.pos.item(); seen.extend(range(p, p+8))
        assert verify.tokens.tolist() == list(range(p, p+8))
        verify.resid[1, :, 0] = torch.arange(p, p+8)
        rt.pos.add_(8)
    verify.replay = replay
    features = [torch.full((n, 1), -1.)]
    executor = SimpleNamespace(chunk_tokens=4096, resume_from_decode=rebuilds.append)
    def run(tokens, *, start):
        assert start % 4096 == 0
        assert tokens == list(range(start, start+len(tokens)))
        chunks.append((start, len(tokens))); seen.extend(tokens)
        features[0][start:start+len(tokens), 0] = torch.tensor(tokens)
    executor.run = run
    timing = {}
    prefill.prefill_tokens([rt], [verify], list(range(n)), features, [0], start=start,
                          mode='native', executor=executor, timing=timing, progress=progress.append)
    assert seen == list(range(start, n))
    assert (features[0][:start] == -1).all()
    assert features[0][start:, 0].tolist() == list(range(start, n))
    assert rt.pos.item() == n
    assert all(p % 4096 == 0 for p in rebuilds)
    if start and chunks: assert rebuilds == [chunks[0][0]]
    else: assert not rebuilds
    if n-1 > start: assert n-1 in progress
    assert timing['tail_tokens'] == 1
