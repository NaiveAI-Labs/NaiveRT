"""Restoring decode KV must not reuse another request's native index keys."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('triton')
from naivert.prefill import adapters
from naivert.prefill.native.executor import NativePrefill


def test_resume_rebuilds_shadow_owned_by_separate_operand_dictionary(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
    executor = NativePrefill.__new__(NativePrefill)
    executor._closed = False
    executor.devices, executor.streams = ['cpu'], [None]
    rows = torch.arange(16 * 128).remainder(256).to(torch.uint8).view(16, 128)
    decode = dict(kind='full_dsa', ki_cache=adapters.dsa_swizzle_rows(rows, torch.arange(16)))
    native = dict(decode, ki_plain=torch.zeros_like(rows), ki_plain_end=16)
    executor.weights = [SimpleNamespace(layer=[native])]
    executor.resume_from_decode(12)
    assert native['ki_plain_end'] == 12 and 'ki_plain_end' not in decode
    assert torch.equal(native['ki_plain'][:12], rows[:12])
    # The restored cache has changed even below the old high-water mark.
    decode['ki_cache'].copy_(adapters.dsa_swizzle_rows(255-rows, torch.arange(16)))
    executor.resume_from_decode(8)
    assert native['ki_plain_end'] == 8
    assert torch.equal(native['ki_plain'][:8], 255-rows[:8])
