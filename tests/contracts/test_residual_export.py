"""Verify-8 window and residual exports the draft stream consumes."""
import os
import re

import pytest

from conftest import ROOT, load_driver
from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS

DRAFT_EXPORT_LAYERS = TARGET_LAYER_IDS   # residual after these target layers feeds the draft fc
VERIFY_ROWS = 8                                          # one anchor + seven draft rows


def test_verify_window_is_eight_rows():
    verify = load_driver("run_naive_n0p5_verify.py")
    assert verify.VROWS == VERIFY_ROWS


def test_draft_export_layers_within_model():
    from naivert.models.naive_n0p5 import model as m
    layers = m.NAIVE_N0P5_GEOMETRY.main_layer_count
    assert layers == 48
    assert all(0 < L < layers for L in DRAFT_EXPORT_LAYERS)
    assert DRAFT_EXPORT_LAYERS == tuple(sorted(set(DRAFT_EXPORT_LAYERS)))


def test_verify_driver_keeps_per_layer_residuals():
    source = (ROOT / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py").read_text()
    assert re.search(r"self\.resid = \[torch\.zeros\(VROWS, HIDDEN.*range\(layers \+ 1\)\]", source), \
        "the verify driver must keep one residual buffer per layer boundary (resid[L] for L in 0..layers)"


def test_feature_export_selects_after_layer_storage():
    verify = load_driver("run_naive_n0p5_verify.py")
    instance = verify.VerifyRuntime.__new__(verify.VerifyRuntime)
    instance.resid = [object() for _ in range(49)]
    for rank, layer in enumerate(TARGET_LAYER_IDS):
        instance.rank = rank
        assert instance.context_feature_for_rank() is instance.resid[layer + 1]


@pytest.mark.skipif(os.environ.get("NAIVERT_CONTRACT_GPU") != "1", reason="GPU box only")
def test_residual_export_values():
    pytest.skip("compare resid[L] for L in DRAFT_EXPORT_LAYERS against the eager reference after one verify-8 replay")
