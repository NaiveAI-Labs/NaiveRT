"""Cache layout contract: what prefill writes and decode reads must not drift without a trunk decision."""
import json
import os
from pathlib import Path

import pytest

from spec_dump import dump

GOLDEN = Path(__file__).with_name("golden") / "naive_n0p5_spec.json"


def test_model_spec_matches_golden():
    current = json.loads(json.dumps(dump(), sort_keys=True))
    golden = json.loads(GOLDEN.read_text())
    assert current == golden, ("Naive N spec / hybrid / kernel geometry changed; if intended, regenerate "
                               "tests/contracts/golden/naive_n0p5_spec.json and say so in the commit message")


def test_cache_dimensions_match_current_attention_geometry():
    model = dump()
    assert model["spec"]["hidden_size"] == 4096
    assert model["spec"]["main_layer_count"] == 48
    layouts = model["hybrid"]["attention_layouts"]
    assert layouts["full_dsa"]["cache_widths"] == [128, 192, 128]
    assert layouts["sliding_window"]["cache_widths"] == [192, 128]
    assert layouts["sliding_window"]["sliding_window"] == 128
    assert model["hybrid"]["indexer"]["head_count"] == 16


@pytest.mark.skipif(os.environ.get("NAIVERT_CONTRACT_GPU") != "1", reason="GPU box only")
def test_decode_after_chunked_prefill_token_exact():
    pytest.skip("GPU comparison is not implemented in this contract test; "
                "use the tests/gpu/prefill_native harness for prefill validation")
