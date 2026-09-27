"""The session engine's choice between the native executor and the verify-8 replay is a pure function."""
import pytest

from naivert.backends.naive_n0p5_session_engine import (NATIVE_MAX_PROMPT, best_prefix_match, over_budget, park_name,
                                                      native_resume_ready, prefill_estimate, prefill_path, select_resume)


def test_without_native_everything_replays():
    assert prefill_path(False, 0, 4000) == "verify8"
    assert prefill_path(False, 3000, 9000) == "verify8"


def test_cold_prompt_goes_native():
    assert prefill_path(True, 0, 1) == "native"
    assert prefill_path(True, 0, 12000) == "native"


def test_cost_policy_follows_the_suffix_length():
    # a 48-token follow-up on a 6.4K prefix extends the parked rows; a 4K paste recomputes
    assert prefill_path(True, 6388, 6436) == "verify8"
    assert prefill_path(True, 6388, 6388 + 4000) == "native"
    # the break-even suffix grows with the prompt length (more chunks, more draft rows to re-ingest)
    def break_even(start):
        return next(s for s in range(1, 20000) if prefill_path(True, start, start + s) == "native")
    assert break_even(1500) < break_even(6000) < break_even(14000)
    assert 150 <= break_even(1500) <= 600


def test_estimate_terms():
    e = prefill_estimate(0, 4000)
    assert e["chunks"] == 1 and e["suffix_tokens"] == 4000 and e["native_ms"] < e["replay_ms"]
    e = prefill_estimate(10000, 10048)
    assert e["chunks"] == 3 and e["suffix_tokens"] == 48 and e["replay_ms"] < e["native_ms"]


def test_batched_ingest_moves_the_break_even_down():
    from naivert.backends.naive_n0p5_session_engine import BATCHED_INGEST_MS_PER_TOKEN
    def break_even(start, rate):
        return next(s for s in range(1, 20000) if prefill_path(True, start, start + s, "cost", rate) == "native")
    assert break_even(1500, BATCHED_INGEST_MS_PER_TOKEN) < break_even(1500, 0.065)


def test_forced_policies_and_validation():
    assert prefill_path(True, 6388, 6436, "native") == "native"
    assert prefill_path(True, 6388, 6388 + 4000, "replay") == "verify8"
    with pytest.raises(ValueError):
        prefill_path(True, 10, 20, "sometimes")


def test_long_resume_requires_complete_chunk_graphs():
    from types import SimpleNamespace
    native = SimpleNamespace(graphs=True, chunk_graphs=True, _chunk_captured=dict.fromkeys(range(256)))
    assert native_resume_ready(native, 256)
    del native._chunk_captured[127]
    native._chunk_captured[999] = None
    assert not native_resume_ready(native, 256)
    assert native_resume_ready(native, 32)
    native.chunk_graphs = False
    assert not native_resume_ready(native, 32)
    native.chunk_graphs = True
    native.graphs = False
    assert not native_resume_ready(native, 32)
    assert not native_resume_ready(native, 16)
    assert not native_resume_ready(None, 32)


def test_short_resume_does_not_recompute_using_uncaptured_warm_estimates():
    from types import SimpleNamespace
    native = SimpleNamespace(graphs=True, chunk_graphs=True, _chunk_captured=dict.fromkeys(range(3)))
    assert native_resume_ready(native, 3)
    assert not native_resume_ready(native, 10)
    # Real Claude Code Read continuation: 38K prompt after a 16.8K parked prefix.
    assert prefill_path(True, 16801, 38316, native_ready=native_resume_ready(native, 10),
                        native_suffix_ready=native_resume_ready(native, 10, first_chunk=5)) == 'verify8'
    native._chunk_captured.update(dict.fromkeys(range(5, 10)))
    assert not native_resume_ready(native, 10)
    assert native_resume_ready(native, 10, first_chunk=5)


def test_uncaptured_graphs_do_not_use_a_warm_resume_estimate():
    assert prefill_path(True, 6388, 10388, native_ready=False) == 'verify8'
    assert prefill_path(True, 6388, 10388, 'native', native_ready=False) == 'native'
    assert prefill_path(True, 0, 10388, native_ready=False) == 'native'


def test_long_estimates_include_growing_indexer_work():
    # Exclude draft ingest to isolate target prefill. A fixed cost per chunk
    # underestimates long prompts: each later chunk scans more index keys.
    short = prefill_estimate(0, 65536, 0)
    medium = prefill_estimate(0, 131073, 0)
    long = prefill_estimate(0, 1048000, 0)
    assert short['chunks'] == 16 and medium['chunks'] == 32 and long['chunks'] == 256
    assert short['native_ms'] < medium['native_ms'] < long['native_ms']
    assert long['native_ms'] / 256 > medium['native_ms'] / 32
    assert prefill_estimate(0, 131074, 0)['chunks'] == 33  # final token uses seq1


def test_long_resume_selects_by_suffix_and_readiness():
    from naivert.backends.naive_n0p5_session_engine import BATCHED_INGEST_MS_PER_TOKEN as rate
    assert prefill_path(True, 900000, 900048, ingest_ms_per_token=rate) == 'verify8'
    assert prefill_path(True, 100000, 900000, ingest_ms_per_token=rate) == 'native'
    assert prefill_path(True, 100000, 900000, ingest_ms_per_token=rate, native_ready=False) == 'verify8'


def test_native_suffix_prices_only_the_bridge_and_new_chunks():
    e = prefill_estimate(900000, 923500, 0)
    assert e['native_bridge_tokens'] == 1120
    assert e['native_suffix_first_chunk'] == 220
    assert e['native_suffix_chunks'] == 6
    assert e['native_suffix_ms'] < e['replay_ms'] < e['native_ms']
    assert prefill_path(True, 900000, 923500, native_ready=False, native_suffix_ready=True) == 'native_suffix'
    assert prefill_path(True, 900000, 900048, native_suffix_ready=True) == 'verify8'
    assert prefill_estimate(900000, 900048)['native_suffix_chunks'] == 0
    # Short full recomputation can be cheaper than decoding to the next grid line.
    assert prefill_path(True, 6388, 10388, native_suffix_ready=True) == 'native'
    from types import SimpleNamespace
    native = SimpleNamespace(graphs=True, chunk_graphs=True, _chunk_captured=dict.fromkeys(range(220, 226)))
    assert native_resume_ready(native, 226, first_chunk=220)
    assert not native_resume_ready(native, 226)


def test_keyed_resume_considers_every_entry_but_prefers_its_own_key():
    parent = list(range(100))
    entries = {"k": (50, parent[:50] + [999] * 10),            # the key holds a subagent's unrelated sequence
               "k#3": (80, parent[:80]),                        # the parent's own sequence, parked aside
               "anon-1": (80, parent[:80])}
    assert select_resume(entries, parent, "k") == ("k#3", 80)   # exact match beats the key's entry
    entries["k"] = (80, parent[:80])
    assert select_resume(entries, parent, "k") == ("k", 80)     # the key wins ties
    assert select_resume(entries, parent[:60], "k") is None      # nothing parked at or below 59
    assert select_resume({}, parent, "k") is None


def test_park_name_keeps_the_parent_prefix_while_subagents_share_the_key():
    parent = list(range(100))
    assert park_name({}, "k", parent, 5) == ("k", [])                                   # free key
    entries = {"k": (60, parent[:60])}
    assert park_name(entries, "k", parent, 6) == ("k", ["k"])                           # extends: replace
    sub = [7] * 40
    assert park_name(entries, "k", sub, 7) == ("k#7", [])                               # unrelated: keep both
    entries["k#7"] = (40, sub)
    assert park_name(entries, "k", parent, 8) == ("k", ["k"])                           # parent continues
    assert park_name(entries, "k", sub + [1, 2], 9) == ("k#9", ["k#7"])                 # subagent continues
    entries["anon-2"] = (60, parent[:60])
    assert park_name(entries, "k", parent, 10) == ("k", ["k"])                          # other families untouched


def test_prompts_beyond_the_native_span_replay():
    """The final padded chunk is validated through 1M visible keys."""
    assert NATIVE_MAX_PROMPT == 1048576
    assert prefill_path(True, 0, NATIVE_MAX_PROMPT) == "native"
    for start in (0, 60000, NATIVE_MAX_PROMPT):
        for policy in ("cost", "native", "replay"):
            assert prefill_path(True, start, NATIVE_MAX_PROMPT + 1, policy) == "verify8"
    assert prefill_path(True, 0, 900_000, "native") == "native"
    assert prefill_path(True, 800_000, 900_000, "native", native_ready=False) == "native"


def test_store_budget_is_per_rank():
    """--store-gb is per rank; the store sums eight ranks, so a 916K-token session (8.3 GB per rank,
    66 GB in all) fits a 64 GB budget instead of being evicted as it is parked."""
    gib = 1 << 30
    assert not over_budget(66 * gib, 8, 64 * gib)
    assert over_budget(8 * 64 * gib + 1, 8, 64 * gib)
    assert over_budget(9 * gib, 1, 8 * gib) and not over_budget(8 * gib, 1, 8 * gib)


def test_best_prefix_match_names_the_closest_entry():
    parent = list(range(100))
    entries = {"k": (50, parent[:40] + [9] * 10), "k#3": (80, parent[:70] + [8] * 10), "anon": (10, parent[:10])}
    assert best_prefix_match(entries, parent) == ("k#3", 70)
    assert best_prefix_match({}, parent) is None
