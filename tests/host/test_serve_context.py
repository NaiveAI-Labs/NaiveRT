"""ServerContext lifecycle on a fake engine (no GPU): admission, context-exceeded, cancel, n_predict."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from naivert.runtime.contract import RuntimeContractError
from naivert.runtime.mtp_contract import DecodeMode
from naivert.serve.context import CONTEXT_EXCEEDED_MARK, ContextExceeded, RequestRejected, ServerContext
from naivert.serve.slot import SlotState
from naivert.serve.task import FinalResult, OpenedResult, PartialResult, ServerTask, TaskParams

EOS = 151645


class FakeTokenizer:
    eos_token_id = EOS

    def decode(self, tokens, skip_special_tokens=False, **_):
        return "".join("<eos>" if t == EOS else chr(96 + t) for t in tokens)


class FakeFormatter:
    tokenizer = FakeTokenizer()
    eos_token_id = EOS

    def encode(self, messages, *, tools=None, enable_thinking=None):
        self.last_kwargs = dict(tools=tools, enable_thinking=enable_thinking)
        return [1, 2, 3]


class FakeRuntime:
    def __init__(self):
        self.prepared_prefill = SimpleNamespace(as_dict=lambda: {"prefill_ms": 1.5})
        self.output_reader = SimpleNamespace(progress_records=[])
        # Part of the runtime contract (every real runtime publishes one), so the fake has it
        # too rather than the serve path reaching for it defensively.  ``abandoned_chain_positions``
        # deliberately is not here: that one is Naive N seq1's, and context.py reads it with a
        # default precisely because a runtime without a device chain has nothing to report.
        self.graph_contract_evidence = MappingProxyType({"schema": "fake_runtime_v1"})


class FakeSession:
    def __init__(self, script, max_decode_tokens):
        self.script, self.max_decode_tokens, self.i = list(script), max_decode_tokens, 0
        self.runtime, self.finished, self.closed = FakeRuntime(), False, False

    def step(self):
        tok = self.script[self.i] if self.i < len(self.script) else 4
        self.i += 1
        self.runtime.output_reader.progress_records.append({"parent_start_ns": 0, "parent_complete_ns": 1_000_000})
        return tok

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, script=(5, 6, 7, EOS), max_decode_tokens=16, limit=20):
        self.config = SimpleNamespace(max_decode_tokens=max_decode_tokens,
                                      model_spec=SimpleNamespace(stop_token_ids=(EOS,)))
        self._backend = SimpleNamespace(seq1_max_cur_pos=limit)
        self.script, self.sessions, self._request_generation = script, [], 0
        self.reserved: list[int] = []
        self._backend.min_prompt_tokens = 1

    @property
    def request_generation(self):
        return self._request_generation

    def open_session(self, tokens, *, n_past=0, max_decode_tokens=None):
        reserved = self.config.max_decode_tokens if max_decode_tokens is None else min(
            int(max_decode_tokens), self.config.max_decode_tokens)
        assert reserved > 0
        self.reserved.append(reserved)
        if len(tokens) + reserved > self._backend.seq1_max_cur_pos + 2:
            raise RuntimeContractError(f"seq1 request would decode at cur_pos 99, {CONTEXT_EXCEEDED_MARK} (x)")
        assert 0 <= n_past <= len(tokens) - 1
        self._request_generation += 1
        s = FakeSession(self.script, reserved)
        s.n_past = n_past
        self.sessions.append(s)
        return s


def _ctx(**kw):
    return ServerContext(FakeEngine(**kw), FakeFormatter())


def test_normal_completion_streams_then_final():
    ctx = _ctx()
    items = list(ctx.handle_completion(ServerTask(id=1, messages=[{"role": "user", "content": "x"}])))
    assert items[0] == OpenedResult(1, 3, 0)
    partials, final = items[1:-1], items[-1]
    assert all(isinstance(p, PartialResult) for p in partials) and isinstance(final, FinalResult)
    assert final.tokens == (5, 6, 7, EOS) and final.stop_reason == "eos" and final.text == "efg"
    assert "".join(p.text for p in partials) == final.text
    t = final.timings
    assert t.prompt_n == 3 and t.predicted_n == 4 and t.prompt_ms == 1.5 and t.capture_ms > 0
    assert t.predicted_ms > 0 and t.decode_tok_s_device == pytest.approx(1000.0)   # 4 replays x 1 ms
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0
    assert ctx.engine.sessions[0].finished and ctx.engine.sessions[0].closed


@pytest.mark.parametrize(
    ("intervals", "expected_tok_s"),
    [
        ([(0, 0)], 0.0),  # EOS from prefill: no timed decode tokens
        ([(0, 0)] + [(10, 1_000_010)] * 2 + [(1_000_010, 4_000_010)] * 5, 1750.0),
    ],
)
def test_native_mtp_timing_counts_tokens_over_distinct_block_intervals(
    monkeypatch, intervals, expected_tok_s,
):
    """Seven published tokens in two blocks taking 4 ms must report 1750 tok/s.

    The prefill anchor is an output token, but contributes neither a decode token nor
    a decode interval. Unequal block sizes expose counting an interval once per token.
    """
    script = [5] * (len(intervals) - 1) + [EOS]
    ctx = _ctx(script=script)
    original_step = FakeSession.step

    def step(session):
        session.mode = DecodeMode.NATIVE_MTP
        token = original_step(session)
        start, complete = intervals[session.i - 1]
        session.runtime.output_reader.progress_records[-1] = {
            "parent_start_ns": start, "parent_complete_ns": complete,
        }
        return token

    monkeypatch.setattr(FakeSession, "step", step)
    final = ctx.complete(ServerTask(id=1, prompt_tokens=[1, 2, 3]))
    assert final.tokens == tuple(script)
    assert final.timings.predicted_n == len(script)
    assert final.timings.decode_tok_s_device == pytest.approx(expected_tok_s)


def test_context_exceeded_wraps_contract_error_and_returns_slot_to_idle():
    ctx = _ctx(limit=20)          # prompt + decode must stay <= 22; a 25-token prompt cannot fit at all
    with pytest.raises(ContextExceeded) as info:
        list(ctx.handle_completion(ServerTask(id=2, prompt_tokens=list(range(1, 26)))))
    assert isinstance(info.value.__cause__, RuntimeContractError) and CONTEXT_EXCEEDED_MARK in str(info.value)
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0
    final = ctx.complete(ServerTask(id=3, prompt_tokens=[1, 2]))
    assert final.tokens == (5, 6, 7, EOS)


def test_cancel_after_partial_results():
    ctx = _ctx(script=[5] * 50, max_decode_tokens=64, limit=200)
    gen = ctx.handle_completion(ServerTask(id=4, prompt_tokens=[1]))
    assert isinstance(next(gen), OpenedResult)
    seen = [next(gen) for _ in range(5)]
    assert all(isinstance(p, PartialResult) for p in seen)
    gen.close()
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0
    assert ctx.engine.sessions[0].finished and ctx.engine.sessions[0].closed
    assert ctx.transcript[-1]["stop_reason"] == "cancel"


def test_n_predict_truncation_and_rejection():
    ctx = _ctx(script=[5] * 50, max_decode_tokens=16, limit=200)
    final = ctx.complete(ServerTask(id=5, prompt_tokens=[1], params=TaskParams(n_predict=8)))
    assert final.timings.predicted_n == 8 and final.stop_reason == "length" and final.truncated
    with pytest.raises(RequestRejected):
        ctx.complete(ServerTask(id=6, prompt_tokens=[1], params=TaskParams(n_predict=17)))
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0


def test_unbounded_n_predict_hits_engine_limit_as_length():
    ctx = _ctx(script=[5] * 50, max_decode_tokens=4, limit=200)
    final = ctx.complete(ServerTask(id=7, prompt_tokens=[1]))
    assert final.tokens == (5, 5, 5, 5) and final.stop_reason == "length"


def test_cache_prompt_reuses_common_prefix_and_recomputes_last_token():
    ctx = _ctx(script=[5, 6, EOS], max_decode_tokens=8, limit=200)
    cached = TaskParams(cache_prompt=True)
    f1 = ctx.complete(ServerTask(id=1, prompt_tokens=[1, 2, 3], params=cached))
    assert f1.n_prompt_tokens_cache == 0 and f1.timings.prompt_n == 3
    assert ctx.slot.tokens == [1, 2, 3, 5, 6, EOS]
    # next turn extends the previous sequence: everything but the last observed token is reused
    f2 = ctx.complete(ServerTask(id=2, prompt_tokens=[1, 2, 3, 5, 6, EOS, 7, 8], params=cached))
    assert f2.n_prompt_tokens_cache == 5 and f2.timings.prompt_n == 3 and ctx.engine.sessions[-1].n_past == 5
    # identical prompt resent: only the last token is recomputed (llama.cpp slot_save case)
    f3 = ctx.complete(ServerTask(id=3, prompt_tokens=[1, 2, 3, 5, 6, EOS, 7, 8], params=cached))
    assert f3.n_prompt_tokens_cache == 7 and f3.timings.prompt_n == 1 and f3.tokens == f2.tokens
    # divergent prompt: prefix stops at the first differing token
    f4 = ctx.complete(ServerTask(id=4, prompt_tokens=[1, 2, 9, 9], params=cached))
    assert f4.n_prompt_tokens_cache == 2 and f4.timings.prompt_n == 2
    # cache_prompt=False never reuses
    f5 = ctx.complete(ServerTask(id=5, prompt_tokens=[1, 2, 9, 9]))
    assert f5.n_prompt_tokens_cache == 0 and f5.timings.prompt_n == 4


def test_cache_invalidated_when_engine_ran_someone_else():
    ctx = _ctx(script=[5, EOS], max_decode_tokens=8, limit=200)
    cached = TaskParams(cache_prompt=True)
    ctx.complete(ServerTask(id=1, prompt_tokens=[1, 2, 3], params=cached))
    ctx.engine.open_session([4, 4])          # foreign request rewrote the caches
    f = ctx.complete(ServerTask(id=2, prompt_tokens=[1, 2, 3, 5, EOS, 7], params=cached))
    assert f.n_prompt_tokens_cache == 0 and f.timings.prompt_n == 6


def test_cache_dropped_after_error():
    ctx = _ctx(script=[5, EOS], max_decode_tokens=8, limit=20)
    cached = TaskParams(cache_prompt=True)
    ctx.complete(ServerTask(id=1, prompt_tokens=[1, 2, 3], params=cached))
    with pytest.raises(ContextExceeded):
        ctx.complete(ServerTask(id=2, prompt_tokens=list(range(1, 30)), params=cached))
    f = ctx.complete(ServerTask(id=3, prompt_tokens=[1, 2, 3, 5, EOS, 7], params=cached))
    assert f.n_prompt_tokens_cache == 0


# --- Stage A: admission reserves per request, not the engine-wide budget ---------------------


def _long_task(prompt_n: int, n_predict: int) -> ServerTask:
    return ServerTask(id=1, prompt_tokens=list(range(1, prompt_n + 1)),
                      params=TaskParams(n_predict=n_predict))


def test_a01_admits_long_prompt_with_small_max_tokens():
    """A0.1: prompt past the engine-budget line is admitted when the request only asks for 128."""
    ctx = _ctx(script=[5] * 200, max_decode_tokens=16384, limit=24575)
    final = ctx.complete(_long_task(9000, 128))
    assert final.stop_reason == "length" and ctx.engine.reserved == [128]
    assert ctx.slot.state is SlotState.IDLE


def test_a02_full_budget_request_is_clamped_to_the_room_left():
    """A0.2: asking for the whole budget on a prompt that fits decodes with the room that is left
    (max_tokens is a cap, not a reservation), and the task records max_tokens_clamped."""
    ctx = _ctx(script=[5] * 200, max_decode_tokens=16384, limit=24575)
    task = _long_task(9000, 16384)
    final = ctx.complete(task)
    assert ctx.engine.reserved == [24575 + 1 - 9000] and final.stop_reason == "length"
    assert "max_tokens_clamped" in task.ignored
    assert ctx.slot.state is SlotState.IDLE


def test_a02b_prompt_that_cannot_fit_is_still_rejected_with_mark():
    """A prompt longer than the sequence itself is ContextExceeded, clamp or not."""
    ctx = _ctx(script=[5] * 200, max_decode_tokens=16384, limit=24575)
    task = _long_task(24580, 16)
    with pytest.raises(ContextExceeded) as exc:
        list(ctx.handle_completion(task))
    assert CONTEXT_EXCEEDED_MARK in str(exc.value) and "max_tokens_clamped" not in task.ignored
    assert ctx.slot.state is SlotState.IDLE


def test_a03_unbounded_n_predict_unchanged():
    """A0.3: n_predict=-1 keeps reserving the engine budget, exactly as before."""
    ctx = _ctx(script=[5] * 200, max_decode_tokens=16384, limit=24575)
    with pytest.raises(ContextExceeded):
        list(ctx.handle_completion(_long_task(9000, -1)))
    assert ctx.engine.reserved == [16384]
    small = _ctx(max_decode_tokens=16, limit=20)
    assert small.complete(ServerTask(id=1, prompt_tokens=[1], params=TaskParams(n_predict=-1))).stop_reason == "eos"
    assert small.engine.reserved == [16]


def test_a04_session_bound_is_the_request_reservation():
    """A0.4: DecodeSession carries the request's reservation and stops on it as `length`."""
    ctx = _ctx(script=[5] * 200, max_decode_tokens=64, limit=200)
    final = ctx.complete(ServerTask(id=1, prompt_tokens=[1], params=TaskParams(n_predict=5)))
    assert ctx.engine.sessions[-1].max_decode_tokens == 5 and ctx.engine.reserved == [5]
    assert final.stop_reason == "length" and len(final.tokens) == 5


# ---------------------------------------------------- the slot must never be stranded busy


def test_a_close_that_raises_still_frees_the_slot():
    """The step-timeout path: ``session.close()`` in the outer finally is the first finish() of
    the turn, so it is the call that raises -- and a raise from inside a finally used to skip
    ``slot.release()`` and the unfinished counter, leaving every later request rejected with
    "slot busy in state ERROR" for the life of the process.  Worse than the error it replaced,
    and it also swallowed the diagnosis: what the client should see is the step timeout (or the
    wedged-backend explanation), not close()'s own 60 s message on the way out."""
    ctx = _ctx()
    boom = RuntimeContractError("Naive N rank 3 did not go idle within 60 s")

    def explode():
        raise boom

    task = ServerTask(id=41, messages=[{"role": "user", "content": "x"}])
    items = ctx.handle_completion(task)
    next(items)                                     # OpenedResult: the session now exists
    ctx.engine.sessions[0].close = explode
    with pytest.raises(RuntimeContractError) as info:
        list(items)
    assert info.value is boom                       # nothing propagating, so close's error is the answer
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0
    # And the slot is genuinely reusable, which is the whole point.
    assert ctx.complete(ServerTask(id=42, prompt_tokens=[1, 2])).stop_reason == "eos"


def test_a_close_that_raises_does_not_replace_the_real_diagnosis():
    """When something *is* already on its way out, close()'s error must not overwrite it."""
    ctx = _ctx()
    original = RuntimeError("step timed out")

    def explode():
        raise RuntimeContractError("Naive N rank 3 did not go idle within 60 s")

    task = ServerTask(id=43, messages=[{"role": "user", "content": "x"}])
    items = ctx.handle_completion(task)
    next(items)
    session = ctx.engine.sessions[0]
    session.close = explode
    session.step = lambda: (_ for _ in ()).throw(original)
    with pytest.raises(RuntimeError) as info:
        list(items)
    assert info.value is original
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0


def test_an_admission_failure_frees_the_slot_too():
    """The same hole one state earlier: formatting and tokenizing run in STARTED, and anything
    they raise that is not RequestRejected leaves the slot in a state release() rejects.  Then
    release() raises inside the finally and the slot is stuck busy -- the error the request
    would have reported is not even what the caller gets back."""
    ctx = _ctx()
    ctx.formatter.encode = lambda *a, **k: (_ for _ in ()).throw(ValueError("bad chat template"))
    with pytest.raises(ValueError, match="bad chat template"):
        list(ctx.handle_completion(ServerTask(id=44, messages=[{"role": "user", "content": "x"}])))
    assert ctx.slot.state is SlotState.IDLE and ctx.n_unfinished == 0
    ctx.formatter = FakeFormatter()
    assert ctx.complete(ServerTask(id=45, prompt_tokens=[1, 2])).stop_reason == "eos"


def test_unclosed_thinking_segments_are_labelled_think():
    from naivert.serve.context import relabel_segments
    stats = {"answer": {"tokens": 32767, "rounds": 8000}, "total": {"tokens": 32767, "rounds": 8000}}
    assert relabel_segments(stats, True) == {"think": stats["answer"], "total": stats["total"]}
    assert relabel_segments(stats, False) is stats
    split = {"think": {"tokens": 5}, "answer": {"tokens": 7}, "total": {"tokens": 12}}
    assert relabel_segments(split, True) is split
    assert relabel_segments(None, True) is None

