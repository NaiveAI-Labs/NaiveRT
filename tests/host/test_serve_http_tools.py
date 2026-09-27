"""Stage 2 T2.1-T2.7 / R2.1-R2.3: the tool and thinking surface over real loopback HTTP.

Everything here goes through the **official** `anthropic` SDK rather than hand-rolled JSON. That
is the whole point of the stage: the block structure, the `stop_reason` and the retry behaviour
only matter insofar as a real client agrees with them, and a bespoke client would happily agree
with a shape no SDK accepts.

The engine is scripted with *text*, not tokens: `PieceTokenizer` hands out one id per slice of a
reply, so a fake decode can produce `<tool_call>` -- which `chr(96 + t)` never could.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import socket
import threading
import time

import anthropic
import httpx
import pytest
import uvicorn

from naivert.serve.context import ServerContext
from naivert.serve.http import build_app
from naivert.serve.slot import SlotState
from test_serve_context import EOS, FakeEngine

WEATHER = {"name": "get_weather", "description": "Look up the weather.",
           "input_schema": {"type": "object",
                            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                            "required": ["city"]}}
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>Beijing</parameter>\n"
        "<parameter=days>3</parameter>\n</function>\n</tool_call>")
ASK = [dict(role="user", content="weather in Beijing?")]


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("condition timed out")


class PieceTokenizer:
    """Ids are indices into `pieces`, so any text at all can be "generated"."""

    eos_token_id = EOS

    def __init__(self):
        self.pieces: list[str] = []

    def ids_for(self, text, chunk=4):
        ids = [len(self.pieces) + i for i in range(0, -(-len(text) // chunk))]
        self.pieces += [text[i:i + chunk] for i in range(0, len(text), chunk)]
        return ids + [EOS]

    def decode(self, tokens, skip_special_tokens=False, **_):
        return "".join("" if t == EOS else self.pieces[t] for t in tokens)


class ScriptedFormatter:
    def __init__(self, tokenizer):
        self.tokenizer, self.eos_token_id = tokenizer, EOS
        self.calls: list[dict] = []

    def encode(self, messages, *, tools=None, enable_thinking=None):
        # Recorded so a test can prove `tools` / `enable_thinking` actually reached the template
        # rather than being dropped between the request and the prompt (Stage 1's open obligation).
        self.calls.append(dict(messages=[dict(m) for m in messages], tools=tools,
                               enable_thinking=enable_thinking))
        return [1, 2, 3]


class ScriptedEngine(FakeEngine):
    """One scripted reply per admitted request; the last one repeats."""

    def __init__(self, tokenizer, replies):
        super().__init__(script=[EOS], max_decode_tokens=4096, limit=8192)
        self.tokenizer, self.replies, self.opened = tokenizer, list(replies), 0
        self.fail_step_once = False

    def open_session(self, tokens, **kw):
        self.script = self.tokenizer.ids_for(self.replies[min(self.opened, len(self.replies) - 1)])
        self.opened += 1
        session = super().open_session(tokens, **kw)
        step = session.step
        def slowed():
            time.sleep(.002)            # let the event loop interleave, as a real decode would
            if session.i == 1 and self.fail_step_once:
                self.fail_step_once = False
                raise ArithmeticError("injected token failure")
            return step()
        session.step = slowed
        return session

    def close(self):
        pass


@contextmanager
def live(replies, *, configure=None):
    tokenizer = PieceTokenizer()
    made = {}
    def factory():
        engine = ScriptedEngine(tokenizer, replies)
        if configure:
            configure(engine)
        formatter = ScriptedFormatter(tokenizer)
        ctx = ServerContext(engine, formatter)
        made.update(engine=engine, formatter=formatter, ctx=ctx)
        return ctx
    app = build_app(factory, served_model="naive-n0.5", queue_depth=8, ping_seconds=5.)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]))
    thread.start()
    until(lambda: server.started)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    client = httpx.Client(base_url=url, timeout=25)
    until(lambda: client.get("/health").status_code == 200)
    try:
        yield made, client, url
    finally:
        app.state.worker.stop()
        server.should_exit = True
        thread.join(10)
        client.close()
        assert not thread.is_alive()


@contextmanager
def sdk(url, *, max_retries=0, on_response=None):
    """`on_response` fires the instant each HTTP response arrives, *before* the SDK decides to retry.

    That instant is the only place a test can look at the server between a failing attempt and its
    retry, which is exactly where L1 lives: the 500 must not arrive while the slot is still ours.
    """
    hooks = {"response": [on_response]} if on_response else {}
    with httpx.Client(timeout=25, event_hooks=hooks) as http:
        with anthropic.Anthropic(base_url=url, api_key="x", max_retries=max_retries,
                                 timeout=25, http_client=http) as client:
            yield client


def idle(made, client):
    """Both the queue's view and the slot object's own state.

    `/health` reports `Worker.slot`, which the worker resets to IDLE after *every* completion
    regardless of what the `ServerSlot` actually did, so on its own it cannot see a leaked slot.
    """
    until(lambda: client.get("/health").json() == dict(status="ok", slot="IDLE", queued=0), timeout=5)
    assert made["ctx"].slot.state is SlotState.IDLE


@pytest.mark.parametrize('stream', [False, True])
def test_omitted_whitespace_replayed_for_messages_and_count_tokens(stream):
    raw = '<think>\n\t </think>answer'
    with live([raw, 'next']) as (made, client, url), sdk(url) as api:
        opts = dict(model='naive-n0.5', max_tokens=2048,
                    thinking={'type': 'enabled', 'budget_tokens': 1024},
                    metadata={'user_id': 'replay-parent'})
        if stream:
            with api.messages.stream(messages=ASK, **opts) as events:
                first = events.get_final_message()
        else:
            first = api.messages.create(messages=ASK, **opts)
        blocks = [b.model_dump() for b in first.content
                  if b.type not in ('thinking', 'text') or getattr(b, b.type).strip()]
        history = ASK + [dict(role='assistant', content=blocks), dict(role='user', content='continue')]
        count_body = dict(opts, messages=history)
        count_body.pop('max_tokens')
        counted = client.post('/v1/messages/count_tokens', json=count_body)
        assert counted.status_code == 200, counted.text
        assert made['formatter'].calls[-1]['messages'][1]['reasoning_content'] == '\n\t '
        api.messages.create(messages=history, **opts)
        assert made['formatter'].calls[-1]['messages'][1] == dict(
            role='assistant', reasoning_content='\n\t ', content='answer')
        idle(made, client)


# --- T2.1 / T2.2: the two projections, as the SDK sees them -----------------------------------

def test_t21_nonstream_tool_use_message():
    with live([f"On it.{CALL}"]) as (made, client, url), sdk(url) as api:
        message = api.messages.create(model="naive-n0.5", max_tokens=256, messages=ASK, tools=[WEATHER])
        idle(made, client)
    assert [b.type for b in message.content] == ["text", "tool_use"]
    call = message.content[1]
    assert call.name == "get_weather"
    assert isinstance(call.input, dict) and call.input == {"city": "Beijing", "days": 3}
    assert call.id.startswith("toolu_")
    assert message.stop_reason == "tool_use" and message.stop_sequence is None
    # the SDK's own helper for the next turn round-trips this block
    assert message.model_dump()["content"][1]["input"] == {"city": "Beijing", "days": 3}
    # Stage 1's open obligation: the tools really reached the prompt builder.
    assert made["formatter"].calls[-1]["tools"] == [
        {"name": "get_weather", "description": "Look up the weather.",
         "parameters": WEATHER["input_schema"]}]
    assert made["formatter"].calls[-1]["enable_thinking"] is False


def test_t22_streaming_final_message_matches_the_nonstream_one():
    reply = f"On it.{CALL}"
    with live([reply]) as (made, client, url), sdk(url) as api:
        direct = api.messages.create(model="naive-n0.5", max_tokens=256, messages=ASK, tools=[WEATHER])
        with api.messages.stream(model="naive-n0.5", max_tokens=256, messages=ASK,
                                 tools=[WEATHER]) as stream:
            streamed = stream.get_final_message()
        idle(made, client)

    def shape(message):
        content = json.loads(message.model_dump_json())["content"]
        for block in content:
            block.pop("id", None)           # random per message
            # The streaming helper stamps `parsed_output` on text blocks it accumulates; it is a
            # client-side field of that helper, never anything this server puts on the wire.
            block.pop("parsed_output", None)
        return content, message.stop_reason, message.stop_sequence

    assert shape(streamed) == shape(direct)
    assert streamed.usage.output_tokens == direct.usage.output_tokens


def test_t23_streaming_thinking_parses_in_the_sdk():
    """§10 contingency check: a thinking block with no `signature` must still parse.

    It does -- `signature` is optional on an inbound `ThinkingBlock`, so the SDK accumulates our
    unsigned block without complaint and `get_final_message()` returns it. No contingency needed.
    """
    reply = "<think>let me look</think>Sunny."
    with live([reply]) as (made, client, url), sdk(url) as api:
        with api.messages.stream(model="naive-n0.5", max_tokens=2048, messages=ASK,
                                 thinking={"type": "enabled", "budget_tokens": 1024}) as stream:
            events = [event.type for event in stream]
            message = stream.get_final_message()
        idle(made, client)
    assert [b.type for b in message.content] == ["thinking", "text"]
    assert message.content[0].thinking == "let me look"
    assert message.content[0].signature in (None, "")
    assert message.content[1].text == "Sunny."
    assert "content_block_delta" in events and "message_stop" in events
    assert made["formatter"].calls[-1]["enable_thinking"] is True


def test_t23_thinking_tokens_ride_the_message_delta_even_though_the_sdk_drops_them():
    """`output_tokens_details` is on the wire but absent from `get_final_message()`.

    The SDK's stream accumulator copies exactly one field out of `message_delta.usage`
    (`output_tokens`, `lib/streaming/_messages.py`), so no per-block usage detail can reach a
    streaming client through it -- ours included. The field is not dropped by this server, and the
    non-streaming path carries it, so a client that wants it reads the raw event or does not stream.
    """
    with live(["<think>let me look</think>Sunny."]) as (made, client, url):
        body = dict(model="naive-n0.5", max_tokens=2048, messages=ASK,
                    thinking={"type": "enabled", "budget_tokens": 1024})
        with client.stream("POST", "/v1/messages", json=dict(body, stream=True)) as response:
            deltas = [json.loads(line[6:]) for line in response.iter_lines()
                      if line.startswith("data: ")]
        streamed = next(d for d in deltas if d.get("type") == "message_delta")
        assert streamed["usage"]["output_tokens_details"]["thinking_tokens"] > 0
        whole = client.post("/v1/messages", json=body).json()
        assert whole["usage"]["output_tokens_details"] == streamed["usage"]["output_tokens_details"]
        idle(made, client)


# --- T2.4: the new rejections keep their header ------------------------------------------------

@pytest.mark.parametrize("body,warning", [
    (dict(tools=[dict(WEATHER, strict=True)]), "strict_tools_unsupported"),
    (dict(tools=[WEATHER], tool_choice={"type": "any"}), "tool_choice_unsupported"),
    (dict(tools=[{"name": "f", "input_schema": {"type": "object",
                                                "properties": {"p": {"type": "str"}}}}]),
     "invalid_tool_schema"),
    (dict(thinking={"type": "enabled", "budget_tokens": 512}), "invalid_thinking_budget"),
    (dict(tools=[dict(WEATHER, type="custom")]), "unsupported_tool_type"),
    (dict(tools=[dict(WEATHER, wat=1)]), "unknown_field"),
    (dict(output_config={"format": {"type": "json_schema"}}), "structured_outputs_unsupported"),
    (dict(messages=ASK + [dict(role="assistant", content=[dict(type="thinking", thinking="t",
                                                               signature="sig")]),
                          dict(role="user", content="q")]), "thinking_signature_unsupported"),
    (dict(service_tier="auto"), "feature_unsupported"),
    (dict(messages=[dict(role="user", content=[dict(type="text", text="x", citations=[])])]),
     "citations_unsupported"),
    (dict(messages=ASK + [dict(role="assistant", content=[dict(type="tool_use", id="toolu_1",
                               name="get_weather", input={}, caller={"type": "direct"})]),
                          dict(role="user", content="q")]), "tool_caller_unsupported"),
])
def test_t24_rejections_carry_warning_type(body, warning):
    with live(["hi"]) as (made, client, url), sdk(url) as api:
        with pytest.raises(anthropic.BadRequestError) as caught:
            api.messages.create(**(dict(model="naive-n0.5", max_tokens=2048, messages=ASK) | body))
        idle(made, client)
    assert caught.value.response.headers["x-naivert-warning"] == warning
    assert caught.value.body["error"]["warning_type"] == warning


def test_t24_an_unsigned_thinking_block_round_trips():
    """The other half of the signature rule: our own block, sent straight back, is accepted."""
    with live(["<think>plan</think>Sure."]) as (made, client, url), sdk(url) as api:
        first = api.messages.create(model="naive-n0.5", max_tokens=2048, messages=ASK,
                                    thinking={"type": "enabled", "budget_tokens": 1024})
        second = api.messages.create(
            model="naive-n0.5", max_tokens=2048,
            messages=ASK + [{"role": "assistant", "content": [b.model_dump() for b in first.content]},
                            {"role": "user", "content": "and tomorrow?"}],
            thinking={"type": "enabled", "budget_tokens": 1024})
        idle(made, client)
    assert second.content[-1].type == "text"
    assert "plan" in made["formatter"].calls[-1]["messages"][1]["reasoning_content"]


def test_t25_the_parse_error_500_also_carries_its_warning_header():
    with live([BAD]) as (made, client, url):
        response = client.post("/v1/messages", json=dict(model="naive-n0.5", max_tokens=256,
                                                         messages=ASK, tools=[WEATHER]))
        assert response.status_code == 500
        assert response.headers["x-naivert-warning"] == "tool_call_parse_error"
        assert response.json()["error"]["warning_type"] == "tool_call_parse_error"
        idle(made, client)


# --- T2.5 / R2.2: a parse error mid-stream ------------------------------------------------------

BAD = "Working.<tool_call>\n<function=nope>\n</function>\n</tool_call>"


def test_t25_mid_stream_parse_error_becomes_an_sse_error_and_frees_the_slot():
    with live([BAD]) as (made, client, url):
        with client.stream("POST", "/v1/messages",
                           json=dict(model="naive-n0.5", max_tokens=256, messages=ASK, stream=True,
                                     tools=[WEATHER])) as response:
            assert response.status_code == 200          # the stream had already begun
            names, bodies = [], []
            for line in response.iter_lines():
                if line.startswith("event: "):
                    names.append(line[7:])
                elif line.startswith("data: "):
                    bodies.append(json.loads(line[6:]))
        assert names[-1] == "error", names
        assert bodies[-1]["error"]["warning_type"] == "tool_call_parse_error"
        # the text that arrived before the malformed call is not retracted, but no block claims the
        # call succeeded: the stream ends on `error`, never `message_stop`
        assert "message_stop" not in names
        idle(made, client)


def test_r22_the_sdk_does_not_retry_an_error_that_arrives_mid_stream():
    """And the slot is free the moment the client learns, not merely some time afterwards."""
    with live([BAD]) as (made, client, url), sdk(url, max_retries=2) as api:
        with pytest.raises(anthropic.APIError):
            with api.messages.stream(model="naive-n0.5", max_tokens=256, messages=ASK,
                                     tools=[WEATHER]) as stream:
                stream.get_final_message()
        # Checked before any polling: a stream error is delivered only after the worker has closed
        # its generator, so the slot object is already IDLE when the exception surfaces (L1). This
        # is the assertion, not `idle()`, which cannot see a leaked slot on its own.
        assert made["ctx"].slot.state is SlotState.IDLE
        assert made["engine"].opened == 1, "the stream was retried, so the error moved a slot"
        idle(made, client)


# --- R2.1: L1 under the SDK's default retry policy ----------------------------------------------

@pytest.mark.parametrize("mode", ["parse_error", "runtime_error"])
def test_r21_a_retryable_500_never_arrives_while_the_slot_is_still_held(mode):
    """L1's whole reason for existing, observed at the only instant that can show it.

    `max_retries=2` is the SDK's default -- what an unconfigured client does. The response hook
    runs when each attempt's response lands, *before* the SDK decides to retry, so it can see the
    server as it was when the 500 was delivered. Two things must hold there: the failing attempt
    is a 500, and the slot is **already** back to IDLE. Polling after the fact cannot distinguish
    that from a slot released half a second late, which is precisely the inversion L1 forbids.
    """
    replies = [BAD, f"On it.{CALL}"] if mode == "parse_error" else [f"On it.{CALL}"]
    configure = None if mode == "parse_error" else (
        lambda engine: setattr(engine, "fail_step_once", True))
    seen = []
    with live(replies, configure=configure) as (made, client, url):
        def watch(response):
            seen.append((response.status_code, made["ctx"].slot.state))
        with sdk(url, max_retries=2, on_response=watch) as api:
            message = api.messages.create(model="naive-n0.5", max_tokens=256, messages=ASK,
                                          tools=[WEATHER])
            assert [status for status, _ in seen] == [500, 200], seen
            assert seen[0][1] is SlotState.IDLE, \
                "the 500 reached the client while the slot was still held (L1 inverted)"
            # and the retry was really admitted -- `opened` only increments past the busy check
            assert made["engine"].opened == 2, "the retry was not admitted"
            idle(made, client)
    assert message.stop_reason == "tool_use"
    assert message.content[-1].input == {"city": "Beijing", "days": 3}


def test_t24_disable_parallel_tool_use_is_a_tool_use_turn_over_http():
    """Closes Stage 1's third handed-down obligation: the `parallel_stop=` passthrough.

    Stage 1 pinned the upgraded stop reason at the assembler; nothing pinned that `http.py` hands
    the injected stop to it, so dropping the argument left the whole suite green while every SDK
    tool loop silently walked past the call.
    """
    reply = f"On it.{CALL} and then some trailing prose the stop string must cut off."
    with live([reply]) as (made, client, url), sdk(url) as api:
        message = api.messages.create(
            model="naive-n0.5", max_tokens=256, messages=ASK, tools=[WEATHER],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True})
        idle(made, client)
    assert message.stop_reason == "tool_use", "the injected stop string swallowed the tool turn"
    assert message.stop_sequence is None            # ours, not the client's
    assert [b.type for b in message.content] == ["text", "tool_use"]
    assert message.content[-1].input == {"city": "Beijing", "days": 3}
    assert "trailing prose" not in message.content[0].text   # the stop really fired


# --- T2.6: a whole tool round trip --------------------------------------------------------------

def test_t26_two_round_tool_trip():
    with live([f"Checking.{CALL}", "It is sunny, 22C."]) as (made, client, url), sdk(url) as api:
        first = api.messages.create(model="naive-n0.5", max_tokens=256, messages=ASK, tools=[WEATHER])
        assert first.stop_reason == "tool_use"
        call = next(b for b in first.content if b.type == "tool_use")
        messages = ASK + [
            {"role": "assistant", "content": [b.model_dump() for b in first.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id,
                                          "content": "sunny, 22C"}]}]
        second = api.messages.create(model="naive-n0.5", max_tokens=256, messages=messages,
                                     tools=[WEATHER])
        idle(made, client)
    assert second.stop_reason == "end_turn"
    assert [b.type for b in second.content] == ["text"]
    assert second.content[0].text == "It is sunny, 22C."
    # the second prompt carried the call and its result, named after the tool it answers
    rendered = made["formatter"].calls[-1]["messages"]
    assert rendered[-1] == {"role": "tool", "content": "sunny, 22C", "name": "get_weather"}
    # rendered as the model wrote it (text, then the call with JSON literals), not via tool_calls:
    # the replayed turn is byte for byte the generated text
    assert "tool_calls" not in rendered[-2]
    assert rendered[-2]["content"] == f"Checking.{CALL}"


def test_t2z_replayed_tool_calls_render_as_the_model_wrote_them():
    """Two calls in one turn are joined by one newline, booleans and objects are JSON literals, and a
    text block the model wrote before the calls keeps its own newlines (Claude Code drops the
    whitespace-only text block, so an empty text becomes the model's blank line)."""
    from naivert.serve.messages_api import MessagesRequest, validate_request
    from naivert.serve.naive_n0p5_syntax import render_tool_calls
    BASE = dict(model="naive-n0.5", max_tokens=16, messages=[dict(role="user", content="x")])
    calls = [{"name": "Read", "arguments": {"file_path": "/tmp/a.txt", "limit": 5, "verbose": True}},
             {"name": "Grep", "arguments": {"pattern": "x", "opts": {"i": True}, "paths": ["a", "b"]}}]
    assert render_tool_calls(calls) == (
        "<tool_call>\n<function=Read>\n<parameter=file_path>/tmp/a.txt</parameter>\n<parameter=limit>5</parameter>\n"
        "<parameter=verbose>true</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=Grep>\n<parameter=pattern>x</parameter>\n<parameter=opts>{\"i\": true}</parameter>\n"
        "<parameter=paths>[\"a\", \"b\"]</parameter>\n</function>\n</tool_call>")
    tools = [{"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]
    history = [dict(role="user", content="go"),
               dict(role="assistant", content=[dict(type="thinking", thinking="\nplan\n", signature=""),
                                               dict(type="tool_use", id="t1", name="Read", input={"file_path": "/tmp/a.txt"})]),
               dict(role="user", content=[dict(type="tool_result", tool_use_id="t1", content="hi")])]
    task = validate_request(MessagesRequest.model_validate(dict(BASE, messages=history, tools=tools,
                                                                thinking={"type": "enabled", "budget_tokens": 1024},
                                                                max_tokens=2048)),
                            max_decode_tokens=4096, served_model=BASE["model"])
    turn = task.messages[1]
    assert turn["reasoning_content"] == "\nplan\n"
    assert turn["content"] == "\n\n<tool_call>\n<function=Read>\n<parameter=file_path>/tmp/a.txt</parameter>\n</function>\n</tool_call>"
    history[1]["content"].insert(1, dict(type="text", text="\n\nok\n\n"))
    task = validate_request(MessagesRequest.model_validate(dict(BASE, messages=history, tools=tools,
                                                                thinking={"type": "enabled", "budget_tokens": 1024},
                                                                max_tokens=2048)),
                            max_decode_tokens=4096, served_model=BASE["model"])
    assert task.messages[1]["content"].startswith("\n\nok\n\n<tool_call>\n<function=Read>")


def test_t26_a_tool_result_for_an_unknown_call_is_refused():
    with live(["hi"]) as (made, client, url), sdk(url) as api:
        with pytest.raises(anthropic.BadRequestError) as caught:
            api.messages.create(model="naive-n0.5", max_tokens=256, messages=ASK + [
                {"role": "assistant", "content": "sure"},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_nope",
                                              "content": "x"}]}], tools=[WEATHER])
        idle(made, client)
    assert caught.value.body["error"]["warning_type"] == "orphan_tool_result"
    assert caught.value.response.headers["x-naivert-warning"] == "orphan_tool_result"


# --- R2.3: the example client uses the default retry policy --------------------------------------

def test_r23_example_client_retries_by_default():
    """Check the constructed client, not the source text: `max_retries = 0` would slip past a grep."""
    from pathlib import Path
    import ast
    source = (Path(__file__).resolve().parents[2] / "examples/chat_http.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "Anthropic"]
    assert calls, "the example no longer constructs an anthropic client"
    for call in calls:
        assert not any(kw.arg == "max_retries" for kw in call.keywords), \
            "the example must leave the SDK's retry policy at its default"
    # and the rationale is documented where a reader would look for it, not only in a code comment
    docs = (Path(__file__).resolve().parents[2] / "docs/messages_http.md").read_text()
    assert "retry policy at its default" in docs


def test_t2x_anyof_enum_and_untyped_properties():
    """Properties typed through anyOf/oneOf unions or enums coerce like explicit unions; a property with no
    typing key at all is served as the raw string and reported in x-naivert-ignored (what Claude Code's
    own tool schemas need)."""
    tool = {"name": "artifact", "description": "d",
            "input_schema": {"type": "object", "properties": {
                "files": {"anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "object"},
                                    {"type": "null"}]},
                "mode": {"enum": ["fast", "slow"]},
                "count": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                "args": {"description": "any value, verbatim"}}}}
    call = ("<tool_call>\n<function=artifact>\n<parameter=files>[{\"a\": 1}]</parameter>\n"
            "<parameter=mode>fast</parameter>\n<parameter=count>7</parameter>\n"
            "<parameter=args>{\"x\": [1, 2]}</parameter>\n</function>\n</tool_call>")
    with live([f"Sure.{call}"]) as (made, client, url):
        response = client.post("/v1/messages", json=dict(model="naive-n0.5", max_tokens=256, messages=ASK, tools=[tool]))
        idle(made, client)
    assert response.status_code == 200, response.text
    block = next(b for b in response.json()["content"] if b["type"] == "tool_use")
    assert block["input"] == {"files": [{"a": 1}], "mode": "fast", "count": 7, "args": '{"x": [1, 2]}'}
    assert response.headers["x-naivert-ignored"] == "untyped_tool_property"
    with live(["hi"]) as (made, client, url):
        bad_branch = dict(tool, input_schema={"type": "object", "properties": {"p": {"anyOf": [{"type": "str"}]}}})
        assert client.post("/v1/messages", json=dict(model="naive-n0.5", max_tokens=8, messages=ASK,
                                                     tools=[bad_branch])).json()["error"]["warning_type"] == "invalid_tool_schema"


def test_t2y_cache_control_on_replayed_tool_blocks_is_ignored():
    """Claude Code marks the assistant's tool_use block (and results) with cache_control when it replays
    the history; the marker is dropped from every block kind and reported."""
    history = ASK + [dict(role="assistant", content=[
                         dict(type="thinking", thinking="t", cache_control={"type": "ephemeral"}),
                         dict(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Beijing"},
                              cache_control={"type": "ephemeral", "ttl": "1h"})]),
                     dict(role="user", content=[dict(type="tool_result", tool_use_id="toolu_1", content="sunny",
                                                     cache_control={"type": "ephemeral"})])]
    with live(["Sunny."]) as (made, client, url):
        response = client.post("/v1/messages", json=dict(model="naive-n0.5", max_tokens=2048, messages=history,
                                                         tools=[WEATHER], thinking={"type": "enabled", "budget_tokens": 1024}))
        idle(made, client)
    assert response.status_code == 200, response.text
    assert response.headers["x-naivert-ignored"] == "cache_control"
    rendered = made["formatter"].calls[-1]["messages"]
    assert all("cache_control" not in json.dumps(m) for m in rendered)
