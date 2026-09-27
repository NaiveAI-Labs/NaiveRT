"""G1: real loopback HTTP tests. Fake engine never imports torch or touches CUDA."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time

import anthropic
import httpx
import pytest
import uvicorn

from naivert.serve.context import ServerContext
from naivert.serve.http import build_app, run_app
from test_serve_context import FakeEngine, FakeFormatter

BASE = dict(model="naive-n0.5", max_tokens=16, messages=[dict(role="user", content="x")])


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("condition timed out")


class Engine(FakeEngine):
    def __init__(self):
        super().__init__(script=[5]*40, max_decode_tokens=64, limit=200)
        self.created_thread = threading.get_ident()
        self.closed_threads = []
        self.calls = []
        self.intervals = []
        self.open_entered = threading.Event()
        self.open_gate = threading.Event()
        self.open_gate.set()
        self.first_delay = 0
        self.step_delay = .01
        self.fail_open = False
        self.fail_step = False

    def open_session(self, tokens, **kw):
        assert threading.get_ident() == self.created_thread
        self.calls.append(list(tokens))
        start = time.monotonic()
        self.open_entered.set()
        assert self.open_gate.wait(15)
        if self.fail_open:
            self.fail_open = False
            raise ValueError("injected admission failure")
        session = super().open_session(tokens, **kw)
        step, close = session.step, session.close
        def delayed_step():
            assert threading.get_ident() == self.created_thread
            time.sleep(self.first_delay if session.i == 0 else self.step_delay)
            if session.i == 1 and self.fail_step:
                self.fail_step = False
                raise ArithmeticError("injected token failure")
            return step()
        def recorded_close():
            assert threading.get_ident() == self.created_thread
            self.intervals.append((start, time.monotonic()))
            close()
        session.step, session.close = delayed_step, recorded_close
        return session

    def close(self):
        self.closed_threads.append(threading.get_ident())


class Formatter(FakeFormatter):
    def encode(self, messages, *, tools=None, enable_thinking=None):
        if messages[-1]["content"] == "overflow":
            return [1]*180
        if messages[-1]["content"] == "overflow-prompt":
            return [1]*240          # longer than the 201-row sequence: no max_tokens clamp can help
        # tools and thinking really do change the rendered prompt, as they do in the template, so a
        # count that ignores them cannot silently agree with the generated one (A0.5).
        return [1,2,ord(messages[-1]["content"][0])] + [7]*(10*len(tools or [])) + [8]*bool(enable_thinking)


@contextmanager
def live(*, depth=8, prepare_gate=None, ping_seconds=5., configure=None, cors_origins=None, cors_credentials=False):
    engines = []
    def factory():
        engine = Engine()
        engines.append(engine)
        if configure:
            configure(engine)
        if prepare_gate:
            assert prepare_gate.wait(15)
        return ServerContext(engine, Formatter())
    app = build_app(factory, served_model="naive-n0.5", queue_depth=depth, ping_seconds=ping_seconds,
                    cors_origins=cors_origins, cors_credentials=cors_credentials)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]))
    thread.start()
    until(lambda: server.started)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    client = httpx.Client(base_url=url, timeout=25)
    if not prepare_gate:
        until(lambda: client.get("/health").status_code == 200)
    try:
        yield app, engines, client, url
    finally:
        for engine in engines:
            engine.open_gate.set()
        if prepare_gate:
            prepare_gate.set()
        app.state.worker.stop()
        server.should_exit = True
        thread.join(10)
        client.close()
        assert not thread.is_alive()
        assert engines[0].closed_threads == [engines[0].created_thread]


def post(client, **kw):
    return client.post("/v1/messages", json=BASE | kw)


def event_records(lines):
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            yield event, json.loads(line[6:]), time.monotonic()


def assert_warning(response, status, warning):
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-naivert-warning"] == warning
    assert response.json()["error"]["warning_type"] == warning


def test_v11_v12_nonstream_and_stream():
    with live() as (_, engines, client, _):
        response = post(client)
        assert response.status_code == 200 and "x-naivert-warning" not in response.headers
        body = response.json()
        assert body["content"] == [{"type":"text","text":"e"*16}]
        usage = body["usage"]                          # Anthropic semantics: the three input fields sum to the prompt
        assert usage["input_tokens"] == 0 and usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"] == 3
        assert usage["output_tokens"] == 16
        assert body["stop_reason"] == "max_tokens" and body["stop_sequence"] is None
        assert body["naivert"]["timings"]["predicted_n"] == 16
        with client.stream("POST", "/v1/messages", json=BASE | {"stream":True}) as response:
            events = list(event_records(response.iter_lines()))
        assert [e for e,d,t in events] == ["message_start", "content_block_start"] + ["content_block_delta"]*16 + [
            "content_block_stop", "message_delta", "message_stop"]
        assert "".join(d["delta"]["text"] for e,d,t in events if e == "content_block_delta") == body["content"][0]["text"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "naive-n0.5"


@pytest.mark.parametrize("stream", [False, True])
def test_v13_rejections(stream):
    with live() as (_, engines, client, _):
        for change,status,warning in [({"max_tokens":65},400,"max_tokens_exceeds_capture"),
            ({"top_k":5},400,"sampling_unsupported"), ({"model":"other"},404,"unknown_model")]:
            assert_warning(post(client, stream=stream, **change), status, warning)
        assert not engines[0].calls
        assert client.get("/health").json() == dict(status="ok",slot="IDLE",queued=0)
        assert_warning(client.post("/v1/messages",content="{"),400,"schema")
        assert_warning(post(client, max_tokens=True),400,"schema")


def test_v14_serialization():
    with live() as (_, engines, client, _), ThreadPoolExecutor(3) as pool:
        replies = list(pool.map(lambda i: post(client, messages=[dict(role="user",content=str(i))]), range(3)))
        assert all(r.status_code == 200 and r.json()["content"][0]["text"] == "e"*16 for r in replies)
        intervals = engines[0].intervals
        assert len(intervals) == 3
        assert all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:]))


def test_v15_queue_full():
    with live(depth=1) as (app, engines, client, _), ThreadPoolExecutor(2) as pool:
        engine = engines[0]
        engine.open_gate.clear()
        a = pool.submit(post, client)
        assert engine.open_entered.wait(3)
        b = pool.submit(post, client)
        until(lambda: app.state.worker.queue.qsize() == 1)
        assert_warning(post(client),529,"queue_full")
        engine.open_gate.set()
        assert a.result().status_code == b.result().status_code == 200


def test_v16_cancel_active():
    with live() as (_, engines, client, _):
        with client.stream("POST", "/v1/messages", json=BASE | {"stream":True,"max_tokens":64}) as response:
            count = 0
            for event,data,_ in event_records(response.iter_lines()):
                count += event == "content_block_delta"
                if count == 3:
                    break
        until(lambda: client.get("/health").json().get("slot") == "IDLE", timeout=2)
        session = engines[0].sessions[0]
        assert session.finished and session.closed and session.i < 64
        assert post(client).status_code == 200


@pytest.mark.parametrize("stream",[False,True])
@pytest.mark.parametrize("where",["open","step"])
def test_v17_runtime_error_recovery(stream, where):
    with live() as (_, engines, client, _):
        setattr(engines[0], f"fail_{where}", True)
        response = post(client, stream=stream)
        if stream and where == "step":
            assert response.status_code == 200
            events = list(event_records(response.text.splitlines()))
            assert events[-1][0] == "error" and events[-1][1]["error"]["warning_type"] == "runtime_error"
            assert "ArithmeticError" in events[-1][1]["error"]["message"]
            assert not any(e == "message_stop" for e,d,t in events)
        else:
            assert_warning(response,500,"runtime_error")
            assert ("ValueError" if where == "open" else "ArithmeticError") in response.json()["error"]["message"]
        assert post(client).status_code == 200


def raw_request(url, body):
    parsed = httpx.URL(url)
    sock = socket.create_connection((parsed.host, parsed.port))
    payload = json.dumps(body).encode()
    sock.sendall(f"POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n".encode()+payload)
    return sock


def test_v18_headers_wait_for_admission():
    with live() as (_, engines, client, url):
        engine = engines[0]
        engine.open_gate.clear()
        # 240 prompt tokens cannot fit the 201-row sequence at all (a shorter prompt would only have
        # its max_tokens clamped to the room left, Stage A).
        sock = raw_request(url, BASE | {"stream":True,"max_tokens":64,
                                        "messages":[dict(role="user",content="overflow-prompt")]})
        try:
            assert engine.open_entered.wait(3)
            assert not select.select([sock],[],[],.3)[0], "headers or SSE leaked before admission"
            engine.open_gate.set()
            sock.settimeout(5)
            data = sock.recv(8192)
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(8192)
                assert chunk, "response ended before headers"
                data += chunk
            headers, body = data.split(b"\r\n\r\n", 1)
            length = int(next(line.split(b":", 1)[1] for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:")))
            while len(body) < length:
                chunk = sock.recv(8192)
                assert chunk, "response body was truncated"
                body += chunk
            data = headers + b"\r\n\r\n" + body
            assert b"400 Bad Request" in data and b"event:" not in data
            assert b"240" in data and b"138" in data and b"max_seq_len=201" in data
            assert b"max_tokens=64" in data
        finally:
            sock.close()
        assert post(client).status_code == 200


def test_v19_keepalive_comments_before_message_start_while_the_session_opens():
    """A cold prefill can take minutes: after one ping interval without an open session the stream
    starts with SSE comment lines (ignored by event parsers) so a client's first-byte timeout is
    satisfied, and the events follow unchanged once the session opens."""
    with live(ping_seconds=0.2) as (_, engines, client, _):
        engine = engines[0]
        engine.open_gate.clear()
        t0 = time.monotonic()
        with client.stream("POST", "/v1/messages", json=BASE | {"stream": True}) as response:
            assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
            lines = iter(response.iter_lines())
            first = next(line for line in lines if line)
            assert first == ": keepalive" and time.monotonic() - t0 < 3        # before the session opened
            time.sleep(0.5)
            engine.open_gate.set()
            rest = list(lines)
        assert len([line for line in rest if line.startswith(":")]) >= 1       # kept coming while held
        names = [e for e, d, t in event_records([first] + rest)]
        assert names[:2] == ["message_start", "content_block_start"] and names[-1] == "message_stop"
        assert client.get("/health").json() == dict(status="ok", slot="IDLE", queued=0)


def test_v18_ping_timing_real_twelve_seconds():
    with live(configure=lambda e: setattr(e,"first_delay",12.)) as (_, engines, client, _):
        with client.stream("POST","/v1/messages",json=BASE | {"stream":True}) as response:
            events = list(event_records(response.iter_lines()))
        names = [e for e,d,t in events]
        first = names.index("content_block_delta")
        pings = [t for e,d,t in events[:first] if e == "ping"]
        # The first block's kind is only known once text arrives (it may be thinking, text or a
        # tool call), so content_block_start now follows the pings instead of preceding them.
        assert names[0] == "message_start" and names[first - 1] == "content_block_start"
        assert len(pings) >= 2
        assert pings[0] - events[0][2] >= 4.8 and all(4.8 <= b-a <= 5.5 for a,b in zip(pings,pings[1:]))
        assert "ping" not in names[first:]


def test_v19_official_sdk():
    with live() as (_, engines, _, url), anthropic.Anthropic(base_url=url,api_key="x",max_retries=0) as sdk:
        msg = sdk.messages.create(**BASE)
        assert msg.content[0].text == "e"*16 and "naivert" in msg.model_extra
        with sdk.messages.stream(**BASE) as stream:
            assert "".join(stream.text_stream) == msg.content[0].text
            assert stream.get_final_message().usage.output_tokens == msg.usage.output_tokens
        with pytest.raises(anthropic.BadRequestError) as error:
            sdk.messages.create(**(BASE | {"top_k":5}))
        assert error.value.body["error"]["warning_type"] == "sampling_unsupported"


def test_v110_loading():
    gate = threading.Event()
    with live(prepare_gate=gate) as (app, engines, client, _):
        response = client.get("/health")
        assert response.status_code == 503 and response.json() == {"status":"loading"}
        assert_warning(post(client),503,"loading")
        assert not engines[0].calls and app.state.worker.queue.qsize() == 0
        gate.set()
        until(lambda: client.get("/health").status_code == 200)
        assert post(client).status_code == 200


@pytest.mark.parametrize("stream",[False,True])
def test_v111_cancel_queued(stream):
    with live() as (app, engines, client, url), ThreadPoolExecutor(1) as pool:
        engine = engines[0]
        engine.open_gate.clear()
        a = pool.submit(post, client, messages=[dict(role="user",content="A")])
        assert engine.open_entered.wait(3)
        sock = raw_request(url, BASE | {"stream":stream,"messages":[dict(role="user",content="B")]})
        until(lambda: app.state.worker.queue.qsize() == 1)
        with app.state.worker.queue.mutex:
            queued = app.state.worker.queue.queue[0]
        assert queued.task.messages[-1]["content"] == "B"
        sock.close()
        assert queued.cancel_event.wait(3)
        engine.open_gate.set()
        assert a.result().status_code == 200
        until(lambda: app.state.worker.queue.qsize() == 0)
        assert post(client,messages=[dict(role="user",content="C")]).status_code == 200
        assert [tokens[-1] for tokens in engine.calls] == [ord("A"), ord("C")]


def test_v112_shutdown_active_and_queued():
    with live() as (app, engines, client, _), ThreadPoolExecutor(2) as pool:
        engine = engines[0]
        engine.open_gate.clear()
        a = pool.submit(post,client)
        assert engine.open_entered.wait(3)
        b = pool.submit(post,client)
        until(lambda: app.state.worker.queue.qsize() == 1)
        app.state.worker.stop()
        assert_warning(post(client),503,"shutting_down")
        assert_warning(b.result(),503,"shutting_down")
        assert len(engine.calls) == 1
        engine.open_gate.set()
        assert a.result().status_code == 200
        app.state.worker.thread.join(3)
        assert not app.state.worker.thread.is_alive()
        assert engine.closed_threads == [engine.created_thread]


def test_v112_real_sigterm(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1",0))
    port = sock.getsockname()[1]
    sock.close()
    evidence = tmp_path / "closed.json"
    log = tmp_path / "child.log"
    with log.open("w") as output:
        child = subprocess.Popen([sys.executable,__file__,str(port),str(evidence)],stdout=output,stderr=subprocess.STDOUT,
                                 env=os.environ | {"PYTHONPATH":str(Path(__file__).resolve().parents[2])})
        try:
            def ready():
                try:
                    return httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200
                except httpx.ConnectError:
                    return False
            until(ready,10)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
                assert post(client).status_code == 200
            t = time.monotonic()
            child.terminate()
            assert child.wait(10) == 0 and time.monotonic()-t < 10
            data = json.loads(evidence.read_text())
            assert data["closed"] == [data["created"]]
            assert "Traceback" not in log.read_text()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    def factory():
        engine = Engine()
        close = engine.close
        def record():
            close()
            Path(sys.argv[2]).write_text(json.dumps(dict(created=engine.created_thread,closed=engine.closed_threads)))
        engine.close = record
        return ServerContext(engine, Formatter())
    run_app(build_app(factory,served_model="naive-n0.5",queue_depth=8),host="127.0.0.1",port=int(sys.argv[1]))


@pytest.mark.parametrize("stream", [False, True])
def test_review_dump_failure_terminates_requests_and_recovers(tmp_path, stream):
    with live() as (app, engines, client, _), ThreadPoolExecutor(2) as pool:
        worker, engine = app.state.worker, engines[0]
        worker.dump_tokens = tmp_path / "missing" / "tokens.jsonl"
        engine.open_gate.clear()
        a = pool.submit(post, client, stream=stream)
        assert engine.open_entered.wait(3)
        b = pool.submit(post, client, stream=stream)
        until(lambda: worker.queue.qsize() == 1)
        engine.open_gate.set()
        for response in (a.result(timeout=5), b.result(timeout=5)):
            if stream:
                assert response.status_code == 200
                events = list(event_records(response.text.splitlines()))
                assert events[-1][0] == "error" and "message_stop" not in [e for e,d,t in events]
                assert "FileNotFoundError" in events[-1][1]["error"]["message"]
            else:
                assert_warning(response, 500, "runtime_error")
                assert "FileNotFoundError" in response.json()["error"]["message"]
        worker.dump_tokens.parent.mkdir()
        assert post(client).status_code == 200
        assert len(worker.dump_tokens.read_text().splitlines()) == 1
        assert worker.thread.is_alive() and worker.queue.qsize() == 0


def test_review_fatal_worker_error_notifies_active_and_queue():
    with live() as (app, engines, client, _), ThreadPoolExecutor(2) as pool:
        entered, release = threading.Event(), threading.Event()
        def broken_complete(ctx, request):
            entered.set()
            assert release.wait(3)
            raise RuntimeError("fatal worker injection")
        app.state.worker.complete = broken_complete
        a = pool.submit(post, client)
        assert entered.wait(3)
        b = pool.submit(post, client)
        until(lambda: app.state.worker.queue.qsize() == 1)
        release.set()
        assert_warning(a.result(timeout=3),500,"runtime_error")
        assert_warning(b.result(timeout=3),500,"runtime_error")
        app.state.worker.thread.join(3)
        assert not app.state.worker.thread.is_alive() and app.state.worker.queue.qsize() == 0
        assert_warning(post(client),500,"runtime_error")


def test_review_fatal_failure_published_before_active_backpressure():
    from naivert.serve.http import QueuedRequest, Worker
    from naivert.serve.task import ServerTask
    entered, release = threading.Event(), threading.Event()
    worker = Worker(lambda: ServerContext(Engine(), Formatter()), queue_depth=1)
    def broken_complete(ctx, request):
        for _ in range(request.out_queue.maxsize):
            request.out_queue.put_nowait("buffered output")
        entered.set()
        assert release.wait(3)
        raise RuntimeError("fatal worker with full output")
    worker.complete = broken_complete
    worker.thread.start()
    a = QueuedRequest(ServerTask(1,prompt_tokens=[1]))
    b = QueuedRequest(ServerTask(2,prompt_tokens=[2]))
    try:
        until(lambda: worker.snapshot()["status"] == "ok")
        worker.submit(a)
        assert entered.wait(3)
        worker.submit(b)
        release.set()
        until(lambda: worker.snapshot()["status"] == "error")
        assert b.out_queue.get(timeout=1).warning_type == "runtime_error"
        assert worker.failure is not None and worker.queue.qsize() == 0
        assert a.out_queue.full()  # failed client still applies backpressure
    finally:
        release.set()
        a.cancel_event.set()
        worker.stop()
        worker.thread.join(3)
        assert not worker.thread.is_alive()


def test_api_only_routes_and_stream_timings():
    with live() as (_, engines, client, url):
        for path in ('/', '/ui/config', '/assets/chat.js', '/assets/chat.css', '/assets/timeline.js',
                     '/demo', '/demo/', '/assets/demo/index.html', '/assets/demo/studio.js'):
            assert client.get(path).status_code == 404
        response = post(client,stream=True)
        events = list(event_records(response.text.splitlines()))
        final = next(d for e,d,t in events if e == 'message_delta')
        assert final['naivert']['timings']['predicted_n'] == final['usage']['output_tokens'] == 16
        assert final['naivert']['timings']['predicted_per_second'] > 0
        with anthropic.Anthropic(base_url=url,api_key='x',max_retries=0) as sdk:
            with sdk.messages.stream(**BASE) as stream:
                deltas = [event for event in stream if event.type == 'message_delta']
            assert deltas[0].model_extra['naivert']['timings']['predicted_n'] == 16


def test_studio_cross_origin_stream_and_preflight():
    origin = 'https://studio.example.com'
    with live(cors_origins=[origin]) as (_, _, client, _):
        headers = {'Origin': origin, 'Access-Control-Request-Method': 'POST',
                   'Access-Control-Request-Headers': 'content-type,x-naivert-token-progress,x-session-id,authorization,x-api-key'}
        preflight = client.options('/v1/messages', headers=headers)
        assert preflight.status_code == 200
        assert preflight.headers['access-control-allow-origin'] == origin
        for path in ('/health', '/v1/models'):
            response = client.get(path, headers={'Origin': origin})
            assert response.headers['access-control-allow-origin'] == origin
        response = client.post('/v1/messages', json=BASE | {'stream': True},
                               headers={'Origin': origin, 'x-naivert-token-progress': '1', 'x-session-id': 'studio-cors-test'})
        assert response.headers['access-control-allow-origin'] == origin
        events = list(event_records(response.text.splitlines()))
        assert events[-1][0] == 'message_stop'
        assert any('naivert' in data for kind, data, _ in events if kind == 'content_block_delta')
        rejected = client.options('/v1/messages', headers=headers | {'Origin': 'https://other.example.com'})
        assert rejected.status_code == 400
        assert 'access-control-allow-origin' not in rejected.headers


def test_studio_cookie_cors_requires_explicit_origin():
    for origins in (None, ['*']):
        with pytest.raises(ValueError, match='explicit allowed origins'):
            build_app(lambda: None, served_model='test', queue_depth=1,
                      cors_origins=origins, cors_credentials=True)
    origin = 'https://studio.example.com'
    with live(cors_origins=[origin], cors_credentials=True) as (_, _, client, _):
        response = client.options('/v1/messages', headers={'Origin': origin,
            'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization'})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == origin
        assert response.headers['access-control-allow-credentials'] == 'true'


def test_studio_cors_is_disabled_by_default():
    with live() as (_, _, client, _):
        response = client.get('/health', headers={'Origin': 'https://studio.example.com'})
        assert 'access-control-allow-origin' not in response.headers


def test_readiness_and_model_discovery_while_loading():
    gate = threading.Event()
    with live(prepare_gate=gate) as (_, _, client, _):
        assert client.get('/health').status_code == 503
        assert client.get('/health').json() == {'status': 'loading'}
        assert client.get('/v1/models').json()['data'][0]['id'] == 'naive-n0.5'


def test_opt_in_stream_token_progress_and_coalescing():
    from naivert.serve.http import coalesce_deltas
    with live() as (_, _, client, _):
        plain = list(event_records(post(client, stream=True).text.splitlines()))
        assert all('naivert' not in data for kind, data, _ in plain if kind == 'content_block_delta')
        response = client.post('/v1/messages', json=BASE | {'stream': True},
                               headers={'x-naivert-token-progress': '1'})
        events = list(event_records(response.text.splitlines()))
        deltas = [data for kind, data, _ in events if kind == 'content_block_delta']
        counts = [data['naivert']['generated_tokens'] for data in deltas]
        assert counts == sorted(counts) and counts[0] == 1 and counts[-1] == 16
        merged = coalesce_deltas([('content_block_delta', data) for data in deltas])
        assert len(merged) == 1
        assert merged[0][1]['delta']['text'] == 'e' * 16
        assert merged[0][1]['naivert']['generated_tokens'] == 16
        assert deltas[0]['naivert']['generated_tokens'] == 1  # inputs are not mutated


# --- Stage A over real HTTP ------------------------------------------------------------------


def count(client, **kw):
    body = {k: v for k, v in (BASE | kw).items() if k not in ("max_tokens", "stream")}
    return client.post("/v1/messages/count_tokens", json=body)


def test_a05_count_tokens_matches_usage_input_tokens():
    with live() as (_, engines, client, _):
        weather = {"name": "get_weather", "input_schema": {"type": "object",
                                                          "properties": {"city": {"type": "string"}}}}
        for kw in [{}, {"system": "sys"},
                   {"messages": [dict(role="user", content="a"), dict(role="assistant", content="b"),
                                 dict(role="user", content="c")]},
                   {"tools": [weather]},                                  # Tool schemas count toward input usage.
                   {"tools": [weather], "tool_choice": {"type": "none"}}]:  # not rendered -> not counted
            counted = count(client, **kw)
            assert counted.status_code == 200, (kw, counted.json())
            generated = post(client, **kw)
            assert generated.status_code == 200, (kw, generated.json())
            usage = generated.json()["usage"]
            # Anthropic semantics: input + cache_read + cache_creation is the prompt (clients sum them)
            assert usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"] \
                == counted.json()["input_tokens"], kw
            assert usage["input_tokens"] == 0, kw           # every prefilled token is parked, so none is uncached
        # and the tools really moved the number, so the agreement above is not vacuous
        assert count(client, tools=[weather]).json()["input_tokens"] > count(client).json()["input_tokens"]
        assert count(client, tools=[weather], tool_choice={"type": "none"}).json()["input_tokens"] == \
            count(client).json()["input_tokens"]
        # thinking cannot be exercised end to end here (budget_tokens >= 1024 > this engine's
        # 16-token capture), but count_tokens has no max_tokens to bound it against, so the
        # passthrough is still checked where it is observable.
        assert count(client, thinking={"type": "enabled", "budget_tokens": 1024}).json()["input_tokens"] > \
            count(client).json()["input_tokens"]


def test_a05b_an_empty_tools_array_is_no_tools():
    """Claude Code's count_tokens calls for subagent transcripts send ``tools: []``; like the Anthropic API,
    that is the same request without tools (a non-list is still a schema error)."""
    with live() as (_, engines, client, _):
        assert count(client, tools=[]).json() == count(client).json()
        assert post(client, tools=[]).status_code == 200
        assert_warning(count(client, tools={"name": "x"}), 400, "invalid_tool_schema")


def test_a06_count_tokens_rejects_exactly_like_messages():
    with live() as (_, engines, client, _):
        for change, status, warning in [({"model": "other"}, 404, "unknown_model"),
                                        ({"top_k": 5}, 400, "sampling_unsupported"),
                                        ({"messages": []}, 400, "invalid_messages"),
                                        ({"frobnicate": 1}, 400, "unknown_field")]:
            assert_warning(count(client, **change), status, warning)
            assert_warning(post(client, **change), status, warning)


def test_a07_count_tokens_does_not_take_the_slot():
    with live() as (_, engines, client, _):
        before = len(engines[0].calls)
        assert count(client).status_code == 200
        assert len(engines[0].calls) == before
        assert client.get("/health").json() == dict(status="ok", slot="IDLE", queued=0)


def test_a07_count_tokens_answers_while_a_generation_is_in_flight():
    """The point of count_tokens is sizing the *next* turn, so it must not queue behind this one."""
    with live() as (_, engines, client, _):
        engine = engines[0]
        engine.open_gate.clear()
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(lambda: post(client))
            assert engine.open_entered.wait(5)
            counted = count(client)
            assert counted.status_code == 200 and counted.json()["input_tokens"] == 3
            assert len(engine.calls) == 1              # still the one in-flight request
            engine.open_gate.set()
            assert running.result(15).status_code == 200
        assert client.get("/health").json() == dict(status="ok", slot="IDLE", queued=0)


def test_a09_a11_unknown_fields_refused_over_http():
    with live() as (_, engines, client, _):
        assert_warning(post(client, frobnicate=1), 400, "unknown_field")
        assert_warning(post(client, output_config={"format": {"type": "json_schema"}}), 400,
                       "structured_outputs_unsupported")
        assert not engines[0].calls


def test_client_hints_are_ignored_and_reported():
    """Prompt-cache breakpoints, adaptive thinking, effort and context edits (what Claude Code sends) are
    served by ignoring them; the response names every dropped hint in x-naivert-ignored."""
    with live() as (_, engines, client, _):
        response = post(client,
                        system=[dict(type="text", text="s", cache_control=dict(type="ephemeral"))],
                        messages=[dict(role="user", content=[dict(type="text", text="hi",
                                                                  cache_control=dict(type="ephemeral"))])],
                        thinking={"type": "adaptive", "display": "omitted"},
                        output_config={"effort": "xhigh"},
                        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
        assert response.status_code == 200, response.text
        assert response.headers["x-naivert-ignored"] == (
            "cache_control,context_management,output_config.effort,thinking.adaptive,thinking.display")
        assert "x-naivert-ignored" not in post(client).headers
        assert_warning(post(client, output_config={"format": {"type": "json_schema"}}), 400,
                       "structured_outputs_unsupported")
        assert_warning(post(client, context_management={"edits": "no"}), 400, "schema")
        assert_warning(post(client, thinking={"type": "adaptive", "wat": 1}), 400, "unknown_field")




def test_system_role_inside_messages_is_rendered_in_place():
    """Claude Code sends an environment block as a `system` message after the user turn and a
    `<total_tokens>` note after every tool result; each stays where it is as a system turn, so the
    prompt prefix of the previous request is unchanged and the parked session resumes."""
    from naivert.serve.messages_api import MessagesRequest, validate_request
    body = dict(BASE, system="top",
                messages=[dict(role="user", content="hi"),
                          dict(role="system", content=[dict(type="text", text="env",
                                                            cache_control=dict(type="ephemeral"))])])
    task = validate_request(MessagesRequest.model_validate(body), max_decode_tokens=64, served_model=BASE["model"])
    assert task.messages == [{"role": "system", "content": "top"}, {"role": "user", "content": "hi"},
                             {"role": "system", "content": "env"}]
    assert task.ignored == ("cache_control",)
    later = dict(BASE, messages=body["messages"] + [dict(role="assistant", content="ok"), dict(role="user", content="more"),
                                                     dict(role="system", content="<total_tokens>5 tokens left</total_tokens>")])
    task2 = validate_request(MessagesRequest.model_validate(later), max_decode_tokens=64, served_model=BASE["model"])
    assert task2.messages[:2] == task.messages[1:] and [m["role"] for m in task2.messages[2:]] == ["assistant", "user", "system"]
    with live() as (_, engines, client, _):
        response = post(client, **{k: v for k, v in body.items() if k not in BASE or k in ("system", "messages")})
        assert response.status_code == 200, response.text
        assert response.headers["x-naivert-ignored"] == "cache_control"
        assert_warning(post(client, messages=[dict(role="system", content="only")]), 400, "invalid_messages")


@pytest.mark.parametrize("stream", [False, True])
def test_queue_timing_excludes_prefill_and_is_stable_in_stream(stream):
    with live() as (app, engines, client, _), ThreadPoolExecutor(2) as pool:
        engine = engines[0]
        engine.open_gate.clear()
        first = pool.submit(post, client)
        assert engine.open_entered.wait(3)

        def queued_response():
            if not stream:
                return post(client).json()["naivert"]["timings"]
            with client.stream("POST", "/v1/messages", json=BASE | {"stream": True}) as response:
                events = list(event_records(response.iter_lines()))
            start = next(data for kind, data, _ in events if kind == "message_start")
            final = next(data for kind, data, _ in events if kind == "message_delta")
            assert start["naivert"]["timings"]["queue_wait_ms"] == final["naivert"]["timings"]["queue_wait_ms"]
            return final["naivert"]["timings"]

        second = pool.submit(queued_response)
        until(lambda: app.state.worker.queue.qsize() == 1)
        # This hold occurs in the first request's admission/prefill, after queue exit.
        held_at = time.monotonic()
        time.sleep(.1)
        hold_ms = (time.monotonic() - held_at) * 1000
        engine.open_gate.set()
        first_timing = first.result().json()["naivert"]["timings"]
        second_timing = second.result()
        assert first_timing["queue_wait_ms"] < second_timing["queue_wait_ms"]
        assert second_timing["queue_wait_ms"] >= hold_ms
        assert second_timing["predicted_n"] == 16
