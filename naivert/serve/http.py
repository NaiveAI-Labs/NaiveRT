"""Single-owner inference worker and text Messages HTTP transport.

The factory, all context/generator operations and engine.close run on the worker.
HTTP reads published host snapshots only. Bounded output queues apply backpressure;
disconnects cancel admission, queue waits, and streaming through the same path.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
import itertools
import json
import logging
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .context import ContextExceeded, RequestRejected
from .messages_api import (ApiError, CountTokensRequest, MessagesRequest, ResponseAssembler, bad,
                           message_id, runtime_error, validate_count_tokens,
                           validate_request, validation_error)
from .replay import AssistantReplay
from .task import FinalResult, OpenedResult, PartialBatch, PartialResult, ServerTask
from .timeline import RequestTimeline, trace_span

log = logging.getLogger(__name__)


def unavailable(state: str) -> ApiError:
    return ApiError(503, "api_error", state, f"server is {state}")


def error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(error.to_body(), status_code=error.status, headers=error.headers())


@dataclass
class QueuedRequest:
    task: ServerTask
    id: str = field(default_factory=message_id)
    out_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=16))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    queued_at_ns: int = 0
    queue_wait_ms: float | None = None
    # Non-streaming requests show nothing until the final body, so their partials skip the
    # per-token queue hand-off entirely (see Worker.complete).  Default True keeps any caller
    # that does not set it on the streaming path, which delivers every token as it is produced.
    stream: bool = True

    def put(self, item) -> bool:
        while not self.cancel_event.is_set():
            try:
                self.out_queue.put(item, timeout=.05)
                return True
            except queue.Full:
                pass
        return False


class Worker:
    def __init__(self, ctx_factory: Callable, *, queue_depth: int, dump_tokens: str | Path | None = None):
        if queue_depth < 1:
            raise ValueError("queue_depth must be positive")
        self.ctx_factory = ctx_factory
        self.queue = queue.Queue(maxsize=queue_depth)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.state = "loading"
        self.slot = "IDLE"
        self.max_decode_tokens = 0
        self.failure: ApiError | None = None
        self.ids = itertools.count(1)
        self.dump_tokens = Path(dump_tokens) if dump_tokens else None
        self.ctx = None
        self.thread = threading.Thread(target=self.run, name="naivert-inference")

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.state, "slot": self.slot, "queued": self.queue.qsize()}

    def submit(self, request: QueuedRequest) -> None:
        with self.lock:
            if self.failure:
                raise self.failure
            if self.state != "ok":
                raise unavailable(self.state)
            request.task.id = next(self.ids)
            try:
                request.queued_at_ns = time.monotonic_ns()
                self.queue.put_nowait(request)
            except queue.Full:
                raise ApiError(529, "overloaded_error", "queue_full", f"queue depth {self.queue.maxsize} reached")

    def count_prompt_tokens(self, task: ServerTask) -> int:
        """A0.7: tokenize only. No slot, no session, no engine call -- the formatter is read-only
        and this runs off the event loop, so it never has to wait for a generation to finish.

        Takes the whole task, not just its messages: `tools` and `enable_thinking` change the
        rendered prompt, so counting without them would under-report by the entire tools section."""
        with self.lock:
            if self.failure:
                raise self.failure
            ctx = self.ctx
            if self.state != "ok" or ctx is None:
                raise unavailable(self.state)
        return len(ctx.prompt_tokens_for(task))

    def stop(self) -> None:
        with self.lock:
            self.state = "shutting_down"
            self.stopping.set()
            self.drain_pending(unavailable("shutting_down"))

    def drain_pending(self, error: ApiError) -> None:
        """Caller holds lock, excluding submit and the worker's queue take."""
        while True:
            try:
                request = self.queue.get_nowait()
            except queue.Empty:
                break
            request.put(error)
            self.queue.task_done()

    def run(self) -> None:
        ctx = None
        request = None
        try:
            ctx = self.ctx_factory()
            with self.lock:
                self.ctx = ctx
                self.max_decode_tokens = ctx.max_decode_tokens
                if not self.stopping.is_set():
                    self.state = "ok"
            log.info("worker ready at %.6f", time.time())
            while not self.stopping.is_set():
                with self.lock:
                    if self.stopping.is_set():
                        break
                    try:
                        request = self.queue.get_nowait()
                    except queue.Empty:
                        request = None
                    if request is not None and not request.cancel_event.is_set():
                        self.slot = "STARTED"
                if request is None:
                    self.stopping.wait(.01)
                    continue
                try:
                    if not request.cancel_event.is_set():
                        self.complete(ctx, request)
                finally:
                    with self.lock:
                        self.slot = "IDLE"
                    self.queue.task_done()
                request = None
        except Exception as exc:
            log.error("worker failed: %s: %s", type(exc).__name__, exc)
            failure = runtime_error(exc)
            with self.lock:
                self.failure = failure
                self.state = "error"
                self.drain_pending(failure)
            if request is not None:
                request.put(failure)
        finally:
            if ctx is not None:
                ctx.engine.close()
                log.info("engine.close() complete on worker thread")
            log.info("worker exited")

    def complete(self, ctx, request: QueuedRequest) -> None:
        worker_start_ns = time.monotonic_ns()
        if request.queued_at_ns:
            request.queue_wait_ms = (worker_start_ns - request.queued_at_ns) / 1e6
        trace = request.task.timeline
        if trace is not None:
            trace.record("Queue", "queue_wait", request.queued_at_ns, worker_start_ns)
        generator = ctx.handle_completion(request.task)
        opened = None
        final = None
        started = time.perf_counter()
        error = None
        buffered: list[PartialResult] = []
        try:
            try:
                for item in generator:
                    if request.cancel_event.is_set():
                        break
                    if isinstance(item, OpenedResult):
                        opened = item
                    with self.lock:
                        self.slot = ctx.slot.state.name
                    if isinstance(item, FinalResult):
                        final = item
                    elif not request.stream and isinstance(item, PartialResult):
                        # Nobody reads this before the final body is assembled.  Cancellation is
                        # still checked every token at the top of the loop, so holding the partial
                        # here costs no responsiveness -- it only removes a queue put and the
                        # consumer-thread wake it would cause.
                        buffered.append(item)
                    elif not request.put(item):
                        break
            finally:
                generator.close()
        except ContextExceeded as exc:
            prompt_n = len(ctx.slot.prompt_tokens)
            max_seq_len = ctx.context_limit + 1
            # The limit is set by what *this* request reserved, not by the engine-wide budget:
            # quoting max_decode_tokens here would tell the client to shrink the wrong number.
            reserved = int(ctx.slot.n_predict)
            error = bad("context_exceeded", f"prompt tokens={prompt_n}; max_tokens={reserved}; available prompt "
                        f"token limit={max_seq_len + 1 - reserved}; max_seq_len={max_seq_len}: {exc}")
        except RequestRejected as exc:
            error = bad("invalid_messages", str(exc))
        except Exception as exc:
            error = runtime_error(exc)
        with self.lock:
            self.slot = ctx.slot.state.name
        if buffered and error is None and not request.cancel_event.is_set():
            # Ordered before the FinalResult put below, so the assembler is fed exactly the
            # sequence it would have seen token by token.
            request.put(PartialBatch(request.task.id, tuple(buffered)))
        finalization_start = time.monotonic_ns()
        if final is not None:
            if self.dump_tokens:
                try:
                    with self.dump_tokens.open("a") as handle:
                        handle.write(json.dumps({"id": request.id, "task_id": request.task.id,
                                                  "tokens": list(final.tokens)}) + "\n")
                except OSError as exc:
                    error = runtime_error(exc)
            segments = final.timings.segments or {}
            log.info("request id=%s session=%s prompt_n=%s cache=%s prefill=%s/%.1fms init=%.1fms est=%s predicted_n=%s "
                     "stop_reason=%s ms=%.3f device_tok_s=%.0f %s start=%.9f end=%.9f",
                     request.id, request.task.session_id, opened.prompt_n, final.n_prompt_tokens_cache,
                     final.timings.prefill_path or "?", final.timings.prompt_ms, final.timings.initialize_ms,
                     final.timings.prefill_estimate,
                     final.timings.predicted_n, final.stop_reason, (time.perf_counter()-started)*1000,
                     final.timings.decode_tok_s_device,
                     " ".join(f"{name}={seg['tokens']}tok/{seg['tokens_per_round']:.2f}acc/{seg['device_tok_s']:.0f}tok_s"
                              for name, seg in segments.items() if name != "total"),
                     started, time.perf_counter())
        else:
            log.info("request id=%s prompt_n=%s cache=%s predicted_n=%s stop_reason=%s ms=%.3f start=%.9f end=%.9f",
                     request.id, len(ctx.slot.prompt_tokens), ctx.slot.timings.n_prompt_tokens_cache,
                     ctx.slot.timings.predicted_n, error.warning_type if error else "cancel",
                     (time.perf_counter()-started)*1000, started, time.perf_counter())
        if error is not None:
            request.put(error)
        elif final is not None:
            if trace is not None:
                trace.record("Worker", "result_finalize", finalization_start, time.monotonic_ns())
            request.put(final)
        elif not request.cancel_event.is_set():
            request.put(runtime_error(RuntimeError("completion ended without FinalResult")))


def coalesce_deltas(events):
    """Merge runs of ``content_block_delta`` events for the same block and delta kind.

    A verify round commits several tokens at once and every token arrives as its own delta; a client
    concatenates consecutive deltas anyway, so one event carrying the round's text is equivalent on
    the wire and costs one JSON encode and one parse instead of six. Deltas of other kinds (input_json,
    signature) and every non-delta event pass through unchanged and end the current run.
    """
    out = []
    for event, data in events:
        if (event == "content_block_delta" and out and out[-1][0] == "content_block_delta"):
            previous = out[-1][1]
            kind = data["delta"].get("type")
            if (previous["index"] == data["index"] and previous["delta"].get("type") == kind
                    and kind in ("text_delta", "thinking_delta")):
                key = "text" if kind == "text_delta" else "thinking"
                merged = dict(previous, delta=dict(previous["delta"]))
                merged["delta"][key] = previous["delta"][key] + data["delta"][key]
                if "naivert" in data:
                    merged["naivert"] = data["naivert"]
                out[-1] = (event, merged)
                continue
        out.append((event, data))
    return out


def encode_events(events):
    for event, data in coalesce_deltas(events):
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def poll_output(request: QueuedRequest):
    try:
        return request.out_queue.get(timeout=.05)
    except queue.Empty:
        return None


# Upper bound on how many already-queued items one SSE write may absorb.  This never *waits*
# for more tokens -- a time window would add latency to every token, and in steady state the
# queue holds at most one item anyway (the generator produces one token per ~2.3 ms while the
# consumer drains one in ~0.2 ms).  The batching only engages when the consumer has fallen
# behind, which is exactly when the per-token thread round trip hurts.
SSE_BATCH_MAX = 32


def drain_output(request: QueuedRequest, limit: int = SSE_BATCH_MAX) -> list:
    """Take whatever is already queued, without blocking."""
    items: list = []
    while len(items) < limit:
        try:
            items.append(request.out_queue.get_nowait())
        except queue.Empty:
            break
    return items


class CompletionResponse(Response):
    """Own ASGI disconnect handling through admission AND the response body.

    StreamingResponse.stream_response handles SSE sending; one receive task watches
    disconnects, including before headers and for non-streaming/queued requests.
    """
    def __init__(self, request: QueuedRequest, *, stream: bool, model: str, ping_seconds: float,
                 token_progress: bool = False):
        super().__init__()
        self.request, self.stream, self.model, self.ping_seconds = request, stream, model, ping_seconds
        # L1: the assembler lives here, on the event loop, never inside Worker.complete's generator
        # loop. Its parse errors must land after generator.close() has returned the slot to IDLE.
        self.assembler = ResponseAssembler(model=model, id=request.id, tools=request.task.tools,
                                           enable_thinking=request.task.enable_thinking,
                                           parallel_stop=request.task.parallel_stop)
        self.deferred: ApiError | None = None
        self.replay = None
        self.token_progress = token_progress
        self.generated_tokens = 0

    def queue_timing(self, data):
        if self.request.queue_wait_ms is None:
            return data
        extension = dict(data.get("naivert", {}))
        extension["timings"] = dict(extension.get("timings", {}),
                                    queue_wait_ms=self.request.queue_wait_ms)
        return dict(data, naivert=extension)

    def progress_events(self, events):
        """Attach the generated-token frontier to visible content, without extra SSE writes.

        Clients subtract frontiers at two receive boundaries; they must not treat an SSE event
        as one token. This is opt-in and does not tokenize text or add device synchronization.
        """
        for kind, data in events:
            if kind in ("message_start", "message_delta"):
                data = self.queue_timing(data)
            if (self.token_progress and kind == "content_block_delta"
                    and data["delta"].get("type") in ("text_delta", "thinking_delta")):
                data = dict(data, naivert={"generated_tokens": self.generated_tokens})
            yield kind, data

    def absorb(self, item: PartialResult):
        """Feed one token and return its events -- or none, if assembly has already failed.

        L1: the assembler raises the moment a malformed <tool_call> closes, which is mid-decode,
        while the worker still owns the slot. Delivering that error there would put a retryable 500
        in front of the client *before* the slot returned to IDLE. So the error is held until the
        FinalResult arrives, which the worker only sends after generator.close() has run.
        """
        if self.deferred is not None:
            return []
        try:
            self.generated_tokens = item.index + 1
            return list(self.progress_events(self.assembler.feed(item)))
        except ApiError as exc:
            self.deferred = exc
            return []

    async def next_item(self):
        while True:
            item = await asyncio.to_thread(poll_output, self.request)
            if item is not None:
                return item

    async def next_item_within(self, seconds: float):
        """The next output item, or None once ``seconds`` pass without one."""
        deadline = time.monotonic() + seconds
        while True:
            item = await asyncio.to_thread(poll_output, self.request)
            if item is not None or time.monotonic() >= deadline:
                return item

    async def early_events(self):
        """SSE comment lines while the session opens (queue wait, prefill), then the stream.

        A cold prefill of a long prompt takes minutes on the replay path (a 916K-token Claude Code
        session: 507 s), and clients abandon a stream that shows no bytes for about five minutes
        (Claude Code's first-byte timeout cannot be raised past that). Comment lines are ignored by
        SSE parsers, so the response can start before message_start exists; an error that arrives
        after the headers went out is delivered as the stream's ``error`` event.
        """
        next_ping = time.monotonic() + self.ping_seconds
        while True:
            item = await self.next_item_within(max(0.0, next_ping - time.monotonic()))
            if item is None:
                yield b": keepalive\n\n"
                next_ping = time.monotonic() + self.ping_seconds
                continue
            if not isinstance(item, OpenedResult):
                error = item if isinstance(item, ApiError) else runtime_error(RuntimeError("missing OpenedResult"))
                yield next(encode_events([("error", error.to_body())]))
                return
            async for event in self.events(item):
                yield event
            return

    async def respond(self, scope, receive, send):
        trace = self.request.task.timeline
        with trace_span(trace, "HTTP", "wait_admission"):
            # A stream that has not opened within one ping interval starts now, with keepalive
            # comments: validation and admission errors surface within milliseconds and keep their
            # HTTP status; only a slow open (long prefill, queue wait) takes the early path.
            opened = await (self.next_item_within(self.ping_seconds) if self.stream else self.next_item())
        if opened is None:
            response = StreamingResponse(self.early_events(), media_type="text/event-stream",
                                         headers={"cache-control": "no-cache", "x-accel-buffering": "no",
                                                  **self._ignored_headers()})
            await response.stream_response(send)
            return
        if isinstance(opened, ApiError):
            await error_response(opened)(scope, receive, send)
            return
        if not isinstance(opened, OpenedResult):
            await error_response(runtime_error(RuntimeError("missing OpenedResult")))(scope, receive, send)
            return
        ignored = self._ignored_headers()
        if self.stream:
            response = StreamingResponse(self.events(opened), media_type="text/event-stream",
                                         headers={"cache-control": "no-cache", "x-accel-buffering": "no", **ignored})
            await response.stream_response(send)
        else:
            wait_start = time.monotonic_ns()
            while True:
                item = await self.next_item()
                if isinstance(item, ApiError):
                    await error_response(item)(scope, receive, send)
                    return
                if isinstance(item, PartialResult):
                    self.absorb(item)              # same state as the stream, events discarded
                    continue
                if isinstance(item, PartialBatch):
                    for partial in item.items:     # same order the stream would have seen
                        self.absorb(partial)
                    continue
                if isinstance(item, FinalResult):
                    if trace is not None:
                        trace.record("HTTP", "wait_completion", wait_start, time.monotonic_ns())
                        item = replace(item, timeline=trace.snapshot())
                    if self.deferred is not None:   # L2: the existing non-stream error channel
                        await error_response(self.deferred)(scope, receive, send)
                        return
                    try:
                        body = self.assembler.message(item, opened.prompt_n)
                    except ApiError as exc:
                        await error_response(exc)(scope, receive, send)
                        return
                    if body is not None:
                        if self.replay is not None:
                            self.replay.remember(self.request.task, self.assembler.blocks)
                        await JSONResponse(self.queue_timing(body), headers=ignored)(scope, receive, send)
                    return

    def _ignored_headers(self) -> dict[str, str]:
        """``x-naivert-ignored``: the request hints this server served by ignoring them (messages_api)."""
        hints = getattr(getattr(self.request, "task", None), "ignored", ())
        return {"x-naivert-ignored": ",".join(hints)} if hints else {}

    async def events(self, opened):
        stream_start = time.monotonic_ns()
        for event in encode_events(self.progress_events(self.assembler.start_events(opened))):
            yield event
        waiting_first = True
        next_ping = time.monotonic() + self.ping_seconds
        buffered: list = []
        while True:
            if buffered:
                item = buffered.pop(0)
            else:
                item = await asyncio.to_thread(poll_output, self.request)
            if item is None:
                if waiting_first and time.monotonic() >= next_ping:
                    yield next(encode_events([("ping", {"type": "ping"})]))
                    next_ping = time.monotonic() + self.ping_seconds
                continue
            if isinstance(item, ApiError):
                yield next(encode_events([("error", item.to_body())]))
                return
            if isinstance(item, PartialResult):
                first = waiting_first
                waiting_first = False
                events = self.absorb(item)
                if not first and not buffered:
                    # The first token always goes out on its own; after that, fold in anything
                    # already sitting in the queue so one SSE write covers several tokens.
                    buffered.extend(drain_output(self.request))
                while buffered and isinstance(buffered[0], PartialResult):
                    events.extend(self.absorb(buffered.pop(0)))
                chunk = b"".join(encode_events(events))
                if chunk:
                    yield chunk
            elif isinstance(item, FinalResult):
                trace = self.request.task.timeline
                if trace is not None:
                    trace.record("HTTP", "stream_until_report", stream_start, time.monotonic_ns(),
                                 includes="SSE sending and waiting; excludes final report transmission")
                    item = replace(item, timeline=trace.snapshot())
                if self.deferred is not None:
                    yield next(encode_events([("error", self.deferred.to_body())]))
                    return
                try:
                    final_events = list(self.assembler.final_events(item))
                    if final_events and self.replay is not None:
                        self.replay.remember(self.request.task, self.assembler.blocks)
                    for event in encode_events(self.progress_events(final_events)):
                        yield event
                except ApiError as exc:
                    yield next(encode_events([("error", exc.to_body())]))
                return

    async def __call__(self, scope, receive, send):
        async def disconnected():
            while (await receive())["type"] != "http.disconnect":
                pass
        tasks = [asyncio.create_task(self.respond(scope, receive, send)), asyncio.create_task(disconnected())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except OSError:
            pass  # ASGI 2.4 send on a closed connection
        finally:
            self.request.cancel_event.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def build_app(ctx_factory, *, served_model: str, queue_depth: int, ping_seconds: float = 5., display_name: str | None = None,
              dump_tokens: str | Path | None = None, cors_origins: list[str] | None = None,
              cors_credentials: bool = False) -> FastAPI:
    if ping_seconds <= 0:
        raise ValueError("ping_seconds must be positive")
    if cors_credentials and (not cors_origins or "*" in cors_origins):
        raise ValueError("credentialed CORS requires explicit allowed origins")
    worker = Worker(ctx_factory, queue_depth=queue_depth, dump_tokens=dump_tokens)
    replay = AssistantReplay()

    @asynccontextmanager
    async def lifespan(app):
        worker.thread.start()
        try:
            yield
        finally:
            worker.stop()
            await asyncio.to_thread(worker.thread.join)

    app = FastAPI(lifespan=lifespan)
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins,
                           allow_credentials=cors_credentials,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "x-naivert-token-progress", "X-Session-Id", "Authorization", "x-api-key"],
                           expose_headers=["x-naivert-warning", "x-naivert-ignored"])
    app.state.worker = worker
    @app.get("/health")
    async def health():
        snapshot = worker.snapshot()
        if snapshot["status"] != "ok":
            return JSONResponse({"status": snapshot["status"]}, status_code=503)
        return snapshot

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": served_model, "type": "model", "display_name": display_name or served_model,
                          "created_at": "1970-01-01T00:00:00Z"}], "has_more": False,
                "first_id": served_model, "last_id": served_model}

    @app.post("/v1/messages")
    async def messages(request: Request):
        trace = RequestTimeline()
        state = worker.snapshot()["status"]
        if state != "ok":
            return error_response(worker.failure or unavailable(state))
        with trace.span("HTTP", "parse_validate"):
            try:
                req = MessagesRequest.model_validate_json(await request.body())
            except ValidationError as exc:
                return error_response(validation_error(exc))
            task = validate_request(req, max_decode_tokens=worker.max_decode_tokens, served_model=served_model)
        if isinstance(task, ApiError):
            return error_response(task)
        task.timeline = trace
        # The session key: the x-session-id header wins over metadata.user_id. It selects the
        # parked KV prefix this request may resume and the one it parks under when it finishes.
        header_session = request.headers.get("x-session-id")
        if header_session:
            task.session_id = header_session
        replay.restore(task)
        queued = QueuedRequest(task, stream=bool(req.stream))
        try:
            worker.submit(queued)
        except ApiError as exc:
            return error_response(exc)
        response = CompletionResponse(queued, stream=req.stream, model=served_model, ping_seconds=ping_seconds,
                                      token_progress=request.headers.get("x-naivert-token-progress") == "1")
        response.replay = replay
        return response

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        """Admission is a hard rule, so the client must be able to size the next turn before sending it."""
        state = worker.snapshot()["status"]
        if state != "ok":
            return error_response(worker.failure or unavailable(state))
        try:
            req = CountTokensRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            return error_response(validation_error(exc))
        task = validate_count_tokens(req, served_model=served_model)
        if isinstance(task, ApiError):
            return error_response(task)
        metadata_session = req.metadata.get('user_id') if isinstance(req.metadata, dict) else None
        task.session_id = request.headers.get('x-session-id') or (str(metadata_session) if metadata_session is not None else None)
        replay.restore(task)
        try:
            n = await asyncio.to_thread(worker.count_prompt_tokens, task)
        except ApiError as exc:
            return error_response(exc)
        return {"input_tokens": n}

    return app


def run_app(app: FastAPI, *, host: str, port: int):
    """Mark stopping at SIGTERM, before uvicorn waits for existing connections."""
    import uvicorn

    class Server(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            # A handled SIGTERM is a successful service shutdown (exit 0).
            # Uvicorn's default re-raises it after cleanup, producing exit -15.
            signals = (signal.SIGINT, signal.SIGTERM)
            previous = {sig: signal.signal(sig, self.handle_exit) for sig in signals}
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

        def handle_exit(self, sig, frame):
            # Do not acquire a lock from a signal handler: it can interrupt a
            # snapshot/submit on this same thread while that lock is held.
            asyncio.get_running_loop().call_soon(app.state.worker.stop)
            super().handle_exit(sig, frame)

    Server(uvicorn.Config(app, host=host, port=port, log_level="info")).run()
