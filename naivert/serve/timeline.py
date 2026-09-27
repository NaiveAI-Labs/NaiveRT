"""Request-scoped host spans. No CUDA calls, synchronization, or clock inference."""
from contextlib import contextmanager, nullcontext
import threading
import time


class RequestTimeline:
    def __init__(self):
        self.origin_ns = time.monotonic_ns()
        self._spans = []
        self._lock = threading.Lock()

    def record(self, lane, name, start_ns, end_ns, **args):
        if not self.origin_ns <= start_ns <= end_ns:
            raise ValueError("timeline spans must use the request's host monotonic clock")
        event = {"lane": lane, "name": name, "start_ms": (start_ns-self.origin_ns)/1e6,
                 "duration_ms": (end_ns-start_ns)/1e6, "args": args}
        with self._lock:
            self._spans.append(event)

    @contextmanager
    def span(self, lane, name, **args):
        start = time.monotonic_ns()
        try:
            yield args
        finally:
            self.record(lane, name, start, time.monotonic_ns(), **args)

    def snapshot(self):
        with self._lock:
            return {"schema": "naivert.request_timeline.v1", "clock": "host_monotonic",
                    "origin": "http_handler_entry", "unit": "ms",
                    "end_ms": (time.monotonic_ns()-self.origin_ns)/1e6,
                    "spans": sorted(self._spans, key=lambda event: event["start_ms"])}


def trace_span(trace, lane, name, **args):
    return trace.span(lane, name, **args) if trace is not None else nullcontext(args)
