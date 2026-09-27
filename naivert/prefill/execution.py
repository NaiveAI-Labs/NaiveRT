"""Bounded stream waits and persistent per-rank graph segments."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable
import torch

STREAM_TIMEOUT_S = 300.0
WAIT_IDLE_SPIN_S = 2e-3

class PrefillError(RuntimeError):
    """A prefill execution or state contract failed."""

def wait_stream(stream: torch.cuda.Stream, *, what: str, timeout_s: float = STREAM_TIMEOUT_S) -> None:
    """Wait for what is queued on ``stream``, bounded -- never a bare ``synchronize()``.

    Used at runtime completion and lifecycle boundaries. Poll an event on the
    requested stream, stop at the deadline, and identify the unfinished operation.
    This bound does not cover other host synchronizations such as CPU routing.
    """
    ev = torch.cuda.Event()
    ev.record(stream)
    deadline = time.monotonic() + float(timeout_s)
    while not ev.query():
        if time.monotonic() > deadline:
            raise PrefillError(f"{what}: the rank stream did not drain in {timeout_s:g}s")
        time.sleep(WAIT_IDLE_SPIN_S)


@dataclass
class Captured:
    """One segment's per-rank graphs, and what its body returned when it was captured.

    The outputs are tensors in the rank's graph pool: a replay rewrites them in place, so a
    caller that wants one has to clone it before the next segment replays -- capture order and
    replay order are not the same, and an earlier graph's temporaries may live where a later
    one keeps its result.
    """

    graph: list[Any]
    out: list[Any]


class GraphSegments:
    """Own segment capture/replay and release graph pools after a bounded drain."""

    def _segment(self, key: tuple, body: Callable[[int], Any], *, graphs: bool) -> list[Any]:
        """Run one segment on every rank: replay it, capture it, or run it eagerly.

        A key is eager the first time it is seen -- every JIT compile, autotune, workspace
        allocation and cuBLAS handle those kernels want happens there, where it is allowed to
        -- captured the second, and replayed after that.  Capture is all-or-nothing across the
        ranks: eight graphs or none, because a segment half in a graph would reduce partial
        sums that the other half never produced.
        """
        done = self.captured.get(key) if graphs else None
        if done is not None:
            for r in range(self.ranks):
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    done.graph[r].replay()
            self._stats["graph_replays"] = self._stats.get("graph_replays", 0) + 1
            return done.out
        seen = self._seen.get(key, 0)
        self._seen[key] = seen + 1
        if graphs and seen and key not in self.eager_only:
            out = self._capture(key, body)
            if out is not None:
                return out
        outs = []
        for r in range(self.ranks):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                outs.append(body(r))
        self._stats["eager_segments"] = self._stats.get("eager_segments", 0) + 1
        return outs

    def reset_graphs(self) -> dict:
        """Drop every captured segment and forget what has been seen; returns what was dropped.

        A key's three states -- eager, captured, replayed -- are reached in that order and then
        never again, because ``_seen`` and ``captured`` are cumulative across requests by
        design: a served conversation should pay for a shape once.  A test that wants to watch
        the progression therefore has to put the model back at the start of it, and so does a
        caller that has invalidated what the graphs recorded (new weights, a moved cache).  The
        refusals are cleared too: a key kept out of the graph path by an allocation the pool
        could not serve then gets another chance.

        Each rank then gets a *fresh* pool handle.  Resetting every graph that shares a private
        pool takes that pool's use count to zero while its entry is still in the allocator's
        table, and the next ``capture_begin`` on the same handle asserts on it
        (``use_count > 0`` in ``beginAllocateToPool``) -- so the handle a full reset leaves
        behind is not reusable, and the captures after one would all refuse.  The old pool's
        blocks are released as the graphs and their outputs go out of scope.
        """
        dropped = {"captured": len(self.captured), "seen": len(self._seen),
                   "eager_only": len(self.eager_only)}
        for done in self.captured.values():
            for g in done.graph:
                g.reset()
        self.captured.clear()
        self._seen.clear()
        self.eager_only.clear()
        for r in range(self.ranks):
            with torch.cuda.device(self.devices[r]):
                self.pool[r] = torch.cuda.graph_pool_handle()
        return dropped

    def _capture(self, key: tuple, body: Callable[[int], Any]) -> list[Any] | None:
        """Capture one segment on every rank, then replay it for the chunk that captured it.

        A capture records launches and runs none of them, so the chunk under way still needs
        its work done: the replay right after is that, not a warm-up.  Anything that refuses
        to be captured -- a host copy, an allocation the pool cannot serve, a kernel that
        synchronises -- takes the key out of the graph path for good, with the reason kept,
        and the caller falls back to the eager loop.
        """
        graphs: list[Any] = []
        outs: list[Any] = []
        try:
            for r in range(self.ranks):
                with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                    g = torch.cuda.CUDAGraph()
                    g.capture_begin(pool=self.pool[r], capture_error_mode="thread_local")
                    graphs.append(g)
                    try:
                        outs.append(body(r))
                    except BaseException:
                        # Keep the reason the body gave, not whatever ending an already
                        # broken capture reports on top of it.
                        try:
                            g.capture_end()
                        except Exception:                              # noqa: BLE001
                            pass
                        raise
                    g.capture_end()
        except Exception as exc:                                       # noqa: BLE001
            # Keep the message.  Reporting only the class turned "this call site copies a host
            # tensor" into "this kernel cannot be captured" once already, and that reached a log.
            self.eager_only[key] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            for g in graphs:
                g.reset()
            return None
        self.captured[key] = Captured(graph=graphs, out=outs)
        for r in range(self.ranks):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                graphs[r].replay()
        self._stats["graph_captures"] = self._stats.get("graph_captures", 0) + 1
        return outs
