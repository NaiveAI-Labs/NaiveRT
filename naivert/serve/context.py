"""Synchronous request handling over a session engine.

``handle_completion`` yields an ``OpenedResult``, streamed ``PartialResult``
records, then one ``FinalResult``. The local common-prefix hint is tracked through
``cache_n_past``. Engines with a PrefixStore can independently resume parked
sessions or prompt checkpoints and report their actual prefill start position.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Generator, Iterator

from naivert.runtime.contract import RuntimeContractError
from naivert.runtime.mtp_contract import DecodeMode
from naivert.serve.chat import ChatFormatter, IncrementalDetokenizer
from naivert.serve.slot import ServerSlot, SlotEvent, SlotState
from naivert.serve.task import FinalResult, OpenedResult, PartialResult, ServerTask, Timings
from naivert.serve.timeline import trace_span

CONTEXT_EXCEEDED_MARK = "past the pinned Seq1 MLA kernel's validated range"


def relabel_segments(segments, enable_thinking: bool):
    """The engine splits its per-segment stats at the first ``</think>``; a thinking request that never
    closed its thinking (the ceiling was hit inside it) is all think, not all answer."""
    if segments and enable_thinking and "think" not in segments and "answer" in segments:
        return {"think": segments["answer"], **{k: v for k, v in segments.items() if k != "answer"}}
    return segments


class RequestRejected(ValueError):
    """The request cannot be admitted (R-c): the slot stays / returns to IDLE."""


class ContextExceeded(RequestRejected):
    """prompt + max_decode_tokens exceeds the engine's validated context (wraps RuntimeContractError)."""


class ServerContext:
    def __init__(self, engine: Any, formatter: ChatFormatter) -> None:
        self.engine = engine
        self.formatter = formatter
        self.slot = ServerSlot(detokenizer=IncrementalDetokenizer(formatter.tokenizer))
        self.n_unfinished = 0
        self.transcript: list[dict[str, Any]] = []
        # Session engine request generation observed right after our last completed session: the KV cache
        # described by slot.tokens is only reusable while nobody else opened a session since.
        self._cache_generation: int | None = None

    @property
    def max_decode_tokens(self) -> int:
        return int(self.engine.config.max_decode_tokens)

    @property
    def context_limit(self) -> int | None:
        """Highest cur_pos a request may decode at in the engine's mode (prompt + decode must stay
        within limit + 2): the seq1 kernel's validated bound, or native MTP's lower one (a block
        anchored at a position writes the caches seven rows past it)."""
        mode = getattr(self.engine.config, "mode", DecodeMode.SEQ1)
        name = "native_mtp_max_cur_pos" if mode == DecodeMode.NATIVE_MTP else "seq1_max_cur_pos"
        limit = getattr(self.engine._backend, name, None)  # noqa: SLF001
        return None if limit is None else int(limit)

    def prompt_tokens_for(self, task: ServerTask) -> list[int]:
        if task.prompt_tokens is not None:
            return [int(t) for t in task.prompt_tokens]
        assert task.messages is not None
        return self.formatter.encode(task.messages, tools=task.tools,
                                     enable_thinking=task.enable_thinking)

    def _effective_n_predict(self, task: ServerTask) -> int:
        n_predict = int(task.params.n_predict)
        if n_predict > self.max_decode_tokens:
            raise RequestRejected(f"n_predict={n_predict} exceeds the engine's max_decode_tokens={self.max_decode_tokens}")
        return self.max_decode_tokens if n_predict < 0 else n_predict

    def cache_n_past(self, tokens: list[int]) -> int:
        """R-a: reuse the common prefix of the last committed sequence, always recomputing the last
        observed token (bounded by both the previous sequence and the new prompt), else 0."""
        if self._cache_generation != self.engine.request_generation or len(self.slot.tokens) == 0:
            return 0
        n_past = min(self.slot.tokens.get_common_prefix(tokens), len(self.slot.tokens) - 1, len(tokens) - 1)
        if n_past < int(getattr(self.engine._backend, "min_prompt_tokens", 1)):  # noqa: SLF001
            return 0
        return n_past

    def _open(self, tokens: list[int], n_past: int, n_predict: int, session_id: str | None = None,
              sampling: dict | None = None) -> Any:
        """Reserve context for this request only: admission uses ``n_predict``, not the engine budget.

        Optional session and sampling arguments are passed only when the engine
        accepts them."""
        # Small test engines may implement only the required open_session arguments.
        import inspect
        accepted = inspect.signature(self.engine.open_session).parameters
        extra = {"session_id": session_id} if session_id is not None and "session_id" in accepted else {}
        if sampling is not None and "sampling" in accepted:
            extra["sampling"] = sampling
        try:
            return self.engine.open_session(tokens, n_past=n_past, max_decode_tokens=n_predict, **extra)
        except RuntimeContractError as exc:
            if CONTEXT_EXCEEDED_MARK in str(exc):
                raise ContextExceeded(str(exc)) from exc
            raise

    def handle_completion(self, task: ServerTask) -> Generator[OpenedResult | PartialResult | FinalResult, None, None]:
        slot = self.slot
        if slot.state is not SlotState.IDLE:
            raise RequestRejected(f"slot busy in state {slot.state.name}")
        slot.transition(SlotEvent.SUBMIT)
        self.n_unfinished += 1
        session: Any | None = None
        trace = task.timeline
        try:
            # ---- admission (STARTED) -------------------------------------------------------
            try:
                with trace_span(trace, "Worker", "format_tokenize"):
                    n_predict = self._effective_n_predict(task)
                    tokens = self.prompt_tokens_for(task)
                    # Incremental tokenization: an engine that parks finished sequences may keep
                    # the model's own ids for the history instead of the canonical re-encoding of
                    # its text, so a sampled token split does not defeat the prefix match.
                    align = getattr(self.engine, "align_prompt", None)
                    if align is not None and task.messages is not None:
                        tokens = align(tokens, text=self.formatter.render(task.messages, tools=task.tools,
                                                                          enable_thinking=task.enable_thinking),
                                       session_id=task.session_id)
                    # max_tokens is a cap, not a reservation the prompt must leave room for: when the
                    # prompt fits but prompt + max_tokens would not, decode with the room that is left
                    # (reported as max_tokens_clamped); a prompt that itself does not fit is refused below.
                    limit = self.context_limit
                    if limit is not None and int(task.params.n_predict) > 0:   # -1 keeps the engine budget (A0.3)
                        # The native MTP engine decodes at cur_pos <= limit, i.e. prompt + n_predict <= limit + 1
                        # (its own message quotes one more, and the seq1 bound is one looser); the tighter one holds.
                        room = limit + 1 - len(tokens)
                        if 0 < room < n_predict:
                            n_predict = room
                            task.ignored = tuple(task.ignored) + ("max_tokens_clamped",)
            except RequestRejected:
                slot.transition(SlotEvent.REJECT)
                raise
            stop_ids = (self.formatter.eos_token_id, *self.engine.config.model_spec.stop_token_ids,
                        *task.params.stop_token_ids)
            with trace_span(trace, "Worker", "cache_lookup"):
                slot.start(task_id=task.id, prompt_tokens=tokens, n_predict=n_predict,
                           stop_strings=task.params.stop_strings, stop_token_ids=stop_ids,
                           include_stop_in_output=task.params.include_stop_in_output,
                           n_past=self.cache_n_past(tokens) if task.params.cache_prompt else 0)
            # ---- prompt (PROCESSING_PROMPT) --------------------------------------------------
            slot.transition(SlotEvent.OPEN_SESSION)
            t0 = time.perf_counter()
            try:
                with trace_span(trace, "Worker", "open_session", includes="cache setup and prefill"):
                    controls = {k: v for k, v in dict(temperature=task.params.temperature, top_p=task.params.top_p,
                                                       seed=task.params.seed, repeat_penalty=task.params.repeat_penalty).items()
                                if v is not None}
                    session = self._open(tokens, slot.n_past, n_predict, task.session_id, controls or None)
            except Exception as exc:
                slot.fail(exc)          # -> ERROR (RUNTIME_ERROR / FINISH_FAILED)
                raise
            capture_ms = (time.perf_counter() - t0) * 1e3
            slot.transition(SlotEvent.SESSION_RETURNED)
            prefill = session.runtime.prepared_prefill.as_dict()
            if trace is not None and "prefill_start_ns" in prefill and "prefill_end_ns" in prefill:
                trace.record("Prefill", "prefill", prefill["prefill_start_ns"], prefill["prefill_end_ns"],
                             measurement="host wall time, including backend synchronization")
            # the backend may have resumed earlier than requested (Naive N SWA ring fallback)
            slot.n_past = int(prefill.get("start_position", slot.n_past))
            slot.timings.n_prompt_tokens_cache = slot.n_past
            slot.timings.prompt_n = len(tokens) - slot.n_past      # tokens actually prefilled this turn
            slot.timings.prompt_ms = float(prefill.get("prefill_ms", 0.0))
            slot.timings.prefill_path = str(prefill.get("prefill_path", ""))
            slot.timings.initialize_ms = float(prefill.get("initialize_ms", 0.0))
            slot.timings.prefill_estimate = prefill.get("prefill_estimate")
            slot.timings.prefill_phases = prefill.get("prefill_phases")
            slot.timings.capture_ms = capture_ms
            # ---- decode (GENERATING) ---------------------------------------------------------
            slot.transition(SlotEvent.FIRST_STEP)
            t1 = time.perf_counter()
            decode_start_ns = time.monotonic_ns()
            reason: str | None = None
            index = 0
            try:
                yield OpenedResult(task.id, len(tokens), slot.n_past)
                while reason is None:
                    if len(slot.generated_tokens) >= session.max_decode_tokens:
                        reason = "length"
                        slot.stop_reason, slot.truncated = "length", True
                        break
                    if trace is None:
                        # Trace off: no span objects, no details dict, and no progress_records
                        # read -- all three exist only to fill the timeline, and all three used
                        # to run on every token regardless.
                        token = session.step()
                        text, reason = slot.process_token(token)
                        yield PartialResult(task_id=task.id, index=index, token=token, text=text)
                    else:
                        with trace_span(trace, "Token step", "step", token_index=index) as details:
                            token = session.step()
                            records = session.runtime.output_reader.progress_records
                            if records and "device_ns" in records[-1]:
                                # This compatibility path reports the device-clock interval
                                # between commits, including host turnaround between replays.
                                # The historical field name is retained for trace consumers;
                                # it does not represent an isolated CUDA-event duration.
                                details["rank0_cuda_event_ms"] = records[-1]["device_ns"] / 1e6
                        with trace_span(trace, "Text", "detokenize_stop", token_index=index):
                            text, reason = slot.process_token(token)
                        with trace_span(trace, "Publish", "publish_partial", token_index=index):
                            yield PartialResult(task_id=task.id, index=index, token=token, text=text)
                    index += 1
                    if reason is None:
                        slot.transition(SlotEvent.TOKEN)
            except GeneratorExit:
                slot.cancel()
                reason = "cancel"
            except Exception as exc:
                slot.fail(exc)          # -> ERROR (RUNTIME_ERROR / FINISH_FAILED)
                raise
            finally:
                if trace is not None:
                    trace.record("Worker", "decode", decode_start_ns, time.monotonic_ns(),
                                 includes="step, text processing and output queue backpressure")
            slot.timings.predicted_ms = (time.perf_counter() - t1) * 1e3
            slot.timings.predicted_n = len(slot.generated_tokens)
            records = session.runtime.output_reader.progress_records
            # Commit-to-commit device-clock time (see the note on rank0_cuda_event_ms):
            # a cadence, not a pure replay duration.
            if getattr(session, "mode", DecodeMode.SEQ1) == DecodeMode.NATIVE_MTP:
                # One commit publishes a whole block, so its tokens share one interval; count
                # the interval once.  Output 0 comes from the prefill and has none.
                blocks: dict[tuple[int, int], int] = {}
                for r in records:
                    key = (int(r["parent_start_ns"]), int(r["parent_complete_ns"]))
                    if key[1] != 0:
                        blocks[key] = blocks.get(key, 0) + 1
                device_ns = sum(complete - start for start, complete in blocks)
                timed = sum(blocks.values())
            else:
                device_ns = sum(int(r["parent_complete_ns"]) - int(r["parent_start_ns"]) for r in records)
                timed = len(records)
            slot.timings.decode_tok_s_device = timed / (device_ns / 1e9) if device_ns > 0 else 0.0
            slot.timings.device_chain = bool(
                session.runtime.graph_contract_evidence.get("device_token_chain", False))
            # ---- finish (FINISHING) ----------------------------------------------------------
            slot.transition(SlotEvent.STOP)
            try:
                with trace_span(trace, "Worker", "finish"):
                    session.finish()
            except Exception as exc:
                slot.fail(exc)          # -> ERROR (RUNTIME_ERROR / FINISH_FAILED)
                raise
            # Only finish() knows this: a stop string or a cancel ends the turn while the
            # device is still generating, and finish() is where the chain is drained and the
            # positions it wrote in the meantime are folded back into the cache accounting.
            # Surfaced because commit_sequence below decides what the next turn may reuse.
            slot.timings.chain_abandoned_n = int(
                getattr(session.runtime, "abandoned_chain_positions", 0))
            slot.timings.chain_canceled = bool(
                getattr(session.runtime, "canceled_chain", False))
            slot.timings.device_steps = int(
                getattr(session.runtime, "device_steps", 0))
            slot.timings.reconcile_error = getattr(session.runtime, "reconcile_error", None)
            slot.timings.segments = relabel_segments(getattr(session.runtime, "segments", None),
                                                     task.enable_thinking)
            bound = getattr(session.runtime, "device_stop_bound", None)
            slot.timings.device_stop_bound = None if bound is None else int(bound)
            slot.commit_sequence()
            self._cache_generation = self.engine.request_generation
            final = FinalResult(task_id=task.id, text=slot.generated_text, tokens=tuple(slot.generated_tokens),
                                stop_reason=reason, truncated=slot.truncated, timings=slot.timings,
                                stop_string=slot.stop_string, n_prompt_tokens_cache=slot.timings.n_prompt_tokens_cache)
            self.transcript.append({"task_id": task.id, "prompt_n": len(tokens) - slot.n_past,
                                    "n_prompt_tokens_cache": slot.n_past, "tokens": list(final.tokens),
                                    "text": final.text, "stop_reason": reason, "timings": slot.timings.as_dict()})
            if reason != "cancel":
                yield final
        finally:
            # close() is finish() for a turn that never reached the finish above -- the step
            # timeout path lands here -- so it can raise, and a raise from inside this finally
            # would skip the two lines after it: the slot would stay ERROR for the life of the
            # process and every later request would be rejected as "slot busy in state ERROR".
            # Release first; re-raise only when nothing better is already on its way out, so a
            # step timeout keeps its own diagnosis instead of being replaced by close()'s.
            close_error: Exception | None = None
            if session is not None:
                with trace_span(trace, "Worker", "close_session"):
                    try:
                        session.close()
                    except Exception as exc:
                        close_error = exc
            # release() is legal only from FINISHING / ERROR / IDLE, and two windows above can
            # raise with the slot in neither: admission (STARTED -- formatting, tokenizing,
            # slot.start) and the gap between the session coming back and the first step
            # (DONE_PROMPT).  Recording the failure first is what ERROR is for; without it
            # release() itself raises here and strands the slot exactly the way a raising
            # close() used to.  The decrement is in a finally so it survives even that.
            try:
                pending = close_error if close_error is not None else sys.exc_info()[1]
                if slot.state not in (SlotState.FINISHING, SlotState.ERROR, SlotState.IDLE):
                    slot.fail(pending if isinstance(pending, BaseException)
                              else RuntimeError(f"request ended in {slot.state.name}"))
                slot.release()
            finally:
                self.n_unfinished -= 1
            if close_error is not None and sys.exc_info()[1] is None:
                raise close_error

    def complete(self, task: ServerTask) -> FinalResult:
        """Run a task to completion, returning only the FinalResult."""
        final: FinalResult | None = None
        for item in self.handle_completion(task):
            if isinstance(item, FinalResult):
                final = item
        assert final is not None
        return final

    def stream(self, task: ServerTask) -> Iterator[OpenedResult | PartialResult | FinalResult]:
        return self.handle_completion(task)


__all__ = ["CONTEXT_EXCEEDED_MARK", "ContextExceeded", "RequestRejected", "ServerContext", "relabel_segments"]
