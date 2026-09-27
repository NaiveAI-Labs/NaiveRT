"""The single serving slot: state machine + per-token processing (llama.cpp ``server_slot``).

States, events and the legal edges are exactly the slot state machine; ``transition`` raises
``IllegalSlotTransition`` for every other (state, event) pair.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from .task import Timings
from .tokens import ServerTokens, StopMatcher


class SlotState(Enum):
    IDLE = "idle"
    STARTED = "started"
    PROCESSING_PROMPT = "processing_prompt"
    DONE_PROMPT = "done_prompt"
    GENERATING = "generating"
    FINISHING = "finishing"
    ERROR = "error"


class SlotEvent(Enum):
    SUBMIT = "submit"                    # IDLE -> STARTED
    REJECT = "reject"                    # STARTED -> IDLE (ContextExceeded | BadTokens | n_predict > max)
    OPEN_SESSION = "open_session"        # STARTED -> PROCESSING_PROMPT
    SESSION_RETURNED = "session_returned"  # PROCESSING_PROMPT -> DONE_PROMPT
    RUNTIME_ERROR = "runtime_error"      # STARTED | PROCESSING_PROMPT | DONE_PROMPT | GENERATING -> ERROR
    FIRST_STEP = "first_step"            # DONE_PROMPT -> GENERATING
    TOKEN = "token"                      # GENERATING -> GENERATING
    STOP = "stop"                        # GENERATING -> FINISHING (eos | stop_string | n_predict | cancel)
    FINISHED = "finished"                # FINISHING -> IDLE
    FINISH_FAILED = "finish_failed"      # FINISHING -> ERROR
    RESET = "reset"                      # ERROR -> IDLE


TRANSITIONS: dict[tuple[SlotState, SlotEvent], SlotState] = {
    (SlotState.IDLE, SlotEvent.SUBMIT): SlotState.STARTED,
    (SlotState.STARTED, SlotEvent.REJECT): SlotState.IDLE,
    (SlotState.STARTED, SlotEvent.OPEN_SESSION): SlotState.PROCESSING_PROMPT,
    # Admission is not failure-free: formatting/tokenizing and slot.start() both run in STARTED
    # and can raise something that is not RequestRejected.  Without this edge the worker cannot
    # even record that failure -- fail() raises, then release() raises, and the slot is stuck
    # busy for the life of the process, which is a worse outcome than the request's own error.
    (SlotState.STARTED, SlotEvent.RUNTIME_ERROR): SlotState.ERROR,
    (SlotState.PROCESSING_PROMPT, SlotEvent.SESSION_RETURNED): SlotState.DONE_PROMPT,
    (SlotState.PROCESSING_PROMPT, SlotEvent.RUNTIME_ERROR): SlotState.ERROR,
    (SlotState.DONE_PROMPT, SlotEvent.FIRST_STEP): SlotState.GENERATING,
    # Same reason, for the window between the session coming back and the first step: prefill
    # bookkeeping and the trace record run there.
    (SlotState.DONE_PROMPT, SlotEvent.RUNTIME_ERROR): SlotState.ERROR,
    (SlotState.GENERATING, SlotEvent.TOKEN): SlotState.GENERATING,
    (SlotState.GENERATING, SlotEvent.STOP): SlotState.FINISHING,
    (SlotState.GENERATING, SlotEvent.RUNTIME_ERROR): SlotState.ERROR,
    (SlotState.FINISHING, SlotEvent.FINISHED): SlotState.IDLE,
    (SlotState.FINISHING, SlotEvent.FINISH_FAILED): SlotState.ERROR,
    (SlotState.ERROR, SlotEvent.RESET): SlotState.IDLE,
}


class IllegalSlotTransition(RuntimeError):
    pass


class ServerSlot:
    """Holds the sequence the KV cache represents and the in-flight generation."""

    def __init__(self, *, detokenizer: Any | None = None) -> None:
        self.state = SlotState.IDLE
        self.tokens = ServerTokens()          # last complete sequence (prompt + generated incl. EOS)
        self.prompt_tokens: list[int] = []
        self.generated_tokens: list[int] = []
        self.generated_text = ""
        self.n_sent_text = 0
        self.n_past = 0
        self.n_predict = -1
        self.stop_token_ids: frozenset[int] = frozenset()
        self.stop_matcher = StopMatcher()
        self.detokenizer = detokenizer
        self.stop_reason: str | None = None
        self.stop_string: str | None = None
        self.truncated = False
        self.timings = Timings()
        self.task_id: int | None = None
        self.error: BaseException | None = None

    # ------------------------------------------------------------- state
    def transition(self, event: SlotEvent) -> SlotState:
        key = (self.state, SlotEvent(event))
        if key not in TRANSITIONS:
            raise IllegalSlotTransition(f"{self.state.name} does not accept {SlotEvent(event).name}")
        self.state = TRANSITIONS[key]
        return self.state

    def can(self, event: SlotEvent) -> bool:
        return (self.state, SlotEvent(event)) in TRANSITIONS

    # ----------------------------------------------------------- request
    def start(self, *, task_id: int, prompt_tokens: Sequence[int], n_predict: int, stop_strings: Sequence[str],
              stop_token_ids: Sequence[int], include_stop_in_output: bool = False, n_past: int = 0) -> None:
        """Bind a request to the slot (the STARTED -> PROCESSING_PROMPT bookkeeping)."""
        self.task_id = int(task_id)
        self.prompt_tokens = [int(t) for t in prompt_tokens]
        self.generated_tokens = []
        self.generated_text = ""
        self.n_sent_text = 0
        self.n_past = int(n_past)
        self.n_predict = int(n_predict)
        self.stop_token_ids = frozenset(int(t) for t in stop_token_ids)
        self.stop_matcher = StopMatcher(stop_strings, include_in_output=include_stop_in_output)
        if self.detokenizer is not None:
            self.detokenizer.reset()
        self.stop_reason = None
        self.stop_string = None
        self.truncated = False
        self.timings = Timings(n_prompt_tokens_cache=int(n_past))
        self.error = None

    def add_token(self, token: int) -> None:
        self.generated_tokens.append(int(token))

    def process_token(self, token: int) -> tuple[str, str | None]:
        """Record one observed token. Returns (text safe to send now, stop_reason or None).

        Order (llama.cpp ``process_token``): append; EOS-class tokens stop before detokenizing;
        otherwise detokenize incrementally, scan for stop strings (truncating the visible text to
        the stop), then apply ``n_predict``. Text is released only past any partial stop-string
        suffix so a caller can stream it verbatim.
        """
        token = int(token)
        self.add_token(token)
        if token in self.stop_token_ids:
            self.stop_reason = "eos"
            return self._release_text(final=True), self.stop_reason
        delta = self.detokenizer.add(token) if self.detokenizer is not None else ""
        if delta:
            self.generated_text += delta
            match = self.stop_matcher.feed(delta)
            if match is not None:
                self.generated_text = self.generated_text[:match.truncate_to(self.stop_matcher.include_in_output)]
                self.stop_reason, self.stop_string = "stop_string", match.stop
                return self._release_text(final=True), self.stop_reason
        if 0 <= self.n_predict <= len(self.generated_tokens):
            self.stop_reason, self.truncated = "length", True
            return self._release_text(final=True), self.stop_reason
        return self._release_text(final=False), None

    def _release_text(self, *, final: bool) -> str:
        hold = 0 if final else self.stop_matcher.holdback()
        end = max(self.n_sent_text, len(self.generated_text) - hold)
        text = self.generated_text[self.n_sent_text:end]
        self.n_sent_text = end
        return text

    def cancel(self) -> None:
        self.stop_reason = "cancel"

    def commit_sequence(self) -> None:
        """After a clean finish: the KV cache now holds prompt + generated (R-f: no overrun token)."""
        self.tokens = ServerTokens(self.prompt_tokens + self.generated_tokens)

    def release(self) -> None:
        """Back to IDLE (FINISHING -> IDLE, or ERROR -> IDLE which also invalidates ``tokens``)."""
        if self.state is SlotState.FINISHING:
            self.transition(SlotEvent.FINISHED)
        elif self.state is SlotState.ERROR:
            self.transition(SlotEvent.RESET)
            self.tokens.clear()
        elif self.state is not SlotState.IDLE:
            raise IllegalSlotTransition(f"release() from {self.state.name}")
        self.generated_tokens = []
        self.generated_text = ""
        self.n_sent_text = 0
        self.task_id = None

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.stop_reason = "error"
        self.transition(SlotEvent.RUNTIME_ERROR if self.state is not SlotState.FINISHING else SlotEvent.FINISH_FAILED)


__all__ = ["IllegalSlotTransition", "ServerSlot", "SlotEvent", "SlotState", "TRANSITIONS"]
