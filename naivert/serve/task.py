"""Task / result records (llama.cpp ``server_task``, ``task_params``, ``server_task_result``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .timeline import RequestTimeline

STOP_REASONS = ("eos", "stop_string", "length", "cancel", "error")


@dataclass(frozen=True)
class TaskParams:
    cache_prompt: bool = False
    n_predict: int = -1                     # -1: up to the engine's max_decode_tokens
    stop_strings: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()    # in addition to the model's stop tokens
    include_stop_in_output: bool = False
    # (repetition theta, frequency alpha, presence beta) for engines with a device-side repeat
    # penalty; None leaves the engine's own default in place.
    repeat_penalty: tuple[float, float, float] | None = None
    # Runtime sampling controls for engines that take them; None keeps the engine's defaults.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_strings", tuple(str(s) for s in self.stop_strings if s))
        object.__setattr__(self, "stop_token_ids", tuple(int(t) for t in self.stop_token_ids))
        if int(self.n_predict) < -1:
            raise ValueError("n_predict must be -1 (unbounded) or >= 0")


@dataclass
class ServerTask:
    """One completion request: chat messages (formatted by ChatFormatter) or raw prompt tokens."""

    id: int
    messages: Sequence[Mapping[str, Any]] | None = None
    prompt_tokens: Sequence[int] | None = None
    params: TaskParams = field(default_factory=TaskParams)
    # Prompt-shaping options passed straight through to the chat template. Changing either one
    # changes the rendered prefix, so it invalidates the cross-turn prefix cache by construction.
    tools: Sequence[Mapping[str, Any]] | None = None
    enable_thinking: bool = False
    # Request hints this server served by ignoring them (reported in x-naivert-ignored).
    ignored: tuple[str, ...] = ()
    # The stop string this server injected itself (``disable_parallel_tool_use``), as opposed to one
    # the client asked for. Stopping on it is a *completed* tool call, not a client stop sequence.
    parallel_stop: str | None = None
    # Messages API ``metadata.user_id`` when the client sent one: the key under which an engine
    # with a prefix store parks this request's KV and looks for a parked prefix to resume.
    session_id: str | None = None
    timeline: RequestTimeline | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.messages is None) == (self.prompt_tokens is None):
            raise ValueError("ServerTask needs exactly one of messages / prompt_tokens")


@dataclass
class Timings:
    prompt_n: int = 0
    prompt_ms: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    capture_ms: float = 0.0
    decode_tok_s_device: float = 0.0
    n_prompt_tokens_cache: int = 0
    # How this turn's new prompt tokens were ingested ("native": the 4K-chunk executor, "verify8":
    # the replay that extends parked rows; "" when the engine does not say) and the draft KV ingest
    # that follows prefill before the first token, in ms.
    prefill_path: str = ""
    initialize_ms: float = 0.0
    # The engine's cost estimate behind the path choice for a resumed prefix (replay_ms / native_ms /
    # suffix_tokens / chunks), None for cold prompts and engines without the executor.
    prefill_estimate: dict | None = None
    # Sub-phase wall times from the engine (executor chunks, seq1 tail, sampling set-up, anchor sample,
    # draft ingest, round initialisation, launch), for tracing where time to the first token goes.
    prefill_phases: dict | None = None
    # Whether this turn's decode was actually driven by the device chain.  Not a constant:
    # a build whose device-launchable instantiation did not hold up serves the host-driven
    # form instead, and a per-turn number that silently means two different things is worse
    # than no number.  Kept next to decode_tok_s_device because it is what explains it.
    device_chain: bool = False
    # How many cache positions the chain wrote after the host stopped stepping -- non-zero
    # exactly when this turn ended on a host-side stop (stop string, cancel, disconnect)
    # while the chain was still live.  It is also the amount of decode work that was done
    # and thrown away, so it is worth seeing in the transcript rather than only in a log.
    chain_abandoned_n: int = 0
    # True when the device chain was still running at finish() and stopped because the
    # host asked it to.  Together with chain_abandoned_n this is how a turn says the
    # cancel path ran: the count is how far it got before the stop took effect.
    chain_canceled: bool = False
    # predicted_n counts tokens delivered to the client; device_steps counts tokens
    # generated on the GPU. A host-side stop can leave generated tokens undelivered.
    device_steps: int = 0
    # Set when finish()'s read-back failed on a turn no chain outlived.  That case does not
    # fail the turn -- the host's own count is already the whole truth about what was written
    # -- but it must not vanish either: it is the one signal that this turn's device_steps and
    # chain_* fields are missing rather than zero.
    reconcile_error: str | None = None
    # The device's own last word on why the chain stopped (kSeq1Bound*, 3 = canceled), read
    # from the record it published for its final commit -- the one a cancelled turn's host
    # never consumed.  Reported for the same reason reconcile_error is: a code the device
    # writes and only a test harness reads is a code nobody can act on.
    device_stop_bound: int | None = None
    # Per-segment decode statistics from engines that publish them: {"think": {...}, "answer":
    # {...}, "total": {...}} with tokens, rounds, tokens_per_round (the accepted length plus the
    # sampled token), device_ms and device_tok_s. "think" is present only when the generation
    # closed a </think> block.
    segments: dict[str, Any] | None = None

    @property
    def prompt_per_second(self) -> float:
        return self.prompt_n / (self.prompt_ms / 1e3) if self.prompt_ms > 0 else 0.0

    @property
    def predicted_per_second(self) -> float:
        return self.predicted_n / (self.predicted_ms / 1e3) if self.predicted_ms > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_n": self.prompt_n, "prompt_ms": self.prompt_ms, "prompt_per_second": self.prompt_per_second,
            "predicted_n": self.predicted_n, "predicted_ms": self.predicted_ms,
            "predicted_per_second": self.predicted_per_second, "capture_ms": self.capture_ms,
            "decode_tok_s_device": self.decode_tok_s_device, "n_prompt_tokens_cache": self.n_prompt_tokens_cache,
            "prefill_path": self.prefill_path, "initialize_ms": self.initialize_ms, "prefill_estimate": self.prefill_estimate,
            "prefill_phases": self.prefill_phases,
            "device_chain": self.device_chain, "chain_abandoned_n": self.chain_abandoned_n,
            "chain_canceled": self.chain_canceled, "device_steps": self.device_steps,
            "reconcile_error": self.reconcile_error, "device_stop_bound": self.device_stop_bound,
            "segments": self.segments,
        }


@dataclass(frozen=True)
class OpenedResult:
    """Admission and prefill succeeded; cache usage is now authoritative."""

    task_id: int
    prompt_n: int
    n_prompt_tokens_cache: int


@dataclass(frozen=True)
class PartialResult:
    """One streamed text chunk (llama.cpp ``server_task_result_cmpl_partial``)."""

    task_id: int
    index: int              # 0-based generated token index this chunk was produced at
    token: int
    text: str               # text safe to show now (UTF-8 complete, past any partial stop string)


@dataclass(frozen=True)
class PartialBatch:
    """Several ``PartialResult``s handed to the HTTP layer in one queue round trip.

    Only produced for non-streaming requests, where nothing is shown to the client before the
    final body is assembled, so the per-token queue hand-off (and the consumer-thread wake it
    causes) is pure cost.  The assembler still sees every partial, in order: this batches the
    *transport*, never the assembly.
    """

    task_id: int
    items: tuple[PartialResult, ...]


@dataclass(frozen=True)
class FinalResult:
    task_id: int
    text: str
    tokens: tuple[int, ...]
    stop_reason: str
    truncated: bool
    timings: Timings
    stop_string: str | None = None
    n_prompt_tokens_cache: int = 0
    timeline: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop_reason {self.stop_reason!r}")
        object.__setattr__(self, "tokens", tuple(int(t) for t in self.tokens))


__all__ = ["FinalResult", "OpenedResult", "PartialBatch", "PartialResult", "STOP_REASONS", "ServerTask",
           "TaskParams", "Timings"]
