"""Token-list helpers and stop detection (llama.cpp ``server_tokens``, vLLM ``check_stop_strings``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class ServerTokens:
    """The token sequence a slot holds: the last full prompt + generation (used for ``n_past``)."""

    def __init__(self, tokens: Iterable[int] = ()) -> None:
        self._tokens: list[int] = [int(t) for t in tokens]

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, index):
        return self._tokens[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ServerTokens):
            return self._tokens == other._tokens
        if isinstance(other, (list, tuple)):
            return self._tokens == [int(t) for t in other]
        return NotImplemented

    def __repr__(self) -> str:
        return f"ServerTokens({self._tokens!r})"

    def as_list(self) -> list[int]:
        return list(self._tokens)

    def push_back(self, token: int) -> None:
        self._tokens.append(int(token))

    def extend(self, tokens: Iterable[int]) -> None:
        self._tokens.extend(int(t) for t in tokens)

    def clear(self) -> None:
        self._tokens.clear()

    def keep_first(self, n: int) -> None:
        """Drop everything after the first ``n`` tokens (llama.cpp ``keep_first``)."""
        n = int(n)
        if n < 0:
            raise ValueError("keep_first needs n >= 0")
        del self._tokens[n:]

    def get_common_prefix(self, other: Sequence[int] | "ServerTokens") -> int:
        """Length of the longest common prefix with ``other``."""
        n = 0
        for a, b in zip(self._tokens, other):
            if a != int(b):
                break
            n += 1
        return n

    def validate(self, vocab_size: int) -> bool:
        """Every token is inside ``[0, vocab_size)``."""
        vocab_size = int(vocab_size)
        return all(0 <= t < vocab_size for t in self._tokens)


@dataclass(frozen=True)
class StopMatch:
    """A stop string found in the generated text."""

    stop: str
    start: int          # index in output_text where the stop string starts
    end: int            # index just past the stop string

    def truncate_to(self, include_in_output: bool) -> int:
        return self.end if include_in_output else self.start


def check_stop_strings(output_text: str, new_char_count: int, stop: Sequence[str],
                       include_in_output: bool = False) -> StopMatch | None:
    """Find the stop string that completes earliest in the newly added characters.

    Semantics (vLLM ``check_stop_strings`` table, made list-order independent): only matches that
    end inside the last ``new_char_count`` characters count; among them the one completing earliest
    wins, ties go to the longest stop string (the one starting earliest). ``None`` when the stop
    list is empty, no new characters were added, or nothing matched. ``include_in_output`` only
    changes where ``truncate_to`` points; the match itself is the same.
    """
    del include_in_output  # kept for signature parity; see StopMatch.truncate_to
    if not new_char_count or not stop:
        return None
    best: StopMatch | None = None
    for stop_str in stop:
        if not stop_str:
            continue
        window_start = max(0, len(output_text) - int(new_char_count) - len(stop_str) + 1)
        index = output_text.find(stop_str, window_start)
        if index == -1:
            continue
        end = index + len(stop_str)
        if best is None or end < best.end or (end == best.end and len(stop_str) > len(best.stop)):
            best = StopMatch(stop=stop_str, start=index, end=end)
    return best


class StopMatcher:
    """Stateful stop-string scanner over an incrementally growing output text."""

    def __init__(self, stop: Sequence[str] = (), *, include_in_output: bool = False) -> None:
        self.stop = tuple(s for s in stop if s)
        self.include_in_output = bool(include_in_output)
        self.text = ""
        self.match: StopMatch | None = None

    @property
    def max_stop_len(self) -> int:
        return max((len(s) for s in self.stop), default=0)

    def reset(self) -> None:
        self.text = ""
        self.match = None

    def feed(self, new_text: str) -> StopMatch | None:
        """Append ``new_text``; return the stop match once one completes (and remember it)."""
        if self.match is not None:
            return self.match
        self.text += new_text
        self.match = check_stop_strings(self.text, len(new_text), self.stop, self.include_in_output)
        return self.match

    def holdback(self) -> int:
        """Characters at the end of ``text`` that could still be the start of a stop string.

        Streaming callers should not send these yet (llama.cpp ``find_partial_stop``).
        """
        if not self.stop or self.match is not None:
            return 0
        best = 0
        for stop_str in self.stop:
            for k in range(min(len(stop_str) - 1, len(self.text)), 0, -1):
                if self.text.endswith(stop_str[:k]):
                    best = max(best, k)
                    break
        return best


class StopTokenSequenceMatcher:
    """Detect a stop *token* sequence with partial-match backtracking (``advance`` per token)."""

    def __init__(self, sequence: Sequence[int]) -> None:
        self.sequence = tuple(int(t) for t in sequence)
        if not self.sequence:
            raise ValueError("stop token sequence must not be empty")
        self._buffer: list[int] = []

    def reset(self) -> None:
        self._buffer.clear()

    def advance(self, token: int) -> bool:
        """Feed one token; True exactly when the buffer now ends with the stop sequence."""
        self._buffer.append(int(token))
        if len(self._buffer) > len(self.sequence):
            del self._buffer[: len(self._buffer) - len(self.sequence)]
        return tuple(self._buffer) == self.sequence


__all__ = ["ServerTokens", "StopMatch", "StopMatcher", "StopTokenSequenceMatcher", "check_stop_strings"]
