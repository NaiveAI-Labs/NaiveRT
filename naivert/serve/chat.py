"""Chat template formatting and UTF-8-safe incremental detokenization.

Both wrap the checkpoint's Hugging Face tokenizer (``transformers`` is imported lazily so the
package stays importable without it). ``ChatFormatter`` renders ``messages`` with the model's
own chat template (thinking off by default: the assistant turn starts with ``<think></think>``,
unless thinking is explicitly enabled); ``IncrementalDetokenizer`` is the vLLM/transformers
``detokenize_incrementally`` scheme: a token whose bytes end mid-codepoint decodes to U+FFFD, so
its text is held back until the codepoint completes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

REPLACEMENT = "�"


def load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


class ChatFormatter:
    """messages -> prompt token ids through the checkpoint's chat template."""

    def __init__(self, tokenizer: Any, *, enable_thinking: bool = False) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = bool(enable_thinking)

    @classmethod
    def from_path(cls, path: str | Path, *, enable_thinking: bool = False) -> "ChatFormatter":
        return cls(load_tokenizer(path), enable_thinking=enable_thinking)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def _kwargs(self, tools: Sequence[Mapping[str, Any]] | None,
                enable_thinking: bool | None) -> dict[str, Any]:
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        # Preserve earlier reasoning when thinking is enabled. Otherwise leave this option
        # to the checkpoint template; forcing it false could strip assistant-history tokens.
        kwargs: dict[str, Any] = {"enable_thinking": thinking}
        if thinking:
            kwargs["keep_all_reasoning"] = True
        if tools:
            kwargs["tools"] = [dict(tool) for tool in tools]
        return kwargs

    def render(self, messages: Sequence[Mapping[str, Any]], *,
               tools: Sequence[Mapping[str, Any]] | None = None,
               enable_thinking: bool | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, tokenize=False,
            **self._kwargs(tools, enable_thinking))

    def encode(self, messages: Sequence[Mapping[str, Any]], *,
               tools: Sequence[Mapping[str, Any]] | None = None,
               enable_thinking: bool | None = None) -> list[int]:
        out = self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, tokenize=True,
            **self._kwargs(tools, enable_thinking))
        if isinstance(out, Mapping) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        return [int(t) for t in out]

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)


class IncrementalDetokenizer:
    """Feed tokens one at a time; ``add`` returns the newly completed text (never a partial codepoint)."""

    def __init__(self, tokenizer: Any, *, skip_special_tokens: bool = False) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = bool(skip_special_tokens)
        self.reset()

    def reset(self) -> None:
        self.tokens: list[int] = []
        self.text = ""                # everything emitted so far
        self.last_segment = ""        # what the last add() returned
        self._prefix_offset = 0       # tokens before this are fully emitted and no longer decoded
        self._read_offset = 0         # tokens in [prefix, read) were emitted; [read, end) pending

    def _decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=self.skip_special_tokens,
                                     clean_up_tokenization_spaces=False)

    def add(self, token: int) -> str:
        self.tokens.append(int(token))
        prefix_text = self._decode(self.tokens[self._prefix_offset:self._read_offset])
        new_text = self._decode(self.tokens[self._prefix_offset:])
        if new_text.endswith(REPLACEMENT):
            # The last token's bytes do not finish a codepoint: wait for the next token.
            self.last_segment = ""
            return ""
        delta = new_text[len(prefix_text):]
        # Keep a short decoded prefix so byte-level merges across the boundary stay consistent.
        self._prefix_offset = max(self._read_offset, len(self.tokens) - 8)
        self._read_offset = len(self.tokens)
        self.text += delta
        self.last_segment = delta
        return delta

    def add_all(self, tokens: Sequence[int]) -> str:
        return "".join(self.add(t) for t in tokens)


__all__ = ["ChatFormatter", "IncrementalDetokenizer", "load_tokenizer"]
