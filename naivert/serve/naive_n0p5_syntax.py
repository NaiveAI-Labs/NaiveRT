"""Naive N assistant-output syntax: incremental segmentation, tool-call parsing, argument coercion.

Pure CPU and dependency-free (no torch, no tokenizer). The wire format is the checkpoint's own
chat template, which instructs the model to emit

    <tool_call>
    <function=NAME>
    <parameter=P>VALUE</parameter>
    </function>
    </tool_call>

and states that "the value enclosed between parameter tags is preserved exactly as-is, including
newlines and spaces". So a parameter value arrives as a **raw string**, never as JSON, and it is
this module -- not the model -- that turns it into the type the tool's ``input_schema`` declares.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Mapping, Sequence

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
DELIMITERS: dict[str, tuple[str, str]] = {
    "text": ("", ""),
    "thinking": (THINK_OPEN, THINK_CLOSE),
    "tool_call": (TOOL_OPEN, TOOL_CLOSE),
}
CALL_SEPARATOR = "\n"          # what the model writes between two calls of one turn
CALLS_AFTER_THINK = "\n\n"    # and between </think> and the first call when it says nothing else


def render_tool_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    """The generated form of ``calls`` (``{"name", "arguments"}`` each), byte for byte.

    The chat template's ``tool_calls`` branch renders the same block but joins calls with nothing
    and prints non-string values through Jinja (``True`` for a boolean the model wrote as ``true``).
    Replayed history is rendered with this instead, so the prompt repeats the tokens the model
    generated and the parked prefix of the previous request resumes (values that are not strings
    are JSON literals, which is how the model writes them).
    """
    out = []
    for call in calls:
        params = "".join(f"<parameter={k}>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</parameter>\n"
                         for k, v in (call.get("arguments") or {}).items())
        out.append(f"{TOOL_OPEN}\n<function={call['name']}>\n{params}</function>\n{TOOL_CLOSE}")
    return CALL_SEPARATOR.join(out)

Segment = tuple[str, str, bool]          # (kind, text, closed)


class ToolCallParseError(ValueError):
    """A ``<tool_call>`` block does not match the template's format, or a value defies its schema.

    Never downgraded to "assume it was text": the model's behaviour already diverged, and hiding
    that leaves the client with a plausible-looking answer and no way to notice.
    """


def _prefix_hold(buffer: str, markers: Sequence[str]) -> int:
    """Length of the buffer's tail that could still grow into one of ``markers``.

    Without this, ``"<tool"`` would be emitted as ordinary text and the client would see a
    fragment of a tag in its ``text`` block moments before the real tool call arrives.
    """
    longest = 0
    for marker in markers:
        for size in range(min(len(marker) - 1, len(buffer)), 0, -1):
            if buffer.endswith(marker[:size]):
                longest = max(longest, size)
                break
    return longest


class SegmentStream:
    """Feed generated text in arbitrary increments; get ``(kind, text, closed)`` segments out.

    ``kind`` is ``text`` / ``thinking`` / ``tool_call``; ``text`` is the segment's *inner* text
    (delimiters stripped, recoverable through :func:`rebuild`); ``closed`` says the closing
    delimiter was actually seen. A segment may be reported over several calls: each call carries
    the newly available text, and only the last one for that segment has ``closed=True``.

    Thinking is recognised **only at the very front** of the reply (§5.2): the template opens the
    assistant turn with ``<think>`` when thinking is on, and a later ``<think>`` is just text.
    """

    def __init__(self, *, thinking: bool = True, tools: bool = True) -> None:
        """``thinking`` / ``tools`` say which delimiters this reply may contain at all.

        A request that declared no tools cannot receive a tool call, so ``<tool_call>`` in its reply
        is the model talking *about* the tag and must stay text -- splitting it would turn ordinary
        prose into a parse error. Same for ``<think>`` when thinking is off.
        """
        self._buffer = ""
        self._kind = "text"
        self._finished = False
        self._open = False                # a segment has had parts emitted but not been closed
        self._allow_tools = bool(tools)
        self._thinking_allowed = bool(thinking)   # and nothing but whitespace emitted so far

    @property
    def kind(self) -> str:
        return self._kind

    def feed(self, text: str) -> list[Segment]:
        if self._finished:
            raise RuntimeError("SegmentStream already finished")
        self._buffer += text
        return list(self._drain(final=False))

    def finish(self) -> list[Segment]:
        """Flush. A trailing ``thinking`` / ``tool_call`` segment is reported with ``closed=False``:
        the model stopped mid-block, which the caller must treat as malformed rather than complete."""
        if self._finished:
            return []
        self._finished = True
        return list(self._drain(final=True))

    def _openers(self) -> list[str]:
        openers = [TOOL_OPEN] if self._allow_tools else []
        return openers + [THINK_OPEN] if self._thinking_allowed else openers

    def _drain(self, *, final: bool) -> Iterator[Segment]:
        while True:
            if self._kind == "text":
                openers = self._openers()
                hits = [(self._buffer.find(o), o) for o in openers if self._buffer.find(o) >= 0]
                if hits:
                    index, opener = min(hits)
                    prefix = self._buffer[:index]
                    if opener == THINK_OPEN and prefix.strip(): 
                        # Thinking is only thinking at the very front. Deciding this from `prefix`
                        # rather than from what earlier feed() calls happened to flush is what makes
                        # the gate chunk-invariant: fed whole or byte by byte, "a<think>x</think>b"
                        # is one text segment either way.
                        self._thinking_allowed = False
                        continue
                    if prefix or self._open:
                        # Close the text segment even when this chunk added nothing to it: a caller
                        # merging the parts must see exactly one `closed` per segment.
                        yield ("text", prefix, True)
                        self._open = False
                    self._buffer = self._buffer[index + len(opener):]
                    self._kind = "thinking" if opener == THINK_OPEN else "tool_call"
                    self._thinking_allowed = False
                    continue
                hold = 0 if final else _prefix_hold(self._buffer, openers)
                cut = len(self._buffer) - hold
                ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
                if ready or (final and self._open):
                    yield ("text", ready, final)
                    self._open = not final
                    self._note_emitted(ready)
                return
            closer = DELIMITERS[self._kind][1]
            index = self._buffer.find(closer)
            if index >= 0:
                yield (self._kind, self._buffer[:index], True)
                self._open = False
                self._buffer = self._buffer[index + len(closer):]
                self._kind = "text"
                continue
            hold = 0 if final else _prefix_hold(self._buffer, [closer])
            cut = len(self._buffer) - hold
            ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
            if final:
                # Truncated block: report what there is, never as `closed`. ``self.kind`` still
                # names the block the model was inside when it stopped.
                # Always yield, even when the block is empty: "a<tool_call>" (the model hit
                # max_tokens right after the opener) must not disappear. rebuild() stays total,
                # and the Messages mapper can reject the unclosed tool_call segment.
                yield (self._kind, ready + self._buffer, False)
                self._buffer = ""
                self._open = True
            elif ready:
                yield (self._kind, ready, False)
                self._open = True
            return

    def _note_emitted(self, text: str) -> None:
        if text.strip():
            self._thinking_allowed = False


def rebuild(segments: Sequence[Segment]) -> str:
    """Reassemble the original generated text, so segmentation is provably lossless (T0.1)."""
    out: list[str] = []
    previous_kind: str | None = None
    for kind, text, closed in segments:
        opener, closer = DELIMITERS[kind]
        if kind != previous_kind:
            out.append(opener)
        out.append(text)
        if closed:
            out.append(closer)
            previous_kind = None
        else:
            previous_kind = kind
    return "".join(out)


_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_call(raw: str) -> tuple[str, dict[str, str]]:
    """Inner text of one ``<tool_call>`` block -> ``(function name, {parameter: raw value})``.

    Values keep every byte between the parameter tags, newlines and spaces included, exactly as
    the template promises the model.
    """
    matches = list(_FUNCTION.finditer(raw))
    if not matches:
        raise ToolCallParseError("tool call has no complete <function=NAME>...</function> block")
    if len(matches) > 1:
        # The template mandates one function per <tool_call>. Two means the model diverged from
        # its own format; returning the first and dropping the rest would lose a real call.
        raise ToolCallParseError(f"tool call has {len(matches)} <function=...> blocks, expected 1")
    match = matches[0]
    outside = raw[:match.start()] + raw[match.end():]
    if outside.strip():
        raise ToolCallParseError(
            f"tool call has text outside the <function> block: {outside.strip()[:80]!r}")
    name, body = match.group(1), match.group(2)
    arguments: dict[str, str] = {}
    consumed: list[tuple[int, int]] = []
    for parameter in _PARAMETER.finditer(body):
        key = parameter.group(1)
        if key in arguments:
            raise ToolCallParseError(f"tool call {name!r} repeats parameter {key!r}")
        arguments[key] = parameter.group(2)
        consumed.append(parameter.span())
    leftover = body
    for start, end in reversed(consumed):
        leftover = leftover[:start] + leftover[end:]
    if leftover.strip():
        raise ToolCallParseError(f"tool call {name!r} has unparsable text between parameters: {leftover.strip()[:80]!r}")
    return name, arguments


# What each JSON-Schema type accepts once the raw string has gone through json.loads. "string"
# is deliberately absent: a declared string keeps its bytes, so it never reaches the parser.
_EXPECTED: dict[str, tuple[type, ...]] = {
    "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,), "null": (type(None),),
}
_PYTHON_BOOLEANS = {"True": True, "False": False}
_SCHEMA_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})
_ENUM_TYPES = {str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}


def property_kinds(declared: Any) -> list[str] | None:
    """The JSON-Schema types a property admits, or None when it declares none.

    ``type`` (a name or a list), the union of ``anyOf`` / ``oneOf`` branches (each resolved the same
    way), or the value types of an ``enum``. A property that carries one of those keys but does not
    resolve (an unknown type name, a branch without a type) also yields None; the caller decides
    whether an untyped property is acceptable."""
    if not isinstance(declared, Mapping):
        return None
    kinds = declared.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if isinstance(kinds, (list, tuple)) and kinds and all(isinstance(k, str) and k in _SCHEMA_TYPES for k in kinds):
        return list(kinds)
    for key in ("anyOf", "oneOf"):
        branches = declared.get(key)
        if isinstance(branches, list) and branches:
            out: list[str] = []
            for branch in branches:
                found = property_kinds(branch)
                if not found:
                    return None
                out.extend(k for k in found if k not in out)
            return out
    values = declared.get("enum")
    if isinstance(values, list) and values:
        out = []
        for value in values:
            kind = _ENUM_TYPES.get(type(value))
            if kind is None:
                return None
            if kind not in out:
                out.append(kind)
        return out
    return None


def declares_type(declared: Any) -> bool:
    """Whether a property carries any typing key at all (``type``, ``anyOf``, ``oneOf``, ``enum``)."""
    return isinstance(declared, Mapping) and any(k in declared for k in ("type", "anyOf", "oneOf", "enum"))


def _accepts(value: Any, kind: str) -> bool:
    # bool is a subclass of int, so "true" must not slip through as an integer *or* a number.
    if isinstance(value, bool) != (kind == "boolean"):
        return False
    return isinstance(value, _EXPECTED[kind])


def coerce_arguments(arguments: Mapping[str, str], schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """Raw parameter strings -> the types ``input_schema`` declares.

    ``string`` and keys the schema does not declare stay verbatim: guessing there would silently
    turn a user's ``"3"`` into ``3``. Everything else goes through JSON, and a value that will not
    parse -- or parses to the wrong type -- raises instead of degrading.
    """
    properties = {}
    if isinstance(schema, Mapping) and isinstance(schema.get("properties"), Mapping):
        properties = schema["properties"]
    out: dict[str, Any] = {}
    for key, raw in arguments.items():
        declared = properties.get(key) if isinstance(properties.get(key), Mapping) else None
        kinds = property_kinds(declared) or []             # a union, e.g. ["integer", "null"], anyOf or enum included
        if not kinds or any(k not in _EXPECTED for k in kinds):
            # No type, a declared "string", or a keyword we do not convert (including a union that
            # admits a string): the bytes stand as they are rather than being guessed at.
            out[key] = raw
            continue
        shown = "/".join(kinds)
        text = raw.strip()
        # The chat template renders Python booleans through `| string`, so a value that came back
        # out of our own rendered history reads "True"/"False" rather than JSON's "true"/"false".
        if "boolean" in kinds and text in _PYTHON_BOOLEANS:
            out[key] = _PYTHON_BOOLEANS[text]
            continue
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ToolCallParseError(f"parameter {key!r} declared {shown!r} is not valid JSON: {raw!r}") from exc
        if not any(_accepts(value, k) for k in kinds):
            raise ToolCallParseError(f"parameter {key!r} declared {shown!r} parsed as {type(value).__name__}: {raw!r}")
        out[key] = value
    return out


__all__ = ["DELIMITERS", "Segment", "SegmentStream", "ToolCallParseError", "coerce_arguments", "declares_type",
           "parse_tool_call", "property_kinds", "rebuild"]
