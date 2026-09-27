"""Anthropic Messages validation and response mapping for text, thinking, and tool calls."""
from __future__ import annotations

import contextvars

import json
import logging
import secrets
import string
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from .naive_n0p5_syntax import (CALLS_AFTER_THINK, CALL_SEPARATOR, SegmentStream, ToolCallParseError, coerce_arguments, declares_type, parse_tool_call,
                          render_tool_calls,
                          property_kinds)
from .task import FinalResult, OpenedResult, PartialResult, ServerTask, TaskParams


@dataclass
class ApiError(Exception):
    status: int
    error_type: str
    warning_type: str
    message: str

    def to_body(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message,
                                          "warning_type": self.warning_type}}

    def headers(self) -> dict[str, str]:
        return {"x-naivert-warning": self.warning_type}


def bad(warning: str, message: str) -> ApiError:
    return ApiError(400, "invalid_request_error", warning, message)


# Hints a request may carry that this server serves by ignoring them (prompt-cache breakpoints, adaptive
# thinking, effort, context edits). They are collected per request and reported back in the
# ``x-naivert-ignored`` response header, so the client learns what was dropped without a refusal.
_IGNORED: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("naivert_ignored", default=None)


def _ignore(hint: str) -> None:
    hints = _IGNORED.get()
    if hints is not None:
        hints.add(hint)


def _collect_ignored():
    """Start a fresh per-request set; returns (token, set)."""
    hints: set[str] = set()
    return _IGNORED.set(hints), hints


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: StrictStr
    content: StrictStr | list[dict[str, Any]]


class CountTokensRequest(BaseModel):
    """Fields shared by ``/v1/messages`` and ``/v1/messages/count_tokens``.

    ``extra="forbid"``: an unknown field is a 400, never a silent drop. Fields we know about but
    cannot execute are declared here for either explicit rejection with a specific
    ``warning_type`` or acceptance as a hint reported in ``x-naivert-ignored``."""

    model_config = ConfigDict(extra="forbid")
    model: StrictStr
    messages: list[ChatMessage]
    system: StrictStr | list[dict[str, Any]] | None = None
    temperature: Any = None
    top_p: Any = None
    top_k: Any = None
    thinking: Any = None
    tools: Any = None
    tool_choice: Any = None
    metadata: Any = None
    output_config: Any = None
    service_tier: Any = None
    container: Any = None
    mcp_servers: Any = None
    betas: Any = None
    context_management: Any = None


class MessagesRequest(CountTokensRequest):
    max_tokens: StrictInt = Field(gt=0)
    # naivert extension: repeat penalty applied on the device to the target logits. repetition_penalty
    # is the multiplicative HF form (1 = off), frequency/presence the additive OpenAI form (0 = off).
    repetition_penalty: Any = None
    frequency_penalty: Any = None
    presence_penalty: Any = None
    # naivert extension: integer seed for this request's sampler (default: the server seed).
    seed: Any = None
    stream: StrictBool = False
    stop_sequences: list[StrictStr] = Field(default_factory=list)


def validation_error(exc: Exception) -> ApiError:
    """Pydantic rejection -> ApiError; unknown fields get their own warning naming each field."""
    unknown = []
    for detail in getattr(exc, "errors", lambda: [])():
        if detail.get("type") == "extra_forbidden" and detail.get("loc"):
            unknown.append(".".join(str(part) for part in detail["loc"]))
    if unknown:
        return bad("unknown_field", "unknown request field(s): " + ", ".join(sorted(set(unknown)))
                   + "; this server rejects unrecognised fields rather than silently ignoring them")
    return bad("schema", str(exc))


TEXT_BLOCK_KEYS = frozenset({"type", "text"})


# Optional block fields the SDK emits as `null` when it round-trips its own reply. A client builds
# the next turn's assistant message with `Message.model_dump()`, which writes every optional field
# including the ones the reply never had; refusing `citations: null` would break the SDK's own tool
# loop over a field nobody set. An explicit null is the same as absent (§5.4). A *non-null* value is
# still refused: this server produced neither, so it cannot honour either.
SDK_NULLABLE = {"citations": "citations_unsupported", "caller": "tool_caller_unsupported",
                "parsed_output": "structured_outputs_unsupported"}


def _without_sdk_nulls(block: Mapping[str, Any]) -> dict:
    """Drop SDK-null fields; a prompt-cache breakpoint on any block kind (text, tool_use, tool_result,
    thinking) is a hint this server honours implicitly, so it is dropped and reported (cache_control)."""
    out = {}
    for key, value in block.items():
        if key in SDK_NULLABLE:
            if value is None:
                continue
            raise bad(SDK_NULLABLE[key], f"{key} on a content block is unsupported")
        if key == "cache_control":
            _ignore("cache_control")
            continue
        out[key] = value
    return out


def text_of(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block = _without_sdk_nulls(block)
        if block.get("type") != "text":
            raise bad("unsupported_content", f"unsupported content block type: {block.get('type')!r}")
        if not isinstance(block.get("text"), str):
            raise bad("schema", "text block requires a string text field")
        if "cache_control" in block:
            # A prompt-cache breakpoint: this server reuses the previous turn's prefix implicitly, so the
            # hint is honoured by construction and dropped from the block.
            block = {k: v for k, v in block.items() if k != "cache_control"}
            _ignore("cache_control")
        extra = sorted(set(block) - TEXT_BLOCK_KEYS)
        if extra:
            raise bad("unknown_field", f"unknown text block field(s): {', '.join(extra)}")
        parts.append(block["text"])
    return "".join(parts)


def _check_model(req: CountTokensRequest, served_model: str) -> None:
    if req.model != served_model:
        raise ApiError(404, "not_found_error", "unknown_model", f"served model is {served_model!r}")


def _check_features(req: CountTokensRequest) -> None:
    # temperature and top_p are runtime controls of the device sampler (validated in _sampling_controls);
    # top_k has no device implementation.
    if "top_k" in req.model_fields_set and req.top_k is not None:
        raise bad("sampling_unsupported", f"top_k={req.top_k!r}: the device sampler implements top_p, not top_k")
    if "output_config" in req.model_fields_set and req.output_config is not None:
        cfg = req.output_config
        if not isinstance(cfg, dict):
            raise bad("schema", "output_config must be an object")
        unknown = sorted(set(cfg) - {"effort", "format"})
        if unknown:
            raise bad("unknown_field", f"unknown output_config field(s): {', '.join(unknown)}")
        if "format" in cfg:
            raise bad("structured_outputs_unsupported", "output_config.format requires constrained decoding, which "
                      "this runtime cannot do: the decode graph advances the token on device and never returns it "
                      "to the host, so no grammar mask can be applied before argmax")
        if "effort" in cfg:
            _ignore("output_config.effort")    # no effort control on the device sampler
    if "context_management" in req.model_fields_set and req.context_management is not None:
        cm = req.context_management
        edits = cm.get("edits") if isinstance(cm, dict) else None
        if not isinstance(edits, list) or any(not isinstance(e, dict) or not isinstance(e.get("type"), str) for e in edits):
            raise bad("schema", "context_management must be an object with an edits list of typed objects")
        # Server-side context edits (clear_thinking / clear_tool_uses) are not applied: the prompt is
        # formatted exactly as sent, which is what the client's own compaction assumes.
        _ignore("context_management")
    for field in ("service_tier", "container", "mcp_servers", "betas"):
        if field in req.model_fields_set:
            raise bad("feature_unsupported", f"{field} is unsupported")



TOOL_KEYS = frozenset({"name", "description", "input_schema", "type", "strict", "cache_control"})
# The types coerce_arguments knows how to convert. A property declaring anything else would be
# handed to the tool as a raw string with no signal, so the schema is refused instead.
SCHEMA_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})
PARALLEL_TOOL_STOP = "</tool_call>"


def _check_tool_schema(name: str, schema: Any) -> None:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise bad("invalid_tool_schema", f"tool {name!r}: input_schema must be a JSON Schema object "
                  'with "type": "object"')
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise bad("invalid_tool_schema", f"tool {name!r}: input_schema.properties must be an object")
    for key, declared in properties.items():
        if not isinstance(declared, dict):
            raise bad("invalid_tool_schema", f"tool {name!r} property {key!r}: must be a schema object")
        if property_kinds(declared) is not None:
            continue                                       # type, a union, anyOf/oneOf branches or an enum
        if declares_type(declared):
            raise bad("invalid_tool_schema", f"tool {name!r} property {key!r}: declare \"type\" as one of "
                      f"{sorted(SCHEMA_TYPES)} (or a list of them), an anyOf/oneOf of such types, or an enum. "
                      "The model emits every parameter as a raw string, so a property this server cannot "
                      "type would reach the tool untyped.")
        # No typing key at all: the tool accepts any value, so the raw string is a faithful value.
        # Served, and reported so the client knows the parameter arrives untyped.
        _ignore("untyped_tool_property")


def _check_tools(req: CountTokensRequest) -> tuple[list[dict] | None, tuple[str, ...]]:
    """-> (tools rendered into the prompt, extra stop strings). Both empty when there are none."""
    tools, choice = req.tools, req.tool_choice
    render = True
    if choice is not None:
        if not isinstance(choice, dict):
            raise bad("tool_choice_unsupported", "tool_choice must be an object")
        kind = choice.get("type")
        if kind in ("any", "tool"):
            raise bad("tool_choice_unsupported", f"tool_choice type {kind!r} forces a tool call, which needs "
                      "constrained decoding: this runtime advances the token on device and never returns it to "
                      "the host, so no grammar mask can be applied before argmax")
        if kind not in ("auto", "none"):
            raise bad("tool_choice_unsupported", f"unknown tool_choice type {kind!r}")
        if not tools:
            raise bad("tool_choice_unsupported", "tool_choice requires tools")
        render = kind != "none"
        extra = sorted(set(choice) - {"type", "disable_parallel_tool_use"})
        if extra:
            raise bad("unknown_field", f"unknown tool_choice field(s): {', '.join(extra)}")
    if tools is None or tools == []:
        # An empty array is "no tools", as on the Anthropic API (Claude Code's count_tokens calls for
        # subagent transcripts send one).
        return None, ()
    if not isinstance(tools, list):
        raise bad("invalid_tool_schema", "tools must be an array")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise bad("invalid_tool_schema", "each tool must be an object")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise bad("invalid_tool_schema", "each tool needs a non-empty string name")
        if tool.get("type") is not None:
            raise bad("unsupported_tool_type", f"tool {name!r} has type {tool['type']!r}: only custom tools "
                      "(name + input_schema) are served; server-side tool types are not")
        if "strict" in tool:
            raise bad("strict_tools_unsupported", f"tool {name!r} requests strict schema adherence, which needs "
                      "constrained decoding this runtime cannot do")
        if "cache_control" in tool:
            _ignore("cache_control")           # prefix reuse is implicit here; the breakpoint hint is dropped
        unknown = sorted(set(tool) - TOOL_KEYS)
        if unknown:
            raise bad("unknown_field", f"tool {name!r} has unknown field(s): {', '.join(unknown)}")
        description = tool.get("description")
        if description is not None and not isinstance(description, str):
            raise bad("invalid_tool_schema", f"tool {name!r}: description must be a string")
        _check_tool_schema(name, tool.get("input_schema"))
        converted.append({"name": name, "description": description or "",
                          "parameters": tool["input_schema"]})
    if len({t["name"] for t in converted}) != len(converted):
        raise bad("invalid_tool_schema", "tool names must be unique")
    stop: tuple[str, ...] = ()
    if isinstance(choice, dict) and choice.get("disable_parallel_tool_use"):
        # One call per reply: stop at the first closing tag, and keep it, so the block is complete
        # and the assembler can parse it (a truncated <tool_call> is an error, not a partial call).
        stop = (PARALLEL_TOOL_STOP,)
    return (converted if render else None), stop


def _check_thinking(req: CountTokensRequest, max_tokens: int | None) -> bool:
    thinking = req.thinking
    if thinking is None:
        return False
    if not isinstance(thinking, dict):
        raise bad("thinking_mode_unsupported", "thinking must be an object")
    kind = thinking.get("type")
    unknown = sorted(set(thinking) - {"type", "budget_tokens", "display"})
    if unknown:
        raise bad("unknown_field", f"unknown thinking field(s): {', '.join(unknown)}")
    if "display" in thinking:
        _ignore("thinking.display")            # how the client shows thinking; nothing to do server-side
    if kind == "disabled":
        return False
    if kind == "adaptive":
        # The model decides how much to think; here thinking is enabled and bounded by max_tokens.
        _ignore("thinking.adaptive")
        return True
    if kind != "enabled":
        raise bad("thinking_mode_unsupported", f"thinking type {kind!r} is unsupported; use "
                  '{"type": "enabled", "budget_tokens": N}')
    budget = thinking.get("budget_tokens")
    if type(budget) is not int:
        raise bad("invalid_thinking_budget", "thinking.budget_tokens must be an integer")
    if budget < 1024:
        raise bad("invalid_thinking_budget", f"budget_tokens={budget} is below the minimum of 1024")
    if max_tokens is not None and budget >= max_tokens:
        raise bad("invalid_thinking_budget", f"budget_tokens={budget} must be less than max_tokens={max_tokens}")
    # The budget is a floor on how much room thinking has, not a device-side cap: decode cannot be
    # interrupted at a token count, so it is validated and then bounded by max_tokens.
    return True


TOOL_RESULT_KEYS = frozenset({"type", "tool_use_id", "content", "is_error"})
TOOL_USE_KEYS = frozenset({"type", "id", "name", "input"})
THINKING_KEYS = frozenset({"type", "thinking", "signature"})


def _blocks_of(content: str | list[dict], role: str) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise bad("schema", f"{role} content must be a string or a list of blocks")
    return content


def _user_message(content: str | list[dict], available: dict[str, str]) -> list[dict]:
    """One API user message -> the tool results it carries (first, in order) then its text.

    The results come first because they answer the assistant turn that precedes them; the text, if
    any, is the new instruction that follows. §5.1.
    """
    results, text_blocks = [], []
    for block in _blocks_of(content, "user"):
        if not isinstance(block, dict):
            raise bad("schema", "content blocks must be objects")
        if block.get("type") != "tool_result":
            text_blocks.append(block)
            continue
        block = _without_sdk_nulls(block)
        unknown = sorted(set(block) - TOOL_RESULT_KEYS)
        if unknown:
            raise bad("unknown_field", f"unknown tool_result field(s): {', '.join(unknown)}")
        reference = block.get("tool_use_id")
        if not isinstance(reference, str) or reference not in available:
            raise bad("orphan_tool_result", f"tool_result references tool_use_id {reference!r}, which the "
                      "preceding assistant message did not produce")
        body = block.get("content", "")
        content = body if isinstance(body, str) else text_of(body)
        failed = block.get("is_error", False)
        if not isinstance(failed, bool):
            raise bad("schema", "tool_result.is_error must be a boolean")
        if failed:
            # The template has no slot for a failure flag, so it is rendered into the result the
            # model reads. Dropping it would tell the model the tool succeeded (group V).
            content = f"[tool error] {content}"
        results.append({"role": "tool", "content": content, "name": available[reference]})
    messages = list(results)
    if text_blocks or not results:
        # A user turn made only of tool results adds no user message: the results *are* the turn.
        messages.append({"role": "user", "content": text_of(text_blocks)})
    return messages


def _assistant_message(content: str | list[dict]) -> tuple[dict, dict[str, str]]:
    """-> (one message with tool_calls / reasoning_content, {tool_use id: name}) (§5.1)."""
    content_parts, reasoning, ids = [], [], {}
    previous_tool = False
    for block in _blocks_of(content, "assistant"):
        if not isinstance(block, dict):
            raise bad("schema", "content blocks must be objects")
        kind = block.get("type")
        if kind in ("thinking", "tool_use"):
            block = _without_sdk_nulls(block)
        if kind == "thinking":
            unknown = sorted(set(block) - THINKING_KEYS)
            if unknown:
                raise bad("unknown_field", f"unknown thinking block field(s): {', '.join(unknown)}")
            if not isinstance(block.get("thinking"), str):
                raise bad("schema", "thinking block requires a string thinking field")
            if block.get("signature"):
                # Empty or absent is this server's own block coming back (we sign nothing), so it
                # passes. A real signature is a verifiable artefact from a provider that does sign,
                # and honouring it is impossible here -- §3 says refuse rather than pretend, and
                # accepting it silently would be exactly the drop group V exists to forbid.
                raise bad("thinking_signature_unsupported",
                          "thinking.signature cannot be verified or reproduced by this server; "
                          "resend the block without it (the reasoning text is kept either way)")
            reasoning.append(block["thinking"])
        elif kind == "tool_use":
            block = _without_sdk_nulls(block)
            unknown = sorted(set(block) - TOOL_USE_KEYS)
            if unknown:
                raise bad("unknown_field", f"unknown tool_use field(s): {', '.join(unknown)}")
            name, arguments, reference = block.get("name"), block.get("input"), block.get("id")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                raise bad("schema", "tool_use requires a string name and an object input")
            if not isinstance(reference, str) or not reference:
                raise bad("schema", "tool_use requires a string id")
            if not content_parts:
                content_parts.append(CALLS_AFTER_THINK)
            elif previous_tool:
                content_parts.append(CALL_SEPARATOR)
            content_parts.append(render_tool_calls([{"name": name, "arguments": arguments}]))
            previous_tool = True
            ids[reference] = name
        else:
            text = text_of([block])
            if text:
                content_parts.append(text)
                previous_tool = False
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    # Preserve text/tool order and explicit separators. Missing separators use the
    # historical defaults; HTTP replay can restore the exact observed blank blocks.
    return message, ids


def _build_messages(req: CountTokensRequest) -> list[dict]:
    # A `system` role inside `messages` is rendered in place as a system turn (the chat template
    # accepts one at any position). Claude Code sends its environment block that way after the first
    # user turn and a running `<total_tokens>` note after every tool result; folding them into the
    # head system turn, as this server once did, moved per-turn content into the prefix, so every
    # request re-prefilled the whole prompt instead of resuming the parked session.
    turns = [m for m in req.messages if m.role != "system"]
    if not turns or any(m.role != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(turns)):
        raise bad("invalid_messages", "messages must start with user and alternate user/assistant")
    if turns[-1].role == "assistant":
        raise bad("assistant_prefill_unsupported", "last message must be user; assistant prefill is unsupported")
    messages: list[dict] = []
    if req.system is not None:
        messages.append({"role": "system", "content": text_of(req.system)})
    available: dict[str, str] = {}
    for message in req.messages:
        if message.role == "system":
            messages.append({"role": "system", "content": text_of(message.content)})
        elif message.role == "assistant":
            built, available = _assistant_message(message.content)
            messages.append(built)
        else:
            messages.extend(_user_message(message.content, available))
            available = {}          # a tool_result may only answer the assistant turn just before it
    if req.metadata is not None:
        # The one field this server reads but does not act on: `user_id` is noted, anything else in
        # `metadata` is ignored by design (there is no request routing to apply it to). Documented in
        # docs/messages_http.md so it is not a silent drop in the sense group V forbids.
        # Never log the object itself: it is caller-supplied and may carry end-user identifiers.
        has_user_id = isinstance(req.metadata, dict) and "user_id" in req.metadata
        logging.getLogger(__name__).info("metadata present (user_id=%s)", "yes" if has_user_id else "no")
    return messages


def _sampling_controls(req: "MessagesRequest") -> tuple[float | None, float | None, int | None]:
    temperature = top_p = seed = None
    if req.temperature is not None:
        if isinstance(req.temperature, bool) or not isinstance(req.temperature, (int, float)) or not 0.0 < req.temperature <= 5.0:
            raise bad("invalid_temperature", f"temperature={req.temperature!r} must be a number in (0, 5]")
        temperature = float(req.temperature)
    if req.top_p is not None:
        if isinstance(req.top_p, bool) or not isinstance(req.top_p, (int, float)) or not 0.0 < req.top_p <= 1.0:
            raise bad("invalid_top_p", f"top_p={req.top_p!r} must be a number in (0, 1]")
        top_p = float(req.top_p)
    if req.seed is not None:
        if isinstance(req.seed, bool) or not isinstance(req.seed, int) or not 0 <= req.seed < 2**63:
            raise bad("invalid_seed", f"seed={req.seed!r} must be a non-negative 63-bit integer")
        seed = int(req.seed)
    return temperature, top_p, seed


def _check_repeat_penalty(req: "MessagesRequest"):
    values = []
    for field, default, low, high in (("repetition_penalty", 1.0, 0.0, 10.0), ("frequency_penalty", 0.0, -10.0, 10.0),
                                      ("presence_penalty", 0.0, -10.0, 10.0)):
        value = getattr(req, field)
        if value is None:
            values.append(default)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (low < value <= high if field == "repetition_penalty" else low <= value <= high):
            raise bad("invalid_repeat_penalty", f"{field}={value!r} must be a number in ({low}, {high}]")
        values.append(float(value))
    return None if values == [1.0, 0.0, 0.0] and all(getattr(req, f) is None for f in ("repetition_penalty", "frequency_penalty", "presence_penalty")) else tuple(values)


def validate_count_tokens(req: CountTokensRequest, *, served_model: str) -> ServerTask | ApiError:
    """Same rejections as ``/v1/messages`` (A0.6), minus the generation-only fields.

    Returns a task, not just messages: the count has to be the count of the prompt the *same*
    request would produce, so `tools` and `enable_thinking` must reach the formatter too (A0.5).
    """
    token, ignored = _collect_ignored()
    try:
        _check_model(req, served_model)
        _check_features(req)
        tools, _ = _check_tools(req)
        enable_thinking = _check_thinking(req, None)   # no max_tokens to bound the budget against
        return ServerTask(id=0, messages=_build_messages(req), tools=tools,
                          enable_thinking=enable_thinking, ignored=tuple(sorted(ignored)))
    except ApiError as exc:
        return exc
    finally:
        _IGNORED.reset(token)


def validate_request(req: MessagesRequest, *, max_decode_tokens: int, served_model: str) -> ServerTask | ApiError:
    token, ignored = _collect_ignored()
    try:
        _check_model(req, served_model)
        if req.max_tokens > max_decode_tokens:
            raise bad("max_tokens_exceeds_capture", f"max_tokens={req.max_tokens} exceeds {max_decode_tokens}; "
                      "restart the server with a larger --max-decode-tokens (and --output-capacity, which caps it)")
        _check_features(req)
        if len(req.stop_sequences) > 8 or any(not s for s in req.stop_sequences):
            raise bad("invalid_stop_sequences", "stop_sequences must contain at most 8 nonempty strings")
        tools, extra_stop = _check_tools(req)
        enable_thinking = _check_thinking(req, req.max_tokens)
        messages = _build_messages(req)
        stop_strings = tuple(req.stop_sequences) + extra_stop
        session_id = req.metadata.get("user_id") if isinstance(req.metadata, dict) else None
        repeat = _check_repeat_penalty(req)
        temperature, top_p, seed = _sampling_controls(req)
        return ServerTask(id=0, messages=messages, tools=tools, enable_thinking=enable_thinking,
                          parallel_stop=extra_stop[0] if extra_stop else None,
                          session_id=str(session_id) if session_id is not None else None,
                          params=TaskParams(cache_prompt=True, n_predict=req.max_tokens,
                                            stop_strings=stop_strings, repeat_penalty=repeat,
                                            temperature=temperature, top_p=top_p, seed=seed,
                                            # `</tool_call>` must stay in the output or the block the
                                            # assembler parses would be missing its closing tag.
                                            include_stop_in_output=bool(extra_stop), stop_token_ids=()),
                          ignored=tuple(sorted(ignored)))
    except ApiError as exc:
        return exc
    finally:
        _IGNORED.reset(token)


def message_id() -> str:
    return "msg_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(22))


def runtime_error(exc: BaseException) -> ApiError:
    return ApiError(500, "api_error", "runtime_error", f"{type(exc).__name__}: {exc}")


def stop_reason_of(final: FinalResult) -> str | None:
    if final.stop_reason == "cancel":
        return None
    if final.stop_reason == "error":
        raise runtime_error(RuntimeError("slot entered ERROR"))
    return {"eos": "end_turn", "stop_string": "stop_sequence", "length": "max_tokens"}[final.stop_reason]


def message_base(prompt_n: int, cache: int, *, model: str, id: str) -> dict:
    # Anthropic's definition: input_tokens counts only prompt tokens neither read from nor written
    # to the cache, so input + cache_read + cache_creation is the prompt length. Clients sum the
    # three (Claude Code does, for its context window and auto-compaction), and reporting the
    # whole prompt in input_tokens as well made them see twice the real context. Every token this
    # server prefills is parked (cache_creation), and the resumed prefix is cache_read.
    return {"id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": cache,
                      "cache_creation_input_tokens": prompt_n - cache}}


def final_extensions(final: FinalResult) -> dict:
    extension = {"timings": final.timings.as_dict()}
    if final.timeline is not None:
        extension["timeline"] = final.timeline
    return extension


class ResponseAssembler:
    """Generated text -> content blocks, as **one state with two projections** (§L3).

    Before tools the block structure was a compile-time constant (always one ``text`` block at index
    0), so streaming and non-streaming could each hard-code it. Now the blocks are discovered as the
    text arrives -- a ``tool_use`` block is not even well-formed until ``</tool_call>`` closes it --
    so the two paths must agree by construction rather than by inspection: ``feed``/``finish`` give
    the SSE projection, ``message`` gives the non-streaming one, and both read the same ``blocks``.

    Lives on the event loop side only (§L1): it raises, and those raises must happen after the
    worker's generator is closed and the slot is back to IDLE.
    """

    def __init__(self, *, model: str, id: str, tools: Sequence[Mapping[str, Any]] | None = None,
                 enable_thinking: bool = False, parallel_stop: str | None = None) -> None:
        self.model, self.id, self.enable_thinking = model, id, bool(enable_thinking)
        self.parallel_stop = parallel_stop
        self.schemas = {t["name"]: t.get("parameters") for t in (tools or [])}
        self.blocks: list[dict] = []
        self.thinking_tokens = 0
        # A reply can only contain what the request allowed: no tools declared means no tool call
        # can be parsed out of it, and thinking off means a <think> the model typed is just text.
        self._stream = SegmentStream(thinking=self.enable_thinking, tools=bool(self.schemas))
        self._pending = ""            # raw text of the tool_call block currently open
        self._in_tool = False         # a <tool_call> was opened and has not closed yet
        self._finished = False        # finish() is idempotent: both projections may ask for it
        self._open: int | None = None  # index of the block open for deltas, if any

    # -- projection 1: SSE ------------------------------------------------------------------
    def start_events(self, opened: OpenedResult):
        yield "message_start", {"type": "message_start", "message": message_base(
            opened.prompt_n, opened.n_prompt_tokens_cache, model=self.model, id=self.id)}

    def feed(self, partial: PartialResult):
        """One decoded token in; the events it completes out."""
        was = self._stream.kind
        parts = self._stream.feed(partial.text)
        if was == "thinking" or self._stream.kind == "thinking" or any(k == "thinking" for k, _, _ in parts):
            self.thinking_tokens += 1
        yield from self._consume(parts, final=False)

    def finish(self):
        if self._finished:
            return                     # the same state may be projected twice; assemble it once
        self._finished = True
        yield from self._consume(self._stream.finish(), final=True)
        if not self.blocks:
            # Never return zero blocks: an empty reply is one empty text block, which is what the
            # pre-tools implementation emitted and what clients index into.
            yield from self._start_block({"type": "text", "text": ""})
            yield from self._stop_block()

    def final_events(self, final: FinalResult):
        reason = self.stop_reason(final)
        if reason is None:
            return
        yield from self.finish()
        yield "message_delta", {"type": "message_delta",
                                "delta": {"stop_reason": reason,
                                          "stop_sequence": self.stop_sequence(final, reason)},
                                "usage": self._usage(final), "naivert": final_extensions(final)}
        yield "message_stop", {"type": "message_stop"}

    # -- projection 2: one JSON body --------------------------------------------------------
    def message(self, final: FinalResult, prompt_n: int) -> dict | None:
        reason = self.stop_reason(final)
        if reason is None:
            return None
        for _ in self.finish():
            pass                       # same state, events discarded
        msg = message_base(prompt_n, final.n_prompt_tokens_cache, model=self.model, id=self.id)
        msg.update(content=self.blocks, stop_reason=reason,
                   stop_sequence=self.stop_sequence(final, reason),
                   naivert=final_extensions(final))
        msg["usage"].update(self._usage(final), cache_creation_input_tokens=final.timings.prompt_n)
        return msg

    # -- shared state -----------------------------------------------------------------------
    def stop_reason(self, final: FinalResult) -> str | None:
        reason = stop_reason_of(final)
        if not any(b["type"] == "tool_use" for b in self.blocks):
            return reason
        if reason == "end_turn":
            return "tool_use"          # §5.2: the turn ended because the model called a tool
        if reason == "stop_sequence" and self.parallel_stop is not None \
                and final.stop_string == self.parallel_stop:
            # `disable_parallel_tool_use` makes us stop the decode at the first `</tool_call>`. That
            # is the call *completing*, not a client stop sequence firing, and reporting it as
            # `stop_sequence` would make every SDK tool loop -- which branches on `tool_use` -- walk
            # straight past a call it asked for. §5.1's stop string must not undo §5.2's stop reason.
            return "tool_use"
        return reason

    def stop_sequence(self, final: FinalResult, reason: str | None) -> str | None:
        """Only a real `stop_sequence` stop carries one; the injected stop is not the client's."""
        return final.stop_string if reason == "stop_sequence" else None

    def _usage(self, final: FinalResult) -> dict:
        usage: dict[str, Any] = {"output_tokens": final.timings.predicted_n}
        if self.enable_thinking:
            usage["output_tokens_details"] = {"thinking_tokens": self.thinking_tokens}
        return usage

    def _start_block(self, block: dict):
        self._open = len(self.blocks)
        self.blocks.append(block)
        if block["type"] == "tool_use":
            opening = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
        else:
            # The block starts empty and is filled by deltas; `blocks` already holds the full text.
            opening = {"type": block["type"], "text" if block["type"] == "text" else "thinking": ""}
        yield "content_block_start", {"type": "content_block_start", "index": self._open,
                                      "content_block": opening}

    def _stop_block(self):
        yield "content_block_stop", {"type": "content_block_stop", "index": self._open}
        self._open = None

    def _consume(self, parts, *, final: bool):
        for kind, text, closed in parts:
            if kind == "tool_call":
                self._pending += text
                self._in_tool = not closed
                if not closed:
                    continue
                yield from self._tool_use(self._pending)
                self._pending = ""
                continue
            field = "text" if kind == "text" else "thinking"
            if self._open is None:
                if not text:
                    continue           # an empty text run between two blocks is not a block
                yield from self._start_block({"type": kind, field: text})
            else:
                self.blocks[self._open][field] += text
            if text:
                yield ("content_block_delta",
                       {"type": "content_block_delta", "index": self._open,
                        "delta": {"type": f"{field}_delta", field: text}})
            if closed and self._open is not None:
                yield from self._stop_block()
        if final and self._in_tool:
            # A block the model never closed cannot be parsed, and §5.2 forbids passing it off as
            # text. The most common cause is max_tokens landing mid-call, which the message says.
            raise ApiError(500, "api_error", "tool_call_parse_error",
                           "the reply ended inside a <tool_call> block (truncated tool call, most likely "
                           f"max_tokens): {self._pending.strip()[:200]!r}")
        if final and self._open is not None:
            # A thinking block the model never closed still has to be closed on the wire.
            yield from self._stop_block()

    def _tool_use(self, raw: str):
        try:
            name, arguments = parse_tool_call(raw)
            if name not in self.schemas:
                raise ToolCallParseError(f"the model called {name!r}, which is not one of the declared tools "
                                         f"({', '.join(sorted(self.schemas))})")
            arguments = coerce_arguments(arguments, self.schemas.get(name))
        except ToolCallParseError as exc:
            raise ApiError(500, "api_error", "tool_call_parse_error", str(exc)) from exc
        block = {"type": "tool_use", "id": "toolu_" + "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(22)),
            "name": name, "input": arguments}
        yield from self._start_block(block)
        # One delta carrying the whole JSON: a parameter's type is only decided once </function>
        # closes the block, and the API only requires the concatenated deltas to be valid JSON.
        yield ("content_block_delta",
               {"type": "content_block_delta", "index": self._open,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(arguments)}})
        yield from self._stop_block()


__all__ = ["ApiError", "CountTokensRequest", "MessagesRequest", "ResponseAssembler", "bad", "message_base",
           "message_id", "runtime_error", "stop_reason_of", "text_of", "validate_count_tokens",
           "validate_request", "validation_error"]
