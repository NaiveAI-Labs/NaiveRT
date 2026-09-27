# Messages API

NaiveRT provides an Anthropic Messages-compatible HTTP API for text generation,
custom tool calls, and thinking output. The default model ID is `naive-n0.5`.
Use `GET /v1/models` to discover the ID configured by your server.

The examples below assume a server at `http://127.0.0.1:8080`. For server setup,
checkpoint selection, and output-limit configuration, see the
[deployment guide](../devops/README.md).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Readiness, active slot state, and queued request count |
| `GET` | `/v1/models` | Served model ID and display name |
| `POST` | `/v1/messages` | Generate a response, optionally streamed |
| `POST` | `/v1/messages/count_tokens` | Count the formatted input tokens without generating |

The server has no built-in authentication or TLS. Configure these at a reverse
proxy if needed. Credentials sent by a client are not validated by NaiveRT itself.
For browser clients, configure `--cors-origin` with the frontend's origin.
Browser requests that include endpoint cookies also require `--cors-credentials`
and an explicit origin. A bearer token or API-key header alone does not require
that flag; its header must be allowed by the CORS configuration.

## Create a message

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "naive-n0.5",
    "max_tokens": 512,
    "temperature": 0.4,
    "top_p": 0.95,
    "seed": 4,
    "messages": [
      {"role": "user", "content": "Explain KV caching in three sentences."}
    ]
  }'
```

### Request fields

| Field | Type | Description |
|---|---|---|
| `model` | string, required | Must match the model ID served by the endpoint |
| `messages` | array, required | Conversation history, including the latest user message |
| `max_tokens` | positive integer, required | Maximum generated tokens, including thinking |
| `stream` | boolean | Enable server-sent events; default `false` |
| `system` | string or text-block array | Optional system instructions |
| `temperature` | number | Sampling temperature in `(0, 5]` |
| `top_p` | number | Nucleus sampling mass in `(0, 1]` |
| `seed` | integer | Sampling seed in `[0, 2^63 - 1]` |
| `stop_sequences` | string array | Up to eight nonempty stop strings; default `[]` |
| `thinking` | object | Enable or disable thinking; see below |
| `tools` | array | Custom tool definitions; see below |
| `tool_choice` | object | `auto` or `none`, with optional `disable_parallel_tool_use` |
| `metadata` | object | `user_id` may identify a session; other keys do not affect generation |
| `repetition_penalty` | number | Multiplicative penalty in `(0, 10]`; `1` disables it |
| `frequency_penalty` | number | Frequency penalty in `[-10, 10]`; `0` disables it |
| `presence_penalty` | number | Presence penalty in `[-10, 10]`; `0` disables it |

When omitted, sampling controls use server defaults: normally temperature `1`,
top-p `0.95`, seed `1234`, and penalties `1`, `0`, `0`. Server startup options can
override these defaults. If specifying penalties, send all three to make the
configuration explicit.

`temperature: 0` and `top_k` sampling are not supported. A requested `top_p: 1`
is applied as `1 - 2^-20` for numerical stability. Repeatable output requires the
same formatted input, checkpoint pair, seed, sampling settings, execution path,
and supported hardware/software configuration.

### Conversation history

Send the complete history on every request. A session ID enables cache lookup;
it does not replace the `messages` array.

Excluding optional `system` messages, turns must start with `user`, alternate
between `user` and `assistant`, and end with `user`. Assistant prefill is not
supported. System messages inside `messages` are preserved in place; the top-level
`system` field supplies instructions before the conversation.

Content can be a string or an array of supported blocks:

| Role | Supported content blocks |
|---|---|
| `system` | `text` |
| `user` | `text`, `tool_result` |
| `assistant` | `text`, `thinking`, `tool_use` |

Image, document, audio, and other non-text input blocks are not supported.

### Output and context limits

`max_tokens` is required; the API does not fill in a default when it is omitted.
The README setup configures a server limit of **524,288 output tokens**. Your
server may use a different limit.

A request above the configured server output limit returns HTTP 400 with
`max_tokens_exceeds_capture`. If the prompt fits but leaves less context space
than the requested output budget, the server reduces the budget and reports
`max_tokens_clamped` in `x-naivert-ignored`. A prompt that cannot fit returns
`context_exceeded`.

### Response

A non-streaming response is a message object with these fields:

| Field | Description |
|---|---|
| `id` | Response identifier |
| `type` | `message` |
| `role` | `assistant` |
| `model` | Served model ID |
| `content` | Ordered `text`, `thinking`, and/or `tool_use` blocks |
| `stop_reason` | `end_turn`, `max_tokens`, `stop_sequence`, or `tool_use` |
| `stop_sequence` | Matched client stop string, or `null` |
| `usage` | Input, cache, output, and optional thinking-token counts |
| `naivert` | Timing and optional request-timeline information |

## Streaming

Set `stream: true` and read the response as `text/event-stream`:

```bash
curl -N http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "naive-n0.5",
    "max_tokens": 512,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain KV caching in three sentences."}
    ]
  }'
```

A successful stream contains:

1. `message_start`: response metadata and initial input/cache usage.
2. `content_block_start`: the beginning of a text, thinking, or tool-use block.
3. `content_block_delta`: content for that block.
4. `content_block_stop`: the end of that block. Steps 2-4 repeat for additional blocks.
5. `message_delta`: stop reason, final output usage, and `naivert` measurements.
6. `message_stop`: completion.

Delta types are `text_delta`, `thinking_delta`, and `input_json_delta`. Tool
arguments are emitted as one complete JSON delta after the call has been parsed.
Concatenate a block's deltas using its `index`. An SSE event is not a token;
events may contain multiple tokens and may arrive in network batches.

The server can send `: keepalive` comment lines before `message_start` and `ping`
events while waiting for the first generated content. The default interval is
five seconds. These indicate an open connection, not generated output. Streaming
is recommended for long requests; non-streaming clients need a timeout covering
the entire request.

Errors after HTTP headers have been sent arrive as an SSE `error` event and end
the stream. Closing the connection cancels queued or active work. A cancelled
stream may not contain final usage or timing information.

### Optional token progress

Send `x-naivert-token-progress: 1` to add
`naivert.generated_tokens` to text and thinking delta events. This is a cumulative
generated-token count, not a count of the characters in that event. It is not
attached to tool-argument deltas.

Clients can use count differences between receive times to measure streamed
generation speed. Final `usage` remains the authoritative completed-response
count; network batching can make streamed rates differ from device rates.

## Python SDK

```python
import anthropic

with anthropic.Anthropic(
    base_url="http://127.0.0.1:8080",
    api_key="unused",  # Replace if your reverse proxy requires a key.
) as client:
    messages = [{"role": "user", "content": "Explain KV caching briefly."}]
    reply = client.messages.create(
        model="naive-n0.5",
        max_tokens=512,
        messages=messages,
        temperature=0.4,
        top_p=0.95,
        extra_body={"seed": 4},
        extra_headers={"x-session-id": "example-session"},
    )
    for block in reply.content:
        if block.type == "text":
            print(block.text)

    messages.extend([
        {"role": "assistant", "content": [
            block.model_dump(exclude_none=True) for block in reply.content
        ]},
        {"role": "user", "content": "What changes when the context grows?"},
    ])
    with client.messages.stream(
        model="naive-n0.5",
        max_tokens=512,
        messages=messages,
        extra_headers={"x-session-id": "example-session"},
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
```

Use `extra_body` for NaiveRT extensions such as `seed` and the penalty fields.
Non-streaming extensions are available through `reply.model_extra["naivert"]`.
For streaming extensions and thinking-token counts, read raw `message_delta`
events; the SDK's accumulated final message may not retain every extension.

## Thinking

Thinking is disabled when `thinking` is omitted or set to `{"type": "disabled"}`.
To enable it:

```json
{
  "model": "naive-n0.5",
  "max_tokens": 4096,
  "thinking": {"type": "enabled", "budget_tokens": 1024},
  "messages": [
    {"role": "user", "content": "Explain the trade-offs of speculative decoding."}
  ]
}
```

Enabled thinking requires an integer budget satisfying
`1024 <= budget_tokens < max_tokens`. The budget is validated but is **not a
separately enforced thinking-token cap**. `max_tokens` bounds the total generated
output, including thinking and the answer.

`{"type": "adaptive"}` enables thinking without a separate budget and is reported
as `thinking.adaptive` in `x-naivert-ignored`. `thinking.display` is also ignored
and reported.

Responses use `thinking` blocks with a `thinking` string and no signature.
Replayed thinking blocks may omit `signature` or use an empty value; nonempty
signatures return `thinking_signature_unsupported`.

When thinking is enabled, final usage includes
`output_tokens_details.thinking_tokens`. When disabled, that field is absent.

## Tools

Tools describe functions that your client executes. NaiveRT returns tool calls;
it does not run tools itself.

```json
{
  "model": "naive-n0.5",
  "max_tokens": 512,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the weather for a city.",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  ],
  "tool_choice": {"type": "auto"},
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
}
```

A tool call is returned as a content block:

```json
{
  "type": "tool_use",
  "id": "toolu_example",
  "name": "get_weather",
  "input": {"city": "Paris"}
}
```

Append the full assistant response to the history, then add a user message with
the result. Keep the same tool definitions in the next request:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_example",
      "content": "Paris: 18 C, clear skies."
    }
  ]
}
```

Use the actual returned call ID. Each `tool_result` must refer to a `tool_use` in
the preceding assistant turn. Result content may be a string or text-block array.
Set `is_error: true` to report a tool failure. Tool results are presented to the
model before any additional text in that user turn.

Supported tool behavior:

- Custom tools require a unique `name` and an object `input_schema`; `description`
  is optional. Built-in/server-side tool types and `strict` schemas are unsupported.
- Property types support strings, integers, numbers, booleans, arrays, objects,
  nulls, type unions, resolvable `anyOf`/`oneOf` branches, and enums. A property
  with no declared type is returned as a raw string and reported as
  `untyped_tool_property` in `x-naivert-ignored`.
- `tool_choice: {"type": "auto"}` allows the model to choose a tool.
  `{"type": "none"}` excludes tools from the prompt. Forced choices (`any` or
  `tool`) are unsupported. A `tool_choice` requires nonempty `tools`.
- `disable_parallel_tool_use: true` stops after one complete tool call. That stop
  reports `stop_reason: "tool_use"`. This option also makes client stop strings
  inclusive in the returned output.
- A malformed or incomplete generated tool call returns `tool_call_parse_error`,
  including when the output limit truncates a call.

Preserve assistant block order and whitespace when sending history back to the
server. This helps retain exact prompt prefixes for cache reuse.

## Count input tokens

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "naive-n0.5",
    "messages": [
      {"role": "user", "content": "Explain KV caching in three sentences."}
    ]
  }'
```

The response is `{"input_tokens": N}`, where `N` is the formatted prompt length,
including system instructions, tool definitions, and thinking-mode formatting.
Supply the same prompt-shaping fields and session ID as the generation request.

Omit generation-only fields: `max_tokens`, `stream`, `stop_sequences`, `seed`, and
the three penalty fields. They are not accepted by this endpoint. Thinking-budget
validation checks the minimum of 1024 but cannot compare against `max_tokens`,
since that field is absent.

Token counting requires a ready server but does not enter the generation queue
or occupy the active generation slot.

## Sessions and token usage

Prefix reuse is automatic. Session lookup uses this order:

1. The `x-session-id` request header.
2. `metadata.user_id` in the request body.
3. A content match against cached prefixes.

Send the complete conversation even when using a session ID. Cache reuse depends
on the actual formatted token prefix. Changes to system instructions, tools,
thinking mode, or earlier messages can reduce reuse. Cached prefixes may be
evicted or lost when the server restarts; they are not durable conversation storage.

| Usage field | Meaning |
|---|---|
| `input_tokens` | Input tokens outside the cache accounting; currently `0` |
| `cache_read_input_tokens` | Prompt tokens reused from a cached prefix |
| `cache_creation_input_tokens` | Prompt tokens processed for this request |
| `output_tokens` | Generated tokens, including thinking |
| `output_tokens_details.thinking_tokens` | Generated thinking tokens; present when thinking is enabled |

The complete prompt length is
`input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
Do not add cache counts to the result of `/v1/messages/count_tokens`;
that endpoint already returns the complete prompt length.

## Timing extensions

Non-streaming responses and the final SSE `message_delta` include `naivert.timings`.
Available fields include:

| Field | Meaning |
|---|---|
| `queue_wait_ms` | Time from enqueue to worker start |
| `prompt_n`, `prompt_ms` | Newly processed prompt tokens and prefill time |
| `n_prompt_tokens_cache` | Reused prompt-token count |
| `initialize_ms` | Draft initialization time after target prefill |
| `prefill_path` | Selected prompt-processing path |
| `predicted_n`, `predicted_ms` | Output-token count and host-measured decode time |
| `decode_tok_s_device` | Device-measured generation rate |
| `segments` | Available `think`, `answer`, and `total` segment measurements |

Segment measurements include `tokens`, `rounds`, `tokens_per_round`, `device_ms`,
and `device_tok_s`. Their generation rates exclude prefill and finalization.
Optional prefill estimates and phase measurements may also be present.

`queue_wait_ms` is also included in the SSE `message_start` extension. Measure
client time to first content separately: `message_start` and keepalives do not
indicate that a generated token has arrived.

`naivert.timeline`, when present, contains `clock`, `origin`, `end_ms`, and `spans`.
Each span has `lane`, `name`, `start_ms`, `duration_ms`, and `args`. Times use the
server's monotonic clock relative to HTTP handler entry. Nested or overlapping
spans must not be added together. The timeline excludes final-report transmission
and client processing; it cannot be directly aligned with the client's clock.

## Health and queue status

```bash
curl http://127.0.0.1:8080/health
```

A ready server returns HTTP 200:

```json
{"status": "ok", "slot": "IDLE", "queued": 0}
```

`queued` counts pending requests; it excludes the active request. Readiness does
not mean the generation slot is idle. During loading or shutdown, `/health`
returns HTTP 503 with the current status. Generation and token-counting requests
also require readiness. `/v1/models` can respond before model loading finishes.

One request runs at a time. Additional generation requests wait in the queue.
A full queue returns HTTP 529 with `queue_full`.

## Compatibility and errors

Unknown fields return HTTP 400 rather than being silently dropped. The following
features are unsupported: image/document inputs, assistant prefill, `top_k`,
structured output formats, forced or strict tool calls, nonempty thinking
signatures, `service_tier`, `container`, `mcp_servers`, and `betas`.

Accepted hints that do not change execution are listed in the comma-separated
`x-naivert-ignored` response header:

| Header value | Behavior |
|---|---|
| `cache_control` | Explicit cache breakpoints are ignored; prefix reuse is automatic |
| `thinking.adaptive` | Served as enabled thinking without a separate budget |
| `thinking.display` | Display preference is ignored |
| `output_config.effort` | No effort control is applied |
| `context_management` | No context edits are applied; send the desired history explicitly |
| `untyped_tool_property` | An untyped tool argument remains a string |
| `max_tokens_clamped` | Requested output budget was reduced to fit remaining context |

`context_management` must still contain a well-formed `edits` list. SDK-generated
null fields such as `citations`, `parsed_output`, and `caller` are accepted as
absent; non-null values for these unsupported features are rejected.

Errors use this shape:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "temperature=0 must be a number in (0, 5]",
    "warning_type": "invalid_temperature"
  }
}
```

For errors sent before streaming headers, `x-naivert-warning` contains the same
code as `error.warning_type`.

| HTTP status | Common warning codes |
|---|---|
| `400` | `schema`, `unknown_field`, `invalid_messages`, `invalid_temperature`, `invalid_top_p`, `invalid_seed`, `invalid_repeat_penalty`, `invalid_stop_sequences`, `max_tokens_exceeds_capture`, `context_exceeded` |
| `400` | `invalid_thinking_budget`, `thinking_mode_unsupported`, `invalid_tool_schema`, `orphan_tool_result`, and unsupported-feature codes |
| `404` | `unknown_model` |
| `503` | `loading`, `shutting_down` |
| `529` | `queue_full` |
| `500` | `runtime_error`, `tool_call_parse_error` |

For an established stream, inspect SSE `error` events even when the HTTP status is
200. Transient availability errors may be retried with backoff. Correct invalid
requests before retrying; a fixed-seed generation may reproduce the same malformed
tool call, so retries alone do not guarantee recovery.

The [Python client example](../examples/chat_http.py) leaves the SDK's retry policy at its default.
