# NaiveRT Studio

Studio is a standalone static frontend for the streaming Anthropic Messages API. It builds without
Python, CUDA, or the NaiveRT runtime. The browser sends requests to the endpoint you configure;
the static hosting service does not proxy inference traffic.

## Run locally

From the repository root:

```sh
node apps/studio/build.mjs
python3 -m http.server 8110 --bind 127.0.0.1 --directory apps/studio/dist
```

Open `http://127.0.0.1:8110`. Expand **API connection**, enter an API origin or a full
`/v1/messages` URL and the model name, then click **Use endpoint**. The default model is
`naive-n0.5`. An explicit endpoint is required; Studio does not treat its static host as an API.

For an HTTP endpoint or one without browser CORS support, use the local development proxy:

```sh
cd apps/studio
npm run local -- --upstream http://your-api-host:20000 --port 8110
```

The proxy binds to `127.0.0.1` and forwards `/v1/messages` and `/health` to one fixed upstream.
It forwards selected authentication and session headers, but not cookies, and rejects redirects.
Credentials are neither logged nor stored by the proxy. Restart it to change the upstream.
This helper is for local testing; an HTTP upstream connection remains unencrypted.

## Deploy on Vercel

Import the repository with **Root Directory** set to `apps/studio`. The included `vercel.json`
builds with `node build.mjs` and publishes `dist`. No npm dependencies or backend functions are
required. The directory can also be copied into a separate repository.

A hosted HTTPS page needs an HTTPS remote API. A loopback endpoint such as
`http://127.0.0.1:18090` refers to each visitor's computer and may require local-network permission.

The API or gateway must allow the Studio origin, methods `GET`, `POST`, and `OPTIONS`, and these
request headers: `Content-Type`, `x-naivert-token-progress`, `X-Session-Id`, and the selected
authentication header (`Authorization` or `x-api-key`). Gateways must answer unauthenticated CORS
preflights before authentication and model routing, and include CORS headers on actual responses.
With NaiveRT, add `--cors-origin https://your-studio.example.com` to the server command, or to
`NAIVERT_EXTRA_ARGS` when using [Compose](../../devops/README.md).

Cookie authentication additionally requires `Access-Control-Allow-Credentials: true` and an
explicit allowed origin. NaiveRT exposes this through `--cors-credentials`. CORS does not validate
credentials; authentication must be enforced by the API or gateway.

## Connection and credentials

Paste a setup line into **Quick setup**, then click **Apply & check**:

```text
https://api.example.com sk-your-key naive-n0.5
```

The key and model are optional. A supplied key selects bearer authentication. The masked paste
field is cleared on submission, and its original contents are not stored. Manual settings also
support `x-api-key`, HTTP Basic, and optional endpoint cookies. Only the selected authentication
header is sent; enabling cookies does not sign you in.

Applied endpoint, model, and credentials are remembered in browser local storage by default.
Credentials are bound to the API origin, including scheme and port, and to the upstream when using
the local proxy. Editing the endpoint or choosing **Clear credentials** removes the saved
credential. If storage is unavailable, settings remain usable in memory. Credentials are not sent
to Vercel or included in exported measurements, and requests reject redirects.

Changing applied connection settings starts a new chat. **Check connection** sends a separate
one-token streaming request with thinking disabled. It verifies inference and authentication
without adding a chat turn. It can enter the server queue and times out after 20 seconds; a timeout
can mean queueing rather than an unavailable endpoint.

**Queued requests** polls `/health` approximately every two seconds. It reports waiting requests
on the selected runtime, excluding its active request. It is neither your queue position nor a
replica-wide aggregate. A busy server can have zero queued requests. Unsupported or failed health
queries show `—`; hover or focus the indicator for details. Health queries use the applied
credentials and session header. A gateway must route them and allow their CORS preflights too.

## Chat, seeds, and examples

The output limit defaults to **524,288 tokens (512K)**. The API validates its own capacity.
Thinking is optional; its budget must be at least 1,024 and below the output limit. On NaiveRT,
that budget is a hint rather than a hard device-side stopping boundary.

The composer seed defaults to **4**. Example cards fill the prompt and settings without sending:

- **Write a tetris** uses seed 4 and requests a self-contained HTML artifact.
- The pelican presets use seed 42, no formatting system message, and a 16,384-token thinking-budget
  hint. The thinking preset enables thinking; the other disables it.

**Send** continues the conversation. **Lucky draw** starts a fresh chat with a new seed and sends
the draft prompt, or the last prompt when the composer is empty. Reusing the last prompt preserves
its generation settings and system prompt while removing preceding conversation turns. Reproducing
an output requires the same model, full context, seed, and settings.

Each new chat gets a session ID in `metadata.user_id` and `X-Session-Id`. Completed assistant
blocks are replayed in order on subsequent turns. Chat contents remain in tab memory and are lost
on reload. Providers must support the request options you select, such as thinking and seed.

## Markdown and interactive previews

Answer prose renders Markdown headings, lists, tables, blockquotes, emphasis, links, and inline
code. Raw HTML in prose is escaped; remote images render as alt text.

Fenced SVG and HTML blocks have Code/Preview tabs, copy, download, and fullscreen controls. Code
appears as it arrives without an artificial typing delay. Scrolling upward pauses auto-follow.
Preview begins after the closing fence; incomplete blocks remain code. Complete raw SVG or HTML
documents can preview after the response finishes.

SVG is displayed as an image. HTML runs in an opaque `sandbox="allow-scripts"` iframe whose CSP
blocks external assets, network requests, nested frames, and form submission. Inline JavaScript,
CSS, and embedded images work; package installs and remote dependencies do not.

**Play / interact** focuses the preview. Arrow and space keys reach the generated app without
scrolling the parent page; text inputs retain their editing behavior. **Restart** reloads the
artifact without regenerating it. Script errors appear outside the sandbox. **Ask to fix** fills
the composer with the source and error for review but does not send automatically.

## Measurements

| Display | Source and interval |
|---|---|
| Thinking / Output device TPS | Final `naivert.timings.segments.think/answer.device_tok_s`; device execution only. Answer-only responses can use `decode_tok_s_device` when thinking is disabled. |
| Streamed TPS | Token-progress counters divided by the browser receive interval, tracked separately for thinking and output. The first receive batch is excluded. |
| Curve and peak | Rolling received throughput over approximately one second, with a minimum 200 ms interval and 200 ms sample buckets. Peak is the highest plotted sample. |
| Generated tokens | Final `usage.output_tokens`, including thinking. |
| Prompt tokens | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`. |
| Total TTFT | Browser request start to first nonempty thinking or text content. |
| Queue wait | Server-reported local worker queue wait; health counts are not used to estimate it. |
| TTFT after queue | Total TTFT minus reported queue wait; still includes network and gateway delays. |
| Elapsed | Whole browser request, including queueing, prefill, and finalization. |

Generation TPS excludes prefill and finalization. Network batching can create curve spikes;
streamed TPS measures delivery, not instantaneous GPU execution. Device TPS and final token counts
appear after completion. Missing measurements show `—`; unsupported token-progress metadata or
one-batch replies do not produce a measurable streamed rate.

Studio requests `x-naivert-token-progress: 1` and reads `naivert.generated_tokens` on content deltas.
Events in the same browser receive batch share a timestamp. Final usage and stop events do not
extend the measured generation interval. Thinking and output have separate curve paths.

The compact panel puts the curve on the left and statistics on the right. **Measurement details**
contains phase timing, streamed averages, backend decode-wall TPS, and browser update costs.
Without queue metadata, TTFT is labeled as including queue. Raw measurements include request
settings, backend fields, curve samples, and a response SHA-256, but not conversation contents.
Historical examples are retained in the [pelican transport comparison](tests/evidence/pelican_20260922/README.md)
and [thinking pelican measurement](tests/evidence/pelican_thinking_20260922/README.md); they are dated
smoke measurements, not current performance guarantees.

## Validation

Run the protocol, parser, and proxy tests with Node.js 18 or later:

```sh
node --test apps/studio/tests/*.test.js
```

For browser checks without a GPU, serve the static build on port 8110 and run the API fixture:

```sh
python3 apps/studio/tests/preview_server.py --port 8099 --cors-origin http://127.0.0.1:8110
```

Configure Studio with `http://127.0.0.1:8099` and any model name. The fixture exposes only the
Messages route; queue telemetry is unavailable. It supports example cards and prompts containing
`plain`, `truncate`, `disconnect`, `error`, or `slow`. Use `scroll slow plain` for a long thinking
stream. `--answer-file PATH` replays a saved answer; `--queue-wait-ms 800` simulates queue timing.
All fixture metrics are synthetic test data.
