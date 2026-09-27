# Pelican: backend and Studio speed agreement

2026-09-22. Existing model server through the `127.0.0.1:18090` tunnel; local Studio/proxy on `8100`.
One sequential run per transport, no overlapping inference. These are smoke measurements, not a
statistical performance comparison. Server configuration was not changed or restarted.

Exact user prompt:

> Generate an SVG of a pelican riding a bicycle. Reply with only the complete SVG markup, no explanation.

All requests used `naive-n0.5`, seed **42**, `max_tokens=131072`, and
`thinking={"type":"disabled","budget_tokens":16384}`. No added system message or prior conversation.
Temperature/top-p were omitted (server defaults). The browser adds a conversation `metadata.user_id`;
this is a session-store key, not model input. All four responses reported zero cache-read tokens.

| Client | Output tokens | Backend answer device TPS | First content | Full request |
|---|---:|---:|---:|---:|
| Direct JSON, equivalent to the supplied curl | 4,872 | 1,699.87 | N/A (not streamed) | 3.482 s |
| Direct SSE | 4,872 | 1,699.53 | 0.172 s | 3.565 s |
| Local proxy SSE | 4,872 | 1,699.90 | 0.196 s | 3.276 s |
| Studio browser through the proxy | 4,872 | 1,699.91 → **1,700 displayed** | 0.195 s | 3.478 s |

Every response has the same SHA-256:
`46573aa762a62df87d913d38c5697e6b768dcbddc94108e23e085f6cd369d01e`.
The browser received 8,902 characters and rendered the raw SVG successfully. The backend segment
counter reports 4,871 answer tokens; total generated usage is 4,872. Studio uses the backend's segment
rate directly, without dividing total usage by a different interval.

Browser work: 33 DOM updates, **46.4 ms cumulative**, **3.0 ms longest update**. The 100 ms refresh
interval caused at most **98.3 ms pending-update latency**. Final DOM update ended at 3.479 s,
about 1.4 ms after the final SSE event. These are DOM-update CPU intervals, not GPU paint timings.

Device TPS is not whole-request TPS: the browser delivered about **1,401 tokens/s over the entire
request**, including admission/prefill/finalization/transport. Its final content arrived at 3.032 s;
the final event arrived at 3.477 s. A similar tail exists without any browser: direct SSE's tail
was 0.547 s and proxy SSE's was 0.211 s. This demonstrates that the tail is not caused solely by DOM
rendering; these client measurements do not separate server finalization from transport time.
The varying tails explain why the proxy's single total happened to be shorter; they are not a
measured proxy speedup. Backend decode intervals were approximately 2.869 s in all four cases.

Files `sse-0-0.json`, `sse-1-0.json`, and `json-0-0.json` are from the CLI harness, retaining the
server's timings. `browser.json` records the visible Raw measurement panel and displayed value.

Reproduce the command-line side with the model server and local proxy already running:

```sh
python3 apps/studio/tests/pelican_bench.py \
  --endpoint http://127.0.0.1:18090 --endpoint http://127.0.0.1:8100 \
  --seed 42 --out /tmp/naivert-pelican
python3 apps/studio/tests/pelican_bench.py \
  --endpoint http://127.0.0.1:18090 --seed 42 --nonstream --out /tmp/naivert-pelican
```

For the browser, start a new conversation and select **Try the pelican benchmark**. It fills
the exact prompt, disables thinking and added formatting instructions, and sets seed 42.
Set the output cap to 131,072 to match this archived run, then send. The current Studio default is
524,288. Expand Measurement details → Raw measurement for
the backend counters, browser timing, output hash, and displayed TPS. Compare matching seeds and
output hashes before drawing performance conclusions.
