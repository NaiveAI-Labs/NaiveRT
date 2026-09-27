# Thinking pelican and measured speed curve — 2026-09-22

One real browser smoke test through `127.0.0.1:8100/demo` → local proxy →
`127.0.0.1:18090` → gpu-test-host:8090. This is one request, not an averaged benchmark.

Request: model `naive-n0.5`, seed 42, max tokens 131072, thinking enabled,
budget hint 16384, one user message, no added system prompt:

> Generate an SVG of a pelican riding a bicycle. Reply with only the complete SVG markup, no explanation.

| Phase | Device tokens | Device interval | Device tok/s | Received tokens after first batch | Receive interval | Streamed tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Thinking | 32713 | 35081.045504 ms | 932.497864 | 32697 | 35064.5 ms | 932.481570 |
| Output | 4030 | 2225.100896 ms | 1811.153826 | 4027 | 2228.900001 ms | 1806.720804 |

Streamed rate differs from device rate by -0.002% for thinking and -0.245% for output.
Streamed counts subtract the whole first receive batch's generated-token frontier. The
metrics therefore compare slightly different token intervals; exact equality is not expected.

- Usage: 36744 output tokens, including 32714 thinking tokens. Device segment counts exclude
  the initial token. The thinking budget is an API hint here, not a hard device stop.
- First content: 235.300000 ms after request; last content: 37528.700001 ms.
- Final report: 40379.900001 ms; browser elapsed: 40399.800000 ms.
- The 2851.2 ms last-content-to-final-report gap is excluded from streamed TPS and the curve.
- Backend prefill: 63.895889 ms; capture: 83.635019 ms. TTFT is shown separately.
- Aggregate received generation rate: 984.812326 tok/s over 37293.400001 ms.
- Curve: rolling approximately 1000 ms window, 200 ms buckets; 174 thinking points and 11
  output points. Maximum plotted sample rounds to 1892 tok/s. This is received throughput,
  including transport batching, not instantaneous device throughput.
- Browser: 375 render updates, 2341.3 ms cumulative DOM work, maximum 12.7 ms per update,
  maximum scheduling delay 104.3 ms. Rendering is batched at approximately 100 ms.
- Completed raw SVG preview loaded at 900×700. Response: 7196 characters;
  SHA-256 `aa89814a9020f83828f5ae395dc10c8307af45258f4ad85879f0954d967f9134`.
- Both gray thinking and orange output paths were visibly rendered. Browser console had
  no warnings/errors, and upstream health returned `ok`, `IDLE`, queue 0 after completion.

The running gpu-test-host checkout received only the opt-in token-progress `serve/http.py` patch
from `dd6be6a2` (23 insertions, 4 deletions). Its original launch script and model/kernel
configuration were reused through the GPU queue. The local proxy forwards
`x-naivert-token-progress: 1`; rates come from generated-token indexes without retokenizing.

Validation: 17 Node tests, two targeted HTTP route tests; the unchanged backend progress
patch previously passed 62 HTTP/HTTP-tools tests. Fixture checks also covered mobile width
390, short phases, and unavailable counters. Full curve samples remain in the completed
browser request's Raw measurement panel; this note records its compact measurement summary.
