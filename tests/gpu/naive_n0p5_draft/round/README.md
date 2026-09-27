# Complete draft/target round composition

`run.py` uses the current Naive N0.5 W8A8 target, BF16 draft and sharded T=1 sampler.
It binds the existing verify driver through `DraftTarget`, with direct pointers
to the post-layer residuals and D56's token buffer. The driver exposes separate
head inboxes and can omit its position advance; the tail advances state once.

It compares these modes:

- The serial schedule as one device-driven graph per rank.
- A device-driven graph with staged D12 alongside sampling after target head.
- With `--native-tail <DSO>`: a host-replayed serial graph using a pinned
  original native tail, which then serves as the reference the other modes are
  compared against. Without it, the modes are compared against each other.

All modes use the same target/draft weights and kernels. Target prefill runs on
real tokens from a repeated technical-text prompt. Cases use prompt lengths
200/2200 and 16/64 **additional** output tokens after the sampled anchor. These
are controlled request tests, not the archived 44-request or served-latency gate.

Checks include exact output tokens, acceptance counters, controls, target and
draft logits, local p/q, all eight selected target layer exports, target caches
and the usable draft cache. The serial path finishes its pending accepted KV
before final cache comparison. The staged graph audit preserves every unrelated
PDL edge, has one root/tail sink, and admits only kernel nodes. Source and binary
hashes must remain unchanged and there must be no new Xids/watchdog trips.

Build target extensions with the verify driver's `--build-only`, and build draft
extensions with `build.py`. Both CPU-only commands require
`CUDA_VISIBLE_DEVICES=''`; they do not configure or launch GPU kernels. Use CUDA
13.2, GCC and the configured CUTLASS include directory. GPU execution needs all
eight GPUs:

```sh
mkdir -p /tmp/naivert-bench
env MAX_JOBS=2 CUDA_HOME=/path/to/cuda-13.2 NAIVERT_CUDA_HOME=/path/to/cuda-13.2 \
    CXX=/usr/bin/g++ CC=/usr/bin/gcc \
    NAIVERT_CUTLASS_INCLUDE=/path/to/cutlass/include \
python3 tests/gpu/naive_n0p5_draft/round/run.py \
  --repo . \
  --target /path/to/target-checkpoint \
  --checkpoint /path/to/draft-checkpoint \
  --out /tmp/naivert-bench/result.json
```

Keep build products, tensor dumps and full graph-name inventories out of Git;
retain only compact evidence after the run completes.

Add `--cycles 24` for fixed-position serial/staged ABBA timing; it needs
`--native-tail`, because the benchmark gate DSO is looked up beside that file.
It restores all inputs outside the events, keeps mailbox generations monotonic
and gates the eight ranks' timed starts together. Every sample is compared
bitwise. Commit publication and recursion are disarmed; acceptance and KV writes
remain active.

Add `--persistent` to test the token-exact persistent sampler and matching
no-ready tail. It adds a fourth request mode with serial D12, checks persistent
anchor p/tokens, and compares that candidate against the existing staged chain
in the timing loop. The cooperative sampler plus concurrent D12 is not enabled.
`protocol.py` checks generation/consumed advancement outside timed intervals;
persistent ready flags stay zero. All protocol epochs survive request resets.
The persistent graph has 207 nodes; the baseline graphs have 238.

For node durations, use `--cycles 4 --profile-fixed` under Nsight Systems with
`--trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi
--capture-range-end=repeat:2 --cuda-graph-trace=node`. This adds one captured
ABBA cycle after the unprofiled cycles at each context. Export both reports to
SQLite, then run `analyze_profile.py --result profile_result.json --database
fixed_trace.1.sqlite --database fixed_trace.2.sqlite --out analysis.json`.
SQLite analysis is CPU-only.

This test does not supply an independent training-math oracle, the greedy gate,
or a trace proving absence of host CUDA calls between device-driven rounds.

## Prompt ingestion (`--prefill`)

Both `run.py` and `request_set.py` ingest prompts through `prefill.py`. The default,
`verify8`, runs the target's eight-row verify graph over eight consecutive prompt tokens per
step (the same kernels and cache writes a draft block gets) and finishes the last 1..8 tokens
with seq1 steps, so the final position's logits and features are the seq1 step's. `seq1` is
the original one-token-per-step replay, about 2.4 ms per token against roughly 0.35 ms for
`verify8`. The rows-8 and rows-1 kernels are not bitwise, so the two modes leave ulp-different
caches; every path within a run shares one prefill, and the gate compares paths against each
other, so both modes are valid gates but their generated tokens differ for a given seed.
Recorded references from before 2026-09-18 used `seq1`.

`native` uses the 4,096-row native prefill executor; `kernel` uses the batched
`DraftFeaturePrefillExecutor` reference. Both leave the final token for a seq1
step. Native ingestion of a resumed prefix uses replay up to the next chunk
boundary before resuming native chunks. Compare execution modes using the same
prefill path; these paths are not assumed to produce bitwise-identical caches.
The serving entrypoint defaults to native prefill, independently of this harness.

Measurements below describe recorded runs, not a fresh validation of the
current checkout or every checkpoint pair.

## Archived 44-request regression

**Recorded archive comparison: failed at request 1.** The archive uses different target math
from this harness's current W8A8 target. Both public modes and a pinned native
draft/tail control agree exactly when using the same current target.
This does not close the 44-request gate.

`request_set.py` loads the current public target/draft kernels at capacity 65536
and compares the existing staged-chain schedule with the serial-D12 persistent
sampler. It validates the archived dataset, checkpoint and tokenizer identities,
then prefills each prompt once and restores the same caches for both modes.
Protocol generations are never reset with request state.

Both modes must match every archived output token, acceptance histogram and
finish reason. They must also match each other's final logits, p/q, selected
exports, controls and usable target/draft caches by SHA-256. Serial draft KV is
flushed before comparing the final state. The first mismatch is written before
stopping; only 44 fully matched requests in both modes can produce a passing
summary. It uses the full 24,576-token output budget, including the initial
anchor, and preserves EOS 151645. It is a correctness test, not serving timing.

Run with the same CUDA/CUTLASS environment as `run.py`, supplying `--repo`,
`--target`, `--checkpoint`, `--baseline` (the archived reference run) and a new
`--out` directory. Keep the full generated tokens, inventories and cache hashes
out of Git; retain a compact summary plus evidence hashes.

For a bounded same-target diagnosis, add `--diagnose-first`, `--native-core`
pointing to the pinned native draft DSO, and `--native-tail` pointing to the
original native tail DSO. This runs the complete first request through pinned
native draft/tail kernels and both public schedules, including native prompt-KV
ingestion. It writes `diagnostic.json` with separate current-target and archive
results; it cannot produce a passing 44-request summary. Its host-replayed
native control is a correctness oracle, not a serving-latency comparison.

`--current-target-reference` extends that native control to all 44 requests.
It requires the same two pinned DSOs and all three paths must match exactly.
The manifest/summary explicitly identify this reference, retain all historical
archive mismatches, and report historical versus current aggregate acceptance.
It does not change the default archive assertion. The recorded matching-target
run passed 386,706 tokens, 83,735 rounds and 92,928 final-state hash comparisons,
with identical 51.69335% acceptance across all three paths. Use
`collect_request_set.py --root COMPLETED_RUN --out compact.json` to independently
recheck records and retain compact evidence.

## Actual device-chain API trace

`run.py --persistent --profile-device-chain` profiles the 64-token persistent
requests at initial contexts 200 and 2200. It retains publication and recursive
launch, and adds NVTX markers around the host's mapped-memory wait. This flag
cannot be combined with fixed-round timing. Use the same Nsight options above
with `--capture-range-end=repeat:2`, export both reports to SQLite, then run:

```sh
python3 tests/gpu/naive_n0p5_draft/round/analyze_device_chain.py --result result.json \
  --database trace.1.sqlite --database trace.2.sqlite --out analysis.json
```

The analyzer requires every captured iteration's complete 207-node order on
all eight ranks, populated startup/drain CUDA API records, and zero host CUDA
calls during both the marked wait and the common interior GPU rounds. It also
checks for memcpy/memset activity in that interior interval. An absent driver
API table is recorded explicitly. This checks the device-driven contract; the
trace is not used as a latency comparison. The recorded collection/analysis covered 14/11
rounds at initial contexts 200/2200: all eight ranks preserve the 207-node order,
zero captured runtime API calls occur during the marked host wait or common
interior interval, and startup/drain API records are populated. Driver API
coverage is absent. All request and protocol records match collection-off.
`collect_device_chain.py --root COMPLETED_RUN --unprofiled-result CONTROL_JSON
--out compact.json` reruns analysis and retains compact evidence.
