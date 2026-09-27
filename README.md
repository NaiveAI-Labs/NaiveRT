# NaiveRT

**Fast single-stream inference for Naive-N0.5-Flash.**

NaiveRT combines fused CUDA kernels, speculative decoding, and GPU-driven execution across eight Blackwell GPUs. It is designed to shorten individual generations—particularly long agentic tasks and RL rollouts—with reproducible sampling.

In our HTML/SVG generation benchmark, NaiveRT achieved **1,719 device tokens/s across requests**, with a **peak one-second rate of 2,122 tokens/s**, with thinking disabled.

[Try Studio](https://ultrafast.naive.ai) · [Tech blog](https://naive.ai/en/research/) · [Target weights](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8) · [Draft weights](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8-Draft) · [Deployment guide](devops/README.md) · [API reference](docs/messages_http.md)

## Why NaiveRT?

Long rollouts can dominate training and agent execution time. Improving aggregate throughput helps serve more requests; improving single-stream speed helps each rollout finish sooner.

NaiveRT focuses on reducing per-stream latency:

- **Specialized CUDA kernels** for the target model and its speculative draft.
- **GPU-driven decoding** without CPU scheduling between speculative rounds.
- **Reproducible sampling**, with bitwise determinism as a correctness requirement for supported execution configurations.
- **Long-context serving** with native prefill and reusable session prefixes.
- **Practical integration** through an Anthropic Messages-compatible API and a standalone web interface.

## Performance

Measured on one eight-GPU Blackwell system with one active sequence, using HTML/SVG generation prompts at temperature **0.4** and top-p **0.95**.

| Metric | Thinking disabled | Thinking enabled |
|---|---:|---:|
| Device generation rate | **1,719 tokens/s** | **1,090 tokens/s** |
| Thinking generation rate | — | 984 tokens/s |
| Answer generation rate | 1,719 tokens/s | 1,816 tokens/s |
| Peak one-second rate | **2,122 tokens/s** | **2,058 tokens/s** |
| Draft acceptance | 70.6% | 43.4% |

These results come from the existing benchmark campaign, rather than a new measurement of every subsequently released checkpoint pair. Repetition-loop requests were excluded.

Device generation rates exclude prefill and finalization. The peak is the best sliding one-second window of committed tokens. These measurements describe a single active stream, not aggregate server throughput or browser delivery speed.

A full speculative round took approximately **3.45 ms** in the thinking-disabled campaign. Generation speed depends on how many draft tokens the target accepts per round, and varies with the checkpoint pair, prompt, context length, sampling settings, and generated content.

## How it works

### Selective kernel fusion

NaiveRT combines normalization, attention projections, positional encoding, cache updates, indexing, attention, and output processing into fused execution paths.

Fusion decisions follow full-model measurements. The MoE router, gate/up, and down computations remain separate kernels because the tested fusion alternatives increased end-to-end latency.

### Overlapping computation and communication

Programmatic Dependent Launch connects kernel stages so downstream work can begin before upstream execution has fully completed, while preserving the required data dependencies.

Inside kernels, asynchronous weight transfers, shared-memory reuse, specialized Tensor Core layouts, and overlapping tensor-parallel communication reduce exposed latency.

### GPU-driven speculative decoding

The draft proposes tokens, the target verifies them, and the sampler and tail commit the accepted output. The device launches the next round directly.

The host handles request setup and streams committed tokens to the client.

### Reproducibility

For fixed input tokens, checkpoint, seed, execution path, and supported hardware/software configuration, NaiveRT targets bitwise-identical tokens, logits, and KV state.

Kernel changes are checked against the established numerical reference. Repeatability and agreement with a training implementation are separate validation requirements.

## Quick start

You need:

- Eight compatible Blackwell GPUs with peer-to-peer access.
- The target checkpoint and its matching draft checkpoint.
- Docker with NVIDIA GPU support.

Use the matching target and draft checkpoints from NaiveAI on Hugging Face:

| Role | Repository |
|---|---|
| Target | [NaiveAI/Naive-N0.5-Flash-FP8](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8) |
| Draft | [NaiveAI/Naive-N0.5-Flash-FP8-Draft](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8-Draft) |

### Build the image

```bash
docker build -f devops/Dockerfile -t naivert:latest .
```

See the [deployment guide](devops/README.md) for the toolchain, dependencies, and environment checks.

### Select the models and start the server

Download both repositories with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then start the server. If access requires authentication, run `hf auth login` first.

```bash
export NAIVERT_IMAGE=naivert:latest

# Checkpoint directories on the host.
export NAIVERT_HOST_MODEL_DIR=/absolute/path/to/Naive-N0.5-Flash-FP8
export NAIVERT_HOST_DRAFT_DIR=/absolute/path/to/Naive-N0.5-Flash-FP8-Draft

hf download NaiveAI/Naive-N0.5-Flash-FP8 --local-dir "$NAIVERT_HOST_MODEL_DIR"
hf download NaiveAI/Naive-N0.5-Flash-FP8-Draft --local-dir "$NAIVERT_HOST_DRAFT_DIR"

# Corresponding paths inside the container.
export NAIVERT_TARGET_CHECKPOINT=/local_models
export NAIVERT_DRAFT_CHECKPOINT=/local_models/drafts

# API model name.
export NAIVERT_MODEL=naive-n0.5

# 1M context capacity and 512K output limit.
export NAIVERT_CAPACITY=1048576
export NAIVERT_OUTPUT_CAPACITY=524288
export NAIVERT_MAX_DECODE_TOKENS=524288
export NAIVERT_EXTRA_ARGS="--cors-origin https://ultrafast.naive.ai"

docker compose -f devops/docker-compose.yml up -d --no-build
```

The target directory must contain `model.safetensors.index.json`; the draft directory must contain `model.safetensors`. Both directories are mounted read-only.

`NAIVERT_MODEL` controls the model name exposed by the API. `NAIVERT_TARGET_CHECKPOINT` and `NAIVERT_DRAFT_CHECKPOINT` select the actual weights.

This setup uses an output limit of **524,288 tokens**, matching Studio’s default, with **1,048,576 context positions**. Generated output must also fit within the remaining context capacity.

The server listens on port **8080**. Cold startup takes several minutes to load weights, prepare kernels, and warm execution paths. `/health` returns HTTP 200 when the server is ready.

### Open Studio

Open **[NaiveRT Studio](https://ultrafast.naive.ai)**, enter your reachable API endpoint and any required credentials, then choose a demo prompt.

Studio supports:

- Streaming thinking and answer text.
- Markdown rendering and SVG/HTML previews.
- Interactive generated applications.
- Explicit seed selection and fresh generations of the same prompt.
- A live received-token speed curve and server-reported generation metrics.

Studio runs independently of the runtime. For local endpoints, you can also [run Studio locally](apps/studio/README.md). Configure the API’s allowed origin to match the Studio instance you use.

The default requested output limit is **524,288 tokens**. You can lower it per request; the server’s configured limit and available context remain authoritative.

## Serving

The server supports:

- Streaming and non-streaming Messages requests.
- Thinking and answer output.
- Temperature, top-p, seed, and repetition/frequency/presence penalties.
- Session-prefix reuse.
- Request queueing.
- Timing information for prefill, generation, and response delivery.

The server processes **one active sequence per instance**. Additional requests queue; requests are rejected with HTTP 529 when the queue is full.

Context capacity, output capacity, and request limits are configurable. Use a reverse proxy for public authentication and TLS.

See the [Messages API reference](docs/messages_http.md) for request formats and integration examples.

## Understanding the metrics

| Metric | Meaning |
|---|---|
| Device generation rate | Committed tokens per unit of device decode time |
| Streamed generation rate | Tokens received by the client during generation |
| Peak one-second rate | Highest rate over a sliding one-second window |
| Time to first token | Latency before the first generated token becomes available |
| Queue wait | Time spent waiting for an available execution slot |
| Total elapsed time | Complete request duration, including waiting, prefill, generation, and finalization |

Generation rates exclude prefill and finalization. Queue wait is tracked separately from generation. Network buffering can make the browser’s streamed curve and peak differ from device measurements.

## Requirements

- **GPU:** eight Blackwell GPUs with UMMA (`tcgen05`) support and peer-to-peer access. The supplied build targets compute capability **10.3a**; other architectures require compatible builds and validation.
- **CUDA:** CUDA 13.2 for NaiveRT extensions.
- **CUTLASS:** v4.7.1 headers.
- **Python dependencies:** the versions specified in the Docker image and [`devops/requirements-serve.txt`](devops/requirements-serve.txt).
- **Checkpoints:** a supported W8A8 target checkpoint and its matching draft.

The Docker image provides the configured toolchain and dependencies. For bare-metal setup, follow the [deployment guide](devops/README.md).

## Repository layout

| Directory | Contents |
|---|---|
| `naivert/kernels`, `naivert/body`, `naivert/entry` | CUDA kernels, bindings, and builders |
| `naivert/backends` | Draft execution, speculative rounds, session engine, and prefix storage |
| `naivert/runtime` | CUDA graph interfaces, dependency handling, and shared control layouts |
| `naivert/models` | Model configuration, checkpoint loading, and weight preparation |
| `naivert/prefill` | Native prefill execution |
| `naivert/serve` | Messages API and request handling |
| `apps/studio` | Standalone web demo |
| `tests` | Host tests, GPU validation, and benchmark harnesses |
| `devops` | Docker image, Compose configuration, and environment checks |

## Validation

Run the host and contract tests:

```bash
python3 -m pytest tests/host tests/contracts -q
```

GPU-only checks require the appropriate hardware and checkpoints.

The [GPU round harness](tests/gpu/naive_n0p5_draft/round/README.md) compares serial, staged, and persistent execution, including output tokens, logits, acceptance counters, control state, and caches.

Served-endpoint checks are available under `tests/gpu/serve`. See the [deployment guide](devops/README.md) for image and end-to-end validation.

## License

NaiveRT is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for vendored components and their licenses.
