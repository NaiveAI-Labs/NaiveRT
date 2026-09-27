# Deployment

NaiveRT serves one active sequence on eight Blackwell GPUs with SM103a support. Additional
requests wait in a bounded queue; a full queue returns HTTP 529. For multiple replicas, route a
session consistently to one replica to reuse its prefix cache.

The image contains the runtime and its dependencies, but no model weights. Download the
target checkpoint [NaiveAI/Naive-N0.5-Flash-FP8](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8)
and its matching draft [NaiveAI/Naive-N0.5-Flash-FP8-Draft](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8-Draft).
See the [quick start](../README.md#select-the-models-and-start-the-server) for download commands.

## Start with Docker Compose

Run these commands from the repository root. Set host directories to the checkpoint directories
that contain the weight files, rather than their parent directories:

```bash
export NAIVERT_HOST_MODEL_DIR=/path/to/Naive-N0.5-Flash-FP8
export NAIVERT_HOST_DRAFT_DIR=/path/to/Naive-N0.5-Flash-FP8-Draft
export NAIVERT_TARGET_CHECKPOINT=/local_models
export NAIVERT_DRAFT_CHECKPOINT=/local_models/drafts

docker build -f devops/Dockerfile -t naivert:latest .
docker compose -f devops/docker-compose.yml up -d --no-build
docker compose -f devops/docker-compose.yml logs -f naivert
```

The target directory must contain `model.safetensors.index.json`; the draft directory must contain
`model.safetensors`. Checkpoint paths are **container paths**. The two host directories are mounted
read-only at `/local_models` and `/local_models/drafts`. Alternatively, mount collections of
checkpoints and select their subdirectories. A checkpoint parent can resolve to its newest
`step_*` child. Explicit `--target` and `--draft` arguments override the environment variables;
`--checkpoint` is an alias for `--draft`.

The container listens on port 8080. `NAIVERT_PORT` selects the published host port in Compose.
Wait for `GET /health` to return HTTP 200 before sending inference requests. `/v1/models` lists the
model before it is ready and is not a readiness check. First startup includes weight loading,
prefill preparation, warm-up, graph capture, and any remaining JIT compilation. The health check
allows 900 seconds for startup; increase that allowance if the host needs more time.

The default container configuration allocates 1,048,576 context rows and a 524,288-token output
buffer, with a 524,288-token request limit. Context capacity covers prompt plus generated tokens;
the output limit is not extra context capacity. Thinking tokens count toward the output limit.
These defaults belong to the container entrypoint; invoking `examples/serve_sessions.py` directly
uses that script's defaults unless flags are supplied.

Both the serving CLI and `SessionEngine.build()` default to native prefill. Resumed requests use
the cost policy to choose replay, native suffix ingestion, or recomputation. For an explicit
replay-only comparison, pass `--prefill verify8` through `NAIVERT_EXTRA_ARGS`.

## Use an existing image

Keep the checkpoint settings above, then select an image tag or digest:

```bash
export NAIVERT_IMAGE=your-registry/naivert:your-tag
docker compose -f devops/docker-compose.yml pull
docker compose -f devops/docker-compose.yml up -d --no-build
```

To publish a build, tag it with the source revision and push it to your registry. Pin a digest in
production manifests when an immutable image reference is required.

## Configuration

| Environment variable | Entrypoint default | Purpose |
|---|---|---|
| `NAIVERT_MODEL` | `naive-n0.5` | Public API model name and default checkpoint selection. |
| `NAIVERT_TARGET_CHECKPOINT` | Model registry target | Target checkpoint directory or short name. |
| `NAIVERT_DRAFT_CHECKPOINT` | Model registry draft | Draft checkpoint directory or short name. |
| `NAIVERT_PORT` | `8080` | Listener port; with Compose, the published host port instead. |
| `NAIVERT_CAPACITY` | `1048576` | Cache rows for prompt plus generation. |
| `NAIVERT_OUTPUT_CAPACITY` | `524288` | Allocated output ring capacity per request. |
| `NAIVERT_MAX_DECODE_TOKENS` | `524288` | Maximum accepted request output limit, bounded by output capacity. |
| `NAIVERT_PREFILL_WARM_CHUNKS` | `3` | Number of 4,096-row prefill chunk positions warmed at startup. Later positions may incur first-use warm-up. |
| `NAIVERT_QUEUE_DEPTH` | `8` | Maximum number of waiting requests, excluding the active request. |
| `NAIVERT_EXTRA_ARGS` | Empty | Space-separated additional server flags; shell quoting inside the value is not interpreted. |
| `NAIVERT_DRY_RUN` | `0` | Set to `1` to print the entrypoint command without starting the server. |

Compose also accepts `NAIVERT_IMAGE`, `NAIVERT_HOST_MODEL_DIR` (default `/local_models`), and
`NAIVERT_HOST_DRAFT_DIR` (default `/local_models/drafts`). For short-name resolution, the Python
server searches the colon-separated `NAIVERT_MODEL_ROOTS` and `NAIVERT_DRAFT_ROOTS`; custom search
roots must be added to the container environment if needed.

Arguments passed directly to `bash devops/entrypoint.sh` come after the environment-derived flags
and `NAIVERT_EXTRA_ARGS`, so explicit arguments take precedence. For example:

```bash
NAIVERT_DRY_RUN=1 bash devops/entrypoint.sh --max-decode-tokens 8192
```

`--store-gb` controls the parked-prefix CPU memory budget **per rank** (default 32 GiB). It is
separate from context capacity and shared memory. Compose uses `ipc: host`, so `/dev/shm` comes
from the host IPC namespace. In Kubernetes, provision `/dev/shm` with a memory-backed `emptyDir`
and set pod memory limits to cover shared memory, prefix storage, and the serving process.

## Studio and browser access

[Studio](../apps/studio/README.md) is a separate static web app. Deploy it independently and enter
the runtime's endpoint URL in its connection settings. Allow its origin on the API:

```bash
export NAIVERT_EXTRA_ARGS="--cors-origin https://your-studio.example.com"
docker compose -f devops/docker-compose.yml up -d --no-build
```

Use `http://127.0.0.1:8110` for a local Studio served on that origin. A gateway must also allow
browser preflight requests and streaming responses. Authentication, TLS, rate limiting, and
multi-replica routing belong at the gateway; CORS does not authenticate requests. The runtime
serves the API only. See the [Messages API](../docs/messages_http.md)
for request formats, queue status, timing fields, and errors.

## Build environment and caches

The image uses `lmsysorg/sglang:v0.5.15.post1`, CUDA toolkit 13.2, CUTLASS v4.7.1 headers, and
[`requirements-serve.txt`](requirements-serve.txt). Keep the dependency pins together; check the
installed versions with `python3 devops/warm_kernels.py --check-env`.

The build attempts to compile runtime extensions. Compilation is best effort because some builders
need visible GPUs; remaining extensions and native-prefill JIT kernels compile at startup. A
successful image build alone does not prove that inference works. `--build-arg NAIVERT_COMPILE=0`
skips extension compilation. Compose preserves JIT and extension caches in named volumes. When
changing the compiler or dependency stack, validate the resulting cached builds before serving.

## Validation

On a Docker build host without GPUs:

```bash
bash devops/tier1.sh naivert:tier1
```

This checks the build, dependency versions, host tests, imports, entrypoint configuration, Compose
configuration, and rejection of an unconfigured startup without model mounts or GPUs. Logs and exit codes are saved under
`tests/results/tier1/`. It does not validate GPU kernels or model output.

On an eight-GPU Docker host, keep the checkpoint environment variables set and run:

```bash
NAIVERT_IMAGE=naivert:latest NAIVERT_PORT=18093 \
  bash devops/validate.sh --no-build --dataset /path/to/prompts.jsonl
```

The dataset uses a `prompt_text` field per JSONL record. Without `--dataset`, validation checks
readiness and one short response. With it, the script also runs the HTTP session and SSE checks.
It uses a separate `naivert-validation` Compose project and removes its containers on completion
or failure. Use this on a host with the required GPUs available.

Additional checks against a running endpoint are under [`tests/gpu/serve`](../tests/gpu/serve/).
For kernel and model validation, see the [test guide](../tests/README.md).

## Run without Docker

Install the same dependencies and set checkpoint paths visible on the host:

```bash
export NAIVERT_CUDA_HOME=/path/to/cuda-13.2
export NAIVERT_CUTLASS_INCLUDE=/path/to/cutlass/include
export NAIVERT_TARGET_CHECKPOINT=/path/to/Naive-N0.5-Flash-FP8
export NAIVERT_DRAFT_CHECKPOINT=/path/to/Naive-N0.5-Flash-FP8-Draft
python3 devops/warm_kernels.py --check-env --compile
bash devops/entrypoint.sh
```

Do not prepend the toolkit's `bin` directory to `PATH`: its linker can interfere with host builds.
The runtime locates `nvcc` through `NAIVERT_CUDA_HOME`.
