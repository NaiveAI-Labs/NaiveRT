#!/bin/bash
# Validate the NaiveRT image on a Docker host with eight SM103a GPUs: build, environment check inside the
# image, compose up, readiness on /health, one request, the served-endpoint checks, compose down.
#   bash devops/validate.sh [--no-build] [--dataset path/to/prompts.jsonl]
# Structural build only (no GPUs, no CUDA compile): docker build --build-arg NAIVERT_COMPILE=0 -f devops/Dockerfile .
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=1; DATASET=""
while [ $# -gt 0 ]; do case "$1" in --no-build) BUILD=0;; --dataset) DATASET="$2"; shift;; *) echo "unknown $1"; exit 2;; esac; shift; done
PORT="${NAIVERT_PORT:-8080}"
export NAIVERT_IMAGE="${NAIVERT_IMAGE:-naivert:latest}"
COMPOSE=(docker compose -p "${NAIVERT_VALIDATE_PROJECT:-naivert-validation}" -f devops/docker-compose.yml)
STARTED=0
cleanup() { if [ "$STARTED" = 1 ]; then "${COMPOSE[@]}" down; fi; }
trap cleanup EXIT
step() { echo; echo "== $*"; }

step "1/6 image build"
if [ "$BUILD" = 1 ]; then docker build -f devops/Dockerfile -t "$NAIVERT_IMAGE" .; fi

step "2/6 environment inside the image (versions, nvcc, CUTLASS, GPUs)"
docker run --rm --gpus all --entrypoint python3 "$NAIVERT_IMAGE" devops/warm_kernels.py --check-env

step "3/6 compose up and readiness (/health, not /v1/models; cold start is minutes)"
STARTED=1
"${COMPOSE[@]}" up -d --no-build
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || { echo "server not healthy after $((i*5)) s"; "${COMPOSE[@]}" logs --tail 50; exit 1; }
echo "healthy after $((i*5)) s"; curl -s "http://127.0.0.1:${PORT}/v1/models"; echo

step "4/6 one request"
MODEL=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
curl -s "http://127.0.0.1:${PORT}/v1/messages" -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}]}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);t=d["naivert"]["timings"];print("text:", "".join(b["text"] for b in d["content"] if b["type"]=="text")[:80]);print("prefill", t["prefill_path"], round(t["prompt_ms"],1), "ms; decode", round(t["decode_tok_s_device"]), "tok/s")'

step "5/6 served-endpoint checks (from this host; anthropic + httpx from devops/requirements-serve.txt)"
if [ -n "$DATASET" ]; then
  python3 tests/gpu/serve/check_sessions_http.py --url "http://127.0.0.1:${PORT}" --dataset "$DATASET" --out /tmp/naivert_sessions.json
  python3 tests/gpu/serve/check_sse.py --url "http://127.0.0.1:${PORT}" --dataset "$DATASET" --max-tokens 2048 --out /tmp/naivert_sse.json
else
  echo "no --dataset given: skipped (a prompt_text jsonl, e.g. the no-think direct set)"
fi

step "6/6 compose down"
"${COMPOSE[@]}" down
STARTED=0
echo "VALIDATE_OK"
