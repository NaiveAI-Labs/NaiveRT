#!/bin/bash
# Validate an image on a build host without GPUs. See devops/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-naivert:tier1}"
OUT="${NAIVERT_VALIDATION_DIR:-$(pwd)/tests/results/tier1}"
mkdir -p "$OUT"
step() {
  local name="$1" expected="$2" rc=0
  shift 2
  echo "== $name"
  "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  echo "$rc" > "$OUT/$name.exit"
  if [ "$rc" -ne "$expected" ]; then
    echo "Failed: $name returned $rc (expected $expected). See $OUT/$name.log" >&2
    return 1
  fi
}
date > "$OUT/started"
step build 0 docker build -f devops/Dockerfile -t "$TAG" .
step checkenv 0 docker run --rm --entrypoint python3 "$TAG" devops/warm_kernels.py --check-env
step host 0 docker run --rm --entrypoint python3 "$TAG" -m pytest tests/host -q -p no:cacheprovider
step imports 0 docker run --rm --entrypoint python3 "$TAG" -c "import naivert.backends.naive_n0p5_session_engine, naivert.serve.http, naivert.prefill.native.executor, naivert.models.registry as r; print('imports ok', sorted(r.MODELS))"
step dryrun 0 docker run --rm -e NAIVERT_DRY_RUN=1 -e NAIVERT_MODEL=naive-n0.5 -e NAIVERT_MAX_DECODE_TOKENS=4096 -e NAIVERT_EXTRA_ARGS="--top-p 0.9" "$TAG"
# A startup without model mounts or GPUs must exit with an application error.
# Timeouts (124), Docker errors (125-127), and unexpected success are failures.
# Remove the named container even if timeout terminates only the Docker client.
CONTAINER="naivert-tier1-$$"
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT
step nogpu 1 timeout 300 docker run --rm --name "$CONTAINER" "$TAG"
step compose 0 docker compose -f devops/docker-compose.yml config -q
step builddir 0 docker run --rm --entrypoint sh "$TAG" -c "ls tests/build; du -sh tests/build"
docker image inspect "$TAG" --format '{{.Size}}' > "$OUT/size_bytes"
date > "$OUT/finished"
echo "TIER1_OK: logs in $OUT"
