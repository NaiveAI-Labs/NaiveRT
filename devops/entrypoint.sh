#!/bin/bash
# NaiveRT serving container entrypoint. Environment knobs (all optional):
#   NAIVERT_MODEL (naive-n0.5)  NAIVERT_PORT (8080)  NAIVERT_MAX_DECODE_TOKENS (524288)
#   NAIVERT_CAPACITY (1048576)  NAIVERT_OUTPUT_CAPACITY (524288)
#   NAIVERT_PREFILL_WARM_CHUNKS (3)  NAIVERT_QUEUE_DEPTH (8)  NAIVERT_EXTRA_ARGS (space-separated flags)
#   NAIVERT_TARGET_CHECKPOINT / NAIVERT_DRAFT_CHECKPOINT are read by serve_sessions.py;
#   explicit --target / --draft arguments take precedence over them.
# Any command-line arguments are appended after the environment-derived ones.
set -euo pipefail
cd "$(dirname "$0")/.."
# NAIVERT_CUDA_HOME wins: base images carry an older CUDA at /usr/local/cuda and export CUDA_HOME for it,
# and the runtime's kernels need the 13.2 toolkit (cuda::ptx::elect_sync). Both variables are set to it.
export NAIVERT_CUDA_HOME="${NAIVERT_CUDA_HOME:-/usr/local/cuda-13.2}"
export CUDA_HOME="$NAIVERT_CUDA_HOME"
export NAIVERT_CUTLASS_INCLUDE="${NAIVERT_CUTLASS_INCLUDE:-/opt/cutlass/include}"
# The system compiler, pinned: the extension build dirs and the FlashInfer JIT cache key the compiler
# command, so an unpinned "c++" would recompile everything a --compile pass built with g++.
export CXX="${CXX:-/usr/bin/g++}" CC="${CC:-/usr/bin/gcc}"
# never put the toolkit's bin first on PATH: its ld breaks host executable links (only nvcc is needed, by path)
CMD=(python3 -u examples/serve_sessions.py
  --host 0.0.0.0 --port "${NAIVERT_PORT:-8080}"
  --model "${NAIVERT_MODEL:-naive-n0.5}"
  --capacity "${NAIVERT_CAPACITY:-1048576}"
  --output-capacity "${NAIVERT_OUTPUT_CAPACITY:-524288}"
  --max-decode-tokens "${NAIVERT_MAX_DECODE_TOKENS:-524288}"
  --prefill-warm-chunks "${NAIVERT_PREFILL_WARM_CHUNKS:-3}"
  --queue-depth "${NAIVERT_QUEUE_DEPTH:-8}"
  ${NAIVERT_EXTRA_ARGS:-} "$@")
if [ "${NAIVERT_DRY_RUN:-0}" = "1" ]; then
  # print the resolved environment and command, run nothing (image validation without GPUs)
  echo "NAIVERT_CUDA_HOME=$NAIVERT_CUDA_HOME CUDA_HOME=$CUDA_HOME NAIVERT_CUTLASS_INCLUDE=$NAIVERT_CUTLASS_INCLUDE CXX=$CXX CC=$CC"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
