# NaiveRT serving image: the Naive N model series on eight Blackwell GPUs with UMMA support (built for SM103a).
#
# Base: the SGLang image the runtime was validated in (torch 2.11.0+cu130, flashinfer 0.6.12,
# sglang-kernel 0.4.4, sgl-deep-gemm 0.1.4, nvidia-cutlass-dsl 4.5.2, DeepEP 1.2.1, triton 3.6.0).
# Added: a CUDA 13.2 toolkit for the runtime's own extensions (sm_103a needs nvcc >= 13.1),
# CUTLASS v4.7.1 headers, the serving Python pins, and a compile pass over the extensions.
# Build (from the repository root):   docker build -f devops/Dockerfile -t naivert:latest .
# Run:                                 see devops/docker-compose.yml or devops/README.md
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.15.post1
FROM ${BASE_IMAGE}

ARG CUDA_PKG_VERSION=13-2
ARG CUTLASS_TAG=v4.7.1
ENV DEBIAN_FRONTEND=noninteractive
# The NVIDIA CUDA apt repository is configured in the base image; only the toolkit pieces nvcc needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_PKG_VERSION} cuda-cudart-dev-${CUDA_PKG_VERSION} cuda-nvrtc-dev-${CUDA_PKG_VERSION} \
        cuda-nvtx-${CUDA_PKG_VERSION} cuda-cccl-${CUDA_PKG_VERSION} cuda-nvml-dev-${CUDA_PKG_VERSION} \
        libcublas-dev-${CUDA_PKG_VERSION} libcusparse-dev-${CUDA_PKG_VERSION} libcusolver-dev-${CUDA_PKG_VERSION} \
        libcurand-dev-${CUDA_PKG_VERSION} libcufft-dev-${CUDA_PKG_VERSION} \
        git ninja-build curl \
    && rm -rf /var/lib/apt/lists/*
ENV NAIVERT_CUDA_HOME=/usr/local/cuda-13.2 \
    CUDA_HOME=/usr/local/cuda-13.2 \
    CXX=/usr/bin/g++ \
    CC=/usr/bin/gcc \
    TORCH_CUDA_ARCH_LIST=10.3a \
    FLASHINFER_CUDA_ARCH_LIST=10.3a \
    LIBRARY_PATH=/usr/local/cuda-13.2/lib64/stubs
# The arch lists let the vendored DeepEP and FlashInfer JIT builds compile with no GPU visible (image build);
# the stubs directory provides libcuda.so for -lcuda links on a build host without a driver. torch's ATen
# CUDA headers include cusparse/cusolver/curand/cufft, hence those -dev packages beside nvcc and cudart.
# Do NOT prepend /usr/local/cuda-13.2/bin to PATH: its ld breaks host executable links (see devops/README.md).

RUN git clone --depth 1 --branch ${CUTLASS_TAG} https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV NAIVERT_CUTLASS_INCLUDE=/opt/cutlass/include

WORKDIR /opt/naivert
COPY devops/requirements-serve.txt devops/requirements-serve.txt
RUN pip install --no-cache-dir -r devops/requirements-serve.txt
COPY . .
RUN python3 devops/warm_kernels.py --check-env || true
# Compile the runtime's extensions into tests/build (kernels that configure devices are skipped
# without GPUs and build at first start); a failure here is not fatal for the image.
# NAIVERT_COMPILE=0 skips it for a structural build (no CUDA compiler run, e.g. on a laptop).
ARG NAIVERT_COMPILE=1
RUN if [ "$NAIVERT_COMPILE" = "1" ]; then python3 devops/warm_kernels.py --compile || echo "extension compile incomplete; remaining kernels build at first start"; fi

ENV NAIVERT_MODEL=naive-n0.5 \
    NAIVERT_PORT=8080 \
    NAIVERT_CAPACITY=1048576 \
    NAIVERT_OUTPUT_CAPACITY=524288 \
    NAIVERT_MAX_DECODE_TOKENS=524288 \
    NAIVERT_PREFILL_WARM_CHUNKS=3 \
    NAIVERT_QUEUE_DEPTH=8
EXPOSE 8080
# /health is the readiness signal (/v1/models answers before the engine is loaded); a cold start takes
# minutes: weights, native-prefill operands, chunk warm-up and any first-use JIT kernels.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${NAIVERT_PORT}/health || exit 1
ENTRYPOINT ["bash", "devops/entrypoint.sh"]
