"""Build the native prefill expert-return extension through the shared kernel builder.

The manifest selects C++17, SM103a device code, compact mailbox geometry, and the
configured CUDA toolchain. Its module name includes a digest of sources and build
flags so edits produce a distinct build. Only ``naivert.kernels.build`` invokes
the PyTorch extension loader.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from naivert.kernels.build import (
    KernelBuildManifest,
    compile_extension,
    file_sha256,
    select_cuda_home,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = (HERE / "native_combine_bindings.cpp", HERE / "native_combine.cu")
GEOMETRY = dict(TP_RANKS=8, HIDDEN=4096, SM_COUNT=148, ROUTED_EXPERTS=256, SHARED_EXPERTS=0,
                SELECTED_EXPERTS=8, ROUTE_KEPT_GROUPS=8, DENSE_INTER=2048, VOCAB_SHARD=19072,
                ROUTE_SCALE=1.0, NORM_EPS=1e-5, EXPERT_WEIGHT_FORMAT=0)
CFLAGS = ("-O3", "-std=c++17")
CUDA_CFLAGS = ("-O3", "-std=c++17", "--expt-relaxed-constexpr",
               "-gencode=arch=compute_103a,code=sm_103a", "-DNAIVERT_COMBINE_COMPACT=1",
               *(f"-DNAIVERT_{k}={v}" for k, v in GEOMETRY.items()))
BUILD_DIRECTORY = ROOT / "tests/build/prefill_native_combine"


def build_manifest(cuda_home: str | os.PathLike | None = None) -> KernelBuildManifest:
    """The combine manifest; ``cuda_home`` then NAIVERT_CUDA_HOME then CUDA_HOME, as before."""
    home = select_cuda_home(
        (cuda_home, os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"),
         "/opt/cuda132-min", "/opt/cuda128"),
        fallback="/opt/cuda128",
    )
    digest = hashlib.sha256(
        "\n".join([*(file_sha256(path) for path in SOURCES), *CFLAGS, *CUDA_CFLAGS]).encode()
    ).hexdigest()
    return KernelBuildManifest(
        module_name=f"naivert_prefill_native_combine_{digest[:16]}",
        sources=SOURCES,
        include_paths=(ROOT,),
        cflags=CFLAGS,
        cuda_cflags=CUDA_CFLAGS,
        ldflags=(),
        build_directory=BUILD_DIRECTORY,
        cuda_home=home,
        suppress_arch_flags=True,
    )


def load_combine_extension(cuda_home: str | os.PathLike | None = None, *, verbose: bool = False) -> Any:
    """Compile (once per source digest) and load the combine kernel through the common builder."""
    return compile_extension(build_manifest(cuda_home), verbose=verbose)


__all__ = ["build_manifest", "load_combine_extension", "BUILD_DIRECTORY", "SOURCES"]
