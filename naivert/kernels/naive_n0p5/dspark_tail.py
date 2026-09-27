"""Device geometry, acceptance and output tail for the Naive N0.5 draft round."""
from functools import lru_cache
import os
from pathlib import Path

from naivert.kernels.build import KernelBuildManifest, compile_extension, select_cuda_home

ROOT = Path(__file__).resolve().parents[3]
OPERATION_ID = "dspark.verify_tail.rows8"


@lru_cache(maxsize=2)
def load_tail(no_ready=False):
    """Build before capture; epilogue's chain/staged choices are capture-time."""
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    suffix = "_no_ready" if no_ready else ""
    flags = ("-DNAIVERT_DSPARK_SAMPLER_NO_READY=1",) if no_ready else ()
    return compile_extension(KernelBuildManifest(
        module_name="naivert_dspark_tail" + suffix,
        sources=(ROOT / "naivert/entry/dspark_tail_entry.cu",
                 ROOT / "naivert/kernels/naive_n0p5/dspark_tail_bindings.cu"),
        include_paths=(ROOT,), cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                     "-gencode=arch=compute_103a,code=sm_103a", *flags),
        ldflags=(), build_directory=ROOT / ("tests/build/naivert_dspark_tail" + suffix),
        cuda_home=cuda, required_symbols=("geometry", "accept", "epilogue",
                                        "geometry_fields", "accept_fields", "argument_bytes", "layouts"),
        suppress_arch_flags=True))
