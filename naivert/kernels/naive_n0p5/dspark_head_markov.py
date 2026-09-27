"""First-class BF16 draft head and seven-step Markov selection; prepare before capture."""
from functools import lru_cache
from pathlib import Path
import os


def temperature_flags():
    """NAIVERT_DSPARK_INV_TEMPERATURE=<float>: sampling temperature 1/T for D56's draw and the target sampler."""
    value = os.environ.get("NAIVERT_DSPARK_INV_TEMPERATURE")
    if not value:
        return (), ""
    inv = float(value)
    if not 0 < inv < 1000:
        raise ValueError("NAIVERT_DSPARK_INV_TEMPERATURE out of range")
    return (f"-DDSPARK_INV_TEMPERATURE={inv!r}f",), "_invt" + value.replace(".", "p").replace("-", "m")

import torch

from naivert.kernels.build import (
    KernelBuildManifest, compile_extension, cutlass_include_root, select_cuda_home,
)
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags

ROOT = Path(__file__).resolve().parents[3]
OPERATION_ID = "dspark.head_markov.rows8"


@lru_cache(maxsize=2)
def load_head_markov(*, diagnostics=False):
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    tflags, tsuffix = temperature_flags()
    name = "naivert_dspark_head_markov" + ("_diagnostics" if diagnostics else "") + tsuffix
    return compile_extension(KernelBuildManifest(
        module_name=name,
        sources=(ROOT / "naivert/entry/dspark_head_markov_entry.cu",
                 ROOT / "naivert/kernels/naive_n0p5/dspark_head_markov_bindings.cu"),
        include_paths=(ROOT, cutlass_include_root()), cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                     "-gencode=arch=compute_103a,code=sm_103a",
                     *geometry_cuda_cflags(sm_count=148),
                     f"-DNAIVERT_DSPARK_DIAGNOSTICS={int(bool(diagnostics))}", *tflags),
        ldflags=(), build_directory=ROOT / "tests/build" / name,
        cuda_home=cuda, required_symbols=("configure", "head_rows8", "markov_block",
            "head_markov", "prepare_markov_bias", "launch_shape"), suppress_arch_flags=True))


def prepare_head_markov(devices, *, diagnostics=False):
    """Configure all three entries on TP8 before capture; peer access is caller-owned."""
    selected = tuple(int(d) for d in devices)
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("D5/D6 requires eight distinct CUDA devices")
    module = load_head_markov(diagnostics=diagnostics)
    for device in selected:
        module.configure(device)
    return module


def pack_head_weights(head_shard, gamma):
    """Pack one BF16 [19072,4096] vocabulary shard and preserve FP32 gamma."""
    if (head_shard.shape != (19072, 4096) or head_shard.dtype != torch.bfloat16 or
            gamma.shape != (4096,) or gamma.dtype not in (torch.bfloat16, torch.float32) or
            gamma.device != head_shard.device):
        raise ValueError("invalid D5 head shard or gamma")
    return dict(
        lm_head=head_shard.reshape(1192, 16, 4, 32, 32).permute(0, 2, 3, 1, 4).contiguous(),
        norm_gamma=gamma.float().reshape(16, 32, 2, 4).permute(0, 2, 1, 3).contiguous().reshape(4096))
