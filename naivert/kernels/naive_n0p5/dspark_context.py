"""First-class BF16 draft context preparation; prepare once before capturing its TP8 launches."""
from functools import lru_cache
from pathlib import Path
import os

import torch

from naivert.kernels.build import (
    KernelBuildManifest, compile_extension, cutlass_include_root, select_cuda_home,
)
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags

ROOT = Path(__file__).resolve().parents[3]
OPERATION_ID = "dspark.context.rows8"


@lru_cache(maxsize=2)
def load_context(*, diagnostics=False):
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    name = "naivert_dspark_context" + ("_diagnostics" if diagnostics else "")
    return compile_extension(KernelBuildManifest(
        module_name=name,
        sources=(ROOT / "naivert/entry/dspark_context_entry.cu",
                 ROOT / "naivert/kernels/naive_n0p5/dspark_context_bindings.cu"),
        include_paths=(ROOT, cutlass_include_root()),
        cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                     "-gencode=arch=compute_103a,code=sm_103a",
                     *geometry_cuda_cflags(sm_count=148),
                     f"-DNAIVERT_DSPARK_DIAGNOSTICS={int(bool(diagnostics))}"),
        ldflags=(), build_directory=ROOT / "tests/build" / name,
        cuda_home=cuda, required_symbols=("configure", "fc_ctx", "ctx_kv", "fc_ctx_kv", "partial_elements", "mailbox_bytes", "launch_shape"),
        suppress_arch_flags=True))


def prepare_context(devices, *, diagnostics=False):
    """Set shared-memory opt-in and verify residency on every selected GPU.

    Peer access and per-rank tensors belong to the caller's runtime domain.
    Every rank must participate with a common epoch/layer and separate buffers.
    """
    selected = tuple(int(d) for d in devices)
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("D12 requires eight distinct CUDA devices")
    module = load_context(diagnostics=diagnostics)
    for device in selected:
        module.configure(device)
    return module


def pack_weights(fc, hidden_norm, layers, *, rank):
    """Pack FC and shard five layers of row-major KV; preserve supplied gamma.

    Inputs are the full BF16 checkpoint matrices after caller-owned dtype
    conversion. Run packing before graph capture.
    """
    if not 0 <= rank < 8 or len(layers) != 5:
        raise ValueError("D12 requires rank 0..7 and five KV layers")
    def check(tensor, shape, dtypes, name):
        if tensor.shape != shape or tensor.dtype not in dtypes or tensor.device != fc.device:
            raise ValueError(f"invalid D12 {name}; expected {shape} on {fc.device}")
    check(fc, (4096, 32768), (torch.bfloat16,), "FC")
    check(hidden_norm, (4096,), (torch.bfloat16, torch.float32), "hidden gamma")
    shard=fc[:, rank*4096:(rank+1)*4096].contiguous()
    out=dict(fc=shard.view(256,16,128,32).permute(0,2,1,3).contiguous(),
             hidden_norm=hidden_norm.float().contiguous(), k=[], v=[], k_norm=[])
    for layer in layers:
        for name in ("k", "v"):
            check(layer[name], (512,4096), (torch.bfloat16,), name)
            out[name].append(layer[name][(rank//2)*128:(rank//2+1)*128].contiguous())
        check(layer["k_norm"], (128,), (torch.bfloat16,torch.float32), "K gamma")
        out["k_norm"].append(layer["k_norm"].float().contiguous())
    return out
