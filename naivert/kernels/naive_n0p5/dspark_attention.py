"""First-class BF16 draft attention; prepare once before capturing its TP8 launches."""
from functools import lru_cache
from pathlib import Path
import os

import torch

from naivert.kernels.build import (
    KernelBuildManifest, compile_extension, cutlass_include_root, select_cuda_home,
)
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags

ROOT = Path(__file__).resolve().parents[3]
OPERATION_ID = "dspark.attention.rows8"


@lru_cache(maxsize=2)
def load_attention(*, diagnostics=False):
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    name = "naivert_dspark_attention" + ("_diagnostics" if diagnostics else "")
    return compile_extension(KernelBuildManifest(
        module_name=name,
        sources=(ROOT / "naivert/entry/dspark_attention_entry.cu",
                 ROOT / "naivert/kernels/naive_n0p5/dspark_attention_bindings.cu"),
        include_paths=(ROOT, cutlass_include_root()),
        cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                     "-gencode=arch=compute_103a,code=sm_103a",
                     *geometry_cuda_cflags(sm_count=148),
                     f"-DNAIVERT_DSPARK_DIAGNOSTICS={int(bool(diagnostics))}"),
        ldflags=(), build_directory=ROOT / "tests/build" / name,
        cuda_home=cuda, required_symbols=("configure", "attn_rows8", "attn_rows8_swa", "split_elems", "launch_shape"),
        suppress_arch_flags=True))


def prepare_attention(devices, *, diagnostics=False):
    """Set shared-memory opt-in and verify residency on every selected GPU.

    Peer access and per-rank tensors belong to the caller's runtime domain.
    Every rank must participate with a common epoch/layer and separate buffers.
    """
    selected = tuple(int(d) for d in devices)
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("D3 requires eight distinct CUDA devices")
    module = load_attention(diagnostics=diagnostics)
    for device in selected:
        module.configure(device)
    return module


def pack_weights(q, k, v, o, input_ln, q_norm, k_norm, *, rank):
    """Pack full BF16 projection matrices for TP8 without changing gamma precision.

    Checkpoint dtype conversion belongs to the caller. All tensors must share
    one device; packing runs outside the captured draft graph.
    """
    if not 0 <= rank < 8:
        raise ValueError("invalid D3 rank")
    for name, tensor, shape in (("q", q, (4096, 4096)), ("k", k, (512, 4096)),
                                ("v", v, (512, 4096)), ("o", o, (4096, 4096))):
        if tensor.shape != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"D3 {name} must be BF16 {shape}")
    for name, tensor, shape in (("input_ln", input_ln, (4096,)),
                                ("q_norm", q_norm, (128,)), ("k_norm", k_norm, (128,))):
        if tensor.shape != shape or tensor.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"D3 {name} must be BF16/FP32 {shape}")
    if len({x.device for x in (q, k, v, o, input_ln, q_norm, k_norm)}) != 1:
        raise ValueError("D3 unpacked weights must share a device")
    def pages(w):
        n, width = w.shape
        return w.contiguous().view(n // 16, 16, width // 32, 32).permute(0, 2, 1, 3).contiguous()
    return dict(q=pages(q[rank*512:(rank+1)*512]),
                k=pages(k[(rank//2)*128:(rank//2+1)*128]), v=pages(v[(rank//2)*128:(rank//2+1)*128]),
                o=pages(o[:, rank*512:(rank+1)*512]),
                input_ln=input_ln.float().contiguous(),
                q_norm=q_norm.float().contiguous(), k_norm=k_norm.float().contiguous())
