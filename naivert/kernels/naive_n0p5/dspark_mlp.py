"""First-class BF16 draft MLP; prepare once before capturing its TP8 launches."""
from functools import lru_cache
from pathlib import Path
import os

import torch

from naivert.kernels.build import (
    KernelBuildManifest, compile_extension, cutlass_include_root, select_cuda_home,
)
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags

ROOT = Path(__file__).resolve().parents[3]
OPERATION_ID = "dspark.mlp.rows8"


@lru_cache(maxsize=2)
def load_mlp(*, diagnostics=False):
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    name = "naivert_dspark_mlp" + ("_diagnostics" if diagnostics else "")
    return compile_extension(KernelBuildManifest(
        module_name=name,
        sources=(ROOT / "naivert/entry/dspark_mlp_entry.cu",
                 ROOT / "naivert/kernels/naive_n0p5/dspark_mlp_bindings.cu"),
        include_paths=(ROOT, cutlass_include_root()),
        cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                     "-gencode=arch=compute_103a,code=sm_103a",
                     *geometry_cuda_cflags(sm_count=148),
                     f"-DNAIVERT_DSPARK_DIAGNOSTICS={int(bool(diagnostics))}"),
        ldflags=(), build_directory=ROOT / "tests/build" / name,
        cuda_home=cuda, required_symbols=("configure", "mlp", "launch_shape"),
        suppress_arch_flags=True))


def prepare_mlp(devices, *, diagnostics=False):
    """Set shared-memory opt-in and verify residency on every selected GPU.

    Peer access and per-rank tensors belong to the caller's runtime domain.
    Every rank must participate with a common epoch/layer and separate buffers.
    """
    selected = tuple(int(d) for d in devices)
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("D4 requires eight distinct CUDA devices")
    module = load_mlp(diagnostics=diagnostics)
    for device in selected:
        module.configure(device)
    return module


def pack_weights(gate, up, down, gamma, *, rank):
    """Pack full BF16 matrices and gamma for one TP rank, outside the graph.

    Gamma is converted to FP32 without first reducing its precision. The caller
    owns checkpoint dtype conversion; this function only changes storage layout.
    """
    if not 0 <= rank < 8:
        raise ValueError("invalid D4 rank")
    for name, tensor, shape in (("gate", gate, (4096, 4096)),
                                ("up", up, (4096, 4096)),
                                ("down", down, (4096, 4096))):
        if tensor.shape != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"D4 {name} must be BF16 {shape}")
    if gamma.shape != (4096,) or gamma.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("D4 gamma must be BF16/FP32 [4096]")
    if len({x.device for x in (gate, up, down, gamma)}) != 1:
        raise ValueError("D4 unpacked weights must share a device")
    gu = torch.stack((gate[rank * 512:(rank + 1) * 512],
                      up[rank * 512:(rank + 1) * 512]), dim=1).reshape(1024, 4096)
    gu = gu.view(64, 16, 128, 32).permute(0, 2, 1, 3).contiguous()
    dn = down[:, rank * 512:(rank + 1) * 512].contiguous()
    dn = dn.view(256, 16, 16, 32).permute(0, 2, 1, 3).contiguous()
    gn = gamma.float().view(16, 32, 2, 4).permute(0, 2, 1, 3).contiguous()
    return {"gate_up": gu, "down_w": dn, "post_ln": gn}
