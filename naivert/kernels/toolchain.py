"""Scoped CUDA toolchain configuration for local extension builds."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
from typing import Iterator

# B300 (sm_103a) streaming multiprocessors. The persistent TP8 kernels launch one
# CTA per SM; kernel builds receive this as -DNAIVERT_SM_COUNT.
SM_COUNT = 148


@contextmanager
def scoped_cuda_toolchain(
    cuda_home: str | Path,
    *,
    suppress_arch_flags: bool = False,
) -> Iterator[None]:
    """Temporarily configure PyTorch's extension builder and restore it."""

    import torch.utils.cpp_extension as cpp_extension

    missing = object()
    selected = str(Path(cuda_home).resolve())
    previous_env = os.environ.get("CUDA_HOME", missing)
    previous_cpp = cpp_extension.CUDA_HOME
    previous_arch_flags = cpp_extension._get_cuda_arch_flags
    try:
        os.environ["CUDA_HOME"] = selected
        cpp_extension.CUDA_HOME = selected
        if suppress_arch_flags:
            cpp_extension._get_cuda_arch_flags = lambda cflags=None: []
        root = Path(selected)
        for directory in (
            root / "lib64",
            root / "targets/x86_64-linux/lib",
        ):
            for cudart in sorted(
                directory.glob("libcudart.so*"),
                reverse=True,
            ):
                if cudart.is_file():
                    ctypes.CDLL(
                        str(cudart),
                        mode=ctypes.RTLD_GLOBAL,
                    )
                    break
            else:
                continue
            break
        yield
    finally:
        cpp_extension._get_cuda_arch_flags = previous_arch_flags
        cpp_extension.CUDA_HOME = previous_cpp
        if previous_env is missing:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = str(previous_env)


__all__ = ["scoped_cuda_toolchain"]
