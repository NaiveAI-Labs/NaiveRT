from __future__ import annotations

import os

import pytest
import torch.utils.cpp_extension as cpp_extension

from naivert.kernels.toolchain import scoped_cuda_toolchain


def test_scoped_cuda_toolchain_restores_process_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    selected = tmp_path / "cuda"
    selected.mkdir()
    monkeypatch.setenv("CUDA_HOME", "/before/cuda")
    previous_cpp = cpp_extension.CUDA_HOME
    previous_arch = cpp_extension._get_cuda_arch_flags

    with scoped_cuda_toolchain(
        selected,
        suppress_arch_flags=True,
    ):
        assert os.environ["CUDA_HOME"] == str(selected.resolve())
        assert cpp_extension.CUDA_HOME == str(selected.resolve())
        assert cpp_extension._get_cuda_arch_flags is not previous_arch

    assert os.environ["CUDA_HOME"] == "/before/cuda"
    assert cpp_extension.CUDA_HOME == previous_cpp
    assert cpp_extension._get_cuda_arch_flags is previous_arch


def test_scoped_cuda_toolchain_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    selected = tmp_path / "cuda"
    selected.mkdir()
    monkeypatch.delenv("CUDA_HOME", raising=False)
    previous_cpp = cpp_extension.CUDA_HOME
    previous_arch = cpp_extension._get_cuda_arch_flags

    with pytest.raises(RuntimeError, match="probe"):
        with scoped_cuda_toolchain(
            selected,
            suppress_arch_flags=True,
        ):
            raise RuntimeError("probe")

    assert "CUDA_HOME" not in os.environ
    assert cpp_extension.CUDA_HOME == previous_cpp
    assert cpp_extension._get_cuda_arch_flags is previous_arch
