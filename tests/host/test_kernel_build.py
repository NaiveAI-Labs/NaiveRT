from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from naivert.kernels import build


def _manifest(tmp_path: Path, **overrides: object) -> build.KernelBuildManifest:
    source = tmp_path / "source.cpp"
    source.write_text("int value = 1;\n", encoding="utf-8")
    values = {
        "module_name": "test_extension",
        "sources": (source,),
        "include_paths": (tmp_path,),
        "cflags": ("-O3",),
        "cuda_cflags": (),
        "ldflags": (),
        "build_directory": tmp_path / "build",
        "cuda_home": tmp_path,
        "with_cuda": False,
        "scoped_toolchain": False,
    }
    values.update(overrides)
    return build.KernelBuildManifest(**values)


def test_manifest_is_immutable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(FrozenInstanceError):
        manifest.module_name = "changed"  # type: ignore[misc]


def test_recursive_inventory_and_digest_are_deterministic(
    tmp_path: Path,
) -> None:
    include = tmp_path / "include.hpp"
    include.write_text("#pragma once\n", encoding="utf-8")
    source = tmp_path / "source.cpp"
    source.write_text('#include "include.hpp"\n', encoding="utf-8")
    inventory = build.recursive_source_inventory(
        tmp_path,
        (source,),
        include_roots=(tmp_path,),
    )
    assert inventory == (include.resolve(), source.resolve())
    assert build.inventory_digest(tmp_path, inventory) == (
        build.inventory_digest(tmp_path, inventory)
    )


def test_compile_extension_uses_manifest_and_checks_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(pointer_count=7, launch=object())

    monkeypatch.setattr(build, "load", fake_load)
    manifest = _manifest(
        tmp_path,
        required_symbols=("launch",),
        integer_attributes=(("pointer_count", 7),),
    )
    module = build.compile_extension(manifest)
    assert module.pointer_count == 7
    assert calls == [
        {
            "name": "test_extension",
            "sources": [str(manifest.sources[0])],
            "extra_include_paths": [str(tmp_path.resolve())],
            "extra_cflags": ["-O3"],
            "extra_cuda_cflags": [],
            "extra_ldflags": [],
            "build_directory": str((tmp_path / "build").resolve()),
            "verbose": False,
            "with_cuda": False,
        }
    ]


def test_compile_extension_rejects_digest_or_abi_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build,
        "load",
        lambda **_kwargs: SimpleNamespace(pointer_count=6),
    )
    with pytest.raises(RuntimeError, match="expected 7"):
        build.compile_extension(
            _manifest(
                tmp_path,
                integer_attributes=(("pointer_count", 7),),
            )
        )
