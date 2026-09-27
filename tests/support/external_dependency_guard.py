"""Fail fast if a retired external runtime is imported during acceptance."""

from __future__ import annotations

from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
import sys


_FORBIDDEN_ROOTS = frozenset({"tilelang", "tvm"})


class _ForbiddenRuntimeFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> ModuleSpec | None:
        del path, target
        root = fullname.partition(".")[0]
        if root in _FORBIDDEN_ROOTS:
            raise ImportError(
                f"retired external runtime import is forbidden: {fullname}"
            )
        return None


def install_external_dependency_guard() -> None:
    loaded = sorted(_FORBIDDEN_ROOTS.intersection(sys.modules))
    if loaded:
        raise RuntimeError(
            "retired external runtimes were imported before the guard: "
            + ", ".join(loaded)
        )
    if not any(
        isinstance(finder, _ForbiddenRuntimeFinder)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _ForbiddenRuntimeFinder())


__all__ = ["install_external_dependency_guard"]
