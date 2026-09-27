"""Package layering: model code below runtime below capture below backends, and kernels never import the runtime."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(ROOT).with_suffix("")
    is_package = relative.name == "__init__"
    parts = relative.parts[:-1] if is_package else relative.parts
    return ".".join(parts), is_package


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    module_name, is_package = _module_name(path)
    package_parts = module_name.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                base = package_parts[:keep]
                if node.module:
                    base.extend(node.module.split("."))
                imported_from = ".".join(base)
            elif node.module:
                imported_from = node.module
            else:
                imported_from = ""
            if imported_from:
                names.add(imported_from)
            names.update(
                (
                    f"{imported_from}.{alias.name}"
                    if imported_from
                    else alias.name
                )
                for alias in node.names
            )
    return names


def test_model_layer_dependency_direction() -> None:
    model_core = ROOT / "naivert/model.py"
    model_files = tuple((ROOT / "naivert/models").rglob("*.py"))
    runtime_files = tuple((ROOT / "naivert/runtime").rglob("*.py"))
    capture_files = tuple((ROOT / "naivert/capture").rglob("*.py"))
    model_forbidden = (
        "naivert.backends",
        "naivert.builder",
        "naivert.capture",
        "naivert.runtime",
    )
    for path in (model_core, *model_files):
        imports = _imports(path)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in model_forbidden
        ), (path, imports)
    for path in runtime_files:
        imports = _imports(path)
        runtime_forbidden = (
            "naivert.backends",
            "naivert.builder",
            "naivert.capture",
                "naivert.kernels",
            "naivert.models",
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in runtime_forbidden
        ), (path, imports)
    capture_forbidden = (
        "naivert.backends",
        "naivert.builder",
    )
    for path in capture_files:
        imports = _imports(path)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in capture_forbidden
        ), (path, imports)
    kernel_files = tuple((ROOT / "naivert/kernels").rglob("*.py"))
    for path in kernel_files:
        imports = _imports(path)
        assert not any(
            name == "naivert.runtime"
            or name.startswith("naivert.runtime.")
            for name in imports
        ), (path, imports)
    # Cross-package checks on files this tree does not carry.
    decode_runtime = ROOT / "naivert/kernels/decode_runtime.py"
    if decode_runtime.exists():
        assert "naivert.models" not in _imports(decode_runtime)
    target = ROOT / "naivert/capture/mtp/target_graph.py"
    if target.exists():
        assert "naivert.capture.mtp.production" not in _imports(target)
