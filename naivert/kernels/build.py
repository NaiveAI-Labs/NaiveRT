"""One inventory, validation, and build path for local CUDA extensions."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from torch.utils.cpp_extension import load

from naivert.kernels.toolchain import scoped_cuda_toolchain


_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOCAL_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*"([^"]+)"',
    re.MULTILINE,
)


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    selected_root = root.resolve()
    return (
        resolved == selected_root
        or selected_root in resolved.parents
    )


def recursive_source_inventory(
    root: Path,
    seeds: Iterable[Path],
    *,
    include_roots: Iterable[Path] = (),
    relative_to_source: bool = True,
    description: str = "kernel source",
) -> tuple[Path, ...]:
    """Discover local quoted includes using deterministic repository paths."""

    selected_root = Path(root).resolve()
    roots = tuple(Path(path).resolve() for path in include_roots)
    pending = [Path(path).resolve() for path in seeds]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        if not path.is_file():
            raise RuntimeError(f"{description} dependency is missing: {path}")
        if not _inside(path, selected_root):
            raise RuntimeError(
                f"{description} dependency escaped its root: {path}"
            )
        discovered.add(path)
        text = path.read_text(encoding="utf-8")
        for include in _LOCAL_INCLUDE_RE.findall(text):
            candidates = []
            if relative_to_source:
                candidates.append(path.parent / include)
            candidates.extend(search_root / include for search_root in roots)
            dependency = next(
                (
                    candidate.resolve()
                    for candidate in candidates
                    if candidate.is_file()
                    and _inside(candidate, selected_root)
                ),
                None,
            )
            if dependency is not None:
                pending.append(dependency)
    return tuple(
        sorted(
            discovered,
            key=lambda path: path.relative_to(selected_root).as_posix(),
        )
    )


def inventory_digest(root: Path, paths: Iterable[Path]) -> str:
    """Hash file bytes and root-relative names in the supplied order."""

    selected_root = Path(root).resolve()
    lines = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"kernel source dependency is missing: {path}")
        if not _inside(path, selected_root):
            raise RuntimeError(
                f"kernel source dependency escaped its root: {path}"
            )
        lines.append(
            f"{file_sha256(path)}  "
            f"{path.relative_to(selected_root).as_posix()}\n"
        )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one source file."""

    selected = Path(path)
    if not selected.is_file():
        raise RuntimeError(f"kernel source dependency is missing: {selected}")
    return hashlib.sha256(selected.read_bytes()).hexdigest()


def digest_records(records: Iterable[tuple[str, str]]) -> str:
    """Hash already-computed digests with stable semantic labels."""

    lines = []
    for raw_digest, raw_label in records:
        digest = str(raw_digest).strip().lower()
        label = str(raw_label).strip()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError("kernel digest record must contain SHA-256")
        if not label or "\n" in label:
            raise ValueError("kernel digest record label must be one line")
        lines.append(f"{digest}  {label}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def select_cuda_home(
    candidates: Iterable[str | Path | None],
    *,
    fallback: str | Path,
) -> Path:
    """Select the first candidate containing nvcc, else return the fallback."""

    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if (path / "bin/nvcc").is_file():
            return path.resolve()
    return Path(fallback).expanduser().resolve()


def cuda_arch_flags(cuda_home: str | Path) -> tuple[str, ...]:
    """Return the production runtime's supported CUDA architecture flags."""

    selected = Path(cuda_home)
    flags = [
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_100,code=compute_100",
    ]
    try:
        output = subprocess.run(
            [str(selected / "bin/nvcc"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"release\s+(\d+)\.(\d+)", output)
    except (OSError, subprocess.CalledProcessError):
        match = None
    if match is not None and tuple(map(int, match.groups())) >= (13, 1):
        flags.append("-gencode=arch=compute_103a,code=sm_103a")
    return tuple(flags)


def cutlass_include_root() -> Path:
    """CUTLASS headers the slot-ring protocol builds on.

    The runtime pins CUTLASS 4.7.1 checked out beside this repository
    (../third_party/cutlass); NAIVERT_CUTLASS_INCLUDE overrides the location.
    """

    override = os.environ.get("NAIVERT_CUTLASS_INCLUDE")
    if override:
        return Path(override).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2].parent
        / "third_party"
        / "cutlass"
        / "include"
    ).resolve()


CUTLASS_INCLUDE_ROOT = cutlass_include_root()


@dataclass(frozen=True)
class KernelBuildManifest:
    """Complete, declarative input to one PyTorch extension build."""

    module_name: str
    sources: tuple[Path, ...]
    include_paths: tuple[Path, ...]
    cflags: tuple[str, ...]
    cuda_cflags: tuple[str, ...]
    ldflags: tuple[str, ...]
    build_directory: Path
    cuda_home: Path
    required_symbols: tuple[str, ...] = ()
    integer_attributes: tuple[tuple[str, int], ...] = ()
    with_cuda: bool = True
    scoped_toolchain: bool = True
    suppress_arch_flags: bool = False

    def __post_init__(self) -> None:
        name = str(self.module_name)
        if not _MODULE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid extension module name: {name!r}")
        sources = tuple(Path(path).resolve() for path in self.sources)
        includes = tuple(
            Path(path).resolve() for path in self.include_paths
        )
        if not sources:
            raise ValueError("kernel build manifest requires source files")
        object.__setattr__(self, "module_name", name)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "include_paths", includes)
        object.__setattr__(
            self,
            "build_directory",
            Path(self.build_directory).resolve(),
        )
        object.__setattr__(
            self,
            "cuda_home",
            Path(self.cuda_home).resolve(),
        )
        for field_name in ("cflags", "cuda_cflags", "ldflags"):
            object.__setattr__(
                self,
                field_name,
                tuple(str(value) for value in getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "required_symbols",
            tuple(str(value) for value in self.required_symbols),
        )
        object.__setattr__(
            self,
            "integer_attributes",
            tuple(
                (str(attribute), int(expected))
                for attribute, expected in self.integer_attributes
            ),
        )


def _validate_manifest(manifest: KernelBuildManifest) -> None:
    missing = [path for path in manifest.sources if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{manifest.module_name} source is missing: {missing[0]}"
        )
    missing_includes = [
        path for path in manifest.include_paths if not path.is_dir()
    ]
    if missing_includes:
        raise RuntimeError(
            f"{manifest.module_name} include root is missing: "
            f"{missing_includes[0]}"
        )
    if manifest.with_cuda and not (
        manifest.cuda_home / "bin/nvcc"
    ).is_file():
        raise RuntimeError(
            f"CUDA toolchain is missing under {manifest.cuda_home}"
        )


def _validate_abi(manifest: KernelBuildManifest, module: Any) -> None:
    missing = [
        name
        for name in manifest.required_symbols
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"{manifest.module_name} ABI is missing: {', '.join(missing)}"
        )
    for name, expected in manifest.integer_attributes:
        try:
            observed = int(getattr(module, name))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{manifest.module_name} ABI attribute {name} is invalid"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"{manifest.module_name} ABI {name} is {observed}, "
                f"expected {expected}"
            )


def compile_extension(
    manifest: KernelBuildManifest,
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    """Validate, compile, load, and ABI-check one extension manifest."""

    _validate_manifest(manifest)
    build = (
        manifest.build_directory
        if build_directory is None
        else Path(build_directory).resolve()
    )
    build.mkdir(parents=True, exist_ok=True)
    context = (
        scoped_cuda_toolchain(
            manifest.cuda_home,
            suppress_arch_flags=manifest.suppress_arch_flags,
        )
        if manifest.scoped_toolchain
        else nullcontext()
    )
    with context:
        module = load(
            name=manifest.module_name,
            sources=[str(path) for path in manifest.sources],
            extra_include_paths=[
                str(path) for path in manifest.include_paths
            ],
            extra_cflags=list(manifest.cflags),
            extra_cuda_cflags=list(manifest.cuda_cflags),
            extra_ldflags=list(manifest.ldflags),
            build_directory=str(build),
            verbose=bool(verbose),
            with_cuda=bool(manifest.with_cuda),
        )
    _validate_abi(manifest, module)
    return module


def environment_cuda_home() -> str | None:
    """Return CUDA_HOME without making loader modules import os."""

    return os.environ.get("CUDA_HOME")


__all__ = [
    "KernelBuildManifest",
    "compile_extension",
    "cuda_arch_flags",
    "digest_records",
    "environment_cuda_home",
    "file_sha256",
    "inventory_digest",
    "recursive_source_inventory",
    "select_cuda_home",
]
