"""The generated entry shims and bindings match their op descriptors."""

from __future__ import annotations

from pathlib import Path

from naivert.kernels import opgen
from naivert.kernels.model_ops import body

ROOT = Path(__file__).resolve().parents[2]


def test_generated_sources_match_descriptors() -> None:
    assert opgen.descriptors(), "no op descriptors found"
    assert opgen.check(ROOT) == []


def test_every_descriptor_is_compiled_by_its_bundle() -> None:
    sources = {path.resolve() for path in body.SOURCES}
    for op in opgen.descriptors():
        assert (ROOT / op.entry_path).resolve() in sources, op.name
        assert (ROOT / op.bindings_cpp_path).resolve() in sources, op.name


def test_descriptor_kernels_follow_the_naming_contract() -> None:
    for op in opgen.descriptors():
        for variant in op.variants:
            assert variant.kernel.startswith("naivert_")
            assert variant.kernel in (ROOT / op.entry_path).read_text(encoding="utf-8")
