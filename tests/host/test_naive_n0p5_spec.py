"""Naive N0.5 host-side contract: current geometry, kernel builds, and package layering."""
from __future__ import annotations

import ast
from pathlib import Path

from naivert.model import HybridModelSpec
from naivert.models.naive_n0p5 import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY
from naivert.models.naive_n0p5.kernel_geometry import ALLREDUCE_GEOMETRY, geometry_cuda_cflags

ROOT = Path(__file__).resolve().parents[2]
FULL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)


def test_layer_kinds_follow_the_checkpoint_pattern() -> None:
    hybrid = NAIVE_N0P5_HYBRID
    assert isinstance(hybrid, HybridModelSpec)
    assert len(hybrid.layer_kinds) == NAIVE_N0P5_GEOMETRY.main_layer_count == 48
    assert hybrid.layers_of("full_dsa") == FULL_LAYERS
    assert len(hybrid.layers_of("sliding_window")) == 39
    assert all(hybrid.layer_has_indexer(layer) for layer in FULL_LAYERS)
    assert not hybrid.layer_has_indexer(1)


def test_tp8_head_math() -> None:
    full = NAIVE_N0P5_HYBRID.attention_layouts["full_dsa"]
    swa = NAIVE_N0P5_HYBRID.attention_layouts["sliding_window"]
    assert full.local_heads(8) == swa.local_heads(8) == 8
    assert full.local_kv_heads(8) == 1 and swa.local_kv_heads(8) == 1
    # 4 full-attention KV heads on 8 ranks: ranks 2h and 2h+1 share head h; SWA has one head per rank.
    assert [full.kv_head_for_rank(r, 8) for r in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [swa.kv_head_for_rank(r, 8) for r in range(8)] == list(range(8))


def test_geometry_flags_carry_naive_n0p5_numbers() -> None:
    flags = geometry_cuda_cflags(sm_count=148)
    assert "-DNAIVERT_HIDDEN=4096" in flags and "-DNAIVERT_TP_RANKS=8" in flags
    assert {
        "-DNAIVERT_ROUTED_EXPERTS=256",
        "-DNAIVERT_SHARED_EXPERTS=0",
        "-DNAIVERT_SELECTED_EXPERTS=8",
        "-DNAIVERT_ROUTE_KEPT_GROUPS=8",
        "-DNAIVERT_DENSE_INTER=2048",
        "-DNAIVERT_VOCAB_SHARD=19072",
        "-DNAIVERT_ROUTE_SCALE=1.0",
        "-DNAIVERT_NORM_EPS=1e-05",
    } <= set(flags)


def test_model_ops_manifest_builds_the_shared_sources_with_naive_n0p5_geometry() -> None:
    from naivert.kernels.naive_n0p5 import model_ops
    from naivert.kernels.model_ops import body
    from naivert.kernels.toolchain import SM_COUNT

    manifest = model_ops._build_manifest()
    flags = set(geometry_cuda_cflags(sm_count=SM_COUNT))
    assert flags <= set(manifest.cuda_cflags) and flags <= set(manifest.cflags)
    assert "-DNAIVERT_HIDDEN=7168" not in manifest.cuda_cflags
    assert manifest.module_name.startswith("naivert_naive_n0p5_model_ops_ext_")
    assert ALLREDUCE_GEOMETRY.unproj_projection_k == 1024
    for width in (ALLREDUCE_GEOMETRY.expert_down_outputs_per_tile, ALLREDUCE_GEOMETRY.down_outputs_per_tile,
                  ALLREDUCE_GEOMETRY.unproj_outputs_per_tile):
        assert 4096 % width == 0


def test_naive_n0p5_package_imports_no_runtime() -> None:
    package = ROOT / "naivert/models/naive_n0p5"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                assert not name.startswith(("naivert.runtime", "naivert.backends", "naivert.capture")), (path.name, name)


def test_current_fp8_indexer_geometry() -> None:
    assert NAIVE_N0P5_HYBRID.indexer.head_count == 16
    assert NAIVE_N0P5_HYBRID.indexer.head_dim == 128
    assert NAIVE_N0P5_HYBRID.weight_dtype == "fp8"
    assert NAIVE_N0P5_GEOMETRY.hidden_size == 4096
