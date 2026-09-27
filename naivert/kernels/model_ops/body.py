"""Build and bind the model-operation CUDA extension (MoE, dense MLP, head, argmax).

The sources are shared; a ModelOpsBuild says which model's geometry they are
compiled with; naivert/kernels/naive_n0p5/model_ops.py builds them for Naive N and passes its
ModelOpsBuild explicitly (this tree has no default build).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from naivert.kernels.build import (
    CUTLASS_INCLUDE_ROOT,
    KernelBuildManifest,
    compile_extension,
    environment_cuda_home,
    inventory_digest,
    recursive_source_inventory,
    select_cuda_home,
)
from naivert.kernels.toolchain import SM_COUNT
from naivert.kernel_variant import KernelVariantSpec, require_kernel_variant


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelOpsBuild:
    """Which model the shared model_ops sources are compiled for."""

    model: str  # module-name and build-directory stem
    geometry_cuda_cflags: Callable[..., tuple[str, ...]]
    variant: KernelVariantSpec

    @property
    def build_dir(self) -> Path:
        return ROOT / "tests" / "build" / f"{self.model}_model_ops"


def default_build() -> ModelOpsBuild:
    """This tree has no default build: pass the model's ModelOpsBuild explicitly
    (naivert/kernels/naive_n0p5/model_ops.py builds the Naive N variant)."""
    raise NotImplementedError("pass build= explicitly; there is no default model_ops build in this tree")

SOURCES = (
    SOURCE_DIR / "extension" / "gen_rmsnorm_expert_proj_prepared_packed_bindings.cu",
    ROOT / "naivert" / "entry" / "rmsnorm_expert_proj_prepared_packed_entry.cu",
    SOURCE_DIR / "extension" / "gen_rmsnorm_expert_proj_prepared_bindings.cu",
    SOURCE_DIR / "extension" / "gen_expert_select_up_gate_silu_rows8_prepared_bindings.cu",
    ROOT / "naivert" / "entry" / "rmsnorm_expert_proj_prepared_entry.cu",
    ROOT / "naivert" / "entry" / "expert_select_up_gate_silu_rows8_prepared_entry.cu",
    SOURCE_DIR / "extension" / "model_ops_bindings.cpp",
    SOURCE_DIR / "extension" / "checkpoint_bf16_bindings.cu",
    SOURCE_DIR / "extension" / "embed_gather_bindings.cu",
    SOURCE_DIR / "extension" / "gen_rmsnorm_head_proj_bindings.cu",
    SOURCE_DIR / "extension" / "gen_rmsnorm_expert_proj_bindings.cu",
    SOURCE_DIR / "extension" / "gen_expert_select_up_gate_silu_rows1_bindings.cu",
    SOURCE_DIR / "extension" / "gen_expert_select_up_gate_silu_rows8_bindings.cu",
    SOURCE_DIR / "extension" / "gen_unproj_o_allreduce_bindings.cu",
    SOURCE_DIR / "extension" / "gen_rmsnorm_up_gate_bindings.cu",
    SOURCE_DIR / "extension" / "gen_down_allreduce_bindings.cu",
    ROOT / "naivert" / "entry" / "rmsnorm_up_gate_entry.cu",
    ROOT / "naivert" / "entry" / "down_allreduce_entry.cu",
    ROOT / "naivert" / "entry" / "unproj_o_allreduce_accepted_row_entry.cu",
    ROOT / "naivert/entry/expert_down_domain.cu",
    ROOT / "naivert" / "entry" / "expert_down_allreduce_entry.cu",
    ROOT / "naivert" / "entry" / "unproj_o_allreduce_entry.cu",
    ROOT / "naivert" / "entry" / "rmsnorm_expert_proj_entry.cu",
    ROOT / "naivert" / "entry" / "rmsnorm_head_proj_entry.cu",
    ROOT / "naivert" / "entry" / "expert_select_up_gate_silu_rows8_entry.cu",
    ROOT / "naivert" / "entry" / "expert_select_up_gate_silu_rows1_entry.cu",
    ROOT / "naivert" / "entry" / "tp8_argmax_rows_entry.cu",
)


def source_inventory() -> tuple[Path, ...]:
    return recursive_source_inventory(
        ROOT,
        SOURCES,
        include_roots=(ROOT,),
        description="model operations",
    )


def source_digest() -> str:
    return inventory_digest(ROOT, source_inventory())


def _build_manifest(
    *,
    expected_variant: KernelVariantSpec | None = None,
    build: ModelOpsBuild | None = None,
) -> KernelBuildManifest:
    build = default_build() if build is None else build
    variant = require_kernel_variant(
        build.variant if expected_variant is None else expected_variant,
        build.variant,
    )
    digest = source_digest()
    geometry = build.geometry_cuda_cflags(sm_count=SM_COUNT)
    return KernelBuildManifest(
        module_name=f"naivert_{build.model}_model_ops_ext_{digest[:20]}",
        sources=SOURCES,
        include_paths=(ROOT, CUTLASS_INCLUDE_ROOT),
        cflags=(
            "-O3",
            "-std=c++17",
            "-pthread",
            "-fno-gnu-unique",
            *geometry,
        ),
        cuda_cflags=(
            "-O3",
            "-lineinfo",
            "-std=c++17",
            "-Xcompiler=-fno-gnu-unique",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_103a,code=sm_103a",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-Xptxas=-O3",
            "-Xptxas=-maxrregcount=255",
            "-Xptxas=--allow-expensive-optimizations=true",
            # The one ExpertDown native switch still worth being a switch:
            # it selects the parallel peer reduction over the serial one,
            # and mtp_draft_ops deliberately does not set it.
            "-DNAIVERT_EXPERT_DOWN_ALLREDUCE_NATIVE_PARALLEL_PEER_POLL=1",
            *geometry,
        ),
        ldflags=("-pthread", "-Wl,-Bsymbolic"),
        build_directory=build.build_dir / digest[:20],
        cuda_home=select_cuda_home(
            (
                environment_cuda_home(),
                "/opt/cuda132-min",
                "/usr/local/cuda",
                "/usr/local/cuda-12.4",
            ),
            fallback=environment_cuda_home() or "/opt/cuda132-min",
        ),
        required_symbols=(
            "prepared_mlp_input_bytes",
            "launch_rmsnorm_expert_proj_prepared_packed",
            "launch_rmsnorm_expert_proj_prepared_packed_pdl_producer",
            "launch_rmsnorm_expert_proj_prepared_packed_only",
            "launch_rmsnorm_expert_proj_prepared_packed_only_pdl_producer",
            "launch_rmsnorm_expert_proj_prepared",
            "launch_rmsnorm_expert_proj_prepared_only",
            "launch_rmsnorm_expert_proj_prepared_only_pdl_producer",
            "launch_expert_select_up_gate_silu_rows8_prepared",
            "launch_rmsnorm_expert_proj_prepared_pdl_producer",
            "launch_expert_select_up_gate_silu_rows8_prepared_pdl_producer",
            "enable_tp8_peer_access",
            "launch_rmsnorm_up_gate",
            "launch_rmsnorm_up_gate_pdl_consumer",
            "launch_down_allreduce",
            "launch_down_allreduce_pdl_consumer",
            "launch_unproj_allreduce",
            "launch_unproj_allreduce_replay",
            "launch_unproj_allreduce_pdl_consumer",
            "launch_unproj_allreduce_replay_pdl_consumer",
            "launch_unproj_allreduce_accepted_row",
            "prepare_expert_down_replay_domain",
            "launch_expert_down_rows_from_replay_domain",
            "launch_expert_down_rows_from_replay_domain_pdl_consumer",
            "expert_down_domain_size",
            "launch_rmsnorm_expert_proj",
            "launch_rmsnorm_expert_proj_pdl_producer",
            "launch_expert_select_up_gate_silu_rows1",
            "launch_expert_select_up_gate_silu_rows1_pdl_consumer",
            "launch_expert_select_up_gate_silu_rows8",
            "launch_expert_select_up_gate_silu_rows8_pdl_producer",
            "launch_rmsnorm_head_proj_rows1",
            "launch_rmsnorm_head_proj_rows1_pdl_consumer",
            "launch_tp8_argmax_rows",
        ),
    )


def load_extension(
    verbose: bool = False,
    *,
    build_directory: str | Path | None = None,
    expected_variant: KernelVariantSpec | None = None,
    build: ModelOpsBuild | None = None,
):
    return compile_extension(
        _build_manifest(expected_variant=expected_variant, build=build),
        build_directory=build_directory,
        verbose=verbose,
    )
