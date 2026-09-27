"""Small, immutable model metadata shared by production components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, TypeVar


class ModelPackageError(ValueError):
    """A checkpoint or prepared model violates its declared contract."""


def _positive(name: str, value: int) -> int:
    selected = int(value)
    if selected <= 0:
        raise ValueError(f"{name} must be positive")
    return selected


def _sha256(name: str, value: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef"
        for character in selected
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return selected


def _names(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(str(value).strip() for value in values)
    if (
        not selected
        or any(not value for value in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError(f"{name} must contain unique non-empty names")
    return selected


@dataclass(frozen=True)
class CheckpointSpec:
    """Identity of one backend-compatible local checkpoint conversion."""

    relative_local_root: str
    index_sha256: str
    shard_count: int
    shard_total_bytes: int
    shard_inventory_sha256: str
    tensor_count: int
    tensor_metadata_sha256: str
    tensor_dtype_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        root = str(self.relative_local_root).strip().strip("/")
        if not root or Path(root).is_absolute() or ".." in Path(root).parts:
            raise ValueError(
                "checkpoint relative_local_root must stay under /local_models"
            )
        object.__setattr__(self, "relative_local_root", root)
        for name in (
            "index_sha256",
            "shard_inventory_sha256",
            "tensor_metadata_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(name, getattr(self, name)),
            )
        for name in (
            "shard_count",
            "shard_total_bytes",
            "tensor_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive(name, getattr(self, name)),
            )
        dtype_counts = {
            str(dtype).strip(): _positive(
                f"tensor_dtype_counts[{dtype!r}]",
                count,
            )
            for dtype, count in self.tensor_dtype_counts.items()
        }
        if (
            not dtype_counts
            or "" in dtype_counts
            or sum(dtype_counts.values()) != self.tensor_count
        ):
            raise ValueError(
                "tensor dtype counts must cover every checkpoint tensor"
            )
        object.__setattr__(
            self,
            "tensor_dtype_counts",
            MappingProxyType(dtype_counts),
        )

    @property
    def local_root(self) -> Path:
        return Path("/local_models") / self.relative_local_root


@dataclass(frozen=True)
class Seq1Plan:
    """One model-owned Seq1 graph/kernel specialization."""

    plan_id: str
    kernel_capability: str
    abi_version: int
    stage_roles: tuple[str, ...]
    max_cur_pos: int | None = None
    # Bound of the fast short-context kernel; requests inside it take that graph.
    short_max_cur_pos: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_cur_pos", "short_max_cur_pos"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"Seq1 {name} must be non-negative")
        if (self.max_cur_pos is not None and self.short_max_cur_pos is not None
                and int(self.short_max_cur_pos) > int(self.max_cur_pos)):
            raise ValueError("Seq1 short_max_cur_pos cannot exceed max_cur_pos")
        for name in ("plan_id", "kernel_capability"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"Seq1 {name} must be non-empty")
            object.__setattr__(self, name, value)
        if int(self.abi_version) != 1:
            raise ValueError("Seq1 currently supports ABI v1")
        object.__setattr__(self, "abi_version", 1)
        object.__setattr__(
            self,
            "stage_roles",
            _names("Seq1 stage_roles", self.stage_roles),
        )
        if len(self.stage_roles) != 3:
            raise ValueError("Seq1 plan requires the three-stage graph ABI")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "naivert-seq1-plan-v1",
            "plan_id": self.plan_id,
            "kernel_capability": self.kernel_capability,
            "abi_version": self.abi_version,
            "stage_roles": list(self.stage_roles),
        }


@dataclass(frozen=True)
class MtpPlan:
    """One native-MTP method bound to a fixed compiled kernel bundle."""

    plan_id: str
    kernel_capability: str
    abi_version: int
    draft_depth: int
    mtp_module_ids: tuple[int, ...]
    proposal_layout: str
    verification_policy: str
    bonus_token_policy: str
    cache_commit_policy: str
    cache_rollback_policy: str
    eos_policy: str
    output_limit_policy: str
    iteration_policy: str
    output_count_policy: str
    stage_roles: tuple[str, ...]
    draft_stage_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        string_fields = (
            "plan_id",
            "kernel_capability",
            "proposal_layout",
            "verification_policy",
            "bonus_token_policy",
            "cache_commit_policy",
            "cache_rollback_policy",
            "eos_policy",
            "output_limit_policy",
            "iteration_policy",
            "output_count_policy",
        )
        for name in string_fields:
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"MTP {name} must be non-empty")
            object.__setattr__(self, name, value)
        abi_version = int(self.abi_version)
        depth = _positive("draft_depth", self.draft_depth)
        if abi_version != 1:
            raise ValueError("only the accepted native-MTP ABI v1 exists")
        if depth > 3:
            raise ValueError(
                "native-MTP draft depth above 3 requires ABI v2"
            )
        module_ids = tuple(int(value) for value in self.mtp_module_ids)
        if (
            not module_ids
            or any(value < 0 for value in module_ids)
            or len(module_ids) != len(set(module_ids))
        ):
            raise ValueError(
                "MTP module IDs must be unique non-negative indices"
            )
        stage_roles = _names("MTP stage_roles", self.stage_roles)
        draft_roles = _names(
            "MTP draft_stage_roles",
            self.draft_stage_roles,
        )
        if len(draft_roles) != depth:
            raise ValueError(
                "MTP draft stage roles must equal draft depth"
            )
        if len(stage_roles) != 3:
            raise ValueError(
                "native MTP plan requires the three-stage graph ABI"
            )
        object.__setattr__(self, "abi_version", abi_version)
        object.__setattr__(self, "draft_depth", depth)
        object.__setattr__(self, "mtp_module_ids", module_ids)
        object.__setattr__(self, "stage_roles", stage_roles)
        object.__setattr__(self, "draft_stage_roles", draft_roles)

    @property
    def proposal_row_capacity(self) -> int:
        return self.draft_depth + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "naivert-native-mtp-plan-v1",
            "plan_id": self.plan_id,
            "kernel_capability": self.kernel_capability,
            "abi_version": self.abi_version,
            "draft_depth": self.draft_depth,
            "mtp_module_ids": list(self.mtp_module_ids),
            "proposal_layout": self.proposal_layout,
            "proposal_row_capacity": self.proposal_row_capacity,
            "verification_policy": self.verification_policy,
            "bonus_token_policy": self.bonus_token_policy,
            "cache_commit_policy": self.cache_commit_policy,
            "cache_rollback_policy": self.cache_rollback_policy,
            "eos_policy": self.eos_policy,
            "output_limit_policy": self.output_limit_policy,
            "iteration_policy": self.iteration_policy,
            "output_count_policy": self.output_count_policy,
            "stage_roles": list(self.stage_roles),
            "draft_stage_roles": list(self.draft_stage_roles),
        }


@dataclass(frozen=True)
class ModelSpec:
    """Only the model facts consumed by the current source runtime."""

    model_id: str
    architecture: str
    checkpoint: CheckpointSpec
    rank_count: int
    replay_slot_count: int
    main_layer_count: int
    dense_layer_count: int
    hidden_size: int
    dense_intermediate_size: int
    vocab_size: int
    vocab_shard_size: int
    routed_expert_count: int
    total_expert_count: int
    selected_expert_count: int
    cache_vars_per_layer: int
    cache_widths: tuple[int, ...]
    temp_vars_size: int
    max_seq_len: int
    seq1_plan: Seq1Plan
    native_mtp_plans: tuple[MtpPlan, ...]
    stop_token_ids: tuple[int, ...] = ()
    # Present for models with several attention kinds (per-layer layouts, indexer, routing rules).
    hybrid: HybridModelSpec | None = None

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        architecture = str(self.architecture).strip()
        if not model_id or not architecture:
            raise ValueError(
                "model_id and architecture must be non-empty"
            )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "architecture", architecture)
        for name in (
            "rank_count",
            "replay_slot_count",
            "main_layer_count",
            "dense_layer_count",
            "hidden_size",
            "dense_intermediate_size",
            "vocab_size",
            "vocab_shard_size",
            "routed_expert_count",
            "total_expert_count",
            "selected_expert_count",
            "cache_vars_per_layer",
            "temp_vars_size",
            "max_seq_len",
        ):
            object.__setattr__(
                self,
                name,
                _positive(name, getattr(self, name)),
            )
        mtp_plans = tuple(self.native_mtp_plans)
        if (
            any(not isinstance(plan, MtpPlan) for plan in mtp_plans)
            or len({plan.plan_id for plan in mtp_plans}) != len(mtp_plans)
            or any(
                plan.plan_id == self.seq1_plan.plan_id
                for plan in mtp_plans
            )
            or len({plan.kernel_capability for plan in mtp_plans})
            != len(mtp_plans)
            or any(
                plan.kernel_capability
                == self.seq1_plan.kernel_capability
                for plan in mtp_plans
            )
            or any(
                module < self.main_layer_count
                for plan in mtp_plans
                for module in plan.mtp_module_ids
            )
        ):
            raise ValueError(
                "native MTP plans must be unique and follow main layers"
            )
        if self.dense_layer_count > self.main_layer_count:
            raise ValueError(
                "dense_layer_count cannot exceed main_layer_count"
            )
        if self.vocab_size != (
            self.rank_count * self.vocab_shard_size
        ):
            raise ValueError(
                "vocab_size must equal rank_count * vocab_shard_size"
            )
        if self.total_expert_count < self.routed_expert_count:
            raise ValueError(
                "total_expert_count cannot be smaller than routed experts"
            )
        if self.selected_expert_count > self.routed_expert_count:
            raise ValueError(
                "selected_expert_count cannot exceed routed experts"
            )
        cache_widths = tuple(
            _positive("cache_width", value)
            for value in self.cache_widths
        )
        if len(cache_widths) != self.cache_vars_per_layer:
            raise ValueError(
                "cache_widths must describe every per-layer cache"
            )
        object.__setattr__(self, "cache_widths", cache_widths)
        object.__setattr__(self, "native_mtp_plans", mtp_plans)
        stop_tokens = tuple(int(value) for value in self.stop_token_ids)
        if (
            len(stop_tokens) != len(set(stop_tokens))
            or any(
                token < 0 or token >= self.vocab_size
                for token in stop_tokens
            )
        ):
            raise ValueError(
                "stop_token_ids must be unique vocabulary indices"
            )
        object.__setattr__(self, "stop_token_ids", stop_tokens)
        if self.hybrid is not None:
            if not isinstance(self.hybrid, HybridModelSpec):
                raise ValueError("hybrid must be a HybridModelSpec")
            if len(self.hybrid.layer_kinds) != self.main_layer_count:
                raise ValueError("hybrid layer_kinds must name every main layer")

    @property
    def moe_layer_count(self) -> int:
        return self.main_layer_count - self.dense_layer_count

    @property
    def cache_vars_size(self) -> int:
        return self.main_layer_count * self.cache_vars_per_layer

    def as_manifest(self) -> dict[str, Any]:
        checkpoint = self.checkpoint
        return {
            "schema": "naivert-model-v1",
            "model_id": self.model_id,
            "architecture": self.architecture,
            "checkpoint": {
                "relative_local_root": checkpoint.relative_local_root,
                "index_sha256": checkpoint.index_sha256,
                "shard_count": checkpoint.shard_count,
                "shard_total_bytes": checkpoint.shard_total_bytes,
                "shard_inventory_sha256":
                    checkpoint.shard_inventory_sha256,
            },
            "tensor_parallel": self.rank_count,
            "main_layers": self.main_layer_count,
            "native_mtp_layers": sorted(
                {
                    module_id
                    for plan in self.native_mtp_plans
                    for module_id in plan.mtp_module_ids
                }
            ),
            "seq1_plan": self.seq1_plan.as_dict(),
            "native_mtp_plans": [
                plan.as_dict() for plan in self.native_mtp_plans
            ],
        }


@dataclass(frozen=True)
class AttentionLayout:
    """Grouped-query attention geometry of one layer kind.

    Query heads are sharded across ranks; a KV head count below the rank count
    means each KV head is replicated on ``rank_count // kv_head_count`` ranks.
    ``cache_roles`` / ``cache_widths`` are the per-token, per-rank cache rows
    this kind keeps (a sliding-window kind keeps ``sliding_window`` of them).
    """

    kind: str
    head_count: int
    kv_head_count: int
    head_dim: int
    v_head_dim: int
    rope_dim: int
    rope_theta: float
    cache_roles: tuple[str, ...]
    cache_widths: tuple[int, ...]
    sliding_window: int | None = None
    sink_bias: bool = False
    value_scale: float | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("attention layout kind must be non-empty")
        object.__setattr__(self, "kind", kind)
        for name in ("head_count", "kv_head_count", "head_dim", "v_head_dim", "rope_dim"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if self.head_count % self.kv_head_count:
            raise ValueError("head_count must be a multiple of kv_head_count")
        if self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("rope_dim must be even and at most head_dim")
        if float(self.rope_theta) <= 0.0:
            raise ValueError("rope_theta must be positive")
        roles = _names("attention cache_roles", tuple(self.cache_roles))
        widths = tuple(_positive("cache_width", value) for value in self.cache_widths)
        if len(roles) != len(widths):
            raise ValueError("cache_roles and cache_widths must pair up")
        object.__setattr__(self, "cache_roles", roles)
        object.__setattr__(self, "cache_widths", widths)
        if self.sliding_window is not None:
            object.__setattr__(self, "sliding_window", _positive("sliding_window", self.sliding_window))
        if self.value_scale is not None and not float(self.value_scale) > 0.0:
            raise ValueError("value_scale must be positive when set")

    def local_heads(self, rank_count: int) -> int:
        if self.head_count % int(rank_count):
            raise ValueError("head_count must be divisible by rank_count")
        return self.head_count // int(rank_count)

    def local_kv_heads(self, rank_count: int) -> int:
        ranks = int(rank_count)
        if self.kv_head_count >= ranks:
            if self.kv_head_count % ranks:
                raise ValueError("kv_head_count must be divisible by rank_count")
            return self.kv_head_count // ranks
        if ranks % self.kv_head_count:
            raise ValueError("rank_count must be a multiple of kv_head_count to replicate KV heads")
        return 1

    def kv_head_for_rank(self, rank: int, rank_count: int) -> int:
        """Global KV head index a rank holds (replicated heads share one index)."""
        ranks = int(rank_count)
        if self.kv_head_count >= ranks:
            return int(rank) * (self.kv_head_count // ranks)
        return int(rank) // (ranks // self.kv_head_count)


@dataclass(frozen=True)
class SparseIndexerSpec:
    """Sparse attention indexer: scores every cached key, keeps top-k."""

    head_count: int
    head_dim: int
    rope_dim: int
    top_k: int
    key_norm_bias: bool = True

    def __post_init__(self) -> None:
        for name in ("head_count", "head_dim", "rope_dim", "top_k"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("indexer rope_dim must be even and at most head_dim")


@dataclass(frozen=True)
class MoeRoutingSpec:
    score_func: str
    group_count: int
    limited_group_count: int
    route_scale: float
    normalize_top_k: bool
    shared_expert_count: int

    def __post_init__(self) -> None:
        if str(self.score_func) not in ("sigmoid", "softmax"):
            raise ValueError("score_func must be sigmoid or softmax")
        for name in ("group_count", "limited_group_count"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if self.limited_group_count > self.group_count:
            raise ValueError("limited_group_count cannot exceed group_count")
        if not float(self.route_scale) > 0.0:
            raise ValueError("route_scale must be positive")
        if int(self.shared_expert_count) < 0:
            raise ValueError("shared_expert_count must be non-negative")
        object.__setattr__(self, "shared_expert_count", int(self.shared_expert_count))


@dataclass(frozen=True)
class HybridModelSpec:
    """Per-layer attention kinds and the model rules the spec leaves implicit."""

    layer_kinds: tuple[str, ...]
    attention_layouts: Mapping[str, AttentionLayout]
    indexer: SparseIndexerSpec | None
    indexer_layer_kinds: tuple[str, ...]
    routing: MoeRoutingSpec
    norm_eps: float
    weight_dtype: str

    def __post_init__(self) -> None:
        layouts = {str(k): v for k, v in self.attention_layouts.items()}
        if not layouts or any(not isinstance(v, AttentionLayout) or v.kind != k for k, v in layouts.items()):
            raise ValueError("attention_layouts must map each kind to its AttentionLayout")
        kinds = tuple(str(k) for k in self.layer_kinds)
        if not kinds or set(kinds) - set(layouts):
            raise ValueError("layer_kinds must name declared attention layouts")
        indexer_kinds = tuple(str(k) for k in self.indexer_layer_kinds)
        if (self.indexer is None) != (not indexer_kinds) or set(indexer_kinds) - set(layouts):
            raise ValueError("indexer_layer_kinds must be non-empty exactly when an indexer exists")
        if not float(self.norm_eps) > 0.0:
            raise ValueError("norm_eps must be positive")
        if str(self.weight_dtype) not in ("bf16", "fp8"):
            raise ValueError("weight_dtype must be bf16 or fp8")
        object.__setattr__(self, "attention_layouts", MappingProxyType(layouts))
        object.__setattr__(self, "layer_kinds", kinds)
        object.__setattr__(self, "indexer_layer_kinds", indexer_kinds)

    def layers_of(self, kind: str) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.layer_kinds) if k == kind)

    def layout_of(self, layer: int) -> AttentionLayout:
        return self.attention_layouts[self.layer_kinds[int(layer)]]

    def layer_has_indexer(self, layer: int) -> bool:
        return self.layer_kinds[int(layer)] in self.indexer_layer_kinds


COMPATIBILITY_CATEGORIES = (
    "config",
    "checkpoint_keys",
    "shapes_dtypes",
    "tp_layout",
    "cache_layout",
    "quantization",
    "kernel_specializations",
)


@dataclass(frozen=True)
class ModelCompatibilityReport:
    """Host-only gate that must pass before model loading or CUDA setup."""

    model_id: str
    candidate_root: str | None
    checks: Mapping[str, Mapping[str, Any]]
    blockers: tuple[str, ...]
    searched_local_roots: tuple[str, ...] = ("/local_models",)

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        if not model_id:
            raise ValueError("compatibility report requires a model_id")
        checks = {
            str(name): MappingProxyType(dict(value))
            for name, value in self.checks.items()
        }
        if set(checks) != set(COMPATIBILITY_CATEGORIES):
            raise ValueError(
                "compatibility report must cover every required category"
            )
        if any(type(row.get("ok")) is not bool for row in checks.values()):
            raise ValueError(
                "every compatibility category requires a boolean ok"
            )
        blockers = tuple(
            str(value).strip()
            for value in self.blockers
            if str(value).strip()
        )
        roots = tuple(
            str(value).strip()
            for value in self.searched_local_roots
            if str(value).strip()
        )
        if not roots:
            raise ValueError(
                "compatibility report requires a local search root"
            )
        if (not blockers) != all(
            bool(row["ok"]) for row in checks.values()
        ):
            raise ValueError(
                "compatibility blockers must agree with category results"
            )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "searched_local_roots", roots)

    @property
    def compatible(self) -> bool:
        return not self.blockers

    @property
    def cuda_context_allowed(self) -> bool:
        return self.compatible

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "naivert-model-compatibility-v1",
            "model_id": self.model_id,
            "candidate_root": self.candidate_root,
            "searched_local_roots": list(self.searched_local_roots),
            "checks": {
                name: dict(self.checks[name])
                for name in COMPATIBILITY_CATEGORIES
            },
            "blockers": list(self.blockers),
            "compatible": self.compatible,
            "cuda_context_allowed": self.cuda_context_allowed,
        }


StateT = TypeVar("StateT")
WeightsT = TypeVar("WeightsT")


@dataclass(frozen=True)
class ModelPackage(Generic[StateT, WeightsT]):
    """One validated model conversion and its prepared kernel operands."""

    spec: ModelSpec
    state: StateT
    weights: WeightsT
    native_mtp_weight_loader: (
        Callable[..., Any] | None
    )
    manifest: Mapping[str, Any]
    keepalive: tuple[Any, ...]

    def __post_init__(self) -> None:
        if (
            self.spec.native_mtp_plans
            and not callable(self.native_mtp_weight_loader)
        ):
            raise ModelPackageError(
                "native MTP capability requires its model weight loader"
            )
        manifest = dict(self.manifest)
        if manifest != self.spec.as_manifest():
            raise ModelPackageError(
                "model package manifest does not match its ModelSpec"
            )
        if not self.keepalive:
            raise ModelPackageError(
                "model package requires tensor-owner keepalive objects"
            )
        object.__setattr__(
            self, "manifest", MappingProxyType(manifest)
        )
        object.__setattr__(self, "keepalive", tuple(self.keepalive))

    def load_native_mtp_weights(self, *, cache_rows: int) -> Any:
        return self.native_mtp_weight_loader(cache_rows=int(cache_rows))


__all__ = [
    "AttentionLayout",
    "CheckpointSpec",
    "HybridModelSpec",
    "MoeRoutingSpec",
    "SparseIndexerSpec",
    "COMPATIBILITY_CATEGORIES",
    "ModelCompatibilityReport",
    "MtpPlan",
    "ModelPackage",
    "ModelPackageError",
    "ModelSpec",
    "Seq1Plan",
]
