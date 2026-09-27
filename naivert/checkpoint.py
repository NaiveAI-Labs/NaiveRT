"""Local safetensors checkpoint identity and verified reads, for any model.

A model binds this to its ``CheckpointSpec`` (see models/<model>/checkpoint.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Mapping

from naivert.model import CheckpointSpec, ModelPackageError



_DTYPE_ITEM_SIZES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_metadata(
    root: Path,
    weight_items: tuple[tuple[str, str], ...],
) -> tuple[int, str, tuple[tuple[str, int], ...]]:
    """Validate safetensors headers without materializing tensor payloads."""

    weight_map = dict(weight_items)
    seen: dict[str, str] = {}
    dtype_counts: dict[str, int] = {}
    rows: list[str] = []
    for relative in sorted(set(weight_map.values())):
        path = root / relative
        with path.open("rb") as handle:
            encoded_size = handle.read(8)
            if len(encoded_size) != 8:
                raise ModelPackageError(
                    f"checkpoint shard has no safetensors header: {relative}"
                )
            header_size = struct.unpack("<Q", encoded_size)[0]
            header = json.loads(handle.read(header_size))
        payload_size = path.stat().st_size - 8 - header_size
        intervals: list[tuple[int, int, str]] = []
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if (
                key in seen
                or weight_map.get(key) != relative
                or not isinstance(metadata, dict)
            ):
                raise ModelPackageError(
                    f"checkpoint tensor inventory changed at {key!r}"
                )
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    for value in offsets
                )
                or not 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise ModelPackageError(
                    f"checkpoint tensor metadata changed at {key!r}"
                )
            item_size = _DTYPE_ITEM_SIZES.get(dtype)
            expected_bytes = (
                item_size * math.prod(shape)
                if item_size is not None
                else None
            )
            if (
                expected_bytes is None
                or offsets[1] - offsets[0] != expected_bytes
            ):
                raise ModelPackageError(
                    f"checkpoint tensor byte extent changed at {key!r}"
                )
            seen[key] = relative
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            intervals.append((offsets[0], offsets[1], key))
            rows.append(
                f"{key}\0{relative}\0{dtype}\0"
                f"{','.join(str(value) for value in shape)}\0"
                f"{offsets[0]},{offsets[1]}\n"
            )
        cursor = 0
        for start, end, key in sorted(intervals):
            if start != cursor:
                raise ModelPackageError(
                    "checkpoint tensor payload is overlapping or "
                    f"non-contiguous at {key!r}"
                )
            cursor = end
        if cursor != payload_size:
            raise ModelPackageError(
                f"checkpoint shard payload is not fully described: {relative}"
            )
    if seen != weight_map:
        missing = sorted(set(weight_map) - set(seen))
        raise ModelPackageError(
            "checkpoint safetensors headers differ from the weight map"
            + (f": missing {missing[0]!r}" if missing else "")
        )
    digest = hashlib.sha256(
        "".join(sorted(rows)).encode("utf-8")
    ).hexdigest()
    return (
        len(rows),
        digest,
        tuple(sorted(dtype_counts.items())),
    )


@dataclass(frozen=True)
class LocalModelSource:
    root: Path
    reason: str
    index_sha256: str
    shard_count: int
    shard_total_bytes: int
    shard_inventory_sha256: str
    tensor_count: int
    tensor_metadata_sha256: str
    tensor_dtype_counts: tuple[tuple[str, int], ...]

    @classmethod
    def resolve(
        cls,
        root: str | Path | None = None,
        *,
        checkpoint: CheckpointSpec,
    ) -> "LocalModelSource":
        """Verify a local checkpoint against ``checkpoint`` and return a reader for it."""
        selected = Path(
            root or checkpoint.local_root
        ).expanduser().resolve()
        local_root = checkpoint.local_root.resolve()
        if selected != local_root:
            raise ModelPackageError(
                "the local checkpoint root must be the frozen root the model "
                "spec names"
            )
        index = selected / "model.safetensors.index.json"
        if not index.is_file():
            raise ModelPackageError(
                "compatible local checkpoint index is missing; no "
                "backend-specific substitute is allowed"
            )
        digest = _sha256(index)
        if digest != checkpoint.index_sha256:
            raise ModelPackageError("local checkpoint index changed")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_weight_map = index_payload.get("weight_map")
        if not isinstance(index_weight_map, dict) or not index_weight_map:
            raise ModelPackageError("local checkpoint index has no weight map")
        # Shard names differ between checkpoints; the index is the authority.
        shards = sorted(
            {selected / str(relative) for relative in index_weight_map.values()}
        )
        if any(not path.is_file() for path in shards):
            raise ModelPackageError("local checkpoint index names a missing shard")
        if len(shards) != checkpoint.shard_count:
            raise ModelPackageError(
                f"local checkpoint has {len(shards)} shards, expected "
                f"{checkpoint.shard_count}"
            )
        escaped = [
            path
            for path in (index, *shards)
            if (
                path.resolve() != local_root
                and local_root not in path.resolve().parents
            )
        ]
        if escaped:
            raise ModelPackageError(
                "checkpoint overlay contains a non-local weight/index file: "
                f"{escaped[0]}"
            )
        shard_total_bytes = sum(path.stat().st_size for path in shards)
        if shard_total_bytes != checkpoint.shard_total_bytes:
            raise ModelPackageError(
                "local checkpoint shard byte total changed"
            )
        inventory = "".join(
            f"{path.name} {path.stat().st_size}\n" for path in shards
        ).encode("utf-8")
        inventory_digest = hashlib.sha256(inventory).hexdigest()
        if inventory_digest != checkpoint.shard_inventory_sha256:
            raise ModelPackageError(
                "local checkpoint shard inventory changed"
            )
        payload = json.loads(index.read_text(encoding="utf-8"))
        raw_weight_map = payload.get("weight_map")
        weight_items = tuple(
            sorted(
                (str(name), str(relative))
                for name, relative in raw_weight_map.items()
            )
        )
        (
            tensor_count,
            tensor_metadata_sha256,
            tensor_dtype_counts,
        ) = _tensor_metadata(selected, weight_items)
        if (
            tensor_count != checkpoint.tensor_count
            or tensor_metadata_sha256
            != checkpoint.tensor_metadata_sha256
            or dict(tensor_dtype_counts)
            != dict(checkpoint.tensor_dtype_counts)
        ):
            raise ModelPackageError(
                "local checkpoint tensor shapes, dtypes, or keys changed"
            )
        return cls(
            root=selected,
            reason="compatible_local_checkpoint_copy",
            index_sha256=digest,
            shard_count=len(shards),
            shard_total_bytes=shard_total_bytes,
            shard_inventory_sha256=inventory_digest,
            tensor_count=tensor_count,
            tensor_metadata_sha256=tensor_metadata_sha256,
            tensor_dtype_counts=tensor_dtype_counts,
        )

    @cached_property
    def weight_map(self) -> Mapping[str, str]:
        payload = json.loads(
            (self.root / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weights = payload.get("weight_map")
        result = {
            str(name): str(relative)
            for name, relative in weights.items()
        }
        if len(set(result.values())) != self.shard_count:
            raise ModelPackageError(
                "local model weight map shard count changed"
            )
        return MappingProxyType(result)

    def load_tensors(
        self,
        keys: tuple[str, ...] | list[str],
        *,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Read an exact tensor batch, opening each backing shard once."""

        from safetensors import safe_open

        selected = tuple(dict.fromkeys(str(key) for key in keys))
        if not selected:
            raise ModelPackageError(
                "checkpoint tensor batch cannot be empty"
            )
        by_shard: dict[str, list[str]] = {}
        for key in selected:
            relative = self.weight_map.get(key)
            if relative is None:
                raise ModelPackageError(
                    f"local model has no source tensor {key}"
                )
            by_shard.setdefault(relative, []).append(key)
        tensors: dict[str, Any] = {}
        for relative, shard_keys in sorted(by_shard.items()):
            with safe_open(
                str(self.root / relative),
                framework="pt",
                device=str(device),
            ) as handle:
                tensors.update(
                    (key, handle.get_tensor(key))
                    for key in shard_keys
                )
        return tensors


__all__ = ["LocalModelSource"]
