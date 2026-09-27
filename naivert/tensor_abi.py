"""Pure-data tensor ABI records a model expands into its parameter, temporary and cache slots."""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSlotAbi:
    """One tensor of a domain (parameter, temporary, cache): its index, shape and layout."""

    slot_id: str
    domain: str
    index: int
    shape: tuple[int | str, ...]
    dtype: str
    alignment: int | None
    layout_id: str
    operation_id: str


@dataclass(frozen=True)
class PackedLayoutAbi:
    layout_id: str


@dataclass(frozen=True)
class TensorViewAbi:
    view_id: str
    domain: str
    slot_id: str


@dataclass(frozen=True)
class ModelTensorAbi:
    abi_id: str
    parameter_count: int
    temporary_count: int
    cache_count: int
    slots: tuple[TensorSlotAbi, ...] = ()
    packed_layouts: tuple[PackedLayoutAbi, ...] = ()
    named_views: tuple[TensorViewAbi, ...] = ()

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        expected = {
            "parameter": self.parameter_count,
            "temporary": self.temporary_count,
            "cache": self.cache_count,
        }
        for domain, count in expected.items():
            indices = sorted(slot.index for slot in slots if slot.domain == domain)
            if indices != list(range(count)):
                raise ValueError(f"{domain} slots do not cover their ABI")
        if len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("tensor slot IDs must be unique")
        layouts = {layout.layout_id for layout in self.packed_layouts}
        if {slot.layout_id for slot in slots} - layouts:
            raise ValueError("tensor slots reference unknown layouts")
        slot_keys = {(slot.domain, slot.slot_id) for slot in slots}
        if any((view.domain, view.slot_id) not in slot_keys for view in self.named_views):
            raise ValueError("tensor views reference unknown slots")


__all__ = ["ModelTensorAbi", "PackedLayoutAbi", "TensorSlotAbi", "TensorViewAbi"]
