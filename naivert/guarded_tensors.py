"""Torch tensors with fixed 4096-byte front and rear canaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping


CANARY_BYTES = 4096
FRONT_CANARY = 0xA5
REAR_CANARY = 0x5A


def guarded_allocation_bytes(payload_bytes: int) -> int:
    size = int(payload_bytes)
    if size < 0:
        raise ValueError("guarded payload size must be non-negative")
    return size + 2 * CANARY_BYTES


def canary_pattern_bytes(pattern: int) -> bytes:
    return bytes([int(pattern)]) * CANARY_BYTES


def guarded_allocation_report(
    *,
    allocation_count: int,
    payload_bytes: int,
) -> dict[str, int | bool]:
    count = int(allocation_count)
    payload = int(payload_bytes)
    if count < 0 or payload < 0:
        raise ValueError("guarded allocation totals must be non-negative")
    return {
        "allocation_count": count,
        "payload_bytes": payload,
        "guard_bytes": count * 2 * CANARY_BYTES,
        "canary_bytes_per_side": CANARY_BYTES,
        "canaries_exact": True,
    }


@dataclass(frozen=True)
class GuardedTorchTensor:
    """One tensor view retained inside a byte allocation with two guards."""

    name: str
    allocation: Any
    tensor: Any
    payload_bytes: int

    @classmethod
    def allocate(
        cls,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        initialization: str,
        fill_value: int | float | None = None,
    ) -> "GuardedTorchTensor":
        import torch

        normalized_shape = tuple(int(value) for value in shape)
        if not normalized_shape or any(value < 0 for value in normalized_shape):
            raise ValueError(
                "guarded tensor shape must be non-empty and non-negative"
            )
        elements = math.prod(normalized_shape)
        element_size = torch.empty((), dtype=dtype).element_size()
        payload_bytes = int(elements * element_size)
        allocation = torch.empty(
            (guarded_allocation_bytes(payload_bytes),),
            dtype=torch.uint8,
            device=device,
        )
        allocation[:CANARY_BYTES].fill_(FRONT_CANARY)
        allocation[
            CANARY_BYTES + payload_bytes :
        ].fill_(REAR_CANARY)
        payload = allocation.narrow(
            0, CANARY_BYTES, payload_bytes
        ).view(dtype).reshape(normalized_shape)
        if initialization == "zeros":
            payload.zero_()
        elif initialization == "full":
            if fill_value is None:
                raise ValueError(
                    "guarded full allocation requires fill_value"
                )
            payload.fill_(fill_value)
        elif initialization != "empty":
            raise ValueError(
                f"unsupported guarded initialization {initialization!r}"
            )
        if int(payload.data_ptr()) != (
            int(allocation.data_ptr()) + CANARY_BYTES
        ):
            raise RuntimeError("guarded tensor payload is not 4096-byte aligned")
        return cls(
            name=str(name),
            allocation=allocation,
            tensor=payload,
            payload_bytes=payload_bytes,
        )

    def verify_canaries(self) -> None:
        front = self.allocation[:CANARY_BYTES]
        rear = self.allocation[
            CANARY_BYTES + self.payload_bytes :
        ]
        if not bool(front.eq(FRONT_CANARY).all().item()):
            raise RuntimeError(f"front canary changed for {self.name}")
        if not bool(rear.eq(REAR_CANARY).all().item()):
            raise RuntimeError(f"rear canary changed for {self.name}")


class GuardedTensorSet:
    """Retain and verify a named set of guarded mutable allocations."""

    def __init__(self, *, scope: str) -> None:
        self.scope = str(scope)
        self._records: dict[str, GuardedTorchTensor] = {}

    def _allocate(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        initialization: str,
        fill_value: int | float | None = None,
    ) -> Any:
        qualified_name = f"{self.scope}:{name}"
        if qualified_name in self._records:
            raise RuntimeError(
                f"guarded allocation {qualified_name!r} already exists"
            )
        record = GuardedTorchTensor.allocate(
            name=qualified_name,
            shape=shape,
            dtype=dtype,
            device=device,
            initialization=initialization,
            fill_value=fill_value,
        )
        self._records[qualified_name] = record
        return record.tensor

    def empty(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
    ) -> Any:
        return self._allocate(
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
            initialization="empty",
        )

    def zeros(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any,
    ) -> Any:
        return self._allocate(
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
            initialization="zeros",
        )

    def full(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        fill_value: int | float,
        dtype: Any,
        device: Any,
    ) -> Any:
        return self._allocate(
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
            initialization="full",
            fill_value=fill_value,
        )

    @property
    def records(self) -> Mapping[str, GuardedTorchTensor]:
        return MappingProxyType(dict(self._records))

    def verify_canaries(self) -> dict[str, object]:
        for record in self._records.values():
            record.verify_canaries()
        return {
            "schema": "naivert_guarded_mutable_allocations_v1",
            "scope": self.scope,
            "front_pattern": FRONT_CANARY,
            "rear_pattern": REAR_CANARY,
            **guarded_allocation_report(
                allocation_count=len(self._records),
                payload_bytes=sum(
                    record.payload_bytes
                    for record in self._records.values()
                ),
            ),
        }


__all__ = [
    "CANARY_BYTES",
    "FRONT_CANARY",
    "GuardedTensorSet",
    "GuardedTorchTensor",
    "REAR_CANARY",
    "canary_pattern_bytes",
    "guarded_allocation_bytes",
    "guarded_allocation_report",
]
