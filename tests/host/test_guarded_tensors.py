from __future__ import annotations

import pytest
import torch

from naivert.guarded_tensors import (
    CANARY_BYTES,
    FRONT_CANARY,
    GuardedTensorSet,
)


def test_guarded_tensor_set_allocates_aligned_views_and_reports_scope() -> None:
    guards = GuardedTensorSet(scope="unit")
    value = guards.zeros(
        name="scratch",
        shape=(3, 5),
        dtype=torch.bfloat16,
        device="cpu",
    )
    value.add_(1)

    report = guards.verify_canaries()

    assert report == {
        "schema": "naivert_guarded_mutable_allocations_v1",
        "scope": "unit",
        "allocation_count": 1,
        "payload_bytes": 30,
        "guard_bytes": 2 * CANARY_BYTES,
        "canary_bytes_per_side": CANARY_BYTES,
        "front_pattern": 0xA5,
        "rear_pattern": 0x5A,
        "canaries_exact": True,
    }
    record = guards.records["unit:scratch"]
    assert value.data_ptr() == record.allocation.data_ptr() + CANARY_BYTES


def test_guarded_tensor_set_detects_front_canary_change() -> None:
    guards = GuardedTensorSet(scope="unit")
    guards.empty(
        name="scratch",
        shape=(1,),
        dtype=torch.int32,
        device="cpu",
    )
    record = guards.records["unit:scratch"]
    record.allocation[0] = FRONT_CANARY ^ 0xFF

    with pytest.raises(RuntimeError, match="front canary changed"):
        guards.verify_canaries()
