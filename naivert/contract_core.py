"""Pure-stdlib primitives shared by model and runtime contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class RuntimeContractError(ValueError):
    """Raised when immutable or runtime data violates an accepted contract."""


def canonical_contract_digest(value: Any) -> str:
    """Hash one JSON-compatible contract deterministically."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "RuntimeContractError",
    "canonical_contract_digest",
]
