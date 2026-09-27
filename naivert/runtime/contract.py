"""Errors and the contract digest shared by the runtime and its callers.

Both names live in ``naivert.contract_core``; this module keeps the
``naivert.runtime.contract`` import path the session engine and serve layer use.
"""

from __future__ import annotations

from naivert.contract_core import (
    RuntimeContractError,
    canonical_contract_digest,
)

__all__ = [
    "RuntimeContractError",
    "canonical_contract_digest",
]
