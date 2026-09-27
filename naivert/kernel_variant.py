"""Accepted identity and integer ABI of one kernel variant (model-neutral).

A model package declares the variants its loaders accept (naivert/models/<model>/
kernel_specs.py); a loader fails closed unless it is asked for exactly the
variant it builds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelVariantSpec:
    """Accepted identity and integer ABI for one kernel variant."""

    variant_id: str
    abi_version: int = 1
    parameter_pointer_count: int | None = None
    parameter_bytes: int | None = None
    # Highest decode position the variant is validated for; None = unbounded.
    max_cur_pos: int | None = None

    def __post_init__(self) -> None:
        variant_id = str(self.variant_id).strip()
        if not variant_id:
            raise ValueError("kernel variant ID must be non-empty")
        if int(self.abi_version) <= 0:
            raise ValueError("kernel variant ABI version must be positive")
        for name in ("parameter_pointer_count", "parameter_bytes"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when specified")
        object.__setattr__(self, "variant_id", variant_id)


def require_kernel_variant(
    expected: KernelVariantSpec,
    actual: KernelVariantSpec,
) -> KernelVariantSpec:
    """Fail closed unless Contract and loader name the same descriptor."""

    if not isinstance(expected, KernelVariantSpec) or expected != actual:
        raise ValueError(
            f"kernel descriptor differs from loader-owned {actual.variant_id!r}"
        )
    return actual


__all__ = ["KernelVariantSpec", "require_kernel_variant"]
