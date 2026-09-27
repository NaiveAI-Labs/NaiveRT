"""The Naive N model series: public names for the checkpoints this runtime serves.

Checkpoint directories are found under the model roots by the ``target`` and ``draft``
short names below (``examples/serve_sessions.py`` resolves them; ``--target`` / ``--draft``
or ``NAIVERT_TARGET_CHECKPOINT`` point at other directories). Everything user-facing uses
the public names below: the served model id, ``/v1/models``, logs and documentation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaiveNModel:
    key: str            # public model id, what clients send as ``model`` and what /v1/models lists
    display: str        # human name
    series: str
    target: str         # target checkpoint short name (examples/serve_sessions.py resolves it on disk)
    draft: str          # draft checkpoint short name
    lineage: str        # where the checkpoints come from, for logs and weights_info


SERIES = "Naive N"

MODELS = {
    "naive-n0.5": NaiveNModel(
        key="naive-n0.5", display="Naive N0.5", series=SERIES,
        target="naive_n0p5_fp8", draft="naive_n0p5_draft",
        lineage="the served checkpoint pair"),
    "naive-n0.5-preview": NaiveNModel(
        key="naive-n0.5-preview", display="Naive N0.5 (preview checkpoint)", series=SERIES,
        target="naive_n0p5_preview_fp8", draft="naive_n0p5_preview_draft",
        lineage="the earlier checkpoint pair, used by the 44-request gate"),
}
DEFAULT_MODEL = "naive-n0.5"


def resolve_model(name: str | None) -> NaiveNModel:
    """The registry entry for a public name (``None``: the default)."""
    key = DEFAULT_MODEL if name is None else str(name)
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {', '.join(sorted(MODELS))}") from None


__all__ = ["DEFAULT_MODEL", "MODELS", "NaiveNModel", "SERIES", "resolve_model"]
