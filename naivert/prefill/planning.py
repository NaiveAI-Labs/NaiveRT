"""CPU-only bucketed chunk planning for the reference prefill executor."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import operator
from typing import Sequence

BUCKETS: tuple[int, ...] = (256, 1024, 4096)
WINDOW = 128
STRATEGIES = ("aligned", "greedy", "budget")


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass(frozen=True)
class Chunk:
    """A contiguous token interval and the compute shape that contains it."""

    start: int
    valid: int
    bucket: int

    @property
    def prefix(self) -> int:
        return min(self.start, WINDOW - 1)


@dataclass(frozen=True)
class PrefillPlanConfig:
    """Immutable shape/policy inputs; construct a new executor to change them."""

    buckets: tuple[int, ...] = BUCKETS
    strategy: str = "aligned"
    max_chunk_tokens: int | None = None

    def __post_init__(self) -> None:
        sizes = tuple(sorted({_integer(b, "bucket", minimum=1) for b in self.buckets}))
        if not sizes:
            raise ValueError("buckets must not be empty")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected {STRATEGIES}")
        object.__setattr__(self, "buckets", sizes)
        if self.max_chunk_tokens is not None:
            cap = _integer(self.max_chunk_tokens, "max_chunk_tokens", minimum=1)
            if cap > sizes[-1]:
                raise ValueError("max_chunk_tokens must not exceed the largest bucket")
            object.__setattr__(self, "max_chunk_tokens", cap)

    @property
    def token_limit(self) -> int:
        return self.buckets[-1] if self.max_chunk_tokens is None else self.max_chunk_tokens

    def as_dict(self) -> dict:
        return {"buckets": list(self.buckets), "strategy": self.strategy,
                "max_chunk_tokens": self.max_chunk_tokens}

    def plan(self, total: int, *, start: int = 0) -> list[Chunk]:
        left = _integer(total, "total")
        pos = _integer(start, "start")
        chunks = []
        smallest = self.buckets[0]
        while left:
            limit = min(left, self.token_limit)
            if self.strategy == "budget":
                bucket = next(b for b in self.buckets if b >= limit)
            else:
                target = ((limit + smallest - 1) // smallest * smallest
                          if self.strategy == "aligned" else limit)
                bucket = max((b for b in self.buckets if b <= target), default=smallest)
            valid = min(limit, bucket)
            chunks.append(Chunk(start=pos, valid=valid, bucket=bucket))
            pos, left = pos + valid, left - valid
        return chunks


def resolve_plan_config(
    *, buckets: Sequence[int] | None = None, config: PrefillPlanConfig | None = None,
) -> PrefillPlanConfig:
    """Keep the existing buckets-only API while giving configuration one owner."""
    if config is not None:
        if buckets is not None:
            raise ValueError("pass either buckets or plan config, not both")
        if not isinstance(config, PrefillPlanConfig):
            raise TypeError("config must be a PrefillPlanConfig")
        return config
    return PrefillPlanConfig(buckets=BUCKETS if buckets is None else tuple(buckets))


def plan_chunks(
    total: int, *, start: int = 0, buckets: Sequence[int] | None = None,
    config: PrefillPlanConfig | None = None,
) -> list[Chunk]:
    return resolve_plan_config(buckets=buckets, config=config).plan(total, start=start)


def main() -> None:
    """Inspect one candidate's token/shape plan; this does not predict GPU latency."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket-sizes", type=int, nargs="+", default=BUCKETS)
    ap.add_argument("--strategy", choices=STRATEGIES, default=PrefillPlanConfig().strategy)
    ap.add_argument("--max-chunk-tokens", type=int)
    ap.add_argument("--tokens", type=int, nargs="+", required=True)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    config = PrefillPlanConfig(tuple(args.bucket_sizes), args.strategy, args.max_chunk_tokens)
    plans = []
    for total in args.tokens:
        chunks = config.plan(total, start=args.start)
        computed = sum(c.bucket for c in chunks)
        plans.append({"tokens": total, "start": args.start, "chunks": [asdict(c) for c in chunks],
                      "chunk_count": len(chunks), "computed_rows": computed,
                      "padding_rows": computed-total,
                      "used_buckets": sorted({c.bucket for c in chunks})})
    print(json.dumps({"config": config.as_dict(), "plans": plans}, indent=2))


if __name__ == "__main__":
    main()
