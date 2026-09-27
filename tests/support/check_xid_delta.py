#!/usr/bin/env python3
"""Reject Xid records newly added between two kernel-log snapshots."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


XID_PATTERN = re.compile(r"\bNVRM:\s+Xid\b", re.IGNORECASE)


def _xids(path: Path) -> Counter[str]:
    return Counter(
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if XID_PATTERN.search(line)
    )


def compare(before: Path, after: Path, *, stage: str) -> dict[str, Any]:
    before_xids = _xids(before)
    after_xids = _xids(after)
    added = after_xids - before_xids
    removed = before_xids - after_xids
    added_records = [
        {"record": record, "count": count}
        for record, count in sorted(added.items())
    ]
    return {
        "ok": not added_records,
        "stage": stage,
        "before_xid_count": sum(before_xids.values()),
        "after_xid_count": sum(after_xids.values()),
        "new_xid_count": sum(added.values()),
        "new_xids": added_records,
        "dropped_from_ring_count": sum(removed.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = compare(args.before, args.after, stage=args.stage)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
