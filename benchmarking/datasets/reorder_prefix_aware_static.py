#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a statically prefix-aware ordering of a JSONL dataset by clustering "
            "requests with the same shared family/prefix together."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["family-phase-branch", "family-only"],
        default="family-phase-branch",
        help="Static ordering strategy.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_no} in {path}") from exc
    return rows


def sort_key(row: dict[str, Any], mode: str) -> tuple[Any, ...]:
    meta = row.get("metadata") or {}
    family = str(meta.get("family", "zzz"))
    phase = str(meta.get("phase", "zzz"))
    branch = int(meta.get("branch", 0))
    seq = int(meta.get("sequence_id", 0))
    kind = str(meta.get("kind", "shared"))

    # shared requests first within each family so clustered reusable prefixes appear together
    kind_rank = 0 if kind == "shared" else 1
    if mode == "family-only":
        return (family, kind_rank, branch, phase, seq)
    return (family, phase, kind_rank, branch, seq)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path)
    rows = sorted(rows, key=lambda row: sort_key(row, args.mode))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
