#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a dynamic offline Belady oracle from a baseline request lookup trace."
    )
    parser.add_argument("--trace", required=True, help="Baseline JSONL trace path.")
    parser.add_argument("--output", required=True, help="Output JSON plan path.")
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.trace))

    block_access_steps: dict[int, list[int]] = defaultdict(list)
    lookup_count = 0
    for event in events:
        if event.get("event") != "request_lookup":
            continue
        for block_hash in event.get("block_hashes", []):
            block_access_steps[int(block_hash)].append(lookup_count)
        lookup_count += 1

    payload = {
        "source_trace": args.trace,
        "num_request_lookups": lookup_count,
        "num_unique_blocks": len(block_access_steps),
        "block_access_steps": block_access_steps,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
