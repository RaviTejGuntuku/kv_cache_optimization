#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def block_signature(block_hashes: list[int]) -> str:
    return ",".join(str(block_hash) for block_hash in block_hashes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an oracle-replay Belady victim plan from a baseline radix trace."
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


def next_use(accesses: dict[str, list[int]], signature: str, seq: int) -> int | None:
    seqs = accesses.get(signature, [])
    idx = bisect.bisect_right(seqs, seq)
    if idx >= len(seqs):
        return None
    return seqs[idx]


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.trace))

    accesses: dict[str, list[int]] = defaultdict(list)
    frontiers = []
    for event in events:
        event_type = event.get("event")
        if event_type == "node_access":
            signature = block_signature([int(x) for x in event.get("block_hashes", [])])
            if signature:
                accesses[signature].append(int(event["seq"]))
        elif event_type == "eviction_frontier":
            frontiers.append(event)

    decisions = []
    fallback_count = 0
    for frontier_index, frontier in enumerate(frontiers):
        frontier_seq = int(frontier["seq"])
        best_signature = None
        best_next_use = -1.0

        for candidate in frontier.get("candidates", []):
            signature = block_signature([int(x) for x in candidate.get("block_hashes", [])])
            if not signature:
                continue
            candidate_next = next_use(accesses, signature, frontier_seq)
            candidate_rank = float("inf") if candidate_next is None else float(candidate_next)
            if candidate_rank > best_next_use:
                best_next_use = candidate_rank
                best_signature = signature

        if best_signature is None:
            fallback_count += 1
            continue

        decisions.append(
            {
                "frontier_index": frontier_index,
                "frontier_seq": frontier_seq,
                "victim_signature": best_signature,
                "victim_next_use": None if best_next_use == float("inf") else int(best_next_use),
                "candidate_count": len(frontier.get("candidates", [])),
            }
        )

    payload = {
        "source_trace": args.trace,
        "num_frontiers": len(frontiers),
        "num_decisions": len(decisions),
        "num_fallback_frontiers": fallback_count,
        "decision_signature_counts": Counter(item["victim_signature"] for item in decisions),
        "decisions": decisions,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
