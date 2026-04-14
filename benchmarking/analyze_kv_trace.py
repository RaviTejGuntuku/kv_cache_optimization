#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrontierDecision:
    frontier_seq: int
    candidate_count: int
    actual_node_id: int
    actual_next_use: int | None
    belady_node_id: int
    belady_next_use: int | None
    actual_block_count: int
    belady_block_count: int
    same_choice: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SGLang radix benchmark traces and approximate Belady headroom."
    )
    parser.add_argument("--trace", required=True, help="Path to JSONL trace file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSON/CSV summaries.",
    )
    parser.add_argument(
        "--block-capacity",
        type=int,
        default=None,
        help="Optional page/block capacity for an additional page-level cache simulation.",
    )
    return parser.parse_args()


def load_events(trace_path: Path) -> list[dict[str, Any]]:
    events = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {trace_path}") from exc
    return events


def build_access_index(events: list[dict[str, Any]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    node_accesses: dict[int, list[int]] = defaultdict(list)
    block_accesses: dict[int, list[int]] = defaultdict(list)

    for event in events:
        if event.get("event") != "node_access":
            continue
        seq = int(event["seq"])
        node_id = int(event["node_id"])
        node_accesses[node_id].append(seq)
        for block_hash in event.get("block_hashes", []):
            block_accesses[int(block_hash)].append(seq)

    return node_accesses, block_accesses


def next_use(access_index: dict[int, list[int]], object_id: int, current_seq: int) -> int | None:
    seqs = access_index.get(object_id, [])
    pos = bisect.bisect_right(seqs, current_seq)
    if pos >= len(seqs):
        return None
    return seqs[pos]


def analyze_frontiers(events: list[dict[str, Any]], node_accesses: dict[int, list[int]]) -> list[FrontierDecision]:
    frontiers = [
        event
        for event in events
        if event.get("event") in {"eviction_frontier", "belady_frontier"}
    ]
    eviction_events = [
        event for event in events if event.get("event") == "node_evicted"
    ]
    decisions: list[FrontierDecision] = []

    for frontier in frontiers:
        frontier_seq = int(frontier["seq"])
        actual = next(
            (
                event
                for event in eviction_events
                if int(event["seq"]) > frontier_seq
            ),
            None,
        )
        if actual is None:
            continue

        candidates = frontier.get("candidates", [])
        if not candidates:
            continue

        belady_candidate = None
        belady_next = -1
        for candidate in candidates:
            candidate_node_id = int(candidate["node_id"])
            candidate_next = next_use(node_accesses, candidate_node_id, frontier_seq)
            candidate_rank = candidate_next if candidate_next is not None else float("inf")
            if candidate_rank > belady_next:
                belady_next = candidate_rank
                belady_candidate = candidate

        actual_node_id = int(actual["node_id"])
        actual_next = next_use(node_accesses, actual_node_id, frontier_seq)
        belady_node_id = int(belady_candidate["node_id"]) if belady_candidate else actual_node_id
        belady_next_use = (
            next_use(node_accesses, belady_node_id, frontier_seq) if belady_candidate else actual_next
        )

        decisions.append(
            FrontierDecision(
                frontier_seq=frontier_seq,
                candidate_count=len(candidates),
                actual_node_id=actual_node_id,
                actual_next_use=actual_next,
                belady_node_id=belady_node_id,
                belady_next_use=belady_next_use,
                actual_block_count=len(actual.get("block_hashes", [])),
                belady_block_count=len(belady_candidate.get("block_hashes", [])) if belady_candidate else len(actual.get("block_hashes", [])),
                same_choice=(actual_node_id == belady_node_id),
            )
        )

    return decisions


def summarize_match_results(events: list[dict[str, Any]]) -> dict[str, Any]:
    match_events = [event for event in events if event.get("event") == "match_result"]
    if not match_events:
        return {}

    request_count = len(match_events)
    matched_tokens = sum(int(event.get("matched_tokens", 0)) for event in match_events)
    missed_tokens = sum(int(event.get("missed_tokens", 0)) for event in match_events)
    matched_blocks = sum(int(event.get("matched_blocks", 0)) for event in match_events)
    missed_blocks = sum(int(event.get("missed_blocks", 0)) for event in match_events)
    total_aligned_tokens = matched_tokens + missed_tokens
    total_blocks = matched_blocks + missed_blocks

    return {
        "request_count": request_count,
        "matched_tokens": matched_tokens,
        "missed_tokens": missed_tokens,
        "token_hit_rate": (
            matched_tokens / total_aligned_tokens if total_aligned_tokens else 0.0
        ),
        "token_miss_rate": (
            missed_tokens / total_aligned_tokens if total_aligned_tokens else 0.0
        ),
        "matched_blocks": matched_blocks,
        "missed_blocks": missed_blocks,
        "block_hit_rate": matched_blocks / total_blocks if total_blocks else 0.0,
        "block_miss_rate": missed_blocks / total_blocks if total_blocks else 0.0,
    }


def simulate_page_cache(
    events: list[dict[str, Any]],
    block_accesses: dict[int, list[int]],
    capacity: int,
) -> dict[str, Any]:
    resident: set[int] = set()
    last_access: dict[int, int] = {}
    lru_hits = 0
    lru_misses = 0
    belady_resident: set[int] = set()
    belady_hits = 0
    belady_misses = 0

    def evict_lru(cache: set[int]) -> None:
        victim = min(cache, key=lambda block: last_access.get(block, -1))
        cache.remove(victim)

    def evict_belady(cache: set[int], seq: int) -> None:
        victim = max(
            cache,
            key=lambda block: (
                next_use(block_accesses, block, seq)
                if next_use(block_accesses, block, seq) is not None
                else float("inf")
            ),
        )
        cache.remove(victim)

    for event in events:
        if event.get("event") != "node_access":
            continue
        seq = int(event["seq"])
        for block_hash in event.get("block_hashes", []):
            block_hash = int(block_hash)

            if block_hash in resident:
                lru_hits += 1
            else:
                lru_misses += 1
                if len(resident) >= capacity:
                    evict_lru(resident)
                resident.add(block_hash)
            last_access[block_hash] = seq

            if block_hash in belady_resident:
                belady_hits += 1
            else:
                belady_misses += 1
                if len(belady_resident) >= capacity:
                    evict_belady(belady_resident, seq)
                belady_resident.add(block_hash)

    total_accesses = lru_hits + lru_misses
    return {
        "capacity_blocks": capacity,
        "total_accesses": total_accesses,
        "lru_hits": lru_hits,
        "lru_misses": lru_misses,
        "lru_hit_rate": (lru_hits / total_accesses) if total_accesses else 0.0,
        "belady_hits": belady_hits,
        "belady_misses": belady_misses,
        "belady_hit_rate": (belady_hits / total_accesses) if total_accesses else 0.0,
    }


def write_frontier_csv(output_dir: Path, decisions: list[FrontierDecision]) -> None:
    csv_path = output_dir / "frontier_decisions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frontier_seq",
                "candidate_count",
                "actual_node_id",
                "actual_next_use",
                "belady_node_id",
                "belady_next_use",
                "actual_block_count",
                "belady_block_count",
                "same_choice",
            ]
        )
        for item in decisions:
            writer.writerow(
                [
                    item.frontier_seq,
                    item.candidate_count,
                    item.actual_node_id,
                    item.actual_next_use,
                    item.belady_node_id,
                    item.belady_next_use,
                    item.actual_block_count,
                    item.belady_block_count,
                    item.same_choice,
                ]
            )


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(trace_path)
    node_accesses, block_accesses = build_access_index(events)
    decisions = analyze_frontiers(events, node_accesses)
    write_frontier_csv(output_dir, decisions)

    unique_nodes = {
        int(event["node_id"])
        for event in events
        if "node_id" in event
    }
    unique_blocks = {
        int(block_hash)
        for event in events
        for block_hash in event.get("block_hashes", [])
    }
    event_counts = Counter(event["event"] for event in events)

    summary: dict[str, Any] = {
        "trace_path": str(trace_path),
        "event_counts": dict(event_counts),
        "unique_nodes": len(unique_nodes),
        "unique_blocks": len(unique_blocks),
        "frontier_decisions": len(decisions),
        "frontier_same_choice_rate": (
            sum(item.same_choice for item in decisions) / len(decisions) if decisions else 0.0
        ),
        "frontier_belady_diff_rate": (
            sum(not item.same_choice for item in decisions) / len(decisions) if decisions else 0.0
        ),
        "most_accessed_nodes": Counter(
            event["node_id"] for event in events if event.get("event") == "node_access"
        ).most_common(20),
        "most_accessed_blocks": Counter(
            int(block_hash)
            for event in events
            if event.get("event") == "node_access"
            for block_hash in event.get("block_hashes", [])
        ).most_common(20),
        "match_summary": summarize_match_results(events),
    }

    if args.block_capacity is not None:
        summary["page_cache_simulation"] = simulate_page_cache(
            events=events,
            block_accesses=block_accesses,
            capacity=args.block_capacity,
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
