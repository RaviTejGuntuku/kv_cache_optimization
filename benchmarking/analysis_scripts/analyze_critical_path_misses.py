#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate critical-path miss impact from request-level match results and TTFTs."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-label", required=True)
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_bench(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No benchmark rows found in {path}")
    return json.loads(lines[-1])


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.trace))
    bench = load_bench(Path(args.bench))

    match_events = [event for event in events if event.get("event") == "match_result"]
    ttfts = bench.get("ttfts", [])
    if len(match_events) != len(ttfts):
        usable = min(len(match_events), len(ttfts))
        match_events = match_events[:usable]
        ttfts = ttfts[:usable]

    missed_blocks = [float(event.get("missed_blocks", 0)) for event in match_events]
    missed_tokens = [float(event.get("missed_tokens", 0)) for event in match_events]
    ttft_ms = [float(value) * 1000.0 for value in ttfts]

    buckets: dict[int, list[float]] = defaultdict(list)
    for miss_blocks, ttft in zip(missed_blocks, ttft_ms):
        buckets[int(miss_blocks)].append(ttft)

    payload = {
        "policy_label": args.policy_label,
        "request_count": len(match_events),
        "critical_path_miss_rate": (
            sum(missed_blocks) / sum(event.get("matched_blocks", 0) + event.get("missed_blocks", 0) for event in match_events)
            if match_events
            else 0.0
        ),
        "mean_missed_blocks_per_request": statistics.fmean(missed_blocks) if missed_blocks else 0.0,
        "mean_missed_tokens_per_request": statistics.fmean(missed_tokens) if missed_tokens else 0.0,
        "mean_ttft_ms": statistics.fmean(ttft_ms) if ttft_ms else 0.0,
        "corr_missed_blocks_vs_ttft": correlation(missed_blocks, ttft_ms),
        "corr_missed_tokens_vs_ttft": correlation(missed_tokens, ttft_ms),
        "ttft_by_missed_block_bucket_ms": {
            str(bucket): {
                "count": len(values),
                "mean_ttft_ms": statistics.fmean(values),
                "median_ttft_ms": statistics.median(values),
            }
            for bucket, values in sorted(buckets.items())
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
