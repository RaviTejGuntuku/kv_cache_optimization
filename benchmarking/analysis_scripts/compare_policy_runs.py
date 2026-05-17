#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark runs with generic policy labels."
    )
    parser.add_argument("--primary-label", default="lru")
    parser.add_argument("--secondary-label", required=True)
    parser.add_argument("--primary-bench", required=True)
    parser.add_argument("--secondary-bench", required=True)
    parser.add_argument("--primary-trace-summary", default=None)
    parser.add_argument("--secondary-trace-summary", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--bytes-per-token", type=int, default=131072)
    return parser.parse_args()


def load_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No rows found in {path}")
    return json.loads(lines[-1])


def load_json(path: Optional[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return new - old


def pct_delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old


def match_summary_metric(trace_summary: Optional[dict[str, Any]], key: str) -> Optional[float]:
    if not trace_summary:
        return None
    return maybe_nested_get(trace_summary, "match_summary", key)


def compulsory_misses(trace_summary: Optional[dict[str, Any]]) -> Optional[int]:
    if not trace_summary:
        return None
    return maybe_nested_get(trace_summary, "page_cache_simulation", "compulsory_misses")


def estimate_transfer_bytes(
    trace_summary: Optional[dict[str, Any]],
    *,
    page_size: int,
    bytes_per_token: int,
) -> Optional[int]:
    if not trace_summary:
        return None
    missed_blocks = match_summary_metric(trace_summary, "missed_blocks")
    if missed_blocks is None:
        return None
    return int(missed_blocks * page_size * bytes_per_token)


def build_metric_block(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> dict[str, Any]:
    a = primary.get(key)
    b = secondary.get(key)
    return {
        "primary": a,
        "secondary": b,
        "delta": delta(b, a),
        "pct_delta": pct_delta(b, a),
    }


def main() -> None:
    args = parse_args()
    primary = load_last_jsonl(Path(args.primary_bench))
    secondary = load_last_jsonl(Path(args.secondary_bench))
    primary_trace = load_json(args.primary_trace_summary)
    secondary_trace = load_json(args.secondary_trace_summary)

    report = {
        "inputs": {
            "primary_label": args.primary_label,
            "secondary_label": args.secondary_label,
            "primary_bench": args.primary_bench,
            "secondary_bench": args.secondary_bench,
            "primary_trace_summary": args.primary_trace_summary,
            "secondary_trace_summary": args.secondary_trace_summary,
        },
        "serving_metrics": {
            "request_throughput": build_metric_block(primary, secondary, "request_throughput"),
            "output_throughput": build_metric_block(primary, secondary, "output_throughput"),
            "median_ttft_ms": build_metric_block(primary, secondary, "median_ttft_ms"),
            "p99_ttft_ms": build_metric_block(primary, secondary, "p99_ttft_ms"),
            "median_itl_ms": build_metric_block(primary, secondary, "median_itl_ms"),
            "p99_itl_ms": build_metric_block(primary, secondary, "p99_itl_ms"),
        },
        "cache_metrics": {
            "compulsory_misses": compulsory_misses(primary_trace),
            "matched_blocks": {
                args.primary_label: match_summary_metric(primary_trace, "matched_blocks"),
                args.secondary_label: match_summary_metric(secondary_trace, "matched_blocks"),
            },
            "missed_blocks": {
                args.primary_label: match_summary_metric(primary_trace, "missed_blocks"),
                args.secondary_label: match_summary_metric(secondary_trace, "missed_blocks"),
            },
            "block_hit_rate": {
                args.primary_label: match_summary_metric(primary_trace, "block_hit_rate"),
                args.secondary_label: match_summary_metric(secondary_trace, "block_hit_rate"),
            },
            "block_miss_rate": {
                args.primary_label: match_summary_metric(primary_trace, "block_miss_rate"),
                args.secondary_label: match_summary_metric(secondary_trace, "block_miss_rate"),
            },
            "primary_trace_summary": primary_trace,
            "secondary_trace_summary": secondary_trace,
        },
        "transfer_proxy_bytes": {
            args.primary_label: estimate_transfer_bytes(
                primary_trace,
                page_size=args.page_size,
                bytes_per_token=args.bytes_per_token,
            ),
            args.secondary_label: estimate_transfer_bytes(
                secondary_trace,
                page_size=args.page_size,
                bytes_per_token=args.bytes_per_token,
            ),
        },
    }

    primary_transfer = report["transfer_proxy_bytes"][args.primary_label]
    secondary_transfer = report["transfer_proxy_bytes"][args.secondary_label]
    report["transfer_proxy_bytes"]["delta"] = delta(secondary_transfer, primary_transfer)
    report["transfer_proxy_bytes"]["pct_delta"] = pct_delta(secondary_transfer, primary_transfer)

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
