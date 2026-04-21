#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LRU and Belady benchmark outputs and produce one summary report."
    )
    parser.add_argument("--lru-bench", required=True)
    parser.add_argument("--belady-bench", required=True)
    parser.add_argument("--lru-trace-summary", default=None)
    parser.add_argument("--belady-trace-summary", default=None)
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


def maybe_server_info_metric(result: dict[str, Any], key: str) -> Any:
    server_info = result.get("server_info") or {}
    if key in server_info:
        return server_info[key]
    internal_states = server_info.get("internal_states")
    if isinstance(internal_states, list):
        for item in internal_states:
            if isinstance(item, dict) and key in item:
                return item[key]
    return None


def delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return new - old


def pct_delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old


def estimate_transfer_bytes(
    trace_summary: Optional[dict[str, Any]],
    *,
    policy: str,
    page_size: int,
    bytes_per_token: int,
) -> Optional[int]:
    if not trace_summary:
        return None
    sim = trace_summary.get("page_cache_simulation")
    if sim:
        misses = sim.get(f"{policy}_misses")
        if misses is not None:
            return int(misses * page_size * bytes_per_token)
    match_summary = trace_summary.get("match_summary") or {}
    missed_blocks = match_summary.get("missed_blocks")
    if missed_blocks is not None:
        return int(missed_blocks * page_size * bytes_per_token)
    return None


def main() -> None:
    args = parse_args()
    lru = load_last_jsonl(Path(args.lru_bench))
    belady = load_last_jsonl(Path(args.belady_bench))
    lru_trace = load_json(args.lru_trace_summary)
    belady_trace = load_json(args.belady_trace_summary)

    report = {
        "inputs": {
            "lru_bench": args.lru_bench,
            "belady_bench": args.belady_bench,
            "lru_trace_summary": args.lru_trace_summary,
            "belady_trace_summary": args.belady_trace_summary,
        },
        "serving_metrics": {
            "output_throughput": {
                "lru": lru.get("output_throughput"),
                "belady": belady.get("output_throughput"),
                "delta": delta(belady.get("output_throughput"), lru.get("output_throughput")),
                "pct_delta": pct_delta(belady.get("output_throughput"), lru.get("output_throughput")),
            },
            "median_ttft_ms": {
                "lru": lru.get("median_ttft_ms"),
                "belady": belady.get("median_ttft_ms"),
                "delta": delta(belady.get("median_ttft_ms"), lru.get("median_ttft_ms")),
                "pct_delta": pct_delta(lru.get("median_ttft_ms"), belady.get("median_ttft_ms")),
            },
            "p99_ttft_ms": {
                "lru": lru.get("p99_ttft_ms"),
                "belady": belady.get("p99_ttft_ms"),
                "delta": delta(belady.get("p99_ttft_ms"), lru.get("p99_ttft_ms")),
                "pct_delta": pct_delta(lru.get("p99_ttft_ms"), belady.get("p99_ttft_ms")),
            },
            "median_itl_ms": {
                "lru": lru.get("median_itl_ms"),
                "belady": belady.get("median_itl_ms"),
                "delta": delta(belady.get("median_itl_ms"), lru.get("median_itl_ms")),
                "pct_delta": pct_delta(lru.get("median_itl_ms"), belady.get("median_itl_ms")),
            },
            "p99_itl_ms": {
                "lru": lru.get("p99_itl_ms"),
                "belady": belady.get("p99_itl_ms"),
                "delta": delta(belady.get("p99_itl_ms"), lru.get("p99_itl_ms")),
                "pct_delta": pct_delta(lru.get("p99_itl_ms"), belady.get("p99_itl_ms")),
            },
        },
        "cache_metrics": {
            "lru_cache_hit_rate_server": maybe_server_info_metric(lru, "cache_hit_rate"),
            "belady_cache_hit_rate_server": maybe_server_info_metric(belady, "cache_hit_rate"),
            "hbm_hit_count_lru": maybe_nested_get(
                lru_trace or {}, "page_cache_simulation", "lru_hits"
            ),
            "hbm_miss_count_lru": maybe_nested_get(
                lru_trace or {}, "page_cache_simulation", "lru_misses"
            ),
            "hbm_hit_rate_lru": maybe_nested_get(
                lru_trace or {}, "page_cache_simulation", "lru_hit_rate"
            ),
            "hbm_miss_rate_lru": maybe_nested_get(
                lru_trace or {}, "page_cache_simulation", "lru_miss_rate"
            ),
            "hbm_hit_count_belady": maybe_nested_get(
                belady_trace or {}, "page_cache_simulation", "belady_hits"
            ),
            "hbm_miss_count_belady": maybe_nested_get(
                belady_trace or {}, "page_cache_simulation", "belady_misses"
            ),
            "hbm_hit_rate_belady": maybe_nested_get(
                belady_trace or {}, "page_cache_simulation", "belady_hit_rate"
            ),
            "hbm_miss_rate_belady": maybe_nested_get(
                belady_trace or {}, "page_cache_simulation", "belady_miss_rate"
            ),
            "lru_trace_summary": lru_trace,
            "belady_trace_summary": belady_trace,
        },
        "transfer_proxy_bytes": {
            "lru": estimate_transfer_bytes(
                lru_trace,
                policy="lru",
                page_size=args.page_size,
                bytes_per_token=args.bytes_per_token,
            ),
            "belady": estimate_transfer_bytes(
                belady_trace,
                policy="belady",
                page_size=args.page_size,
                bytes_per_token=args.bytes_per_token,
            ),
        },
    }

    lru_transfer = report["transfer_proxy_bytes"]["lru"]
    belady_transfer = report["transfer_proxy_bytes"]["belady"]
    report["transfer_proxy_bytes"]["delta"] = delta(belady_transfer, lru_transfer)
    report["transfer_proxy_bytes"]["pct_delta"] = pct_delta(belady_transfer, lru_transfer)

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
