#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LRU vs Belady metrics from synced FCFS static-prefix runs."
    )
    parser.add_argument("--synced-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No rows found in {path}")
    return json.loads(lines[-1])


def workload_label(name: str) -> str:
    mapping = {
        "natural-bursty-return-hbm": "Bursty Return",
        "natural-hotset-one-shot-hbm": "Hotset vs One-Shot",
        "natural-zipf-bursty-hbm": "Zipf Bursty",
    }
    return mapping.get(name, name)


def gather_rows(synced_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(synced_root.glob("*/mc-128__20260419T*Z")):
        workload = run_dir.parent.name
        lru_bench = load_last_jsonl(run_dir / "benchmarks" / "lru.jsonl")
        belady_bench = load_last_jsonl(run_dir / "benchmarks" / "belady.jsonl")
        comparison = load_json(run_dir / "reports" / "comparison.json")

        rows.append(
            {
                "workload": workload,
                "label": workload_label(workload),
                "policy": "lru",
                "request_throughput": lru_bench["request_throughput"],
                "output_throughput": lru_bench["output_throughput"],
                "median_ttft_ms": lru_bench["median_ttft_ms"],
                "p99_ttft_ms": lru_bench["p99_ttft_ms"],
                "median_itl_ms": lru_bench["median_itl_ms"],
                "p99_itl_ms": lru_bench["p99_itl_ms"],
                "median_e2e_latency_ms": lru_bench["median_e2e_latency_ms"],
                "p99_e2e_latency_ms": lru_bench["p99_e2e_latency_ms"],
                "hbm_hit_rate": comparison["cache_metrics"]["hbm_hit_rate_lru"],
                "hbm_miss_rate": comparison["cache_metrics"]["hbm_miss_rate_lru"],
                "hbm_hits": comparison["cache_metrics"]["hbm_hit_count_lru"],
                "hbm_misses": comparison["cache_metrics"]["hbm_miss_count_lru"],
            }
        )
        rows.append(
            {
                "workload": workload,
                "label": workload_label(workload),
                "policy": "belady",
                "request_throughput": belady_bench["request_throughput"],
                "output_throughput": belady_bench["output_throughput"],
                "median_ttft_ms": belady_bench["median_ttft_ms"],
                "p99_ttft_ms": belady_bench["p99_ttft_ms"],
                "median_itl_ms": belady_bench["median_itl_ms"],
                "p99_itl_ms": belady_bench["p99_itl_ms"],
                "median_e2e_latency_ms": belady_bench["median_e2e_latency_ms"],
                "p99_e2e_latency_ms": belady_bench["p99_e2e_latency_ms"],
                "hbm_hit_rate": comparison["cache_metrics"]["hbm_hit_rate_belady"],
                "hbm_miss_rate": comparison["cache_metrics"]["hbm_miss_rate_belady"],
                "hbm_hits": comparison["cache_metrics"]["hbm_hit_count_belady"],
                "hbm_misses": comparison["cache_metrics"]["hbm_miss_count_belady"],
            }
        )
    return rows


def write_grouped_bars(rows: list[dict[str, Any]], output_dir: Path, metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    groups = []
    seen = set()
    for row in rows:
        if row["label"] not in seen:
            groups.append(row["label"])
            seen.add(row["label"])

    x = np.arange(len(groups))
    width = 0.34
    lru_vals = []
    belady_vals = []
    for group in groups:
        lru_vals.append(next(row[metric] for row in rows if row["label"] == group and row["policy"] == "lru"))
        belady_vals.append(next(row[metric] for row in rows if row["label"] == group and row["policy"] == "belady"))

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.8), 5.2))
    ax.bar(x - width / 2, lru_vals, width, label="FCFS-LRU", color="#4C78A8", edgecolor="black", linewidth=0.5)
    bars = ax.bar(x + width / 2, belady_vals, width, label="FCFS-Belady", color="#9ecae9", edgecolor="black", linewidth=0.5)
    for bar in bars:
        bar.set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_grouped_bars.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    synced_root = Path(args.synced_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = gather_rows(synced_root)
    if not rows:
        raise ValueError(f"No synced runs found under {synced_root}")

    (output_dir / "fcfs_static_prefix_raw_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    specs = [
        ("request_throughput", "req/s"),
        ("output_throughput", "tok/s"),
        ("median_ttft_ms", "ms"),
        ("p99_ttft_ms", "ms"),
        ("median_itl_ms", "ms"),
        ("p99_itl_ms", "ms"),
        ("median_e2e_latency_ms", "ms"),
        ("p99_e2e_latency_ms", "ms"),
        ("hbm_hit_rate", "rate"),
        ("hbm_miss_rate", "rate"),
        ("hbm_hits", "blocks"),
        ("hbm_misses", "blocks"),
    ]
    for metric, ylabel in specs:
        write_grouped_bars(rows, output_dir, metric, ylabel)


if __name__ == "__main__":
    main()
