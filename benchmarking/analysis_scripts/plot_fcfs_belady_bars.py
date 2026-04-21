#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FCFS-only LRU vs Belady grouped bar charts from benchmark JSON outputs."
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_rows(experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lru_path in sorted(experiment_root.glob("workloads/*/sched-fcfs/mc-*/benchmarks/lru.jsonl")):
        belady_path = lru_path.parent / "belady.jsonl"
        if not belady_path.exists():
            continue
        rel = lru_path.relative_to(experiment_root / "workloads")
        workload, _sched, mc, _bench, _file = rel.parts
        lru = load_json(lru_path)
        belady = load_json(belady_path)
        for policy, payload in (("lru", lru), ("belady", belady)):
            rows.append(
                {
                    "group": f"{workload}\nmc={int(mc.replace('mc-', ''))}",
                    "workload": workload,
                    "mc": int(mc.replace("mc-", "")),
                    "policy": policy,
                    "request_throughput": payload["request_throughput"],
                    "output_throughput": payload["output_throughput"],
                    "median_e2e_latency_ms": payload["median_e2e_latency_ms"],
                    "p99_e2e_latency_ms": payload["p99_e2e_latency_ms"],
                    "median_ttft_ms": payload["median_ttft_ms"],
                    "p99_ttft_ms": payload["p99_ttft_ms"],
                    "median_itl_ms": payload["median_itl_ms"],
                    "p99_itl_ms": payload["p99_itl_ms"],
                }
            )
    return rows


def write_grouped_bars(rows: list[dict[str, Any]], output_dir: Path, metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    groups = sorted({row["group"] for row in rows}, key=lambda x: (x.split("\n")[0], int(x.split("=")[1])))
    x = np.arange(len(groups))
    width = 0.34
    lru_vals = []
    belady_vals = []
    for group in groups:
        lru_vals.append(next(row[metric] for row in rows if row["group"] == group and row["policy"] == "lru"))
        belady_vals.append(next(row[metric] for row in rows if row["group"] == group and row["policy"] == "belady"))

    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 1.25), 5.3))
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
    fig.savefig(output_dir / f"{metric}_fcfs_grouped_bars.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = gather_rows(experiment_root)
    (output_dir / "fcfs_raw_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    specs = [
        ("request_throughput", "req/s"),
        ("output_throughput", "tok/s"),
        ("median_e2e_latency_ms", "ms"),
        ("p99_e2e_latency_ms", "ms"),
        ("median_ttft_ms", "ms"),
        ("p99_ttft_ms", "ms"),
        ("median_itl_ms", "ms"),
        ("p99_itl_ms", "ms"),
    ]
    for metric, ylabel in specs:
        write_grouped_bars(rows, output_dir, metric, ylabel)


if __name__ == "__main__":
    main()
