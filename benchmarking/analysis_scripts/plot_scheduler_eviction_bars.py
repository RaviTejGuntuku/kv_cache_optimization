#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar charts comparing FCFS-LRU, FCFS-Belady, "
            "Prefix-LRU, and Prefix-Belady across workload/concurrency groups."
        )
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_rows(experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_path in sorted(experiment_root.rglob("reports/comparison.json")):
        run_root = comparison_path.parent.parent
        metadata = load_json(run_root / "run_metadata.json")
        comparison = load_json(comparison_path)
        args = metadata.get("args", {})
        dataset_path = Path(args["dataset_path"])
        workload_name = dataset_path.stem.replace("_", "-")
        scheduler = args["schedule_policy"]
        mc = int(args["max_concurrency"])

        for policy in ("lru", "belady"):
            rows.append(
                {
                    "group": f"{workload_name}\nmc={mc}",
                    "workload": workload_name,
                    "max_concurrency": mc,
                    "scheduler": scheduler,
                    "policy": policy,
                    "throughput": comparison["serving_metrics"]["output_throughput"][policy],
                    "median_ttft_ms": comparison["serving_metrics"]["median_ttft_ms"][policy],
                    "p99_ttft_ms": comparison["serving_metrics"]["p99_ttft_ms"][policy],
                    "median_itl_ms": comparison["serving_metrics"]["median_itl_ms"][policy],
                    "p99_itl_ms": comparison["serving_metrics"]["p99_itl_ms"][policy],
                    "hbm_hit_rate": comparison["cache_metrics"][f"hbm_hit_rate_{policy}"],
                    "hbm_miss_rate": comparison["cache_metrics"][f"hbm_miss_rate_{policy}"],
                    "transfer_proxy_bytes": comparison["transfer_proxy_bytes"][policy],
                }
            )
    return rows


def write_grouped_bar(rows: list[dict[str, Any]], output_dir: Path, metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ordering = [
        ("fcfs", "lru"),
        ("fcfs", "belady"),
        ("prefix-coverage", "lru"),
        ("prefix-coverage", "belady"),
    ]
    labels = {
        ("fcfs", "lru"): "FCFS-LRU",
        ("fcfs", "belady"): "FCFS-Belady",
        ("prefix-coverage", "lru"): "Prefix-LRU",
        ("prefix-coverage", "belady"): "Prefix-Belady",
    }
    colors = {
        ("fcfs", "lru"): "#4C78A8",
        ("fcfs", "belady"): "#9ecae9",
        ("prefix-coverage", "lru"): "#E45756",
        ("prefix-coverage", "belady"): "#f4a6a6",
    }
    hatches = {
        ("fcfs", "lru"): "",
        ("fcfs", "belady"): "//",
        ("prefix-coverage", "lru"): "",
        ("prefix-coverage", "belady"): "//",
    }

    groups = sorted({row["group"] for row in rows}, key=lambda x: (x.split("\n")[0], int(x.split("=")[1])))
    width = 0.18
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.3), 5.5))
    for idx, key in enumerate(ordering):
        vals = []
        for group in groups:
            match = next(
                (
                    row[metric]
                    for row in rows
                    if row["group"] == group and row["scheduler"] == key[0] and row["policy"] == key[1]
                ),
                None,
            )
            vals.append(match)
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=labels[key], color=colors[key], edgecolor="black", linewidth=0.5)
        for bar in bars:
            bar.set_hatch(hatches[key])

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_grouped_bars.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = gather_rows(experiment_root)
    (output_dir / "grouped_bar_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    specs = [
        ("throughput", "tokens/sec"),
        ("median_ttft_ms", "ms"),
        ("p99_ttft_ms", "ms"),
        ("median_itl_ms", "ms"),
        ("p99_itl_ms", "ms"),
        ("hbm_hit_rate", "fraction"),
        ("hbm_miss_rate", "fraction"),
        ("transfer_proxy_bytes", "bytes"),
    ]
    for metric, ylabel in specs:
        write_grouped_bar(rows, output_dir, metric, ylabel)


if __name__ == "__main__":
    main()
