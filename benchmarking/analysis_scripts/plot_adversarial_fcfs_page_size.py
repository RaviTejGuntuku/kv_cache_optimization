#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Belady vs LRU results for the adversarial FCFS page-size experiment."
        )
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(experiment_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_root in sorted(experiment_root.iterdir()):
        if not run_root.is_dir():
            continue
        metadata_path = run_root / "run_metadata.json"
        comparison_path = run_root / "reports" / "comparison.json"
        if not metadata_path.exists() or not comparison_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        run_args = metadata.get("args", {})
        serving = comparison.get("serving_metrics", {})
        cache = comparison.get("cache_metrics", {})

        dataset_path = Path(run_args["dataset_path"])
        workload = dataset_path.stem
        page_size = int(run_args.get("page_size", 16))

        rows.append(
            {
                "run_root": str(run_root),
                "workload": workload,
                "page_size": page_size,
                "throughput_lru": serving["output_throughput"]["lru"],
                "throughput_belady": serving["output_throughput"]["belady"],
                "throughput_pct_delta": 100.0 * serving["output_throughput"]["pct_delta"],
                "ttft_lru": serving["median_ttft_ms"]["lru"],
                "ttft_belady": serving["median_ttft_ms"]["belady"],
                "ttft_pct_improvement": 100.0 * serving["median_ttft_ms"]["pct_delta"],
                "itl_lru": serving["median_itl_ms"]["lru"],
                "itl_belady": serving["median_itl_ms"]["belady"],
                "itl_pct_improvement": 100.0 * serving["median_itl_ms"]["pct_delta"],
                "miss_rate_lru": cache["hbm_miss_rate_lru"],
                "miss_rate_belady": cache["hbm_miss_rate_belady"],
                "hit_rate_lru": cache["hbm_hit_rate_lru"],
                "hit_rate_belady": cache["hbm_hit_rate_belady"],
                "miss_count_lru": cache["hbm_miss_count_lru"],
                "miss_count_belady": cache["hbm_miss_count_belady"],
            }
        )
    return rows


def make_metric_bar(
    df: pd.DataFrame,
    *,
    workload: str,
    metric_lru: str,
    metric_belady: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    subset = df[df["workload"] == workload].sort_values("page_size")
    if subset.empty:
        return

    x = range(len(subset))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], subset[metric_lru], width=width, label="LRU")
    ax.bar([i + width / 2 for i in x], subset[metric_belady], width=width, label="Belady")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in subset["page_size"]])
    ax.set_xlabel("Page Size (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_delta_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    workloads = sorted(df["workload"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    for workload in workloads:
        subset = df[df["workload"] == workload].sort_values("page_size")
        ax.plot(
            subset["page_size"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=workload,
        )
    ax.set_xlabel("Page Size (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(experiment_root)
    if not rows:
        raise SystemExit(f"No completed runs found under {experiment_root}")

    df = pd.DataFrame(rows).sort_values(["workload", "page_size"])
    df.to_csv(output_dir / "summary.csv", index=False)

    for workload in sorted(df["workload"].unique()):
        safe = workload.replace("_", "-")
        make_metric_bar(
            df,
            workload=workload,
            metric_lru="throughput_lru",
            metric_belady="throughput_belady",
            ylabel="Output Throughput (tok/s)",
            title=f"{workload}: LRU vs Belady Throughput",
            output_path=output_dir / f"{safe}__throughput_bar.png",
        )
        make_metric_bar(
            df,
            workload=workload,
            metric_lru="miss_rate_lru",
            metric_belady="miss_rate_belady",
            ylabel="HBM Miss Rate",
            title=f"{workload}: LRU vs Belady HBM Miss Rate",
            output_path=output_dir / f"{safe}__miss_rate_bar.png",
        )
        make_metric_bar(
            df,
            workload=workload,
            metric_lru="ttft_lru",
            metric_belady="ttft_belady",
            ylabel="Median TTFT (ms)",
            title=f"{workload}: LRU vs Belady Median TTFT",
            output_path=output_dir / f"{safe}__ttft_bar.png",
        )

    make_delta_plot(
        df,
        metric="throughput_pct_delta",
        ylabel="Belady vs LRU Throughput Delta (%)",
        title="Belady Throughput Gain vs Page Size",
        output_path=output_dir / "throughput_pct_delta_vs_page_size.png",
    )
    make_delta_plot(
        df,
        metric="ttft_pct_improvement",
        ylabel="Belady Median TTFT Improvement (%)",
        title="Belady TTFT Improvement vs Page Size",
        output_path=output_dir / "ttft_pct_improvement_vs_page_size.png",
    )
    make_delta_plot(
        df,
        metric="itl_pct_improvement",
        ylabel="Belady Median ITL Improvement (%)",
        title="Belady ITL Improvement vs Page Size",
        output_path=output_dir / "itl_pct_improvement_vs_page_size.png",
    )


if __name__ == "__main__":
    main()
