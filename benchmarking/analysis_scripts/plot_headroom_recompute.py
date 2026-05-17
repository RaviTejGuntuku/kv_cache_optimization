#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from headroom_plot_common import (
    ensure_dir,
    load_json,
    plot_grouped_bars,
    plot_lines,
    write_csv,
    write_json,
)


BASELINE_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__baselines$")
MICROBENCH_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__microbench$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full recomputation microbenchmark results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def serving_metric(report: dict, metric: str, branch: str) -> float | None:
    return report["serving_metrics"][metric][branch]


def build_baseline_rows(experiment_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(experiment_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = BASELINE_RE.match(run_dir.name)
        if not match:
            continue
        workload = match.group("workload")
        page_size = int(match.group("page"))
        for policy_dir_name, policy_name in (("opt", "opt"),):
            report_path = run_dir / policy_dir_name / "reports" / "comparison.json"
            if not report_path.exists():
                continue
            report = load_json(report_path)
            rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "policy": policy_name,
                    "output_throughput": serving_metric(report, "output_throughput", "secondary"),
                    "request_throughput": serving_metric(report, "request_throughput", "secondary"),
                    "median_ttft_ms": serving_metric(report, "median_ttft_ms", "secondary"),
                    "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "secondary"),
                    "median_itl_ms": serving_metric(report, "median_itl_ms", "secondary"),
                    "p99_itl_ms": serving_metric(report, "p99_itl_ms", "secondary"),
                    "block_miss_rate": report["cache_metrics"]["block_miss_rate"][report["inputs"]["secondary_label"]],
                    "block_hit_rate": report["cache_metrics"]["block_hit_rate"][report["inputs"]["secondary_label"]],
                    "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
                    "transfer_proxy_bytes": report["transfer_proxy_bytes"][report["inputs"]["secondary_label"]],
                }
            )
            if policy_name == "opt":
                rows.append(
                    {
                        "workload": workload,
                        "page_size": page_size,
                        "policy": "lru",
                        "output_throughput": serving_metric(report, "output_throughput", "primary"),
                        "request_throughput": serving_metric(report, "request_throughput", "primary"),
                        "median_ttft_ms": serving_metric(report, "median_ttft_ms", "primary"),
                        "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "primary"),
                        "median_itl_ms": serving_metric(report, "median_itl_ms", "primary"),
                        "p99_itl_ms": serving_metric(report, "p99_itl_ms", "primary"),
                        "block_miss_rate": report["cache_metrics"]["block_miss_rate"]["lru"],
                        "block_hit_rate": report["cache_metrics"]["block_hit_rate"]["lru"],
                        "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
                        "transfer_proxy_bytes": report["transfer_proxy_bytes"]["lru"],
                    }
                )
    return rows


def build_microbench_rows(experiment_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(experiment_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = MICROBENCH_RE.match(run_dir.name)
        if not match:
            continue
        workload = match.group("workload")
        page_size = int(match.group("page"))
        summary_path = run_dir / "microbench_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        for row in summary.get("rows", []):
            rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "row_index": row["row_index"],
                    "target_recompute_blocks": row["target_recompute_blocks"],
                    "prompt_len": row["prompt_len"],
                    "cold_median_ttft_ms": row["cold_median_ttft_ms"],
                    "warm_median_ttft_ms": row["warm_median_ttft_ms"],
                    "delta_median_ttft_ms": row["delta_median_ttft_ms"],
                    "cold_output_throughput": row["cold_output_throughput"],
                    "warm_output_throughput": row["warm_output_throughput"],
                    "delta_output_throughput": row["warm_output_throughput"] - row["cold_output_throughput"],
                    "has_target_recompute_blocks": row["target_recompute_blocks"] is not None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    metrics_dir = experiment_root / "metrics"
    graphs_dir = experiment_root / "graphs"
    ensure_dir(metrics_dir)
    ensure_dir(graphs_dir)

    baseline_rows = build_baseline_rows(experiment_root)
    microbench_rows = build_microbench_rows(experiment_root)

    write_csv(metrics_dir / "baseline_metrics.csv", baseline_rows)
    write_csv(metrics_dir / "microbench_metrics.csv", microbench_rows)
    write_json(
        metrics_dir / "aggregated_metrics.json",
        {"baseline_rows": baseline_rows, "microbench_rows": microbench_rows},
    )

    policy_order = ["lru", "opt"]
    for workload in sorted({row["workload"] for row in baseline_rows}):
        workload_baselines = [row for row in baseline_rows if row["workload"] == workload]
        workload_microbench = [row for row in microbench_rows if row["workload"] == workload]

        for metric, ylabel in (
            ("output_throughput", "Output Throughput (tok/s)"),
            ("median_ttft_ms", "Median TTFT (ms)"),
            ("p99_ttft_ms", "P99 TTFT (ms)"),
            ("block_miss_rate", "Block Miss Rate"),
        ):
            plot_grouped_bars(
                rows=workload_baselines,
                category_key="page_size",
                value_key=metric,
                series_key="policy",
                title=f"{workload}: {metric} by page size",
                xlabel="page_size",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{metric}_by_page_size.png",
                category_order=[str(page) for page in sorted({row["page_size"] for row in workload_baselines})],
                series_order=policy_order,
            )

        for metric, ylabel in (
            ("delta_median_ttft_ms", "Warm - Cold Median TTFT (ms)"),
            ("delta_output_throughput", "Warm - Cold Throughput (tok/s)"),
        ):
            rows_with_targets = [row for row in workload_microbench if row["has_target_recompute_blocks"]]
            if rows_with_targets:
                plot_lines(
                    rows=rows_with_targets,
                    x_key="target_recompute_blocks",
                    y_key=metric,
                    series_key="page_size",
                    title=f"{workload}: {metric} vs recompute blocks",
                    xlabel="target_recompute_blocks",
                    ylabel=ylabel,
                    output_path=graphs_dir / f"{workload}__{metric}_vs_recompute_blocks.png",
                    series_order=[str(page) for page in sorted({row["page_size"] for row in rows_with_targets})],
                )
            plot_lines(
                rows=workload_microbench,
                x_key="prompt_len",
                y_key=metric,
                series_key="page_size",
                title=f"{workload}: {metric} vs prompt length",
                xlabel="prompt_len",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{metric}_vs_prompt_len.png",
                series_order=[str(page) for page in sorted({row["page_size"] for row in workload_microbench})],
            )


if __name__ == "__main__":
    main()
