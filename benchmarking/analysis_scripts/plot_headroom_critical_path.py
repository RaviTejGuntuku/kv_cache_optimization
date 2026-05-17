#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from headroom_plot_common import ensure_dir, load_json, plot_lines, write_csv, write_json


RUN_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__mem(?P<mem>\d+)__(?P<policy>belady|belady_bypass)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full critical-path miss-attribution results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def serving_metric(report: dict, metric: str, branch: str) -> float | None:
    return report["serving_metrics"][metric][branch]


def critical_payload(run_dir: Path, name: str) -> dict:
    return load_json(run_dir / "reports" / name)


def build_rows(experiment_root: Path) -> list[dict]:
    rows: list[dict] = []
    lru_seen: set[tuple[str, float, int]] = set()
    for run_dir in sorted(experiment_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        report = load_json(run_dir / "reports" / "comparison.json")
        workload = match.group("workload")
        page_size = int(match.group("page"))
        mem_fraction = int(match.group("mem")) / 100.0
        secondary_policy = "opt" if match.group("policy") == "belady" else "opt_bypass"
        secondary_report_name = "belady_critical_path.json" if secondary_policy == "opt" else "belady_bypass_critical_path.json"
        lru_cp = critical_payload(run_dir, "lru_critical_path.json")
        secondary_cp = critical_payload(run_dir, secondary_report_name)

        key = (workload, mem_fraction, page_size)
        if key not in lru_seen:
            rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "mem_fraction": mem_fraction,
                    "policy": "lru",
                    "output_throughput": serving_metric(report, "output_throughput", "primary"),
                    "median_ttft_ms": serving_metric(report, "median_ttft_ms", "primary"),
                    "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "primary"),
                    "median_itl_ms": serving_metric(report, "median_itl_ms", "primary"),
                    "p99_itl_ms": serving_metric(report, "p99_itl_ms", "primary"),
                    **{k: lru_cp.get(k) for k in (
                        "critical_path_miss_rate",
                        "mean_missed_blocks_per_request",
                        "mean_missed_tokens_per_request",
                        "mean_ttft_ms",
                        "corr_missed_blocks_vs_ttft",
                        "corr_missed_tokens_vs_ttft",
                    )},
                }
            )
            lru_seen.add(key)

        rows.append(
            {
                "workload": workload,
                "page_size": page_size,
                "mem_fraction": mem_fraction,
                "policy": secondary_policy,
                "output_throughput": serving_metric(report, "output_throughput", "secondary"),
                "median_ttft_ms": serving_metric(report, "median_ttft_ms", "secondary"),
                "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "secondary"),
                "median_itl_ms": serving_metric(report, "median_itl_ms", "secondary"),
                "p99_itl_ms": serving_metric(report, "p99_itl_ms", "secondary"),
                **{k: secondary_cp.get(k) for k in (
                    "critical_path_miss_rate",
                    "mean_missed_blocks_per_request",
                    "mean_missed_tokens_per_request",
                    "mean_ttft_ms",
                    "corr_missed_blocks_vs_ttft",
                    "corr_missed_tokens_vs_ttft",
                )},
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

    rows = build_rows(experiment_root)
    write_csv(metrics_dir / "aggregated_metrics.csv", rows)
    write_json(metrics_dir / "aggregated_metrics.json", {"rows": rows})

    lru_headroom_rows: list[dict] = []

    for workload in sorted({row["workload"] for row in rows}):
        workload_rows = [row for row in rows if row["workload"] == workload]
        lru_rows = sorted(
            [row for row in workload_rows if row["policy"] == "lru"],
            key=lambda row: float(row["mem_fraction"]),
        )
        if lru_rows:
            baseline = lru_rows[0]
            for row in lru_rows:
                delta_throughput = float(row["output_throughput"]) - float(baseline["output_throughput"])
                delta_ttft = float(row["median_ttft_ms"]) - float(baseline["median_ttft_ms"])
                delta_itl = float(row["median_itl_ms"]) - float(baseline["median_itl_ms"])
                lru_headroom_rows.append(
                    {
                        **row,
                        "baseline_mem_fraction": baseline["mem_fraction"],
                        "throughput_delta_vs_baseline": delta_throughput,
                        "throughput_pct_gain_vs_baseline": (
                            delta_throughput / float(baseline["output_throughput"]) * 100.0
                        ),
                        "median_ttft_delta_vs_baseline": delta_ttft,
                        "median_ttft_pct_change_vs_baseline": (
                            delta_ttft / float(baseline["median_ttft_ms"]) * 100.0
                        ),
                        "median_itl_delta_vs_baseline": delta_itl,
                        "median_itl_pct_change_vs_baseline": (
                            delta_itl / float(baseline["median_itl_ms"]) * 100.0
                        ),
                    }
                )
        for metric, ylabel in (
            ("critical_path_miss_rate", "Critical-Path Miss Rate"),
            ("mean_missed_blocks_per_request", "Mean Missed Blocks / Request"),
            ("output_throughput", "Output Throughput (tok/s)"),
            ("median_ttft_ms", "Median TTFT (ms)"),
            ("p99_itl_ms", "P99 ITL (ms)"),
        ):
            plot_lines(
                rows=workload_rows,
                x_key="mem_fraction",
                y_key=metric,
                series_key="policy",
                title=f"{workload}: {metric} vs mem_fraction",
                xlabel="mem_fraction_static",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{metric}_vs_mem_fraction.png",
                series_order=["lru", "opt", "opt_bypass"],
            )

        if lru_rows:
            for metric, ylabel, filename in (
                ("critical_path_miss_rate", "LRU Critical-Path Miss Rate", "lru_headroom__critical_path_miss_rate_vs_mem_fraction.png"),
                ("output_throughput", "LRU Output Throughput (tok/s)", "lru_headroom__output_throughput_vs_mem_fraction.png"),
                ("median_ttft_ms", "LRU Median TTFT (ms)", "lru_headroom__median_ttft_vs_mem_fraction.png"),
                ("median_itl_ms", "LRU Median ITL (ms)", "lru_headroom__median_itl_vs_mem_fraction.png"),
            ):
                plot_lines(
                    rows=lru_rows,
                    x_key="mem_fraction",
                    y_key=metric,
                    series_key="policy",
                    title=f"{workload}: LRU headroom vs mem_fraction",
                    xlabel="mem_fraction_static",
                    ylabel=ylabel,
                    output_path=graphs_dir / f"{workload}__{filename}",
                    series_order=["lru"],
                )

            lru_workload_rows = [row for row in lru_headroom_rows if row["workload"] == workload]
            for metric, ylabel, filename in (
                ("throughput_pct_gain_vs_baseline", "LRU Throughput Gain vs Lowest Capacity (%)", "lru_headroom__throughput_pct_gain_vs_baseline.png"),
                ("median_ttft_pct_change_vs_baseline", "LRU Median TTFT Change vs Lowest Capacity (%)", "lru_headroom__median_ttft_pct_change_vs_baseline.png"),
                ("median_itl_pct_change_vs_baseline", "LRU Median ITL Change vs Lowest Capacity (%)", "lru_headroom__median_itl_pct_change_vs_baseline.png"),
            ):
                plot_lines(
                    rows=lru_workload_rows,
                    x_key="critical_path_miss_rate",
                    y_key=metric,
                    series_key="policy",
                    title=f"{workload}: LRU impact vs critical-path misses",
                    xlabel="critical_path_miss_rate",
                    ylabel=ylabel,
                    output_path=graphs_dir / f"{workload}__{filename}",
                    series_order=["lru"],
                )

    write_csv(metrics_dir / "lru_headroom.csv", lru_headroom_rows)
    write_json(metrics_dir / "lru_headroom.json", {"rows": lru_headroom_rows})


if __name__ == "__main__":
    main()
