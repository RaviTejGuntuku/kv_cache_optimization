#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from headroom_plot_common import ensure_dir, load_json, plot_lines, write_csv, write_json


RUN_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__mem(?P<mem>\d+)__(?P<policy>belady|belady_bypass)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full effective-residency sweep results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def serving_metric(report: dict, metric: str, branch: str) -> float | None:
    return report["serving_metrics"][metric][branch]


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

        trace_summary = report["cache_metrics"]["primary_trace_summary"]
        page_cache = trace_summary["page_cache_simulation"]
        total_accesses = page_cache["total_accesses"]
        compulsory_rate = page_cache["compulsory_misses"] / total_accesses if total_accesses else 0.0

        key = (workload, mem_fraction, page_size)
        if key not in lru_seen:
            rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "mem_fraction": mem_fraction,
                    "policy": "lru",
                    "output_throughput": serving_metric(report, "output_throughput", "primary"),
                    "request_throughput": serving_metric(report, "request_throughput", "primary"),
                    "median_ttft_ms": serving_metric(report, "median_ttft_ms", "primary"),
                    "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "primary"),
                    "median_itl_ms": serving_metric(report, "median_itl_ms", "primary"),
                    "p99_itl_ms": serving_metric(report, "p99_itl_ms", "primary"),
                    "block_hit_rate": report["cache_metrics"]["block_hit_rate"]["lru"],
                    "block_miss_rate": report["cache_metrics"]["block_miss_rate"]["lru"],
                    "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
                    "compulsory_miss_rate": compulsory_rate,
                    "matched_blocks": report["cache_metrics"]["matched_blocks"]["lru"],
                    "missed_blocks": report["cache_metrics"]["missed_blocks"]["lru"],
                    "transfer_proxy_bytes": report["transfer_proxy_bytes"]["lru"],
                }
            )
            rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "mem_fraction": mem_fraction,
                    "policy": "compulsory",
                    "output_throughput": None,
                    "request_throughput": None,
                    "median_ttft_ms": None,
                    "p99_ttft_ms": None,
                    "median_itl_ms": None,
                    "p99_itl_ms": None,
                    "block_hit_rate": None,
                    "block_miss_rate": compulsory_rate,
                    "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
                    "compulsory_miss_rate": compulsory_rate,
                    "matched_blocks": None,
                    "missed_blocks": report["cache_metrics"]["compulsory_misses"],
                    "transfer_proxy_bytes": None,
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
                "request_throughput": serving_metric(report, "request_throughput", "secondary"),
                "median_ttft_ms": serving_metric(report, "median_ttft_ms", "secondary"),
                "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "secondary"),
                "median_itl_ms": serving_metric(report, "median_itl_ms", "secondary"),
                "p99_itl_ms": serving_metric(report, "p99_itl_ms", "secondary"),
                "block_hit_rate": report["cache_metrics"]["block_hit_rate"][report["inputs"]["secondary_label"]],
                "block_miss_rate": report["cache_metrics"]["block_miss_rate"][report["inputs"]["secondary_label"]],
                "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
                "compulsory_miss_rate": compulsory_rate,
                "matched_blocks": report["cache_metrics"]["matched_blocks"][report["inputs"]["secondary_label"]],
                "missed_blocks": report["cache_metrics"]["missed_blocks"][report["inputs"]["secondary_label"]],
                "transfer_proxy_bytes": report["transfer_proxy_bytes"][report["inputs"]["secondary_label"]],
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

    policy_order = ["lru", "opt", "opt_bypass", "compulsory"]
    for workload in sorted({row["workload"] for row in rows}):
        workload_rows = [row for row in rows if row["workload"] == workload]
        serving_rows = [row for row in workload_rows if row["policy"] != "compulsory"]
        miss_rows = workload_rows
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
                        "reuse_miss_gap_to_compulsory": (
                            float(row["block_miss_rate"]) - float(row["compulsory_miss_rate"])
                        ),
                    }
                )
        for metric, ylabel in (
            ("output_throughput", "Output Throughput (tok/s)"),
            ("median_ttft_ms", "Median TTFT (ms)"),
            ("p99_ttft_ms", "P99 TTFT (ms)"),
            ("median_itl_ms", "Median ITL (ms)"),
            ("p99_itl_ms", "P99 ITL (ms)"),
        ):
            plot_lines(
                rows=serving_rows,
                x_key="mem_fraction",
                y_key=metric,
                series_key="policy",
                title=f"{workload}: {metric} vs mem_fraction",
                xlabel="mem_fraction_static",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{metric}_vs_mem_fraction.png",
                series_order=["lru", "opt", "opt_bypass"],
            )
        plot_lines(
            rows=miss_rows,
            x_key="mem_fraction",
            y_key="block_miss_rate",
            series_key="policy",
            title=f"{workload}: block_miss_rate vs mem_fraction",
            xlabel="mem_fraction_static",
            ylabel="Block Miss Rate",
                output_path=graphs_dir / f"{workload}__block_miss_rate_vs_mem_fraction.png",
                series_order=policy_order,
        )

        if lru_rows:
            for metric, ylabel, filename in (
                ("output_throughput", "LRU Output Throughput (tok/s)", "lru_headroom__output_throughput_vs_mem_fraction.png"),
                ("median_ttft_ms", "LRU Median TTFT (ms)", "lru_headroom__median_ttft_vs_mem_fraction.png"),
                ("median_itl_ms", "LRU Median ITL (ms)", "lru_headroom__median_itl_vs_mem_fraction.png"),
                ("block_miss_rate", "LRU Block Miss Rate", "lru_headroom__block_miss_rate_vs_mem_fraction.png"),
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
                ("reuse_miss_gap_to_compulsory", "LRU Miss Rate - Compulsory Miss Rate", "lru_headroom__reuse_gap_to_compulsory.png"),
            ):
                plot_lines(
                    rows=lru_workload_rows,
                    x_key="mem_fraction",
                    y_key=metric,
                    series_key="policy",
                    title=f"{workload}: LRU headroom relative to baseline",
                    xlabel="mem_fraction_static",
                    ylabel=ylabel,
                    output_path=graphs_dir / f"{workload}__{filename}",
                    series_order=["lru"],
                )

    write_csv(metrics_dir / "lru_headroom.csv", lru_headroom_rows)
    write_json(metrics_dir / "lru_headroom.json", {"rows": lru_headroom_rows})


if __name__ == "__main__":
    main()
