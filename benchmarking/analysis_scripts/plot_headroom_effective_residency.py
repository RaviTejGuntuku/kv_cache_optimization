#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from headroom_plot_common import ensure_dir, load_json, plot_lines, write_csv, write_json


RUN_RE_CAP = re.compile(
    r"(?P<workload>.+)__ps(?P<page>\d+)__cap(?P<cap>\d+)__(?P<policy>belady)$"
)
RUN_RE_MEM = re.compile(
    r"(?P<workload>.+)__ps(?P<page>\d+)__mem(?P<mem>\d+)__(?P<policy>belady)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot effective-residency sweep results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def serving_metric(report: dict, metric: str, branch: str) -> float | None:
    return report["serving_metrics"][metric][branch]


def parse_run_dir_name(name: str) -> dict | None:
    match = RUN_RE_CAP.match(name)
    if match:
        return {
            "workload": match.group("workload"),
            "page_size": int(match.group("page")),
            "capacity_blocks": int(match.group("cap")),
            "capacity_label": f"cap_{match.group('cap')}",
        }
    match = RUN_RE_MEM.match(name)
    if match:
        mem_fraction = int(match.group("mem")) / 100.0
        return {
            "workload": match.group("workload"),
            "page_size": int(match.group("page")),
            "capacity_blocks": None,
            "capacity_label": f"mem_{mem_fraction:.2f}",
            "mem_fraction": mem_fraction,
        }
    return None


def build_rows(experiment_root: Path) -> list[dict]:
    rows: list[dict] = []
    lru_seen: set[tuple[str, int | None, str, int]] = set()
    for run_dir in sorted(experiment_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_dir_name(run_dir.name)
        if parsed is None:
            continue
        report_path = run_dir / "reports" / "comparison.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        workload = parsed["workload"]
        page_size = parsed["page_size"]
        capacity_blocks = parsed.get("capacity_blocks")
        capacity_label = parsed["capacity_label"]

        trace_summary = report["cache_metrics"]["primary_trace_summary"]
        page_cache = trace_summary["page_cache_simulation"]
        total_accesses = page_cache["total_accesses"]
        compulsory_rate = (
            page_cache["compulsory_misses"] / total_accesses if total_accesses else 0.0
        )

        common = {
            "workload": workload,
            "page_size": page_size,
            "capacity_blocks": capacity_blocks,
            "capacity_label": capacity_label,
            "compulsory_misses": report["cache_metrics"]["compulsory_misses"],
            "compulsory_miss_rate": compulsory_rate,
        }
        if "mem_fraction" in parsed:
            common["mem_fraction"] = parsed["mem_fraction"]

        key = (workload, capacity_blocks, capacity_label, page_size)
        if key not in lru_seen:
            rows.append(
                {
                    **common,
                    "policy": "lru",
                    "output_throughput": serving_metric(
                        report, "output_throughput", "primary"
                    ),
                    "request_throughput": serving_metric(
                        report, "request_throughput", "primary"
                    ),
                    "median_ttft_ms": serving_metric(report, "median_ttft_ms", "primary"),
                    "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "primary"),
                    "median_itl_ms": serving_metric(report, "median_itl_ms", "primary"),
                    "p99_itl_ms": serving_metric(report, "p99_itl_ms", "primary"),
                    "block_hit_rate": report["cache_metrics"]["block_hit_rate"]["lru"],
                    "block_miss_rate": report["cache_metrics"]["block_miss_rate"]["lru"],
                    "matched_blocks": report["cache_metrics"]["matched_blocks"]["lru"],
                    "missed_blocks": report["cache_metrics"]["missed_blocks"]["lru"],
                    "transfer_proxy_bytes": report["transfer_proxy_bytes"]["lru"],
                }
            )
            rows.append(
                {
                    **common,
                    "policy": "compulsory",
                    "output_throughput": None,
                    "request_throughput": None,
                    "median_ttft_ms": None,
                    "p99_ttft_ms": None,
                    "median_itl_ms": None,
                    "p99_itl_ms": None,
                    "block_hit_rate": None,
                    "block_miss_rate": compulsory_rate,
                    "matched_blocks": None,
                    "missed_blocks": report["cache_metrics"]["compulsory_misses"],
                    "transfer_proxy_bytes": None,
                }
            )
            lru_seen.add(key)

        rows.append(
            {
                **common,
                "policy": "opt",
                "output_throughput": serving_metric(
                    report, "output_throughput", "secondary"
                ),
                "request_throughput": serving_metric(
                    report, "request_throughput", "secondary"
                ),
                "median_ttft_ms": serving_metric(report, "median_ttft_ms", "secondary"),
                "p99_ttft_ms": serving_metric(report, "p99_ttft_ms", "secondary"),
                "median_itl_ms": serving_metric(report, "median_itl_ms", "secondary"),
                "p99_itl_ms": serving_metric(report, "p99_itl_ms", "secondary"),
                "block_hit_rate": report["cache_metrics"]["block_hit_rate"][
                    report["inputs"]["secondary_label"]
                ],
                "block_miss_rate": report["cache_metrics"]["block_miss_rate"][
                    report["inputs"]["secondary_label"]
                ],
                "matched_blocks": report["cache_metrics"]["matched_blocks"][
                    report["inputs"]["secondary_label"]
                ],
                "missed_blocks": report["cache_metrics"]["missed_blocks"][
                    report["inputs"]["secondary_label"]
                ],
                "transfer_proxy_bytes": report["transfer_proxy_bytes"][
                    report["inputs"]["secondary_label"]
                ],
            }
        )
    return rows


def numeric_capacity(row: dict) -> float:
    if row.get("capacity_blocks") is not None:
        return float(row["capacity_blocks"])
    return float(row.get("mem_fraction", 0.0))


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    metrics_dir = experiment_root / "metrics"
    graphs_dir = experiment_root / "graphs"
    ensure_dir(metrics_dir)
    ensure_dir(graphs_dir)

    rows = build_rows(experiment_root)
    rows = sorted(rows, key=lambda row: (row["workload"], numeric_capacity(row), row["policy"]))
    write_csv(metrics_dir / "aggregated_metrics.csv", rows)
    write_json(metrics_dir / "aggregated_metrics.json", {"rows": rows})

    lru_headroom_rows: list[dict] = []

    for workload in sorted({row["workload"] for row in rows}):
        workload_rows = [row for row in rows if row["workload"] == workload]
        serving_rows = [row for row in workload_rows if row["policy"] != "compulsory"]
        miss_rows = workload_rows
        lru_rows = sorted(
            [row for row in workload_rows if row["policy"] == "lru"],
            key=numeric_capacity,
        )
        if lru_rows:
            baseline = lru_rows[0]
            for row in lru_rows:
                delta_throughput = float(row["output_throughput"]) - float(
                    baseline["output_throughput"]
                )
                delta_ttft = float(row["median_ttft_ms"]) - float(
                    baseline["median_ttft_ms"]
                )
                delta_itl = float(row["median_itl_ms"]) - float(
                    baseline["median_itl_ms"]
                )
                lru_headroom_rows.append(
                    {
                        **row,
                        "baseline_capacity_blocks": baseline.get("capacity_blocks"),
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
                        "distance_above_compulsory": (
                            float(row["block_miss_rate"]) - float(row["compulsory_miss_rate"])
                        ),
                    }
                )

        for metric, ylabel, suffix in (
            ("output_throughput", "Output Throughput (tok/s)", "output_throughput"),
            ("median_ttft_ms", "Median TTFT (ms)", "median_ttft"),
            ("p99_ttft_ms", "P99 TTFT (ms)", "p99_ttft"),
            ("median_itl_ms", "Median ITL (ms)", "median_itl"),
            ("p99_itl_ms", "P99 ITL (ms)", "p99_itl"),
        ):
            plot_lines(
                rows=serving_rows,
                x_key="capacity_blocks" if any(r.get("capacity_blocks") is not None for r in serving_rows) else "mem_fraction",
                y_key=metric,
                series_key="policy",
                title=f"{workload}: {metric} vs reusable capacity",
                xlabel="reusable_capacity_blocks",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{suffix}_vs_capacity.png",
                series_order=["lru", "opt"],
            )

        plot_lines(
            rows=miss_rows,
            x_key="capacity_blocks" if any(r.get("capacity_blocks") is not None for r in miss_rows) else "mem_fraction",
            y_key="block_miss_rate",
            series_key="policy",
            title=f"{workload}: block miss rate vs reusable capacity",
            xlabel="reusable_capacity_blocks",
            ylabel="Block Miss Rate",
            output_path=graphs_dir / f"{workload}__block_miss_rate_vs_capacity.png",
            series_order=["lru", "opt", "compulsory"],
        )

        lru_workload_rows = [row for row in lru_headroom_rows if row["workload"] == workload]
        if lru_workload_rows:
            for metric, ylabel, suffix in (
                ("output_throughput", "LRU Output Throughput (tok/s)", "lru_headroom__output_throughput_vs_capacity.png"),
                ("median_ttft_ms", "LRU Median TTFT (ms)", "lru_headroom__median_ttft_vs_capacity.png"),
                ("median_itl_ms", "LRU Median ITL (ms)", "lru_headroom__median_itl_vs_capacity.png"),
                ("block_miss_rate", "LRU Block Miss Rate", "lru_headroom__block_miss_rate_vs_capacity.png"),
                ("distance_above_compulsory", "LRU Miss Rate - Compulsory Miss Rate", "lru_headroom__distance_above_compulsory.png"),
                ("throughput_pct_gain_vs_baseline", "LRU Throughput Gain vs Lowest Capacity (%)", "lru_headroom__throughput_pct_gain_vs_baseline.png"),
                ("median_ttft_pct_change_vs_baseline", "LRU Median TTFT Change vs Lowest Capacity (%)", "lru_headroom__median_ttft_pct_change_vs_baseline.png"),
                ("median_itl_pct_change_vs_baseline", "LRU Median ITL Change vs Lowest Capacity (%)", "lru_headroom__median_itl_pct_change_vs_baseline.png"),
            ):
                plot_lines(
                    rows=lru_workload_rows,
                    x_key="capacity_blocks" if any(r.get("capacity_blocks") is not None for r in lru_workload_rows) else "mem_fraction",
                    y_key=metric,
                    series_key="policy",
                    title=f"{workload}: {metric} vs reusable capacity",
                    xlabel="reusable_capacity_blocks",
                    ylabel=ylabel,
                    output_path=graphs_dir / f"{workload}__{suffix}",
                    series_order=["lru"],
                )

            plot_lines(
                rows=lru_workload_rows,
                x_key="distance_above_compulsory",
                y_key="output_throughput",
                series_key="policy",
                title=f"{workload}: throughput vs distance above compulsory",
                xlabel="distance_above_compulsory",
                ylabel="LRU Output Throughput (tok/s)",
                output_path=graphs_dir / f"{workload}__lru_headroom__throughput_vs_distance_above_compulsory.png",
                series_order=["lru"],
            )
            plot_lines(
                rows=lru_workload_rows,
                x_key="distance_above_compulsory",
                y_key="median_ttft_ms",
                series_key="policy",
                title=f"{workload}: median TTFT vs distance above compulsory",
                xlabel="distance_above_compulsory",
                ylabel="LRU Median TTFT (ms)",
                output_path=graphs_dir / f"{workload}__lru_headroom__median_ttft_vs_distance_above_compulsory.png",
                series_order=["lru"],
            )
            plot_lines(
                rows=lru_workload_rows,
                x_key="distance_above_compulsory",
                y_key="median_itl_ms",
                series_key="policy",
                title=f"{workload}: median ITL vs distance above compulsory",
                xlabel="distance_above_compulsory",
                ylabel="LRU Median ITL (ms)",
                output_path=graphs_dir / f"{workload}__lru_headroom__median_itl_vs_distance_above_compulsory.png",
                series_order=["lru"],
            )

    write_csv(metrics_dir / "lru_headroom.csv", lru_headroom_rows)
    write_json(metrics_dir / "lru_headroom.json", {"rows": lru_headroom_rows})


if __name__ == "__main__":
    main()
