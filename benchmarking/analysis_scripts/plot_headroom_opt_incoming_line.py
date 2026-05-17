#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


SWEEP_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__n(?P<frac>\d+)$")
BASELINE_RE = re.compile(r"(?P<workload>.+)__ps(?P<page>\d+)__opt_baseline$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full incoming-line-aware OPT+bypass results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def serving_metric(report: dict, metric: str, branch: str) -> float | None:
    return report["serving_metrics"][metric][branch]


def build_rows(experiment_root: Path) -> tuple[list[dict], list[dict]]:
    sweep_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for run_dir in sorted(experiment_root.iterdir()):
        if not run_dir.is_dir():
            continue

        sweep_match = SWEEP_RE.match(run_dir.name)
        if sweep_match:
            report = load_json(run_dir / "reports" / "comparison.json")
            workload = sweep_match.group("workload")
            page_size = int(sweep_match.group("page"))
            cache_fraction = int(sweep_match.group("frac")) / 10.0
            sweep_rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "cache_fraction": cache_fraction,
                    "policy": "opt_bypass",
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
            continue

        baseline_match = BASELINE_RE.match(run_dir.name)
        if baseline_match:
            report = load_json(run_dir / "reports" / "comparison.json")
            workload = baseline_match.group("workload")
            page_size = int(baseline_match.group("page"))
            baseline_rows.append(
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
            baseline_rows.append(
                {
                    "workload": workload,
                    "page_size": page_size,
                    "policy": "opt",
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
    return sweep_rows, baseline_rows


def build_best_rows(experiment_root: Path, sweep_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for best_path in sorted(experiment_root.glob("*__best_fraction.json")):
        workload = best_path.name.replace("__best_fraction.json", "")
        best_payload = load_json(best_path)
        best_run_root = Path(best_payload["best"]["run_root"]).name
        best_match = SWEEP_RE.match(best_run_root)
        if not best_match:
            continue
        page_size = int(best_match.group("page"))
        best_cache_fraction = int(best_match.group("frac")) / 10.0
        best_row = next(
            row
            for row in sweep_rows
            if row["workload"] == workload
            and row["page_size"] == page_size
            and abs(row["cache_fraction"] - best_cache_fraction) < 1e-9
        )
        rows.append(best_row | {"policy": "best_opt_bypass"})
        rows.extend(
            row
            for row in baseline_rows
            if row["workload"] == workload and row["page_size"] == page_size
        )
    return rows


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    metrics_dir = experiment_root / "metrics"
    graphs_dir = experiment_root / "graphs"
    ensure_dir(metrics_dir)
    ensure_dir(graphs_dir)

    sweep_rows, baseline_rows = build_rows(experiment_root)
    best_rows = build_best_rows(experiment_root, sweep_rows, baseline_rows)

    write_csv(metrics_dir / "bypass_sweep_metrics.csv", sweep_rows)
    write_csv(metrics_dir / "best_vs_baselines.csv", best_rows)
    write_json(
        metrics_dir / "aggregated_metrics.json",
        {"bypass_sweep_rows": sweep_rows, "best_rows": best_rows, "baseline_rows": baseline_rows},
    )

    for workload in sorted({row["workload"] for row in sweep_rows}):
        workload_sweep = [row for row in sweep_rows if row["workload"] == workload]
        workload_best = [row for row in best_rows if row["workload"] == workload]

        for metric, ylabel in (
            ("output_throughput", "Output Throughput (tok/s)"),
            ("median_ttft_ms", "Median TTFT (ms)"),
            ("block_miss_rate", "Block Miss Rate"),
        ):
            plot_lines(
                rows=workload_sweep,
                x_key="cache_fraction",
                y_key=metric,
                series_key="page_size",
                title=f"{workload}: {metric} vs cache fraction",
                xlabel="cache_fraction n",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{metric}_vs_cache_fraction.png",
                series_order=[str(page) for page in sorted({row["page_size"] for row in workload_sweep})],
            )
            plot_grouped_bars(
                rows=workload_best,
                category_key="page_size",
                value_key=metric,
                series_key="policy",
                title=f"{workload}: best OPT+bypass vs baselines ({metric})",
                xlabel="page_size",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__best_vs_baselines__{metric}.png",
                category_order=[str(page) for page in sorted({row["page_size"] for row in workload_best})],
                series_order=["lru", "opt", "best_opt_bypass"],
            )


if __name__ == "__main__":
    main()
