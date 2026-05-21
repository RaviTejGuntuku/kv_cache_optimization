#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from headroom_plot_common import ensure_dir, load_json, plot_lines, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recomputation microbenchmark results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append(
                {
                    **row,
                    "k_blocks": int(row["k_blocks"]),
                    "batch_size": int(row["batch_size"]),
                    "wall_mean_ms": float(row["wall_mean_ms"]),
                    "wall_trimmed_mean_ms": float(row["wall_trimmed_mean_ms"]),
                    "wall_p50_ms": float(row["wall_p50_ms"]),
                    "wall_p95_ms": float(row["wall_p95_ms"]),
                    "wall_std_ms": float(row["wall_std_ms"]),
                    "cuda_mean_ms": float(row["cuda_mean_ms"]),
                    "cuda_trimmed_mean_ms": float(row["cuda_trimmed_mean_ms"]),
                    "cuda_p50_ms": float(row["cuda_p50_ms"]),
                    "cuda_p95_ms": float(row["cuda_p95_ms"]),
                    "cuda_std_ms": float(row["cuda_std_ms"]),
                }
            )
        return rows


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    metrics_dir = experiment_root / "metrics"
    graphs_dir = experiment_root / "graphs"
    ensure_dir(graphs_dir)

    rows = load_rows(metrics_dir / "recovery_times.csv")
    crossovers = load_json(metrics_dir / "crossover_points.json")
    write_json(metrics_dir / "aggregated_metrics.json", {"rows": rows, "crossovers": crossovers})

    for metric, ylabel, suffix in (
        ("wall_trimmed_mean_ms", "Wall-Clock Recovery Trimmed Mean (ms)", "wall_trimmed_mean"),
        ("wall_p50_ms", "Wall-Clock Recovery Latency P50 (ms)", "wall_p50"),
        ("cuda_trimmed_mean_ms", "CUDA-Event Recovery Trimmed Mean (ms)", "cuda_trimmed_mean"),
        ("cuda_p50_ms", "CUDA-Event Recovery Latency P50 (ms)", "cuda_p50"),
        ("wall_p95_ms", "Wall-Clock Recovery P95 (ms)", "wall_p95"),
        ("cuda_p95_ms", "CUDA-Event Recovery P95 (ms)", "cuda_p95"),
    ):
        plot_lines(
            rows=rows,
            x_key="k_blocks",
            y_key=metric,
            series_key="method",
            title=f"{metric} vs missing blocks",
            xlabel="missing_blocks_k",
            ylabel=ylabel,
            output_path=graphs_dir / f"{suffix}_vs_k.png",
            series_order=["HBM_reuse", "DRAM_fetch", "SSD_fetch", "recompute"],
        )


if __name__ == "__main__":
    main()
