#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from headroom_plot_common import ensure_dir, load_json, plot_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot queue information visibility results.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    metrics_path = experiment_root / "metrics" / "visibility_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file at {metrics_path}")
    payload = load_json(metrics_path)
    rows = payload["rows"]
    graphs_dir = experiment_root / "graphs"
    ensure_dir(graphs_dir)

    for workload in sorted({row["workload"] for row in rows}):
        workload_rows = [row for row in rows if row["workload"] == workload]
        total_requests = max(int(row["queue_size_numeric"]) for row in workload_rows)
        note_text = (
            f"Full future for this workload = {total_requests} queued requests. "
            "x-axis shows the visible queue size q as a subset of that future."
        )
        for metric, ylabel, suffix in (
            ("reuse_event_visibility_fraction", "Reuse Event Visibility Fraction", "reuse_event_visibility"),
            ("reuse_block_visibility_fraction", "Reuse Block Visibility Fraction", "reuse_block_visibility"),
            ("next_reuse_visibility_fraction", "Next-Reuse Visibility Fraction", "next_reuse_visibility"),
        ):
            plot_lines(
                rows=workload_rows,
                x_key="queue_size_numeric",
                y_key=metric,
                series_key="page_size",
                title=f"{workload}: {metric} vs queue size",
                xlabel="visible_queue_size_q (requests)",
                ylabel=ylabel,
                output_path=graphs_dir / f"{workload}__{suffix}_vs_queue_size.png",
                series_order=[str(page) for page in sorted({row["page_size"] for row in workload_rows})],
                note_text=note_text,
            )
        plot_lines(
            rows=workload_rows,
            x_key="queue_size_numeric",
            y_key="reuse_block_visibility_fraction",
            series_key="page_size",
            title=f"{workload}: visibility saturation",
            xlabel="visible_queue_size_q (requests)",
            ylabel="Reuse Block Visibility Fraction",
            output_path=graphs_dir / f"{workload}__saturation_curve.png",
            series_order=[str(page) for page in sorted({row["page_size"] for row in workload_rows})],
            note_text=note_text,
        )


if __name__ == "__main__":
    main()
