#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark reports and plot LRU vs Belady curves."
    )
    parser.add_argument(
        "--sweep-manifest",
        default=None,
        help="Path to sweep_manifest.json from run_benchmark_sweep.py.",
    )
    parser.add_argument(
        "--run-roots",
        nargs="*",
        default=None,
        help="Optional list of run roots, each containing reports/comparison.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for aggregated CSV/JSON and PNG plots.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["memory_pressure", "max_concurrency", "request_rate"],
        default="memory_pressure",
        help="Primary x-axis for the generated plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def infer_run_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    if args.sweep_manifest:
        manifest = load_json(Path(args.sweep_manifest))
        for item in manifest:
            output_root = Path(item["output_root"])
            entries.append(
                {
                    "run_name": item.get("run_name", output_root.name),
                    "schedule_policy": item.get("schedule_policy"),
                    "request_rate": item.get("request_rate"),
                    "max_concurrency": item.get("max_concurrency"),
                    "output_root": output_root,
                }
            )

    if args.run_roots:
        for run_root in args.run_roots:
            output_root = Path(run_root)
            entries.append(
                {
                    "run_name": output_root.name,
                    "request_rate": None,
                    "max_concurrency": None,
                    "output_root": output_root,
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        deduped[str(entry["output_root"])] = entry
    return list(deduped.values())


def build_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for entry in entries:
        output_root = Path(entry["output_root"])
        comparison = load_json(output_root / "reports" / "comparison.json")
        pressure = load_json(output_root / "reports" / "memory_pressure.json")
        run_metadata_path = output_root / "run_metadata.json"
        run_metadata = load_json(run_metadata_path) if run_metadata_path.exists() else {}

        max_concurrency = entry.get("max_concurrency")
        if max_concurrency is None:
            max_concurrency = maybe_nested_get(
                comparison, "inputs", "max_concurrency"
            )
        pressure_points = pressure.get("pressure_by_concurrency", [])
        matched_pressure = next(
            (
                item
                for item in pressure_points
                if max_concurrency is not None
                and int(item.get("concurrency", -1)) == int(max_concurrency)
            ),
            None,
        )
        if matched_pressure is None and pressure_points:
            matched_pressure = pressure_points[-1]

        row = {
            "run_name": entry["run_name"],
            "schedule_policy": entry.get("schedule_policy")
            or maybe_nested_get(run_metadata, "args", "schedule_policy"),
            "request_rate": entry.get("request_rate"),
            "max_concurrency": max_concurrency,
            "memory_pressure_prompt_only": coerce_float(
                matched_pressure.get("prompt_only_pressure") if matched_pressure else None
            ),
            "memory_pressure_full_request": coerce_float(
                matched_pressure.get("full_request_pressure") if matched_pressure else None
            ),
            "throughput_lru": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "output_throughput", "lru")
            ),
            "throughput_belady": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "output_throughput", "belady")
            ),
            "median_ttft_ms_lru": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "median_ttft_ms", "lru")
            ),
            "median_ttft_ms_belady": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "median_ttft_ms", "belady")
            ),
            "p99_ttft_ms_lru": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "p99_ttft_ms", "lru")
            ),
            "p99_ttft_ms_belady": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "p99_ttft_ms", "belady")
            ),
            "median_itl_ms_lru": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "median_itl_ms", "lru")
            ),
            "median_itl_ms_belady": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "median_itl_ms", "belady")
            ),
            "p99_itl_ms_lru": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "p99_itl_ms", "lru")
            ),
            "p99_itl_ms_belady": coerce_float(
                maybe_nested_get(comparison, "serving_metrics", "p99_itl_ms", "belady")
            ),
            "cache_hit_rate_lru_server": coerce_float(
                maybe_nested_get(comparison, "cache_metrics", "lru_cache_hit_rate_server")
            ),
            "cache_hit_rate_belady_server": coerce_float(
                maybe_nested_get(comparison, "cache_metrics", "belady_cache_hit_rate_server")
            ),
            "block_hit_rate_lru": coerce_float(
                first_non_none(
                    maybe_nested_get(
                        comparison, "cache_metrics", "hbm_hit_rate_lru"
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "lru_trace_summary",
                        "page_cache_simulation",
                        "lru_hit_rate",
                    ),
                )
            ),
            "block_hit_rate_belady": coerce_float(
                first_non_none(
                    maybe_nested_get(
                        comparison, "cache_metrics", "hbm_hit_rate_belady"
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "belady_trace_summary",
                        "page_cache_simulation",
                        "belady_hit_rate",
                    ),
                )
            ),
            "block_miss_rate_lru": coerce_float(
                first_non_none(
                    maybe_nested_get(
                        comparison, "cache_metrics", "hbm_miss_rate_lru"
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "lru_trace_summary",
                        "page_cache_simulation",
                        "lru_miss_rate",
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "lru_trace_summary",
                        "match_summary",
                        "block_miss_rate",
                    ),
                )
            ),
            "block_miss_rate_belady": coerce_float(
                first_non_none(
                    maybe_nested_get(
                        comparison, "cache_metrics", "hbm_miss_rate_belady"
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "belady_trace_summary",
                        "page_cache_simulation",
                        "belady_miss_rate",
                    ),
                    maybe_nested_get(
                        comparison,
                        "cache_metrics",
                        "belady_trace_summary",
                        "match_summary",
                        "block_miss_rate",
                    ),
                )
            ),
            "transfer_proxy_bytes_lru": coerce_float(
                maybe_nested_get(comparison, "transfer_proxy_bytes", "lru")
            ),
            "transfer_proxy_bytes_belady": coerce_float(
                maybe_nested_get(comparison, "transfer_proxy_bytes", "belady")
            ),
            "frontier_diff_rate_lru_trace": coerce_float(
                maybe_nested_get(
                    comparison,
                    "cache_metrics",
                    "lru_trace_summary",
                    "frontier_belady_diff_rate",
                )
            ),
            "selection_fallback_rate_belady_trace": _fallback_rate(
                maybe_nested_get(comparison, "cache_metrics", "belady_trace_summary")
            ),
        }
        rows.append(row)

    return rows


def _fallback_rate(belady_summary: Any) -> float | None:
    if not isinstance(belady_summary, dict):
        return None
    event_counts = belady_summary.get("event_counts") or {}
    belady_frontiers = event_counts.get("belady_frontier")
    if not belady_frontiers:
        return None
    return None


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows found to write.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, Any]], x_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get(x_key) is None else float(row[x_key]),
            row.get("run_name", ""),
        ),
    )


def make_plots(rows: list[dict[str, Any]], output_dir: Path, x_key: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    sorted_rows = sort_rows(rows, x_key)
    x_values = [row.get(x_key) for row in sorted_rows]
    x_labels = [row["run_name"] for row in sorted_rows]

    metric_specs = [
        ("throughput", "Output Throughput", "tokens/sec"),
        ("median_ttft_ms", "Median TTFT", "ms"),
        ("p99_ttft_ms", "P99 TTFT", "ms"),
        ("median_itl_ms", "Median ITL", "ms"),
        ("p99_itl_ms", "P99 ITL", "ms"),
        ("block_hit_rate", "Block Hit Rate", "fraction"),
        ("block_miss_rate", "Block Miss Rate", "fraction"),
        ("transfer_proxy_bytes", "Transfer Proxy Bytes", "bytes"),
    ]

    for metric_key, title, ylabel in metric_specs:
        y_lru = [row.get(f"{metric_key}_lru") for row in sorted_rows]
        y_belady = [row.get(f"{metric_key}_belady") for row in sorted_rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        if all(value is not None for value in x_values):
            ax.plot(x_values, y_lru, marker="o", label="LRU")
            ax.plot(x_values, y_belady, marker="o", label="Belady")
            ax.set_xlabel(x_key.replace("_", " "))
        else:
            positions = list(range(len(sorted_rows)))
            ax.plot(positions, y_lru, marker="o", label="LRU")
            ax.plot(positions, y_belady, marker="o", label="Belady")
            ax.set_xticks(positions)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_xlabel("run")

        ax.set_title(f"{title}: LRU vs Belady")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}_vs_{x_key}.png", dpi=180)
        plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    summary = {
        "num_runs": len(rows),
        "best_throughput_gain_tokens_per_s": max(
            (
                row["throughput_belady"] - row["throughput_lru"]
                for row in rows
                if row.get("throughput_lru") is not None
                and row.get("throughput_belady") is not None
            ),
            default=None,
        ),
        "best_ttft_reduction_ms": max(
            (
                row["median_ttft_ms_lru"] - row["median_ttft_ms_belady"]
                for row in rows
                if row.get("median_ttft_ms_lru") is not None
                and row.get("median_ttft_ms_belady") is not None
            ),
            default=None,
        ),
        "best_block_miss_rate_reduction": max(
            (
                row["block_miss_rate_lru"] - row["block_miss_rate_belady"]
                for row in rows
                if row.get("block_miss_rate_lru") is not None
                and row.get("block_miss_rate_belady") is not None
            ),
            default=None,
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.sweep_manifest and not args.run_roots:
        raise ValueError("Provide either --sweep-manifest or --run-roots.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = infer_run_entries(args)
    rows = build_rows(entries)
    if not rows:
        raise ValueError("No completed benchmark runs were found.")

    x_key = {
        "memory_pressure": "memory_pressure_full_request",
        "max_concurrency": "max_concurrency",
        "request_rate": "request_rate",
    }[args.x_axis]

    write_csv(rows, output_dir / "aggregated_metrics.csv")
    (output_dir / "aggregated_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary(rows, output_dir / "summary.json")
    make_plots(rows, output_dir, x_key=x_key)


if __name__ == "__main__":
    main()
