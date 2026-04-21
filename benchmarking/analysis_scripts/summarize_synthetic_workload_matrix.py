#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a synthetic workload experiment root and plot LRU vs Belady under "
            "both FCFS and prefix-coverage scheduling."
        )
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
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


def gather_rows(experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_path in sorted(experiment_root.rglob("reports/comparison.json")):
        run_root = comparison_path.parent.parent
        metadata_path = run_root / "run_metadata.json"
        pressure_path = run_root / "reports" / "memory_pressure.json"
        if not metadata_path.exists() or not pressure_path.exists():
            continue

        comparison = load_json(comparison_path)
        metadata = load_json(metadata_path)
        pressure = load_json(pressure_path)

        args = metadata.get("args", {})
        dataset_path = Path(args["dataset_path"])
        workload_name = "__".join(list(dataset_path.parts[:-1])[-2:] + [dataset_path.stem]).replace("_", "-")
        max_concurrency = int(args["max_concurrency"])

        matched_pressure = None
        for item in pressure.get("pressure_by_concurrency", []):
            if int(item["concurrency"]) == max_concurrency:
                matched_pressure = item
                break

        rows.append(
            {
                "workload_name": workload_name,
                "dataset_path": str(dataset_path),
                "schedule_policy": args["schedule_policy"],
                "max_concurrency": max_concurrency,
                "request_rate": args["request_rate"],
                "num_prompts": args["num_prompts"],
                "memory_pressure_full_request": coerce_float(
                    matched_pressure.get("full_request_pressure") if matched_pressure else None
                ),
                "memory_pressure_prompt_only": coerce_float(
                    matched_pressure.get("prompt_only_pressure") if matched_pressure else None
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
                "block_hit_rate_lru": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "hbm_hit_rate_lru")
                ),
                "block_hit_rate_belady": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "hbm_hit_rate_belady")
                ),
                "block_miss_rate_lru": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "hbm_miss_rate_lru")
                ),
                "block_miss_rate_belady": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "hbm_miss_rate_belady")
                ),
                "server_hit_rate_lru": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "lru_cache_hit_rate_server")
                ),
                "server_hit_rate_belady": coerce_float(
                    maybe_nested_get(comparison, "cache_metrics", "belady_cache_hit_rate_server")
                ),
                "transfer_proxy_bytes_lru": coerce_float(
                    maybe_nested_get(comparison, "transfer_proxy_bytes", "lru")
                ),
                "transfer_proxy_bytes_belady": coerce_float(
                    maybe_nested_get(comparison, "transfer_proxy_bytes", "belady")
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    metric_specs = [
        ("throughput", "Output Throughput", "tokens/sec"),
        ("median_ttft_ms", "Median TTFT", "ms"),
        ("p99_ttft_ms", "P99 TTFT", "ms"),
        ("median_itl_ms", "Median ITL", "ms"),
        ("block_hit_rate", "Block Hit Rate", "fraction"),
        ("block_miss_rate", "Block Miss Rate", "fraction"),
    ]
    policy_colors = {
        "fcfs": "#1f77b4",
        "prefix-coverage": "#d62728",
    }
    policy_labels = {
        "fcfs": "FCFS",
        "prefix-coverage": "Prefix-Coverage",
    }
    line_styles = {
        "lru": "-",
        "belady": "--",
    }

    workloads = sorted({row["workload_name"] for row in rows})
    for workload_name in workloads:
        workload_rows = [row for row in rows if row["workload_name"] == workload_name]
        workload_dir = output_dir / workload_name
        workload_dir.mkdir(parents=True, exist_ok=True)

        for metric_key, title, ylabel in metric_specs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for schedule_policy in sorted({row["schedule_policy"] for row in workload_rows}):
                policy_rows = sorted(
                    [row for row in workload_rows if row["schedule_policy"] == schedule_policy],
                    key=lambda row: row["max_concurrency"],
                )
                x = [row["max_concurrency"] for row in policy_rows]
                ax.plot(
                    x,
                    [row[f"{metric_key}_lru"] for row in policy_rows],
                    marker="o",
                    linestyle=line_styles["lru"],
                    color=policy_colors.get(schedule_policy, "#444444"),
                    label=f"{policy_labels.get(schedule_policy, schedule_policy)} LRU",
                )
                ax.plot(
                    x,
                    [row[f"{metric_key}_belady"] for row in policy_rows],
                    marker="o",
                    linestyle=line_styles["belady"],
                    color=policy_colors.get(schedule_policy, "#444444"),
                    label=f"{policy_labels.get(schedule_policy, schedule_policy)} Belady",
                )

            ax.set_title(f"{title}: {workload_name}")
            ax.set_xlabel("max concurrency")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(workload_dir / f"{metric_key}_vs_max_concurrency.png", dpi=180)
            plt.close(fig)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    summary: dict[str, Any] = {"num_rows": len(rows), "workloads": {}}
    for workload_name in sorted({row["workload_name"] for row in rows}):
        workload_rows = [row for row in rows if row["workload_name"] == workload_name]
        summary["workloads"][workload_name] = {
            "num_runs": len(workload_rows),
            "best_throughput_gain_pct": max(
                (
                    (row["throughput_belady"] - row["throughput_lru"]) / row["throughput_lru"]
                    for row in workload_rows
                    if row.get("throughput_lru") not in (None, 0)
                    and row.get("throughput_belady") is not None
                ),
                default=None,
            ),
            "best_block_miss_reduction": max(
                (
                    row["block_miss_rate_lru"] - row["block_miss_rate_belady"]
                    for row in workload_rows
                    if row.get("block_miss_rate_lru") is not None
                    and row.get("block_miss_rate_belady") is not None
                ),
                default=None,
            ),
        }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = gather_rows(experiment_root)
    if not rows:
        raise ValueError(f"No completed runs found under {experiment_root}")

    rows = sorted(rows, key=lambda row: (row["workload_name"], row["schedule_policy"], row["max_concurrency"]))
    write_csv(rows, output_dir / "aggregated_metrics.csv")
    (output_dir / "aggregated_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary(rows, output_dir / "summary.json")
    make_plots(rows, output_dir)


if __name__ == "__main__":
    main()
