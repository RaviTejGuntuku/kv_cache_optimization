#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot empirical headroom pilot outputs.")
    parser.add_argument("--pilot-root", required=True, help="Pilot root containing Oracle 0, marginal, and optional accounting outputs.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def newest(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for {pattern} under {root}")
    return matches[-1]


def newest_any(root: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    matches = sorted(set(matches))
    if not matches:
        joined = ", ".join(patterns)
        raise FileNotFoundError(f"No matches for {joined} under {root}")
    return matches[-1]


def load_bundle_indices(bundle_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    request_rows = load_jsonl(bundle_root / "requests.jsonl")
    object_rows = load_jsonl(bundle_root / "objects.jsonl")
    request_index = {row["request_id"]: row for row in request_rows}
    object_index = {row["object_id"]: row for row in object_rows}
    return request_index, object_index


def object_token_size(object_row: dict) -> int:
    metadata = object_row.get("metadata") or {}
    if metadata.get("object_size_tokens") is not None:
        return int(metadata["object_size_tokens"])
    if object_row.get("seed_prompt_tokens") is not None:
        return int(object_row["seed_prompt_tokens"])
    return 0


def reusable_token_budget(request_row: dict, object_index: dict[str, dict]) -> int:
    return sum(object_token_size(object_index[obj_id]) for obj_id in request_row["reusable_object_ids"] if obj_id in object_index)


def runtime_prompt_tokens(row: dict, request_index: dict[str, dict]) -> int:
    value = row.get("prompt_tokens")
    if value is not None:
        return int(value)
    metadata = row.get("metadata") or {}
    runtime_value = metadata.get("runtime_prompt_tokens")
    if runtime_value is not None:
        return int(runtime_value)
    return int(request_index[row["request_id"]]["prompt_tokens"])


def runtime_reusable_token_budget(
    row: dict,
    request_index: dict[str, dict],
    object_index: dict[str, dict],
) -> int:
    metadata = row.get("metadata") or {}
    execution_plan = metadata.get("execution_plan") or {}
    occurrences = execution_plan.get("ordered_occurrences") or []
    if occurrences:
        by_object: dict[str, int] = {}
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            object_id = occurrence.get("object_id")
            if object_id is None:
                continue
            token_length = max(
                0,
                int(occurrence.get("token_end", 0)) - int(occurrence.get("token_start", 0)),
            )
            by_object[object_id] = max(by_object.get(object_id, 0), token_length)
        if by_object:
            return sum(by_object.values())
    return reusable_token_budget(request_index[row["request_id"]], object_index)


def pick_prefill_or_wall(row: dict) -> float:
    return float(row["prefill_time_ms"] or row["wall_time_ms"])


def pick_ttft_or_wall(row: dict) -> float:
    return float(row["ttft_ms"] or row["wall_time_ms"])


def short_request_label(request_id: str) -> str:
    parts = request_id.split("_")
    if len(parts) >= 4 and parts[0] == "prefix" and parts[1] == "req":
        return f"g{parts[2]}-b{parts[3]}"
    if len(parts) >= 3 and parts[0] == "broad" and parts[1] == "req":
        return f"r{parts[2]}"
    return request_id


def bar_oracle_vs_baseline(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    system_name: str,
) -> None:
    request_ids = [row["request_id"] for row in oracle_rows]
    oracle = [float(row["prefill_time_ms"] or row["wall_time_ms"]) for row in oracle_rows]
    baseline_index = {row["request_id"]: row for row in baseline_rows}
    baseline = [float(baseline_index[rid]["prefill_time_ms"] or baseline_index[rid]["wall_time_ms"]) for rid in request_ids]

    x = range(len(request_ids))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], baseline, width=0.4, label="Baseline replay", color="#c44e52")
    plt.bar([i + 0.2 for i in x], oracle, width=0.4, label="Oracle 0", color="#4c72b0")
    plt.xticks(list(x), [short_request_label(rid) for rid in request_ids], rotation=30, ha="right")
    plt.xlabel("Request (group/branch or request index)")
    plt.ylabel("Measured prefill latency (ms)")
    plt.title(f"Baseline vs Oracle 0\nWorkload: {bundle_name} | System: {system_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_metric_vs_baseline(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    system_name: str,
    metric_name: str,
    ylabel: str,
) -> None:
    request_ids = [row["request_id"] for row in oracle_rows]
    baseline_index = {row["request_id"]: row for row in baseline_rows}
    if metric_name == "ttft":
        oracle = [pick_ttft_or_wall(row) for row in oracle_rows]
        baseline = [pick_ttft_or_wall(baseline_index[rid]) for rid in request_ids]
        title_metric = "TTFT"
    else:
        oracle = [pick_prefill_or_wall(row) for row in oracle_rows]
        baseline = [pick_prefill_or_wall(baseline_index[rid]) for rid in request_ids]
        title_metric = "Prefill"

    x = range(len(request_ids))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], baseline, width=0.4, label="Baseline replay", color="#c44e52")
    plt.bar([i + 0.2 for i in x], oracle, width=0.4, label="Oracle 0", color="#4c72b0")
    plt.xticks(list(x), [short_request_label(rid) for rid in request_ids], rotation=30, ha="right")
    plt.xlabel("Request (group/branch or request index)")
    plt.ylabel(ylabel)
    plt.title(f"Baseline vs Oracle 0 {title_metric}\nWorkload: {bundle_name} | System: {system_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_cached_vs_prefill(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, rows, color in [
        ("Baseline replay", baseline_rows, "#c44e52"),
        ("Oracle 0", oracle_rows, "#4c72b0"),
    ]:
        xs = [float(row.get("num_cached_tokens") or 0) for row in rows]
        ys = [float(row["prefill_time_ms"] or row["wall_time_ms"]) for row in rows]
        plt.scatter(xs, ys, label=label, s=70, alpha=0.85, color=color)
    plt.xlabel("Cached tokens reported by serving backend")
    plt.ylabel("Measured prefill latency (ms)")
    plt.title(f"Cached Tokens vs Prefill Latency\nWorkload: {bundle_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_prefill_vs_sequence_length(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    request_index: dict[str, dict],
    object_index: dict[str, dict],
    out_path: Path,
    *,
    bundle_name: str,
    x_mode: str,
) -> None:
    plt.figure(figsize=(8, 5))
    if x_mode == "reusable_tokens":
        xlabel = "Reusable-token budget for request"
        title_suffix = "Reusable Tokens"
        x_getter = lambda row: runtime_reusable_token_budget(row, request_index, object_index)
    else:
        xlabel = "Total prompt tokens"
        title_suffix = "Prompt Length"
        x_getter = lambda row: runtime_prompt_tokens(row, request_index)

    for label, rows, color in [
        ("Baseline replay", baseline_rows, "#c44e52"),
        ("Oracle 0", oracle_rows, "#4c72b0"),
    ]:
        xs = [x_getter(row) for row in rows]
        ys = [pick_prefill_or_wall(row) for row in rows]
        plt.scatter(xs, ys, label=label, s=45, alpha=0.75, color=color)

    plt.xlabel(xlabel)
    plt.ylabel("Measured prefill latency (ms)")
    plt.title(f"Prefill Latency vs {title_suffix}\nWorkload: {bundle_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def binned_gap_plot(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    request_index: dict[str, dict],
    object_index: dict[str, dict],
    out_path: Path,
    *,
    bundle_name: str,
    x_mode: str,
) -> None:
    oracle_index = {row["request_id"]: row for row in oracle_rows}
    baseline_index = {row["request_id"]: row for row in baseline_rows}
    shared_ids = [rid for rid in oracle_index if rid in baseline_index]
    if x_mode == "reusable_tokens":
        xlabel = "Reusable-token budget for request"
        title_suffix = "Reusable Tokens"
        x_getter = lambda row: runtime_reusable_token_budget(row, request_index, object_index)
    else:
        xlabel = "Total prompt tokens"
        title_suffix = "Prompt Length"
        x_getter = lambda row: runtime_prompt_tokens(row, request_index)

    points = []
    for rid in shared_ids:
        x = x_getter(baseline_index[rid])
        gap = pick_prefill_or_wall(baseline_index[rid]) - pick_prefill_or_wall(oracle_index[rid])
        points.append((x, gap))
    points.sort(key=lambda item: item[0])
    if not points:
        return

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=35, alpha=0.4, color="#7a7a7a", label="Request-level gap")

    num_bins = min(8, max(3, int(math.sqrt(len(points)))))
    min_x = min(xs)
    max_x = max(xs)
    if min_x == max_x:
        centers = [min_x]
        means = [sum(ys) / len(ys)]
    else:
        bin_width = (max_x - min_x) / num_bins
        buckets: list[list[tuple[int, float]]] = [[] for _ in range(num_bins)]
        for x, y in points:
            idx = min(num_bins - 1, int((x - min_x) / bin_width))
            buckets[idx].append((x, y))
        centers = []
        means = []
        for bucket in buckets:
            if not bucket:
                continue
            centers.append(sum(x for x, _ in bucket) / len(bucket))
            means.append(sum(y for _, y in bucket) / len(bucket))
    plt.plot(centers, means, color="#2f6db3", linewidth=2.5, marker="o", label="Binned mean gap")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Baseline minus Oracle 0 prefill (ms)")
    plt.title(f"Prefill Gap vs {title_suffix}\nWorkload: {bundle_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_oracle_baseline_table(
    oracle_rows: list[dict],
    baseline_rows: list[dict],
    request_index: dict[str, dict],
    object_index: dict[str, dict],
    out_path: Path,
) -> None:
    baseline_index = {row["request_id"]: row for row in baseline_rows}
    fieldnames = [
        "request_id",
        "prompt_tokens",
        "reusable_tokens",
        "baseline_prefill_ms",
        "oracle0_prefill_ms",
        "prefill_gap_ms",
        "baseline_ttft_ms",
        "oracle0_ttft_ms",
        "ttft_gap_ms",
        "baseline_num_cached_tokens",
        "oracle0_num_cached_tokens",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for oracle_row in oracle_rows:
            request_id = oracle_row["request_id"]
            baseline_row = baseline_index[request_id]
            writer.writerow(
                {
                    "request_id": request_id,
                    "prompt_tokens": runtime_prompt_tokens(oracle_row, request_index),
                    "reusable_tokens": runtime_reusable_token_budget(
                        oracle_row,
                        request_index,
                        object_index,
                    ),
                    "baseline_prefill_ms": pick_prefill_or_wall(baseline_row),
                    "oracle0_prefill_ms": pick_prefill_or_wall(oracle_row),
                    "prefill_gap_ms": pick_prefill_or_wall(baseline_row) - pick_prefill_or_wall(oracle_row),
                    "baseline_ttft_ms": pick_ttft_or_wall(baseline_row),
                    "oracle0_ttft_ms": pick_ttft_or_wall(oracle_row),
                    "ttft_gap_ms": pick_ttft_or_wall(baseline_row) - pick_ttft_or_wall(oracle_row),
                    "baseline_num_cached_tokens": baseline_row.get("num_cached_tokens"),
                    "oracle0_num_cached_tokens": oracle_row.get("num_cached_tokens"),
                }
            )


def bar_marginal_gains(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    missed_only: bool,
) -> None:
    plotted_rows = [
        row for row in counterfactual_rows
        if (not missed_only) or row.get("was_missed_in_baseline")
    ]
    if not plotted_rows:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No rows matched this filter.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    labels = [
        f'{short_request_label(row["request_id"])} | {row["object_type"]}'
        for row in plotted_rows
    ]
    gains = [float(row["marginal_gain_ms"]) for row in plotted_rows]
    colors = ["#55a868" if gain >= 0 else "#8172b3" for gain in gains]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), gains, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.xlabel("Counterfactual case (request | object type)")
    plt.ylabel("Latency saved if block were on-time (ms)")
    qualifier = "missed blocks only" if missed_only else "all candidate blocks"
    plt.title(f"Marginal Counterfactual Gains ({qualifier})\nWorkload: {bundle_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_cached_token_gain(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
) -> None:
    plotted_rows = [row for row in counterfactual_rows if row.get("was_missed_in_baseline")]
    if not plotted_rows:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No missed baseline blocks were detected.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    labels = [short_request_label(row["request_id"]) for row in plotted_rows]
    deltas = [float(row.get("cached_token_gain") or 0.0) for row in plotted_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), deltas, color="#64b5cd")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.xlabel("Request with a missed reusable block")
    plt.ylabel("Additional cached tokens in counterfactual")
    plt.title(f"Counterfactual Cached-Token Gain\nWorkload: {bundle_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_marginal_gain_by_object_type(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    missed_only: bool,
) -> None:
    plotted_rows = [
        row for row in counterfactual_rows
        if (not missed_only) or row.get("was_missed_in_baseline")
    ]
    grouped: dict[str, list[float]] = {}
    for row in plotted_rows:
        grouped.setdefault(row["object_type"], []).append(float(row["marginal_gain_ms"]))
    if not grouped:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No rows matched this filter.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    counts = [len(grouped[label]) for label in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), values, color="#4c9f70")
    plt.xticks(range(len(labels)), [f"{label}\n(n={count})" for label, count in zip(labels, counts)], rotation=0)
    plt.ylabel("Mean marginal gain (ms)")
    qualifier = "missed blocks only" if missed_only else "all candidate blocks"
    plt.title(f"Marginal Gain by Object Type ({qualifier})\nWorkload: {bundle_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_marginal_gain_by_group(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    group_field: str,
    missed_only: bool,
    aggregate: str,
) -> None:
    plotted_rows = [
        row for row in counterfactual_rows
        if (not missed_only) or row.get("was_missed_in_baseline")
    ]
    grouped: dict[str, list[float]] = {}
    for row in plotted_rows:
        grouped.setdefault(str(row.get(group_field, "unknown")), []).append(float(row["marginal_gain_ms"]))
    if not grouped:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No rows matched this filter.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    labels = sorted(grouped)
    if aggregate == "mean":
        values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
        ylabel = "Mean marginal gain (ms)"
        title_agg = "Mean"
    else:
        values = [sum(grouped[label]) for label in labels]
        ylabel = "Total marginal gain (ms)"
        title_agg = "Total"
    counts = [len(grouped[label]) for label in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), values, color="#4c9f70" if aggregate == "mean" else "#dd8452")
    plt.xticks(range(len(labels)), [f"{label}\n(n={count})" for label, count in zip(labels, counts)], rotation=0)
    plt.ylabel(ylabel)
    qualifier = "missed blocks only" if missed_only else "all candidate blocks"
    group_name = group_field.replace("_", " ")
    plt.title(f"{title_agg} Marginal Gain by {group_name.title()} ({qualifier})\nWorkload: {bundle_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_marginal_gain_cdf(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    missed_only: bool,
) -> None:
    plotted_rows = [
        row for row in counterfactual_rows
        if (not missed_only) or row.get("was_missed_in_baseline")
    ]
    gains = sorted(float(row["marginal_gain_ms"]) for row in plotted_rows)
    if not gains:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No rows matched this filter.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    ys = [(idx + 1) / len(gains) for idx in range(len(gains))]
    plt.figure(figsize=(8, 5))
    plt.plot(gains, ys, color="#2f6db3", linewidth=2)
    plt.xlabel("Marginal gain (ms)")
    plt.ylabel("CDF")
    qualifier = "missed blocks only" if missed_only else "all candidate blocks"
    plt.title(f"CDF of Marginal Gain ({qualifier})\nWorkload: {bundle_name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk_cumulative_gain(
    counterfactual_rows: list[dict],
    out_path: Path,
    *,
    bundle_name: str,
    missed_only: bool,
) -> None:
    plotted_rows = [
        row for row in counterfactual_rows
        if (not missed_only) or row.get("was_missed_in_baseline")
    ]
    gains = sorted((max(0.0, float(row["marginal_gain_ms"])) for row in plotted_rows), reverse=True)
    if not gains:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No rows matched this filter.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    xs = list(range(1, len(gains) + 1))
    running = []
    total = 0.0
    for gain in gains:
        total += gain
        running.append(total)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, running, color="#d17c2f", linewidth=2)
    plt.xlabel("Top-k missed blocks")
    plt.ylabel("Cumulative latency saved (ms)")
    qualifier = "missed blocks only" if missed_only else "all candidate blocks"
    plt.title(f"Cumulative Top-k Marginal Value ({qualifier})\nWorkload: {bundle_name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_cause_breakdown(cause_rows: list[dict], out_path: Path) -> None:
    labels = [row["cause"] for row in cause_rows]
    values = [float(row["total_marginal_gain_ms"]) for row in cause_rows]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(labels)), values, color="#dd8452")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Total marginal gain (ms)")
    plt.title("Missed-Opportunity Gain by Cause")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_markdown_summary(
    *,
    output_path: Path,
    oracle_summary: dict,
    marginal_summary: dict,
    accounting_summary: dict | None,
) -> None:
    lines = [
        "# Empirical Headroom Pilot Summary",
        "",
        "## Oracle 0",
        f'- requests measured: `{oracle_summary.get("requests_measured")}`',
        f'- mean prefill time (ms): `{oracle_summary.get("prefill_time_ms_mean")}`',
        f'- mean cached tokens: `{oracle_summary.get("num_cached_tokens_mean")}`',
        "",
        "## Marginal Counterfactuals",
        f'- counterfactuals measured: `{marginal_summary.get("counterfactuals_measured")}`',
        f'- counterfactuals that were actual baseline misses: `{marginal_summary.get("missed_counterfactuals")}`',
        f'- mean marginal gain (ms): `{marginal_summary.get("marginal_gain_ms_mean")}`',
        f'- mean marginal gain over actual misses (ms): `{marginal_summary.get("missed_marginal_gain_ms_mean")}`',
        f'- max marginal gain (ms): `{marginal_summary.get("marginal_gain_ms_max")}`',
    ]
    if accounting_summary:
        lines.extend(
            [
                "",
                "## Accounting",
                f'- total Oracle 0 gap (ms): `{accounting_summary.get("total_oracle0_gap_ms")}`',
                f'- total positive marginal gain (ms): `{accounting_summary.get("total_positive_marginal_gain_ms")}`',
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cause_rows(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_root = newest_any(
        pilot_root,
        ["oracle0__*", "oracle0_fcfs__*", "raw/oracle0__*", "raw/oracle0_fcfs__*"],
    )
    marginal_root = newest_any(
        pilot_root,
        [
            "marginal_counterfactuals__*",
            "marginal_counterfactuals_fcfs__*",
            "raw/marginal_counterfactuals__*",
            "raw/marginal_counterfactuals_fcfs__*",
        ],
    )
    accounting_root = pilot_root / "analysis" / "missed_opportunity_accounting"
    if not accounting_root.exists():
        accounting_root = pilot_root / "missed_opportunity_accounting"
    if not accounting_root.exists():
        accounting_root = pilot_root / "accounting_manual"

    oracle_rows = load_jsonl(oracle_root / "oracle0_measurements.jsonl")
    baseline_rows = load_jsonl(marginal_root / "baseline_replay_measurements.jsonl")
    marginal_rows = load_jsonl(marginal_root / "marginal_counterfactuals.jsonl")

    oracle_summary = load_json(oracle_root / "summary.json")
    marginal_summary = load_json(marginal_root / "summary.json")
    accounting_summary = load_json(accounting_root / "summary.json") if (accounting_root / "summary.json").exists() else None
    oracle_manifest = load_json(oracle_root / "run_manifest.json")

    if "parameters" in oracle_manifest:
        bundle_root = oracle_manifest["parameters"]["bundle_root"]
        system_name = oracle_manifest["parameters"]["system"]
    else:
        bundle_root = oracle_manifest["bundle_root"]
        system_name = oracle_manifest["system"]
    bundle_name = bundle_root.split("/")[-1]
    bundle_path = Path(bundle_root)
    request_index, object_index = load_bundle_indices(bundle_path)

    bar_oracle_vs_baseline(
        oracle_rows,
        baseline_rows,
        output_dir / "oracle_vs_baseline_by_request.png",
        bundle_name=bundle_name,
        system_name=system_name,
    )
    bar_metric_vs_baseline(
        oracle_rows,
        baseline_rows,
        output_dir / "ttft_oracle_vs_baseline_by_request.png",
        bundle_name=bundle_name,
        system_name=system_name,
        metric_name="ttft",
        ylabel="Measured TTFT (ms)",
    )
    write_oracle_baseline_table(
        oracle_rows,
        baseline_rows,
        request_index,
        object_index,
        output_dir / "oracle0_vs_baseline_per_request.csv",
    )
    scatter_cached_vs_prefill(
        oracle_rows,
        baseline_rows,
        output_dir / "cached_tokens_vs_prefill.png",
        bundle_name=bundle_name,
    )
    scatter_prefill_vs_sequence_length(
        oracle_rows,
        baseline_rows,
        request_index,
        object_index,
        output_dir / "prefill_vs_reusable_tokens.png",
        bundle_name=bundle_name,
        x_mode="reusable_tokens",
    )
    scatter_prefill_vs_sequence_length(
        oracle_rows,
        baseline_rows,
        request_index,
        object_index,
        output_dir / "prefill_vs_prompt_tokens.png",
        bundle_name=bundle_name,
        x_mode="prompt_tokens",
    )
    binned_gap_plot(
        oracle_rows,
        baseline_rows,
        request_index,
        object_index,
        output_dir / "prefill_gap_vs_reusable_tokens.png",
        bundle_name=bundle_name,
        x_mode="reusable_tokens",
    )
    binned_gap_plot(
        oracle_rows,
        baseline_rows,
        request_index,
        object_index,
        output_dir / "prefill_gap_vs_prompt_tokens.png",
        bundle_name=bundle_name,
        x_mode="prompt_tokens",
    )
    bar_marginal_gains(
        marginal_rows,
        output_dir / "marginal_gains_missed_only.png",
        bundle_name=bundle_name,
        missed_only=True,
    )
    bar_marginal_gains(
        marginal_rows,
        output_dir / "marginal_gains_all_candidates.png",
        bundle_name=bundle_name,
        missed_only=False,
    )
    bar_cached_token_gain(
        marginal_rows,
        output_dir / "counterfactual_cached_token_gain.png",
        bundle_name=bundle_name,
    )
    bar_marginal_gain_by_object_type(
        marginal_rows,
        output_dir / "marginal_gain_by_object_type.png",
        bundle_name=bundle_name,
        missed_only=True,
    )
    bar_marginal_gain_by_object_type(
        marginal_rows,
        output_dir / "marginal_gain_by_object_type_all_candidates.png",
        bundle_name=bundle_name,
        missed_only=False,
    )
    bar_marginal_gain_by_group(
        marginal_rows,
        output_dir / "total_marginal_gain_by_object_type.png",
        bundle_name=bundle_name,
        group_field="object_type",
        missed_only=True,
        aggregate="total",
    )
    bar_marginal_gain_by_group(
        marginal_rows,
        output_dir / "total_marginal_gain_by_source_tier.png",
        bundle_name=bundle_name,
        group_field="source_tier",
        missed_only=True,
        aggregate="total",
    )
    bar_marginal_gain_by_group(
        marginal_rows,
        output_dir / "mean_marginal_gain_by_source_tier.png",
        bundle_name=bundle_name,
        group_field="source_tier",
        missed_only=True,
        aggregate="mean",
    )
    plot_marginal_gain_cdf(
        marginal_rows,
        output_dir / "marginal_gain_cdf.png",
        bundle_name=bundle_name,
        missed_only=True,
    )
    plot_topk_cumulative_gain(
        marginal_rows,
        output_dir / "topk_cumulative_gain.png",
        bundle_name=bundle_name,
        missed_only=True,
    )

    if (accounting_root / "cause_breakdown.csv").exists():
        cause_rows = load_cause_rows(accounting_root / "cause_breakdown.csv")
        bar_cause_breakdown(cause_rows, output_dir / "cause_breakdown.png")

    write_markdown_summary(
        output_path=output_dir / "summary.md",
        oracle_summary=oracle_summary,
        marginal_summary=marginal_summary,
        accounting_summary=accounting_summary,
    )


if __name__ == "__main__":
    main()
