#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heuristic missed-opportunity accounting over Oracle 0 + marginal-counterfactual outputs."
    )
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--oracle0-root", required=True)
    parser.add_argument("--counterfactual-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--gain-threshold-ms",
        type=float,
        default=2.0,
        help="Counterfactual gain at or below this threshold is treated as already realized or negligible.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_prefill_or_wall(row: dict[str, Any]) -> float:
    value = row.get("prefill_time_ms")
    if value is None:
        value = row["wall_time_ms"]
    return float(value)


def pick_ttft_or_wall(row: dict[str, Any]) -> float:
    value = row.get("ttft_ms")
    if value is None:
        value = row["wall_time_ms"]
    return float(value)


def cause_for_row(
    *,
    object_type: str,
    gain_ms: float,
    oracle0_gap_ms: float,
    gain_threshold_ms: float,
) -> str:
    if object_type == "approximate" and gain_ms <= gain_threshold_ms:
        return "approx_not_worth_it"
    if gain_ms <= gain_threshold_ms:
        return "already_realized_or_low_value"
    if oracle0_gap_ms <= gain_threshold_ms:
        return "no_request_level_gap"
    if object_type == "prefix_exact":
        return "prefix_realization_gap"
    if object_type == "nonprefix_exact":
        return "nonprefix_realization_gap"
    if object_type == "approximate":
        return "approximate_realization_gap"
    return "unclassified_gap"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root)
    oracle0_root = Path(args.oracle0_root)
    counterfactual_root = Path(args.counterfactual_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_manifest = load_json(bundle_root / "manifest.json")
    requests = load_jsonl(bundle_root / "requests.jsonl")
    objects = load_jsonl(bundle_root / "objects.jsonl")
    oracle0_rows = load_jsonl(oracle0_root / "oracle0_measurements.jsonl")
    baseline_rows = load_jsonl(counterfactual_root / "baseline_replay_measurements.jsonl")
    counterfactual_rows = load_jsonl(counterfactual_root / "marginal_counterfactuals.jsonl")

    request_index = {row["request_id"]: row for row in requests}
    object_index = {row["object_id"]: row for row in objects}
    oracle0_index = {row["request_id"]: row for row in oracle0_rows}
    baseline_index = {row["request_id"]: row for row in baseline_rows}

    missed_rows: list[dict[str, Any]] = []
    request_rollups: list[dict[str, Any]] = []

    counterfactuals_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_rows:
        counterfactuals_by_request[row["request_id"]].append(row)

    for request_id, baseline_row in baseline_index.items():
        oracle0_row = oracle0_index.get(request_id)
        if oracle0_row is None:
            continue
        baseline_ms = pick_prefill_or_wall(baseline_row)
        oracle0_ms = pick_prefill_or_wall(oracle0_row)
        baseline_ttft_ms = pick_ttft_or_wall(baseline_row)
        oracle0_ttft_ms = pick_ttft_or_wall(oracle0_row)
        oracle0_gap_ms = baseline_ms - oracle0_ms
        analyzed_rows = counterfactuals_by_request.get(request_id, [])
        positive_gain_sum_ms = sum(max(0.0, float(row["marginal_gain_ms"])) for row in analyzed_rows)
        request_obj_count = len(request_index[request_id]["reusable_object_ids"])
        request_rollups.append(
            {
                "request_id": request_id,
                "track": request_index[request_id]["track"],
                "bundle_name": bundle_manifest["bundle_name"],
                "baseline_prefill_or_wall_ms": baseline_ms,
                "oracle0_prefill_or_wall_ms": oracle0_ms,
                "oracle0_gap_ms": oracle0_gap_ms,
                "baseline_ttft_ms": baseline_ttft_ms,
                "oracle_ttft_ms": oracle0_ttft_ms,
                "oracle0_ttft_gap_ms": baseline_ttft_ms - oracle0_ttft_ms,
                "baseline_num_cached_tokens": baseline_row.get("num_cached_tokens"),
                "oracle0_num_cached_tokens": oracle0_row.get("num_cached_tokens"),
                "reusable_object_count": request_obj_count,
                "counterfactual_objects_analyzed": len(analyzed_rows),
                "analysis_coverage_frac": (
                    float(len(analyzed_rows)) / float(request_obj_count) if request_obj_count else 0.0
                ),
                "positive_marginal_gain_sum_ms": positive_gain_sum_ms,
                "max_marginal_gain_ms": max((float(row["marginal_gain_ms"]) for row in analyzed_rows), default=0.0),
            }
        )

    for row in counterfactual_rows:
        request_id = row["request_id"]
        object_id = row["object_id"]
        baseline_row = baseline_index[request_id]
        oracle0_row = oracle0_index.get(request_id)
        if oracle0_row is None:
            continue
        request_row = request_index[request_id]
        object_row = object_index[object_id]
        baseline_ms = pick_prefill_or_wall(baseline_row)
        oracle0_ms = pick_prefill_or_wall(oracle0_row)
        baseline_ttft_ms = pick_ttft_or_wall(baseline_row)
        oracle0_ttft_ms = pick_ttft_or_wall(oracle0_row)
        oracle0_gap_ms = baseline_ms - oracle0_ms
        gain_ms = float(row["marginal_gain_ms"])
        cause = cause_for_row(
            object_type=row["object_type"],
            gain_ms=gain_ms,
            oracle0_gap_ms=oracle0_gap_ms,
            gain_threshold_ms=args.gain_threshold_ms,
        )
        missed_rows.append(
            {
                "bundle_name": bundle_manifest["bundle_name"],
                "request_id": request_id,
                "track": row["track"],
                "object_id": object_id,
                "object_type": row["object_type"],
                "source_tier": row["source_tier"],
                "object_position": row["object_position"],
                "request_reusable_object_count": row["request_reusable_object_count"],
                "analysis_coverage_frac": (
                    float(len(counterfactuals_by_request[request_id])) / float(row["request_reusable_object_count"])
                    if row["request_reusable_object_count"]
                    else 0.0
                ),
                "baseline_prefill_or_wall_ms": baseline_ms,
                "oracle0_prefill_or_wall_ms": oracle0_ms,
                "oracle0_gap_ms": oracle0_gap_ms,
                "baseline_ttft_ms": baseline_ttft_ms,
                "oracle_ttft_ms": oracle0_ttft_ms,
                "oracle0_ttft_gap_ms": baseline_ttft_ms - oracle0_ttft_ms,
                "baseline_num_cached_tokens": row.get("baseline_num_cached_tokens"),
                "counterfactual_num_cached_tokens": row.get("counterfactual_num_cached_tokens"),
                "cached_token_gain": row.get("cached_token_gain"),
                "was_missed_in_baseline": row.get("was_missed_in_baseline", False),
                "marginal_gain_ms": gain_ms,
                "marginal_prefill_gain_ms": row.get("marginal_prefill_gain_ms", gain_ms),
                "marginal_ttft_gain_ms": row.get("marginal_ttft_gain_ms"),
                "gain_frac_of_oracle0_gap": (
                    gain_ms / oracle0_gap_ms if oracle0_gap_ms > 0.0 else None
                ),
                "cause": cause,
                "request_metadata": request_row.get("metadata", {}),
                "object_metadata": object_row.get("metadata", {}),
                "counterfactual_metadata": row.get("metadata", {}),
            }
        )

    cause_value = Counter()
    cause_count = Counter()
    object_type_value = Counter()
    for row in missed_rows:
        value = max(0.0, float(row["marginal_gain_ms"]))
        cause_value[row["cause"]] += value
        cause_count[row["cause"]] += 1
        object_type_value[(row["cause"], row["object_type"])] += value

    cause_rows = [
        {
            "cause": cause,
            "missed_object_count": cause_count[cause],
            "total_marginal_gain_ms": cause_value[cause],
        }
        for cause in sorted(cause_count)
    ]
    object_type_rows = [
        {
            "cause": cause,
            "object_type": object_type,
            "total_marginal_gain_ms": total_value,
        }
        for (cause, object_type), total_value in sorted(object_type_value.items())
    ]

    summary = {
        "bundle_name": bundle_manifest["bundle_name"],
        "track": bundle_manifest["track"],
        "requests_with_baseline": len(baseline_index),
        "requests_with_oracle0": len(oracle0_index),
        "counterfactual_rows": len(counterfactual_rows),
        "missed_counterfactual_rows": sum(1 for row in missed_rows if row.get("was_missed_in_baseline")),
        "missed_rows_written": len(missed_rows),
        "gain_threshold_ms": args.gain_threshold_ms,
        "total_positive_marginal_gain_ms": sum(max(0.0, float(row["marginal_gain_ms"])) for row in missed_rows),
        "total_oracle0_gap_ms": sum(max(0.0, float(row["oracle0_gap_ms"])) for row in request_rollups),
    }

    write_jsonl(output_root / "missed_opportunities.jsonl", missed_rows)
    write_jsonl(output_root / "request_gap_summary.jsonl", request_rollups)
    write_csv(
        output_root / "cause_breakdown.csv",
        cause_rows,
        ["cause", "missed_object_count", "total_marginal_gain_ms"],
    )
    write_csv(
        output_root / "cause_by_object_type.csv",
        object_type_rows,
        ["cause", "object_type", "total_marginal_gain_ms"],
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
