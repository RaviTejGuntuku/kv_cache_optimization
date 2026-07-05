#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether Oracle 0 measurements are safe to interpret."
    )
    parser.add_argument(
        "--case-root",
        required=True,
        help="Case directory containing raw/baseline_replay_fcfs and raw/oracle0_fcfs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--negative-gap-tolerance-ms",
        type=float,
        default=5.0,
        help="Small noise tolerance before negative Oracle gaps are flagged.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reusable_tokens(row: dict[str, Any]) -> int:
    plan = ((row.get("metadata") or {}).get("execution_plan") or {})
    by_object: dict[str, int] = {}
    for occ in plan.get("ordered_occurrences") or []:
        object_id = occ.get("object_id")
        if not object_id:
            continue
        length = max(0, int(occ.get("token_end", 0)) - int(occ.get("token_start", 0)))
        by_object[object_id] = max(by_object.get(object_id, 0), length)
    return sum(by_object.values())


def metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    return float(value)


def cohort_key(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = row.get("metadata") or {}
    cohort = metadata.get("cohort_request_ids") or [row.get("request_id")]
    return tuple(str(item) for item in cohort)


def distinct_values(values: list[float | None], ndigits: int = 3) -> set[float | None]:
    return {None if value is None else round(float(value), ndigits) for value in values}


def detect_shared_batch_timings(rows: list[dict[str, Any]], *, raw: bool = False) -> dict[str, Any]:
    cohorts: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cohorts[cohort_key(row)].append(row)

    repeated_prefill = 0
    repeated_ttft = 0
    multi_request_cohorts = 0
    examples: list[dict[str, Any]] = []
    for cohort, cohort_rows in cohorts.items():
        if len(cohort_rows) <= 1:
            continue
        multi_request_cohorts += 1
        prefill_field = "prefill_time_ms"
        prefill_values = [metric(row, prefill_field) for row in cohort_rows]
        if raw:
            raw_values = [
                metric(row.get("metadata") or {}, "raw_prefill_time_ms")
                for row in cohort_rows
            ]
            if any(value is not None for value in raw_values):
                prefill_values = raw_values
        ttft_values = [metric(row, "ttft_ms") for row in cohort_rows]
        prefill_shared = len(distinct_values(prefill_values)) == 1
        ttft_shared = len(distinct_values(ttft_values)) == 1
        repeated_prefill += int(prefill_shared)
        repeated_ttft += int(ttft_shared)
        if len(examples) < 3 and (prefill_shared or ttft_shared):
            examples.append(
                {
                    "cohort_request_ids": list(cohort),
                    "prefill_values_ms": prefill_values,
                    "ttft_values_ms": ttft_values,
                }
            )

    return {
        "multi_request_cohorts": multi_request_cohorts,
        "cohorts_with_shared_prefill": repeated_prefill,
        "cohorts_with_shared_ttft": repeated_ttft,
        "has_batch_shared_prefill": multi_request_cohorts > 0
        and repeated_prefill == multi_request_cohorts,
        "has_batch_shared_ttft": multi_request_cohorts > 0
        and repeated_ttft == multi_request_cohorts,
        "examples": examples,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_jsonl(
        case_root / "raw" / "baseline_replay_fcfs" / "baseline_replay_measurements.jsonl"
    )
    oracle_rows = load_jsonl(
        case_root / "raw" / "oracle0_fcfs" / "oracle0_measurements.jsonl"
    )
    baseline_by_id = {str(row["request_id"]): row for row in baseline_rows}
    oracle_by_id = {str(row["request_id"]): row for row in oracle_rows}
    request_ids = sorted(set(baseline_by_id) & set(oracle_by_id))

    pair_rows: list[dict[str, Any]] = []
    for request_id in request_ids:
        baseline = baseline_by_id[request_id]
        oracle = oracle_by_id[request_id]
        baseline_prefill = metric(baseline, "prefill_time_ms")
        oracle_prefill = metric(oracle, "prefill_time_ms")
        baseline_ttft = metric(baseline, "ttft_ms")
        oracle_ttft = metric(oracle, "ttft_ms")
        pair_rows.append(
            {
                "request_id": request_id,
                "cohort_request_ids": "|".join(cohort_key(oracle)),
                "prompt_tokens": int(oracle.get("prompt_tokens") or 0),
                "reusable_tokens": reusable_tokens(oracle),
                "baseline_prefill_ms": baseline_prefill,
                "oracle_prefill_ms": oracle_prefill,
                "prefill_gap_ms": (
                    baseline_prefill - oracle_prefill
                    if baseline_prefill is not None and oracle_prefill is not None
                    else None
                ),
                "baseline_ttft_ms": baseline_ttft,
                "oracle_ttft_ms": oracle_ttft,
                "ttft_gap_ms": (
                    baseline_ttft - oracle_ttft
                    if baseline_ttft is not None and oracle_ttft is not None
                    else None
                ),
                "baseline_cached_tokens": baseline.get("num_cached_tokens"),
                "oracle_cached_tokens": oracle.get("num_cached_tokens"),
                "oracle_raw_prefill_ms": (oracle.get("metadata") or {}).get(
                    "raw_prefill_time_ms"
                ),
                "oracle_excluded_hbm_materialization_ms": (
                    oracle.get("metadata") or {}
                ).get("excluded_hbm_materialization_ms"),
                "oracle_repair_compute_ms": (oracle.get("metadata") or {}).get(
                    "oracle_repair_compute_ms"
                ),
                "oracle_load_mode": (oracle.get("metadata") or {}).get(
                    "oracle_load_mode"
                ),
            }
        )

    system = None
    concurrency = 1
    if oracle_rows:
        metadata = oracle_rows[0].get("metadata") or {}
        system = metadata.get("system") or oracle_rows[0].get("system")
        concurrency = int(metadata.get("concurrency") or 1)

    negative_prefill = [
        row for row in pair_rows
        if row["prefill_gap_ms"] is not None
        and float(row["prefill_gap_ms"]) < -args.negative_gap_tolerance_ms
    ]
    negative_ttft = [
        row for row in pair_rows
        if row["ttft_gap_ms"] is not None
        and float(row["ttft_gap_ms"]) < -args.negative_gap_tolerance_ms
    ]
    async_flags = [
        (row.get("metadata") or {}).get("lmcache_async_loading_enabled")
        for row in baseline_rows + oracle_rows
        if str(row.get("system", "")).startswith("lmcache_")
    ]
    oracle_fetch_flags = [
        (row.get("metadata") or {}).get("oracle0_fetch_exclusion_applied")
        for row in oracle_rows
        if str(row.get("system", "")).startswith("lmcache_")
    ]
    materialization_bad = [
        row for row in pair_rows
        if row["oracle_raw_prefill_ms"] is not None
        and row["oracle_excluded_hbm_materialization_ms"] is not None
        and float(row["oracle_raw_prefill_ms"])
        < float(row["oracle_excluded_hbm_materialization_ms"])
    ]
    baseline_shared = detect_shared_batch_timings(baseline_rows)
    oracle_shared = detect_shared_batch_timings(oracle_rows, raw=True)
    lmcache_broad = str(system or "").startswith("lmcache_")
    lmcache_oracle_no_materialization = [
        row for row in pair_rows
        if lmcache_broad
        and int(row["oracle_cached_tokens"] or 0) <= 0
        and float(row["oracle_excluded_hbm_materialization_ms"] or 0.0) <= 0.0
    ]
    unsafe_per_request_concurrency = (
        lmcache_broad
        and concurrency > 1
        and (
            baseline_shared["has_batch_shared_prefill"]
            or oracle_shared["has_batch_shared_prefill"]
        )
    )

    validity_errors: list[str] = []
    validity_warnings: list[str] = []
    if len(request_ids) != len(baseline_rows) or len(request_ids) != len(oracle_rows):
        validity_errors.append("baseline/oracle request IDs do not pair exactly")
    if lmcache_broad and async_flags:
        if any(flag is False for flag in async_flags):
            validity_errors.append("LMCache async loading was disabled on some rows")
        if any(flag is None for flag in async_flags):
            validity_warnings.append(
                "LMCache async-loading flag is missing on some rows; old runs need log/config confirmation"
            )
    if lmcache_broad and oracle_fetch_flags and not all(flag is True for flag in oracle_fetch_flags):
        validity_errors.append("LMCache Oracle rows did not all apply fetch exclusion")
    if materialization_bad:
        validity_errors.append("Oracle materialization exceeded raw prefill on some rows")
    if lmcache_oracle_no_materialization:
        validity_errors.append(
            "LMCache Oracle rows did not materialize cached KV; CacheBlend/retrieve path was not exercised"
        )
    if unsafe_per_request_concurrency:
        validity_errors.append(
            "LMCache broad run has batch-shared timings under concurrency > 1; "
            "per-request Oracle gaps are not safe to interpret"
        )
    if negative_prefill:
        validity_warnings.append(
            f"{len(negative_prefill)} rows have prefill gaps below "
            f"-{args.negative_gap_tolerance_ms} ms"
        )
    if negative_ttft:
        validity_warnings.append(
            f"{len(negative_ttft)} rows have TTFT gaps below "
            f"-{args.negative_gap_tolerance_ms} ms"
        )

    report = {
        "case_root": str(case_root),
        "system": system,
        "concurrency": concurrency,
        "paired_request_count": len(request_ids),
        "valid_for_request_level_headroom": not validity_errors,
        "validity_errors": validity_errors,
        "validity_warnings": validity_warnings,
        "baseline_batch_timing_audit": baseline_shared,
        "oracle_batch_timing_audit": oracle_shared,
        "negative_prefill_gap_rows": len(negative_prefill),
        "negative_ttft_gap_rows": len(negative_ttft),
        "mean_prefill_gap_ms": (
            fmean(float(row["prefill_gap_ms"]) for row in pair_rows if row["prefill_gap_ms"] is not None)
            if pair_rows
            else None
        ),
        "mean_ttft_gap_ms": (
            fmean(float(row["ttft_gap_ms"]) for row in pair_rows if row["ttft_gap_ms"] is not None)
            if pair_rows
            else None
        ),
        "mean_prompt_tokens": (
            fmean(int(row["prompt_tokens"]) for row in pair_rows) if pair_rows else None
        ),
        "mean_reusable_tokens": (
            fmean(int(row["reusable_tokens"]) for row in pair_rows) if pair_rows else None
        ),
        "mean_baseline_cached_tokens": (
            fmean(
                int(row["baseline_cached_tokens"])
                for row in pair_rows
                if row["baseline_cached_tokens"] is not None
            )
            if any(row["baseline_cached_tokens"] is not None for row in pair_rows)
            else None
        ),
        "mean_oracle_cached_tokens": (
            fmean(
                int(row["oracle_cached_tokens"])
                for row in pair_rows
                if row["oracle_cached_tokens"] is not None
            )
            if any(row["oracle_cached_tokens"] is not None for row in pair_rows)
            else None
        ),
    }

    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(pair_rows, output_dir / "audit_pairs.csv")
    status = "VALID" if report["valid_for_request_level_headroom"] else "INVALID"
    lines = [
        f"# Oracle 0 Measurement Audit: {status}",
        "",
        f"- case_root: `{case_root}`",
        f"- system: `{system}`",
        f"- concurrency: `{concurrency}`",
        f"- paired requests: `{len(request_ids)}`",
        f"- mean prefill gap ms: `{report['mean_prefill_gap_ms']}`",
        f"- mean TTFT gap ms: `{report['mean_ttft_gap_ms']}`",
        "",
        "## Errors",
        *(f"- {item}" for item in validity_errors),
        *(["- none"] if not validity_errors else []),
        "",
        "## Warnings",
        *(f"- {item}" for item in validity_warnings),
        *(["- none"] if not validity_warnings else []),
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if validity_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
