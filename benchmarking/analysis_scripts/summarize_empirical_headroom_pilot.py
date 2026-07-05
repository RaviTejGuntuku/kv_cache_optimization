#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact pilot coverage table and summary.")
    parser.add_argument("--pilot-name", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


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

    oracle_rows = load_jsonl(oracle_root / "oracle0_measurements.jsonl")
    marginal_rows = load_jsonl(marginal_root / "marginal_counterfactuals.jsonl")
    accounting_rows = (
        load_jsonl(accounting_root / "missed_opportunities.jsonl")
        if (accounting_root / "missed_opportunities.jsonl").exists()
        else []
    )
    oracle_manifest = load_json(oracle_root / "run_manifest.json")
    manifest_params = oracle_manifest.get("parameters", oracle_manifest)
    system = manifest_params["system"]
    bundle_root = Path(manifest_params["bundle_root"])
    bundle_name = bundle_root.name
    objects = load_jsonl(bundle_root / "objects.jsonl")

    object_types_in_bundle = sorted({row["object_type"] for row in objects})
    object_types_observed = sorted({row["object_type"] for row in marginal_rows})
    true_missed_rows = [row for row in marginal_rows if row.get("was_missed_in_baseline")]
    positive_marginal_rows = [row for row in true_missed_rows if float(row["marginal_gain_ms"]) > 0.0]
    approximate_rows = [row for row in marginal_rows if row["object_type"] == "approximate"]
    repair_expected_rows = [row for row in marginal_rows if row.get("repair_expected")]
    oracle0_gaps = [
        float(row["baseline_prefill_or_wall_ms"]) - float(row["oracle0_prefill_or_wall_ms"])
        for row in accounting_rows
    ]
    ttft_gaps = [
        float(row["baseline_ttft_ms"]) - float(row["oracle_ttft_ms"])
        for row in accounting_rows
        if row.get("baseline_ttft_ms") is not None and row.get("oracle_ttft_ms") is not None
    ]
    marginal_true_miss_gains = [float(row["marginal_gain_ms"]) for row in true_missed_rows]

    coverage_row = {
        "pilot_name": args.pilot_name,
        "workload_name": bundle_name,
        "system": system,
        "request_count": len(oracle_rows),
        "object_types_in_bundle": ",".join(object_types_in_bundle),
        "object_types_observed": ",".join(object_types_observed),
        "true_missed_rows": len(true_missed_rows),
        "positive_marginal_rows": len(positive_marginal_rows),
        "approximate_rows": len(approximate_rows),
        "repair_expected_rows": len(repair_expected_rows),
        "mean_oracle0_gap_ms": round(mean(oracle0_gaps), 3),
        "mean_oracle0_ttft_gap_ms": round(mean(ttft_gaps), 3),
        "mean_marginal_gain_ms_over_true_misses": round(mean(marginal_true_miss_gains), 3),
    }
    write_csv(output_dir / "coverage_table.csv", coverage_row)

    missed_by_type = Counter(row["object_type"] for row in true_missed_rows)
    positive_by_type = Counter(row["object_type"] for row in positive_marginal_rows)

    lines = [
        f"# {args.pilot_name}",
        "",
        f"- workload: `{bundle_name}`",
        f"- system: `{system}`",
        f"- requests measured: `{len(oracle_rows)}`",
        f"- object types in bundle: `{', '.join(object_types_in_bundle)}`",
        f"- object types observed in marginal outputs: `{', '.join(object_types_observed)}`",
        f"- true missed rows: `{len(true_missed_rows)}`",
        f"- positive marginal rows over true misses: `{len(positive_marginal_rows)}`",
        f"- approximate rows: `{len(approximate_rows)}`",
        f"- repair-expected rows: `{len(repair_expected_rows)}`",
        f"- mean Oracle 0 gap (ms): `{coverage_row['mean_oracle0_gap_ms']}`",
        f"- mean Oracle 0 TTFT gap (ms): `{coverage_row['mean_oracle0_ttft_gap_ms']}`",
        f"- mean marginal gain over true misses (ms): `{coverage_row['mean_marginal_gain_ms_over_true_misses']}`",
        "",
        "## True Misses by Object Type",
    ]
    if missed_by_type:
        for object_type in sorted(missed_by_type):
            lines.append(f"- `{object_type}`: `{missed_by_type[object_type]}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Positive Marginal Rows by Object Type"])
    if positive_by_type:
        for object_type in sorted(positive_by_type):
            lines.append(f"- `{object_type}`: `{positive_by_type[object_type]}`")
    else:
        lines.append("- none")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
