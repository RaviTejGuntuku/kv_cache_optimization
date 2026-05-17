#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best cache fraction n from a set of OPT+bypass runs."
    )
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric",
        default="output_throughput",
        choices=["output_throughput", "request_throughput", "median_ttft_ms", "median_itl_ms"],
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(report: dict[str, Any], metric: str) -> float:
    block = report["serving_metrics"][metric]
    secondary = block.get("secondary")
    if secondary is None:
        raise ValueError(f"Missing secondary metric {metric} in {report}")
    return float(secondary)


def main() -> None:
    args = parse_args()
    results = []
    for run_root_str in args.run_roots:
        run_root = Path(run_root_str)
        report = load_report(run_root / "reports" / "comparison.json")
        results.append(
            {
                "run_root": str(run_root),
                "metric": metric_value(report, args.metric),
            }
        )
    reverse = args.metric in {"output_throughput", "request_throughput"}
    best = sorted(results, key=lambda item: item["metric"], reverse=reverse)[0]
    Path(args.output).write_text(
        json.dumps({"metric": args.metric, "best": best, "all_results": results}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
