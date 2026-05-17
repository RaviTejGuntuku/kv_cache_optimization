#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SliceSpec:
    experiment: str
    output_name: str
    description: str
    target_size: int
    min_prompt_len: int
    min_output_len: int


SPECS: tuple[SliceSpec, ...] = (
    SliceSpec(
        experiment="effective_residency_sweep",
        output_name="effective_residency_sweep__realworld_sequence",
        description=(
            "Natural request sequence slice for effective residency studies. Prefers long prompts "
            "with non-trivial outputs while preserving observed request order."
        ),
        target_size=512,
        min_prompt_len=2048,
        min_output_len=128,
    ),
    SliceSpec(
        experiment="critical_path_miss_attribution",
        output_name="critical_path_miss_attribution__realworld_sequence",
        description=(
            "Natural request sequence slice for critical-path miss attribution. Prefers long prompts "
            "with non-trivial outputs while preserving observed request order."
        ),
        target_size=384,
        min_prompt_len=1536,
        min_output_len=192,
    ),
    SliceSpec(
        experiment="recomputation_microbenchmark",
        output_name="recomputation_microbenchmark__realworld_sequence",
        description=(
            "Natural request sequence slice for recomputation-cost measurement. Prefers long prompts "
            "while preserving observed request order."
        ),
        target_size=256,
        min_prompt_len=2048,
        min_output_len=64,
    ),
    SliceSpec(
        experiment="opt_with_incoming_line",
        output_name="opt_with_incoming_line__realworld_sequence",
        description=(
            "Natural request sequence slice for admission-control studies. Prefers medium-to-long prompts "
            "while preserving observed request order."
        ),
        target_size=512,
        min_prompt_len=1024,
        min_output_len=128,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build near-real workload slices for the headroom studies from a naturally occurring "
            "request sequence such as LMSYS-Chat-1M or ShareGPT."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path in the repo's custom conversations/prompt_len/output_len format.",
    )
    parser.add_argument(
        "--dataset-name",
        default="sharegpt_subset",
        help="Human-readable dataset/source name recorded in the output manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/headroom_studies",
        help="Directory where the real-world slices are written.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_row(row: dict[str, Any], spec: SliceSpec) -> float:
    prompt_len = int(row.get("prompt_len", 0))
    output_len = int(row.get("output_len", 0))
    score = 0.0
    if prompt_len >= spec.min_prompt_len:
        score += 4.0
    if output_len >= spec.min_output_len:
        score += 2.0
    score += min(prompt_len / max(1, spec.min_prompt_len), 3.0)
    score += min(output_len / max(1, spec.min_output_len), 2.0)
    return score


def choose_window(rows: list[dict[str, Any]], spec: SliceSpec) -> tuple[int, int]:
    if len(rows) <= spec.target_size:
        return 0, len(rows)

    best_start = 0
    best_score = float("-inf")
    window = spec.target_size

    # Sliding-score window over the natural request sequence.
    current = sum(score_row(row, spec) for row in rows[:window])
    best_score = current
    for start in range(1, len(rows) - window + 1):
        current += score_row(rows[start + window - 1], spec)
        current -= score_row(rows[start - 1], spec)
        if current > best_score:
            best_score = current
            best_start = start
    return best_start, best_start + window


def enrich_row(row: dict[str, Any], *, dataset_name: str, experiment: str, original_index: int) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    metadata = copied.setdefault("metadata", {})
    metadata["source_dataset"] = dataset_name
    metadata["source_experiment"] = experiment
    metadata["source_sequence_index"] = original_index
    return copied


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    top_index: dict[str, dict[str, str]] = {}

    for spec in SPECS:
        experiment_dir = output_dir / spec.experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        start, end = choose_window(rows, spec)
        selected = [
            enrich_row(
                row,
                dataset_name=args.dataset_name,
                experiment=spec.experiment,
                original_index=start + offset,
            )
            for offset, row in enumerate(rows[start:end])
        ]

        out_path = experiment_dir / f"{spec.output_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in selected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest = {
            "name": spec.output_name,
            "experiment": spec.experiment,
            "description": spec.description,
            "source_dataset": args.dataset_name,
            "source_path": str(input_path),
            "preserved_sequence_window": {"start_index": start, "end_index_exclusive": end},
            "selection_policy": {
                "target_size": spec.target_size,
                "min_prompt_len": spec.min_prompt_len,
                "min_output_len": spec.min_output_len,
            },
            "selected_requests": len(selected),
        }
        manifest_path = experiment_dir / f"{spec.output_name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        top_index[spec.experiment] = {
            "path": str(out_path),
            "manifest": str(manifest_path),
        }

    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "source_dataset": args.dataset_name,
                "source_path": str(input_path),
                "experiments": top_index,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
