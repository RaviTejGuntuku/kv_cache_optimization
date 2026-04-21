#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from generate_recency_trap_workloads import (
    make_family_prompt,
    make_row,
    make_unique_prompt,
)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    description: str
    family: str
    hot_families: int
    cold_families: int
    rounds: int
    interference_span: int
    prefix_tokens: int
    suffix_tokens: int
    output_len: int


def build_tenant_rotation_gap(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    hot = [f"tenant_hot_{i:02d}" for i in range(spec.hot_families)]
    cold = [f"tenant_cold_{i:02d}" for i in range(spec.cold_families)]

    # Realistic pattern:
    # a small set of important tenants/users recur, but their next request is
    # separated by many other tenants' long prompts. That is realistic
    # multi-tenant traffic and is unfriendly to pure recency.
    for round_id in range(spec.rounds):
        for hot_idx, family in enumerate(hot):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        round_id,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=hot_idx * 31,
                        phase_label=f"rotation-{round_id}",
                    ),
                    family=family,
                    branch=round_id,
                    sequence_id=seq,
                    phase=f"rotation_{round_id}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

            for j in range(spec.interference_span):
                cold_idx = (round_id * spec.interference_span + hot_idx + j) % len(cold)
                cold_family = cold[cold_idx]
                cold_branch = round_id * spec.interference_span + j
                rows.append(
                    make_row(
                        make_family_prompt(
                            cold_family,
                            cold_branch,
                            prefix_tokens=spec.prefix_tokens,
                            suffix_tokens=spec.suffix_tokens,
                            offset=700 + cold_idx * 17,
                            phase_label=f"cold-{round_id}",
                        ),
                        family=cold_family,
                        branch=cold_branch,
                        sequence_id=seq,
                        phase=f"cold_{round_id}",
                        kind="shared",
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        output_len=spec.output_len,
                    )
                )
                seq += 1

    return rows


def build_periodic_refinement_gap(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    jobs = [f"agent_job_{i:02d}" for i in range(spec.hot_families)]

    # Realistic pattern:
    # iterative agent / analyst workflows revisit the same large context across
    # stages, but every stage is separated by tool output, retrieval, or side
    # analyses. This creates repeated delayed reuse instead of grouped reuse.
    for stage in range(spec.rounds):
        for job_idx, family in enumerate(jobs):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        stage,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=job_idx * 29,
                        phase_label=f"stage-{stage}",
                    ),
                    family=family,
                    branch=stage,
                    sequence_id=seq,
                    phase=f"stage_{stage}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

        for tool_idx in range(spec.interference_span * spec.hot_families):
            tag = f"tool_{stage:02d}_{tool_idx:03d}"
            rows.append(
                make_row(
                    make_unique_prompt(
                        tag,
                        spec.prefix_tokens + spec.suffix_tokens,
                        1200 + seq * 5,
                    ),
                    family=tag,
                    branch=0,
                    sequence_id=seq,
                    phase=f"tool_{stage}",
                    kind="unique",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.prefix_tokens + spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

    return rows


SPECS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        name="natural_tenant_rotation_gap",
        description=(
            "Recurring high-value tenants revisit large shared prompts after a wide "
            "rotation of other tenants. This mimics multi-tenant traffic where the "
            "next useful prefix is often not the most recently used one."
        ),
        family="tenant-rotation-gap",
        hot_families=8,
        cold_families=24,
        rounds=16,
        interference_span=4,
        prefix_tokens=4096,
        suffix_tokens=512,
        output_len=512,
    ),
    WorkloadSpec(
        name="natural_periodic_refinement_gap",
        description=(
            "Iterative refinement workflows revisit large shared contexts across "
            "stages, but each revisit is separated by many side analyses and tool "
            "outputs. This mimics agent or analyst loops with delayed reuse."
        ),
        family="periodic-refinement-gap",
        hot_families=12,
        cold_families=0,
        rounds=10,
        interference_span=5,
        prefix_tokens=4096,
        suffix_tokens=512,
        output_len=512,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate delayed-reuse FCFS workloads intended to expose LRU gaps."
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic/adversarial_fcfs",
        help="Directory where JSONL workloads and manifests are written.",
    )
    parser.add_argument(
        "--gpu-kv-capacity-blocks",
        type=int,
        default=16000,
        help="Used only for simple manifest pressure estimates.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Used only for manifest pressure estimates.",
    )
    parser.add_argument(
        "--candidate-concurrencies",
        type=int,
        nargs="+",
        default=[64, 96, 128, 160],
        help="Candidate concurrency points reported in manifests.",
    )
    return parser.parse_args()


def block_count(spec: WorkloadSpec, page_size: int) -> int:
    tokens = spec.prefix_tokens + spec.suffix_tokens + spec.output_len
    return math.ceil(tokens / page_size)


def estimated_pressure(
    spec: WorkloadSpec, concurrency: int, page_size: int, capacity_blocks: int
) -> float:
    return block_count(spec, page_size) * concurrency / capacity_blocks


def recommend_concurrency(
    spec: WorkloadSpec, candidates: list[int], page_size: int, capacity_blocks: int
) -> int:
    return min(
        candidates,
        key=lambda concurrency: (
            abs(estimated_pressure(spec, concurrency, page_size, capacity_blocks) - 2.0),
            -estimated_pressure(spec, concurrency, page_size, capacity_blocks),
        ),
    )


def build_rows(spec: WorkloadSpec) -> list[dict]:
    if spec.family == "tenant-rotation-gap":
        return build_tenant_rotation_gap(spec)
    if spec.family == "periodic-refinement-gap":
        return build_periodic_refinement_gap(spec)
    raise ValueError(f"Unknown workload family {spec.family}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for spec in SPECS:
        rows = build_rows(spec)
        jsonl_path = output_dir / f"{spec.name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        recommended_mc = recommend_concurrency(
            spec,
            args.candidate_concurrencies,
            args.page_size,
            args.gpu_kv_capacity_blocks,
        )
        manifest = {
            "name": spec.name,
            "description": spec.description,
            "family": spec.family,
            "selected_requests": len(rows),
            "page_size": args.page_size,
            "gpu_kv_capacity_blocks": args.gpu_kv_capacity_blocks,
            "per_request_full_blocks_estimate": block_count(spec, args.page_size),
            "recommended_concurrency": recommended_mc,
            "recommended_concurrency_pressure": estimated_pressure(
                spec, recommended_mc, args.page_size, args.gpu_kv_capacity_blocks
            ),
            "pressure_by_concurrency": [
                {
                    "concurrency": concurrency,
                    "estimated_full_request_pressure": estimated_pressure(
                        spec, concurrency, args.page_size, args.gpu_kv_capacity_blocks
                    ),
                }
                for concurrency in args.candidate_concurrencies
            ],
            "spec": asdict(spec),
        }
        manifest_path = output_dir / f"{spec.name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        index.append(
            {
                "name": spec.name,
                "path": str(jsonl_path),
                "manifest": str(manifest_path),
                "recommended_concurrency": recommended_mc,
                "recommended_concurrency_pressure": manifest["recommended_concurrency_pressure"],
            }
        )

    (output_dir / "index.json").write_text(
        json.dumps({"output_dir": str(output_dir), "workloads": index}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
