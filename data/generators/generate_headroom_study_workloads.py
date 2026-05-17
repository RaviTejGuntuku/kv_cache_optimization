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
    experiment: str
    variant: str
    name: str
    description: str
    builder: str
    prefix_tokens: int
    suffix_tokens: int
    output_len: int
    hot_families: int
    cold_families: int
    rounds: int
    interference_span: int
    concurrency_hint: int
    page_sizes: tuple[int, ...]
    capacity_sweep_blocks: tuple[int, ...]


def _estimate_blocks(spec: WorkloadSpec, page_size: int) -> int:
    total_tokens = spec.prefix_tokens + spec.suffix_tokens + spec.output_len
    return math.ceil(total_tokens / page_size)


def _manifest(spec: WorkloadSpec, rows: list[dict]) -> dict:
    return {
        "name": spec.name,
        "experiment": spec.experiment,
        "variant": spec.variant,
        "description": spec.description,
        "recommended_max_concurrency": spec.concurrency_hint,
        "recommended_page_sizes": list(spec.page_sizes),
        "recommended_capacity_sweep_blocks": list(spec.capacity_sweep_blocks),
        "selected_requests": len(rows),
        "per_request_full_blocks_by_page_size": {
            str(page_size): _estimate_blocks(spec, page_size) for page_size in spec.page_sizes
        },
        "spec": asdict(spec),
    }


def build_residency_hotset_ladder(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    hot = [f"resident_hot_{idx:02d}" for idx in range(spec.hot_families)]
    cold = [f"resident_cold_{idx:02d}" for idx in range(spec.cold_families)]

    for round_id in range(spec.rounds):
        for hot_idx, family in enumerate(hot):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        round_id,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=hot_idx * 19,
                        phase_label=f"ladder-{round_id}",
                    ),
                    family=family,
                    branch=round_id,
                    sequence_id=seq,
                    phase=f"hot_round_{round_id}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

        for cold_idx in range(spec.interference_span):
            family = cold[(round_id * spec.interference_span + cold_idx) % len(cold)]
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        round_id,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=900 + cold_idx * 11,
                        phase_label=f"pressure-{round_id}",
                    ),
                    family=family,
                    branch=round_id,
                    sequence_id=seq,
                    phase=f"pressure_round_{round_id}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

    return rows


def build_residency_tenant_backlog(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    hot = [f"tenant_hot_{idx:02d}" for idx in range(spec.hot_families)]
    cold = [f"tenant_cold_{idx:02d}" for idx in range(spec.cold_families)]

    for round_id in range(spec.rounds):
        for hot_idx, family in enumerate(hot):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        round_id,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=hot_idx * 23,
                        phase_label=f"service-{round_id}",
                    ),
                    family=family,
                    branch=round_id,
                    sequence_id=seq,
                    phase=f"service_round_{round_id}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

            for gap_idx in range(spec.interference_span):
                cold_family = cold[(round_id * spec.interference_span + gap_idx + hot_idx) % len(cold)]
                rows.append(
                    make_row(
                        make_family_prompt(
                            cold_family,
                            round_id,
                            prefix_tokens=spec.prefix_tokens,
                            suffix_tokens=spec.suffix_tokens,
                            offset=1200 + gap_idx * 13,
                            phase_label=f"backlog-{round_id}",
                        ),
                        family=cold_family,
                        branch=round_id,
                        sequence_id=seq,
                        phase=f"backlog_round_{round_id}",
                        kind="shared",
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        output_len=spec.output_len,
                    )
                )
                seq += 1

    return rows


def build_critical_path_serial_resume(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    families = [f"resume_serial_{idx:02d}" for idx in range(spec.hot_families)]

    for stage in range(spec.rounds):
        for family_idx, family in enumerate(families):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        stage,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=family_idx * 17,
                        phase_label=f"resume-{stage}",
                    ),
                    family=family,
                    branch=stage,
                    sequence_id=seq,
                    phase=f"resume_{stage}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

        for unique_idx in range(spec.interference_span):
            tag = f"serial_gap_{stage:02d}_{unique_idx:03d}"
            rows.append(
                make_row(
                    make_unique_prompt(tag, spec.prefix_tokens + spec.suffix_tokens, 1800 + seq * 3),
                    family=tag,
                    branch=0,
                    sequence_id=seq,
                    phase=f"serial_gap_{stage}",
                    kind="unique",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.prefix_tokens + spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

    return rows


def build_critical_path_agent_resume(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    families = [f"agent_resume_{idx:02d}" for idx in range(spec.hot_families)]

    for stage in range(spec.rounds):
        for family_idx, family in enumerate(families):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        stage,
                        prefix_tokens=spec.prefix_tokens,
                        suffix_tokens=spec.suffix_tokens,
                        offset=family_idx * 29,
                        phase_label=f"agent-stage-{stage}",
                    ),
                    family=family,
                    branch=stage,
                    sequence_id=seq,
                    phase=f"agent_stage_{stage}",
                    kind="shared",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

        for tool_idx in range(spec.interference_span * spec.hot_families):
            tag = f"tool_resume_{stage:02d}_{tool_idx:03d}"
            rows.append(
                make_row(
                    make_unique_prompt(tag, spec.prefix_tokens + spec.suffix_tokens, 2500 + seq * 5),
                    family=tag,
                    branch=0,
                    sequence_id=seq,
                    phase=f"tool_stage_{stage}",
                    kind="unique",
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=spec.prefix_tokens + spec.suffix_tokens,
                    output_len=spec.output_len,
                )
            )
            seq += 1

    return rows


def build_recompute_ladder(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    block_ladder = (1, 2, 4, 8, 16)

    for blocks in block_ladder:
        family = f"recompute_k{blocks:02d}"
        suffix_tokens = spec.suffix_tokens + blocks * 32
        for sample_idx in range(spec.rounds):
            row = make_row(
                make_family_prompt(
                    family,
                    sample_idx,
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    offset=blocks * 41,
                    phase_label=f"k{blocks}",
                ),
                family=family,
                branch=sample_idx,
                sequence_id=seq,
                phase=f"k_{blocks}",
                kind="shared",
                prefix_tokens=spec.prefix_tokens,
                suffix_tokens=suffix_tokens,
                output_len=spec.output_len,
            )
            row["metadata"]["target_recompute_blocks"] = blocks
            row["metadata"]["microbench_family"] = "recompute_ladder"
            rows.append(row)
            seq += 1

    return rows


def build_recompute_resume_mix(spec: WorkloadSpec) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    families = [f"resume_mix_{idx:02d}" for idx in range(spec.hot_families)]
    ladder = (1, 2, 4, 8)

    for family_idx, family in enumerate(families):
        for blocks in ladder:
            suffix_tokens = spec.suffix_tokens + blocks * 24
            row = make_row(
                make_family_prompt(
                    family,
                    blocks,
                    prefix_tokens=spec.prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    offset=family_idx * 37,
                    phase_label=f"resume-k{blocks}",
                ),
                family=family,
                branch=blocks,
                sequence_id=seq,
                phase=f"resume_k_{blocks}",
                kind="shared",
                prefix_tokens=spec.prefix_tokens,
                suffix_tokens=suffix_tokens,
                output_len=spec.output_len,
            )
            row["metadata"]["target_recompute_blocks"] = blocks
            row["metadata"]["microbench_family"] = "resume_mix"
            rows.append(row)
            seq += 1

    return rows


BUILDERS = {
    "residency_hotset_ladder": build_residency_hotset_ladder,
    "residency_tenant_backlog": build_residency_tenant_backlog,
    "critical_path_serial_resume": build_critical_path_serial_resume,
    "critical_path_agent_resume": build_critical_path_agent_resume,
    "recompute_ladder": build_recompute_ladder,
    "recompute_resume_mix": build_recompute_resume_mix,
}


SPECS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        experiment="effective_residency_sweep",
        variant="optimistic",
        name="residency_hotset_capacity_ladder",
        description=(
            "Large reusable hot families recur under controlled interference so the working set "
            "straddles the HBM capacity boundary. This is the optimistic workload for measuring "
            "the value of increasing effective KV residency."
        ),
        builder="residency_hotset_ladder",
        prefix_tokens=6144,
        suffix_tokens=512,
        output_len=512,
        hot_families=10,
        cold_families=20,
        rounds=20,
        interference_span=10,
        concurrency_hint=96,
        page_sizes=(16, 32, 64, 128),
        capacity_sweep_blocks=(4000, 6000, 8000, 10000, 12000, 16000),
    ),
    WorkloadSpec(
        experiment="critical_path_miss_attribution",
        variant="optimistic",
        name="critical_path_serial_resume",
        description=(
            "Serial resumptions are separated by unique interference so misses tend to occur on "
            "the request's critical path. This is the optimistic workload for exposed miss cost."
        ),
        builder="critical_path_serial_resume",
        prefix_tokens=4096,
        suffix_tokens=384,
        output_len=384,
        hot_families=12,
        cold_families=0,
        rounds=16,
        interference_span=16,
        concurrency_hint=32,
        page_sizes=(16, 32, 64, 128),
        capacity_sweep_blocks=(4000, 6000, 8000, 10000),
    ),
    WorkloadSpec(
        experiment="recomputation_microbenchmark",
        variant="optimistic",
        name="recompute_k_block_ladder",
        description=(
            "Controlled ladder workload where requests are tagged to induce 1, 2, 4, 8, or 16 "
            "recomputed blocks. This is the optimistic microbenchmark for measuring recompute cost."
        ),
        builder="recompute_ladder",
        prefix_tokens=4096,
        suffix_tokens=128,
        output_len=256,
        hot_families=0,
        cold_families=0,
        rounds=24,
        interference_span=0,
        concurrency_hint=1,
        page_sizes=(16, 32, 64, 128),
        capacity_sweep_blocks=(4000, 6000, 8000),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate workload bundles for the KV-cache headroom studies."
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic/headroom_studies",
        help="Directory where per-experiment workload bundles are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_index: dict[str, list[dict]] = {}
    for spec in SPECS:
        rows = BUILDERS[spec.builder](spec)
        experiment_dir = output_dir / spec.experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = experiment_dir / f"{spec.name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest_path = experiment_dir / f"{spec.name}.manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest(spec, rows), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        top_index.setdefault(spec.experiment, []).append(
            {
                "name": spec.name,
                "variant": spec.variant,
                "path": str(jsonl_path),
                "manifest": str(manifest_path),
            }
        )

    for experiment, entries in top_index.items():
        experiment_dir = output_dir / experiment
        (experiment_dir / "index.json").write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "workloads": sorted(entries, key=lambda item: (item["variant"], item["name"])),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "experiments": {
                    experiment: sorted(entries, key=lambda item: (item["variant"], item["name"]))
                    for experiment, entries in sorted(top_index.items())
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
