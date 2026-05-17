#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkloadRef:
    label: str
    path: str


def run_subprocess(command: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return
    subprocess.run(command, check=True, cwd=str(cwd or ROOT))


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scaled_block_capacity(
    *,
    base_capacity_blocks: int,
    mem_fraction_static: float,
    reference_mem_fraction_static: float,
) -> int:
    scaled = int(base_capacity_blocks * mem_fraction_static / reference_mem_fraction_static)
    return max(1, scaled)


def profile_payload(name: str, mode: str, extra: dict[str, Any]) -> dict[str, Any]:
    payload = {"experiment": name, "mode": mode}
    payload.update(extra)
    return payload


def call_two_pass(
    *,
    model_path: str,
    dataset_path: str,
    output_root: Path,
    page_size: int,
    num_prompts: int,
    request_rate: str,
    max_concurrency: int,
    mem_fraction_static: float,
    gpu_kv_capacity_blocks: int,
    schedule_policy: str,
    bench_seed: int,
    second_policy: str,
    server_extra_args: str,
    skip_analysis: bool,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "benchmarking" / "runners" / "run_two_pass_benchmark.py"),
        "--model-path",
        model_path,
        "--dataset-path",
        dataset_path,
        "--output-root",
        str(output_root),
        "--page-size",
        str(page_size),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--max-concurrency",
        str(max_concurrency),
        "--schedule-policy",
        schedule_policy,
        "--bench-seed",
        str(bench_seed),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--gpu-kv-capacity-blocks",
        str(gpu_kv_capacity_blocks),
        "--second-policy",
        second_policy,
    ]
    if server_extra_args:
        command.extend(["--server-extra-args", server_extra_args])
    if skip_analysis:
        command.append("--skip-analysis")
    run_subprocess(command, dry_run=dry_run, cwd=ROOT)
