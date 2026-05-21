#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
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


def max_prompt_len_from_jsonl(path: Path) -> int:
    max_prompt_len = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_len = row.get("prompt_len")
            if prompt_len is not None:
                max_prompt_len = max(max_prompt_len, int(prompt_len))
    return max_prompt_len


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def utc_timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_timestamped_output_root(output_root: str | Path, *, no_timestamp: bool = False) -> Path:
    root = Path(output_root)
    if no_timestamp:
        return root
    return root.parent / f"{root.name}__{utc_timestamp_tag()}"


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


def wait_until_ready(base_url: str, timeout_s: int = 600) -> None:
    import requests

    deadline = time.time() + timeout_s
    last_exc = None
    while time.time() < deadline:
        try:
            response = requests.get(base_url + "/v1/models", timeout=5)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(2)
    raise RuntimeError(f"Server did not become ready at {base_url}. Last error: {last_exc}")


def launch_server(command: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


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
