#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from headroom_common import (
    ROOT,
    WorkloadRef,
    call_two_pass,
    count_jsonl_rows,
    profile_payload,
    run_subprocess,
    write_manifest,
)


WORKLOADS = [
    WorkloadRef(
        label="optimistic",
        path="data/synthetic/headroom_studies/recomputation_microbenchmark/recompute_k_block_ladder.jsonl",
    ),
    WorkloadRef(
        label="near_real",
        path="data/processed/headroom_studies/recomputation_microbenchmark/recomputation_microbenchmark__realworld_sequence.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [32],
        "sample_rows": 2,
        "max_concurrency": 1,
        "mem_fraction_static": 0.24,
        "estimated_runtime_hours": 0.08,
    },
    "full": {
        "page_sizes": [16, 64],
        "sample_rows": 8,
        "max_concurrency": 1,
        "mem_fraction_static": 0.24,
        "estimated_runtime_hours": 1.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the recomputation microbenchmark in pilot or full mode."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def wait_until_ready(base_url: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"Server at {base_url} did not become ready")


def launch_server(command: list[str], *, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
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
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def canonical_conversation_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    convs = row.get("conversations", row.get("conversation", []))
    if len(convs) < 2:
        return None
    prompt = convs[0].get("content", convs[0].get("value", "")) or ""
    answer = convs[1].get("content", convs[1].get("value", "")) or ""
    if not prompt or not answer:
        return None
    if int(row.get("output_len", 0) or 0) < 2:
        return None
    return prompt, answer


def is_benchmarkable_row(row: dict[str, Any]) -> bool:
    return canonical_conversation_pair(row) is not None


def select_rows(rows: list[dict[str, Any]], sample_rows: int) -> list[dict[str, Any]]:
    rows = [row for row in rows if is_benchmarkable_row(row)]
    targeted = [row for row in rows if row.get("metadata", {}).get("target_recompute_blocks") is not None]
    if targeted:
        picked: dict[int, dict[str, Any]] = {}
        for row in targeted:
            target = int(row["metadata"]["target_recompute_blocks"])
            picked.setdefault(target, row)
        selected = [picked[key] for key in sorted(picked)[:sample_rows]]
        if selected:
            return selected
    rows = sorted(rows, key=lambda row: int(row.get("prompt_len", 0)), reverse=True)
    return rows[:sample_rows]


def write_temp_dataset(row: dict[str, Any]) -> str:
    pair = canonical_conversation_pair(row)
    if pair is None:
        raise ValueError("Row is not benchmarkable and cannot be written as a temp dataset")
    prompt, answer = pair
    payload = {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
    fd, path = tempfile.mkstemp(prefix="recompute_single_", suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def run_single_request(
    *,
    model_path: str,
    dataset_path: str,
    output_file: str,
    base_url: str,
    bench_seed: int,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": model_path,
            "DATASET_PATH": dataset_path,
            "BASE_URL": base_url,
            "NUM_PROMPTS": "1",
            "REQUEST_RATE": "inf",
            "BENCH_SEED": str(bench_seed),
            "OUTPUT_FILE": output_file,
            "MAX_CONCURRENCY": "1",
        }
    )
    command = ["/bin/bash", str(ROOT / "benchmarking" / "launchers" / "run_serving_benchmark.sh")]
    subprocess.run(command, check=True, cwd=str(ROOT), env=env)


def load_last_json(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    server_extra_args = args.server_extra_args
    if args.mode == "pilot" and not server_extra_args:
        server_extra_args = "--disable-cuda-graph --disable-piecewise-cuda-graph"
    output_root = Path(args.output_root)
    manifest = profile_payload(
        "recomputation_microbenchmark",
        args.mode,
        {"workloads": [workload.path for workload in WORKLOADS], **profile},
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    for workload in WORKLOADS:
        dataset_path = ROOT / workload.path
        rows = select_rows(load_rows(dataset_path), profile["sample_rows"])
        sampled_dataset = output_root / f"{workload.label}_sampled_dataset.jsonl"
        sampled_dataset.parent.mkdir(parents=True, exist_ok=True)
        with sampled_dataset.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        for page_size in profile["page_sizes"]:
            baseline_root = output_root / f"{workload.label}__ps{page_size}__baselines"
            call_two_pass(
                model_path=args.model_path,
                dataset_path=str(sampled_dataset),
                output_root=baseline_root / "opt",
                page_size=page_size,
                num_prompts=min(profile["sample_rows"], count_jsonl_rows(sampled_dataset)),
                request_rate="inf",
                max_concurrency=1,
                mem_fraction_static=profile["mem_fraction_static"],
                gpu_kv_capacity_blocks=16000,
                schedule_policy=args.schedule_policy,
                bench_seed=args.bench_seed,
                second_policy="belady",
                server_extra_args=server_extra_args,
                skip_analysis=False,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                continue

            trace_dir = output_root / f"{workload.label}__ps{page_size}__microbench"
            trace_dir.mkdir(parents=True, exist_ok=True)
            base_url = "http://127.0.0.1:30000"
            env = os.environ.copy()
            env.update(
                {
                    "MODEL_PATH": args.model_path,
                    "TRACE_DIR": str(trace_dir / "traces"),
                    "RUN_LABEL": f"{workload.label}_microbench",
                    "PAGE_SIZE": str(page_size),
                    "MEM_FRACTION_STATIC": str(profile["mem_fraction_static"]),
                    "SCHEDULE_POLICY": args.schedule_policy,
                }
            )
            command = [
                "/bin/bash",
                str(ROOT / "benchmarking" / "launchers" / "launch_sglang_server.sh"),
            ]
            if server_extra_args:
                command.extend(server_extra_args.split())
            server = launch_server(command, env=env)
            try:
                wait_until_ready(base_url)
                summary_rows = []
                for idx, row in enumerate(rows):
                    temp_dataset = write_temp_dataset(row)
                    cold_file = trace_dir / f"cold_{idx}.jsonl"
                    warm_file = trace_dir / f"warm_{idx}.jsonl"
                    run_single_request(
                        model_path=args.model_path,
                        dataset_path=temp_dataset,
                        output_file=str(cold_file),
                        base_url=base_url,
                        bench_seed=args.bench_seed,
                    )
                    run_single_request(
                        model_path=args.model_path,
                        dataset_path=temp_dataset,
                        output_file=str(warm_file),
                        base_url=base_url,
                        bench_seed=args.bench_seed,
                    )
                    cold = load_last_json(cold_file)
                    warm = load_last_json(warm_file)
                    summary_rows.append(
                        {
                            "row_index": idx,
                            "target_recompute_blocks": row.get("metadata", {}).get("target_recompute_blocks"),
                            "prompt_len": row.get("prompt_len"),
                            "cold_median_ttft_ms": cold.get("median_ttft_ms"),
                            "warm_median_ttft_ms": warm.get("median_ttft_ms"),
                            "delta_median_ttft_ms": (cold.get("median_ttft_ms") or 0.0)
                            - (warm.get("median_ttft_ms") or 0.0),
                            "cold_output_throughput": cold.get("output_throughput"),
                            "warm_output_throughput": warm.get("output_throughput"),
                        }
                    )
                    os.remove(temp_dataset)
                write_manifest(trace_dir / "microbench_summary.json", {"rows": summary_rows})
            finally:
                stop_server(server)


if __name__ == "__main__":
    main()
