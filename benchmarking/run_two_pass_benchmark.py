#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full two-pass LRU then Belady benchmark.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", default="0.7")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--server-extra-args", default="")
    return parser.parse_args()


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


def run_command(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def launch_server(command: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=str(cwd), env=env, stdout=sys.stdout, stderr=sys.stderr)


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    traces_dir = output_root / "traces"
    plans_dir = output_root / "plans"
    bench_dir = output_root / "benchmarks"
    analysis_dir = output_root / "analysis"
    reports_dir = output_root / "reports"
    for directory in (traces_dir, plans_dir, bench_dir, analysis_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    common_env = os.environ.copy()
    common_env.update(
        {
            "MODEL_PATH": args.model_path,
            "DATASET_PATH": args.dataset_path,
            "PORT": str(args.port),
            "BASE_URL": base_url,
            "PAGE_SIZE": str(args.page_size),
            "MEM_FRACTION_STATIC": str(args.mem_fraction_static),
            "NUM_PROMPTS": str(args.num_prompts),
            "REQUEST_RATE": str(args.request_rate),
            "BENCH_SEED": str(args.bench_seed),
        }
    )
    if args.max_concurrency is not None:
        common_env["MAX_CONCURRENCY"] = str(args.max_concurrency)

    extra_args = shlex.split(args.server_extra_args) if args.server_extra_args else []

    lru_trace = traces_dir / "lru.jsonl"
    lru_bench = bench_dir / "lru.jsonl"
    lru_analysis = analysis_dir / "lru"
    plan_path = plans_dir / "belady_plan.json"
    belady_trace = traces_dir / "belady.jsonl"
    belady_bench = bench_dir / "belady.jsonl"
    belady_analysis = analysis_dir / "belady"
    final_report = reports_dir / "comparison.json"
    pressure_report = reports_dir / "memory_pressure.json"

    lru_env = common_env | {"RUN_LABEL": "lru", "TRACE_DIR": str(traces_dir)}
    lru_server_cmd = ["/bin/bash", str(root / "benchmarking" / "launch_sglang_server.sh"), *extra_args]
    lru_proc = launch_server(lru_server_cmd, env=lru_env, cwd=root)
    try:
        wait_until_ready(base_url)
        run_command(
            ["/bin/bash", str(root / "benchmarking" / "run_serving_benchmark.sh")],
            env=common_env | {"OUTPUT_FILE": str(lru_bench)},
            cwd=root,
        )
    finally:
        stop_server(lru_proc)

    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "compile_belady_plan.py"),
            "--trace",
            str(lru_trace),
            "--output",
            str(plan_path),
        ],
        env=common_env,
        cwd=root,
    )

    belady_env = common_env | {
        "RUN_LABEL": "belady",
        "TRACE_DIR": str(traces_dir),
        "BELADY_PLAN_PATH": str(plan_path),
    }
    belady_server_cmd = ["/bin/bash", str(root / "benchmarking" / "launch_belady_server.sh"), *extra_args]
    belady_proc = launch_server(belady_server_cmd, env=belady_env, cwd=root)
    try:
        wait_until_ready(base_url)
        run_command(
            ["/bin/bash", str(root / "benchmarking" / "run_serving_benchmark.sh")],
            env=common_env | {"OUTPUT_FILE": str(belady_bench)},
            cwd=root,
        )
    finally:
        stop_server(belady_proc)

    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analyze_kv_trace.py"),
            "--trace",
            str(lru_trace),
            "--output-dir",
            str(lru_analysis),
            "--block-capacity",
            str(args.gpu_kv_capacity_blocks),
        ],
        env=common_env,
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analyze_kv_trace.py"),
            "--trace",
            str(belady_trace),
            "--output-dir",
            str(belady_analysis),
            "--block-capacity",
            str(args.gpu_kv_capacity_blocks),
        ],
        env=common_env,
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "compare_benchmark_runs.py"),
            "--lru-bench",
            str(lru_bench),
            "--belady-bench",
            str(belady_bench),
            "--lru-trace-summary",
            str(lru_analysis / "summary.json"),
            "--belady-trace-summary",
            str(belady_analysis / "summary.json"),
            "--output",
            str(final_report),
            "--page-size",
            str(args.page_size),
        ],
        env=common_env,
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "estimate_memory_pressure.py"),
            "--dataset",
            args.dataset_path,
            "--gpu-kv-capacity-blocks",
            str(args.gpu_kv_capacity_blocks),
            "--page-size",
            str(args.page_size),
            "--output",
            str(pressure_report),
        ],
        env=common_env,
        cwd=root,
    )


if __name__ == "__main__":
    main()
