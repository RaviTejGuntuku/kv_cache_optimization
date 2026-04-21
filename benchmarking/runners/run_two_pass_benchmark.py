#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
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
    parser.add_argument(
        "--schedule-policy",
        default="fcfs",
        help="SGLang schedule policy, e.g. fcfs or lpm.",
    )
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", default="0.7")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix appended to the output-root directory name.",
    )
    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="Append a UTC timestamp suffix to output-root so repeated runs are preserved.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help=(
            "Stop after producing traces, benchmark outputs, and the Belady plan. "
            "Use this when post-processing will happen on another machine."
        ),
    )
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


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        parent_listing = []
        if path.parent.exists():
            parent_listing = sorted(item.name for item in path.parent.iterdir())
        raise FileNotFoundError(
            f"Expected {description} at {path}, but it does not exist. "
            f"Directory contents of {path.parent}: {parent_listing}"
        )


def resolve_output_root(output_root: Path, *, run_tag: str | None, auto_version: bool) -> Path:
    suffix = run_tag
    if auto_version and not suffix:
        suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if suffix:
        resolved = output_root.parent / f"{output_root.name}__{suffix}"
    else:
        resolved = output_root

    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory {resolved} already exists and is not empty. "
            "Pass --run-tag or --auto-version to preserve previous runs."
        )
    return resolved


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_root = resolve_output_root(
        Path(args.output_root), run_tag=args.run_tag, auto_version=args.auto_version
    )
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
            "SCHEDULE_POLICY": str(args.schedule_policy),
            "BENCH_SEED": str(args.bench_seed),
        }
    )
    if args.max_concurrency is not None:
        common_env["MAX_CONCURRENCY"] = str(args.max_concurrency)

    run_metadata = {
        "invoked_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolved_output_root": str(output_root),
        "args": vars(args),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

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
    lru_server_cmd = [
        "/bin/bash",
        str(root / "benchmarking" / "launchers" / "launch_sglang_server.sh"),
        *extra_args,
    ]
    lru_proc = launch_server(lru_server_cmd, env=lru_env, cwd=root)
    try:
        wait_until_ready(base_url)
        run_command(
            ["/bin/bash", str(root / "benchmarking" / "launchers" / "run_serving_benchmark.sh")],
            env=common_env | {"OUTPUT_FILE": str(lru_bench)},
            cwd=root,
        )
    finally:
        stop_server(lru_proc)
    require_file(lru_trace, "LRU trace")
    require_file(lru_bench, "LRU benchmark output")

    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "compile_belady_plan.py"),
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
    belady_server_cmd = [
        "/bin/bash",
        str(root / "benchmarking" / "launchers" / "launch_belady_server.sh"),
        *extra_args,
    ]
    belady_proc = launch_server(belady_server_cmd, env=belady_env, cwd=root)
    try:
        wait_until_ready(base_url)
        run_command(
            ["/bin/bash", str(root / "benchmarking" / "launchers" / "run_serving_benchmark.sh")],
            env=common_env | {"OUTPUT_FILE": str(belady_bench)},
            cwd=root,
        )
    finally:
        stop_server(belady_proc)
    require_file(belady_trace, "Belady trace")
    require_file(belady_bench, "Belady benchmark output")

    if args.skip_analysis:
        return

    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "analyze_kv_trace.py"),
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
            str(root / "benchmarking" / "analysis_scripts" / "analyze_kv_trace.py"),
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
            str(root / "benchmarking" / "analysis_scripts" / "compare_benchmark_runs.py"),
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
            str(root / "benchmarking" / "analysis_scripts" / "estimate_memory_pressure.py"),
            "--dataset",
            args.dataset_path,
            "--gpu-kv-capacity-blocks",
            str(args.gpu_kv_capacity_blocks),
            "--page-size",
            str(args.page_size),
            "--concurrency",
            str(args.max_concurrency or args.num_prompts),
            "--output",
            str(pressure_report),
        ],
        env=common_env,
        cwd=root,
    )


if __name__ == "__main__":
    main()
