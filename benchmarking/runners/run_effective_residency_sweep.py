#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from headroom_common import (
    ROOT,
    WorkloadRef,
    call_two_pass,
    count_jsonl_rows,
    launch_server,
    max_prompt_len_from_jsonl,
    profile_payload,
    resolve_timestamped_output_root,
    stop_server,
    wait_until_ready,
    write_manifest,
)


WORKLOADS = [
    WorkloadRef(
        label="optimistic",
        path="datasets/synthetic/headroom_studies/effective_residency_sweep/residency_compulsory_reachable_hotset.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [16],
        "capacity_blocks": [500, 2000, 4000],
        "num_prompts": 16,
        "max_concurrency": 16,
        "request_rate": "8",
        "estimated_runtime_hours": 0.08,
    },
    "full": {
        "page_sizes": [16],
        "capacity_blocks": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
        "num_prompts": 384,
        "max_concurrency": 96,
        "request_rate": "16",
        "estimated_runtime_hours": 2.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the effective-residency headroom sweep in pilot or full mode."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--capacity-blocks", nargs="+", type=int, default=None)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _extract_token_capacity(server_info: dict) -> int | None:
    if not isinstance(server_info, dict):
        return None
    if isinstance(server_info.get("max_total_num_tokens"), int):
        return int(server_info["max_total_num_tokens"])
    internal_states = server_info.get("internal_states")
    if isinstance(internal_states, list) and internal_states:
        state = internal_states[0]
        if isinstance(state, dict):
            for key in ("token_capacity", "max_total_num_tokens"):
                if isinstance(state.get(key), int):
                    return int(state[key])
    return None


def _extract_max_req_input_len(server_info: dict) -> int | None:
    if not isinstance(server_info, dict):
        return None
    if isinstance(server_info.get("max_req_input_len"), int):
        return int(server_info["max_req_input_len"])
    internal_states = server_info.get("internal_states")
    if isinstance(internal_states, list) and internal_states:
        state = internal_states[0]
        if isinstance(state, dict) and isinstance(state.get("max_req_input_len"), int):
            return int(state["max_req_input_len"])
    return None


def _preflight_capacity(
    *,
    model_path: str,
    page_size: int,
    mem_fraction_static: float,
    schedule_policy: str,
    requested_max_tokens: int,
    max_prompt_len: int,
    server_extra_args: str,
    port: int,
) -> dict:
    import requests

    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="residency_preflight_") as tmpdir:
        env = {
            **dict(os.environ),
            "MODEL_PATH": model_path,
            "PORT": str(port),
            "PAGE_SIZE": str(page_size),
            "MEM_FRACTION_STATIC": str(mem_fraction_static),
            "SCHEDULE_POLICY": schedule_policy,
            "RUN_LABEL": "preflight",
            "TRACE_DIR": str(Path(tmpdir) / "traces"),
        }
        command = [
            "/bin/bash",
            str(ROOT / "benchmarking" / "launchers" / "launch_sglang_server.sh"),
        ]
        if server_extra_args:
            command.extend(shlex.split(server_extra_args))
        command.extend(["--max-total-tokens", str(requested_max_tokens)])
        proc = launch_server(command, env=env, cwd=ROOT)
        try:
            wait_until_ready(base_url, timeout_s=900)
            resp = requests.get(base_url + "/server_info", timeout=10)
            resp.raise_for_status()
            server_info = resp.json()
        finally:
            stop_server(proc)

    token_capacity = _extract_token_capacity(server_info)
    max_req_input_len = _extract_max_req_input_len(server_info)
    if token_capacity is None:
        raise RuntimeError(
            "Residency preflight could not determine token capacity from /server_info."
        )
    if token_capacity < requested_max_tokens:
        raise RuntimeError(
            "Residency sweep invalid: SGLang profiled only "
            f"{token_capacity} tokens, below the requested top sweep point "
            f"{requested_max_tokens}. Increase --mem-fraction-static or lower the sweep."
        )
    if max_req_input_len is not None and max_req_input_len <= max_prompt_len:
        raise RuntimeError(
            "Residency sweep invalid: server max_req_input_len="
            f"{max_req_input_len} does not admit the workload's max prompt length "
            f"{max_prompt_len}. Increase --mem-fraction-static or shorten the workload."
        )
    return {
        "requested_max_tokens": requested_max_tokens,
        "profiled_token_capacity": token_capacity,
        "max_req_input_len": max_req_input_len,
        "max_prompt_len": max_prompt_len,
    }


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    capacity_blocks = args.capacity_blocks or profile["capacity_blocks"]
    server_extra_args = args.server_extra_args.strip()
    if args.mode == "pilot" and not server_extra_args:
        server_extra_args = "--disable-cuda-graph --disable-piecewise-cuda-graph"
    output_root = resolve_timestamped_output_root(
        args.output_root, no_timestamp=args.no_timestamp
    )
    max_prompt_len = max(
        max_prompt_len_from_jsonl(ROOT / workload.path) for workload in WORKLOADS
    )
    min_capacity_tokens = min(capacity_blocks) * profile["page_sizes"][0]
    if min_capacity_tokens <= max_prompt_len:
        raise RuntimeError(
            "Residency sweep invalid: smallest capacity point "
            f"{min_capacity_tokens} tokens does not exceed the workload max prompt "
            f"length {max_prompt_len}. Increase the minimum capacity or shorten prompts."
        )
    preflight = None
    if not args.skip_preflight and not args.dry_run:
        preflight = _preflight_capacity(
            model_path=args.model_path,
            page_size=profile["page_sizes"][0],
            mem_fraction_static=args.mem_fraction_static,
            schedule_policy=args.schedule_policy,
            requested_max_tokens=max(capacity_blocks) * profile["page_sizes"][0],
            max_prompt_len=max_prompt_len,
            server_extra_args=server_extra_args,
            port=args.port,
        )
    manifest = profile_payload(
        "effective_residency_sweep",
        args.mode,
        {
            "invoked_at_utc": output_root.name.rsplit("__", 1)[-1]
            if "__" in output_root.name
            else None,
            "resolved_output_root": str(output_root),
            "workloads": [workload.path for workload in WORKLOADS],
            "max_prompt_len": max_prompt_len,
            "preflight": preflight,
            **{
                **profile,
                "capacity_blocks": capacity_blocks,
                "mem_fraction_static": args.mem_fraction_static,
            },
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    for workload in WORKLOADS:
        dataset_path = ROOT / workload.path
        num_prompts = min(profile["num_prompts"], count_jsonl_rows(dataset_path))
        for page_size in profile["page_sizes"]:
            for block_capacity in capacity_blocks:
                max_total_tokens = block_capacity * page_size
                extra_args = " ".join(
                    part
                    for part in [
                        server_extra_args,
                        f"--max-total-tokens {max_total_tokens}",
                    ]
                    if part
                )
                for second_policy in ("belady",):
                    run_name = (
                        f"{workload.label}__ps{page_size}__cap{block_capacity}__{second_policy}"
                    )
                    call_two_pass(
                        model_path=args.model_path,
                        dataset_path=str(dataset_path),
                        output_root=output_root / run_name,
                        page_size=page_size,
                        num_prompts=num_prompts,
                        request_rate=profile["request_rate"],
                        max_concurrency=profile["max_concurrency"],
                        mem_fraction_static=args.mem_fraction_static,
                        gpu_kv_capacity_blocks=block_capacity,
                        schedule_policy=args.schedule_policy,
                        bench_seed=args.bench_seed,
                        second_policy=second_policy,
                        server_extra_args=extra_args,
                        skip_analysis=args.skip_analysis,
                        dry_run=args.dry_run,
                    )

    if not args.dry_run:
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarking" / "analysis_scripts" / "plot_headroom_effective_residency.py"),
                    "--experiment-root",
                    str(output_root),
                ],
                check=True,
                cwd=str(ROOT),
            )
        except subprocess.CalledProcessError as exc:
            write_manifest(
                output_root / "plotting_status.json",
                {
                    "status": "failed",
                    "reason": str(exc),
                    "note": "Runs completed, but plotting failed. Install matplotlib in the active Python environment to materialize graphs.",
                },
            )


if __name__ == "__main__":
    main()
