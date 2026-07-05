#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.adapters import AdapterConfig, build_adapter
from benchmarking.headroom_empirical.case_runner import run_replay_case, run_replay_case_with_adapter
from benchmarking.headroom_empirical.schema import RequestMeasurement, load_workload_bundle, write_measurements
from benchmarking.runners.headroom_common import profile_payload, resolve_timestamped_output_root, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Oracle 0 empirical headroom measurements.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--profile-every", type=int, default=0, help="Profile every Nth request with nsys. 0 disables profiling.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--warmup", action="store_true", help="Run one unmeasured warmup request before measured cases.")
    parser.add_argument("--repeat-count", type=int, default=1, help="Repeat each measurement this many times and aggregate by median.")
    parser.add_argument("--isolated", action="store_true", help="Use a fresh engine for every measured trial.")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _summary(rows: list[dict]) -> dict:
    wall = [row["wall_time_ms"] for row in rows]
    prefill = [row["prefill_time_ms"] for row in rows if row["prefill_time_ms"] is not None]
    cached = [row["num_cached_tokens"] for row in rows if row["num_cached_tokens"] is not None]
    return {
        "requests_measured": len(rows),
        "wall_time_ms_mean": statistics.fmean(wall) if wall else 0.0,
        "wall_time_ms_p50": statistics.median(wall) if wall else 0.0,
        "prefill_time_ms_mean": statistics.fmean(prefill) if prefill else None,
        "prefill_time_ms_p50": statistics.median(prefill) if prefill else None,
        "num_cached_tokens_mean": statistics.fmean(cached) if cached else None,
    }


def _run_profiled_case(
    args: argparse.Namespace,
    request_id: str,
    preload_ids: list[str],
    history_request_ids: list[str],
    warmup_request_ids: list[str],
    out_dir: Path,
) -> None:
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("Requested Nsight profiling, but `nsys` is not installed.")
    rep_path = out_dir / f"{request_id}.nsys-rep"
    json_path = out_dir / f"{request_id}.json"
    cmd = [
        nsys,
        "profile",
        "--force-overwrite",
        "true",
        "-t",
        "cuda,nvtx,osrt",
        "-o",
        str(rep_path.with_suffix("")),
        sys.executable,
        str(ROOT / "benchmarking" / "runners" / "run_empirical_headroom_case.py"),
        "--system",
        args.system,
        "--model",
        args.model,
        "--bundle-root",
        args.bundle_root,
        "--request-id",
        request_id,
        "--mode",
        "oracle0",
        "--output",
        str(json_path),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if preload_ids:
        cmd.extend(["--preload-object-ids", *preload_ids])
    if history_request_ids:
        cmd.extend(["--history-request-ids", *history_request_ids])
    if warmup_request_ids:
        cmd.extend(["--warmup-request-ids", *warmup_request_ids])
        cmd.append("--warmup-full-path")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _aggregate_trial_rows(rows: list[dict]) -> dict:
    if len(rows) == 1:
        row = dict(rows[0])
        row.setdefault("metadata", {})
        row["metadata"]["trial_count"] = 1
        return row
    first = dict(rows[0])
    first.setdefault("metadata", {})
    numeric_float_fields = ("wall_time_ms", "prefill_time_ms", "decode_time_ms", "ttft_ms")
    numeric_int_fields = ("num_cached_tokens", "prompt_tokens", "output_len")
    for field in numeric_float_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        first[field] = statistics.median(values) if values else None
    for field in numeric_int_fields:
        values = [int(row[field]) for row in rows if row.get(field) is not None]
        first[field] = int(round(statistics.median(values))) if values else None
    first["metadata"]["trial_count"] = len(rows)
    first["metadata"]["trial_wall_time_ms"] = [float(row["wall_time_ms"]) for row in rows]
    if any(row.get("prefill_time_ms") is not None for row in rows):
        first["metadata"]["trial_prefill_time_ms"] = [
            None if row.get("prefill_time_ms") is None else float(row["prefill_time_ms"])
            for row in rows
        ]
    return first


def main() -> None:
    args = parse_args()
    output_root = resolve_timestamped_output_root(args.output_root, no_timestamp=args.no_timestamp)
    bundle = load_workload_bundle(Path(args.bundle_root))
    requests = bundle.requests[args.request_offset :]
    if args.request_limit is not None:
        requests = requests[: args.request_limit]
    manifest = profile_payload(
        "oracle0_empirical",
        "pilot",
        {
            "system": args.system,
            "model": args.model,
            "bundle_root": args.bundle_root,
            "request_count": len(requests),
            "profile_every": args.profile_every,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "track": bundle.track,
            "repeat_count": args.repeat_count,
            "isolated": args.isolated,
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    prof_dir = output_root / "nsys"
    prof_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    warmup_request_ids = [requests[0].request_id] if args.warmup and requests else []
    adapter_config = AdapterConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.isolated:
        for idx, request in enumerate(requests):
            history_ids = [item.request_id for item in requests[:idx]]
            trial_rows: list[dict] = []
            for trial_idx in range(args.repeat_count):
                should_profile = args.profile_every and trial_idx == 0 and idx % args.profile_every == 0
                if should_profile:
                    _run_profiled_case(
                        args,
                        request.request_id,
                        request.reusable_object_ids,
                        history_ids,
                        warmup_request_ids,
                        prof_dir,
                    )
                    row = json.loads((prof_dir / f"{request.request_id}.json").read_text(encoding="utf-8"))
                else:
                    row = asdict(
                        run_replay_case(
                            system=args.system,
                            model=args.model,
                            bundle_root=Path(args.bundle_root),
                            request_id=request.request_id,
                            history_request_ids=history_ids,
                            preload_object_ids=request.reusable_object_ids,
                            mode="oracle0",
                            max_model_len=args.max_model_len,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            warmup_request_ids=warmup_request_ids,
                            warmup_full_path=bool(warmup_request_ids),
                        )
                    )
                trial_rows.append(row)
            measurements.append(_aggregate_trial_rows(trial_rows))
    else:
        with build_adapter(args.system, adapter_config) as adapter:
            if args.warmup and requests:
                adapter.clear_state()
                adapter.prewarm([requests[0].prompt])
                adapter.clear_state()
            for idx, request in enumerate(requests):
                history_ids = [item.request_id for item in requests[:idx]]
                if args.profile_every and idx % args.profile_every == 0:
                    _run_profiled_case(
                        args,
                        request.request_id,
                        request.reusable_object_ids,
                        history_ids,
                        warmup_request_ids,
                        prof_dir,
                    )
                    result = json.loads((prof_dir / f"{request.request_id}.json").read_text(encoding="utf-8"))
                else:
                    result = asdict(
                        run_replay_case_with_adapter(
                            adapter=adapter,
                            bundle=bundle,
                            system=args.system,
                            bundle_root=Path(args.bundle_root),
                            request_id=request.request_id,
                            history_request_ids=history_ids,
                            preload_object_ids=request.reusable_object_ids,
                            mode="oracle0",
                        )
                    )
                measurements.append(result)

    write_measurements(
        output_root / "oracle0_measurements.jsonl",
        [
            RequestMeasurement(**row)
            for row in measurements
        ],
    )
    (output_root / "summary.json").write_text(
        json.dumps(_summary(measurements), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
