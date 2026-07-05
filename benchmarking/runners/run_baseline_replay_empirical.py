#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
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
    parser = argparse.ArgumentParser(description="Run baseline FCFS replay measurements without marginal counterfactuals.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _summary(rows: list[dict]) -> dict:
    wall = [float(row["wall_time_ms"]) for row in rows]
    prefill = [float(row["prefill_time_ms"]) for row in rows if row.get("prefill_time_ms") is not None]
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    cached = [int(row["num_cached_tokens"]) for row in rows if row.get("num_cached_tokens") is not None]
    return {
        "requests_measured": len(rows),
        "wall_time_ms_mean": statistics.fmean(wall) if wall else 0.0,
        "wall_time_ms_p50": statistics.median(wall) if wall else 0.0,
        "prefill_time_ms_mean": statistics.fmean(prefill) if prefill else None,
        "prefill_time_ms_p50": statistics.median(prefill) if prefill else None,
        "ttft_ms_mean": statistics.fmean(ttft) if ttft else None,
        "ttft_ms_p50": statistics.median(ttft) if ttft else None,
        "num_cached_tokens_mean": statistics.fmean(cached) if cached else None,
    }


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
    return first


def main() -> None:
    args = parse_args()
    output_root = resolve_timestamped_output_root(args.output_root, no_timestamp=args.no_timestamp)
    bundle = load_workload_bundle(Path(args.bundle_root))
    requests = bundle.requests[args.request_offset :]
    if args.request_limit is not None:
        requests = requests[: args.request_limit]

    manifest = profile_payload(
        "baseline_replay_empirical",
        "oracle0_baseline",
        {
            "system": args.system,
            "model": args.model,
            "bundle_root": args.bundle_root,
            "request_count": len(requests),
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
            trial_rows = []
            for _ in range(args.repeat_count):
                row = asdict(
                    run_replay_case(
                        system=args.system,
                        model=args.model,
                        bundle_root=Path(args.bundle_root),
                        request_id=request.request_id,
                        history_request_ids=history_ids,
                        preload_object_ids=[],
                        mode="baseline_replay",
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
                row = asdict(
                    run_replay_case_with_adapter(
                        adapter=adapter,
                        bundle=bundle,
                        system=args.system,
                        bundle_root=Path(args.bundle_root),
                        request_id=request.request_id,
                        history_request_ids=history_ids,
                        preload_object_ids=[],
                        mode="baseline_replay",
                    )
                )
                measurements.append(row)

    write_measurements(
        output_root / "baseline_replay_measurements.jsonl",
        [RequestMeasurement(**row) for row in measurements],
    )
    (output_root / "summary.json").write_text(
        json.dumps(_summary(measurements), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
