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
from benchmarking.headroom_empirical.interface import candidate_object_ids
from benchmarking.headroom_empirical.schema import RequestMeasurement, load_workload_bundle, write_measurements
from benchmarking.runners.headroom_common import profile_payload, resolve_timestamped_output_root, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run empirical marginal counterfactual analysis.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--max-counterfactuals-per-request", type=int, default=4)
    parser.add_argument("--profile-baseline-every", type=int, default=0)
    parser.add_argument("--profile-counterfactual-every", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--warmup", action="store_true", help="Run one unmeasured warmup request before baseline/counterfactual stages.")
    parser.add_argument("--repeat-count", type=int, default=1, help="Repeat each measurement this many times and aggregate by median.")
    parser.add_argument("--isolated", action="store_true", help="Use a fresh engine for every measured trial.")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_profiled_case(
    *,
    args: argparse.Namespace,
    request_id: str,
    preload_ids: list[str],
    history_request_ids: list[str],
    warmup_request_ids: list[str],
    out_dir: Path,
    mode: str,
) -> dict:
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("Requested Nsight profiling, but `nsys` is not installed.")
    rep_path = out_dir / f"{request_id}__{mode}.nsys-rep"
    json_path = out_dir / f"{request_id}__{mode}.json"
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
        mode,
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
    return json.loads(json_path.read_text(encoding="utf-8"))


def _pick_prefill_or_wall(row: dict) -> float:
    value = row.get("prefill_time_ms")
    if value is None:
        value = row["wall_time_ms"]
    return float(value)


def _measurement_meta_value(row: dict, key: str) -> float | int | None:
    metadata = row.get("metadata", {}) or {}
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _object_size_tokens(object_row) -> int:
    metadata = object_row.metadata or {}
    if metadata.get("object_size_tokens") is not None:
        return int(metadata["object_size_tokens"])
    return int(object_row.seed_prompt_tokens)


def _plan_occurrences(row: dict) -> list[dict]:
    metadata = row.get("metadata", {}) or {}
    execution_plan = metadata.get("execution_plan", {}) or {}
    occurrences = execution_plan.get("ordered_occurrences") or []
    return [occ for occ in occurrences if isinstance(occ, dict)]


def _object_coverage(baseline_row: dict, object_id: str) -> dict[str, float | int | None]:
    occurrences = [
        occurrence
        for occurrence in _plan_occurrences(baseline_row)
        if occurrence.get("object_id") == object_id
    ]
    if not occurrences:
        return {
            "token_position_start": None,
            "token_position_end": None,
            "token_position_coverage_tokens": None,
            "token_position_coverage_fraction": None,
        }
    start = min(int(occurrence["token_start"]) for occurrence in occurrences)
    end = max(int(occurrence["token_end"]) for occurrence in occurrences)
    coverage_tokens = max(0, end - start)
    prompt_tokens = max(1, int(baseline_row.get("prompt_tokens") or 1))
    return {
        "token_position_start": start,
        "token_position_end": end,
        "token_position_coverage_tokens": coverage_tokens,
        "token_position_coverage_fraction": coverage_tokens / prompt_tokens,
    }


def _object_size_tokens_from_row(
    baseline_row: dict,
    object_id: str,
    object_row,
) -> int:
    occurrences = [
        occurrence
        for occurrence in _plan_occurrences(baseline_row)
        if occurrence.get("object_id") == object_id
    ]
    if occurrences:
        return max(
            max(0, int(occurrence["token_end"]) - int(occurrence["token_start"]))
            for occurrence in occurrences
        )
    return _object_size_tokens(object_row)


def _lookup_based_object_status(
    baseline_row: dict,
    *,
    object_start: int | None,
    object_end: int | None,
) -> dict[str, int | bool | None]:
    metadata = baseline_row.get("metadata", {}) or {}
    spans = metadata.get("lmcache_lookup_token_spans")
    hit_span_count = metadata.get("lmcache_lookup_hit_span_count")
    hit_token_end = metadata.get("lmcache_lookup_hit_token_end")
    if (
        object_start is None
        or object_end is None
        or not isinstance(spans, list)
        or not isinstance(hit_span_count, int)
    ):
        return {
            "was_missed_in_baseline_lookup": None,
            "lookup_overlapped_span_count": None,
            "lookup_hit_overlapped_span_count": None,
            "lookup_first_missed_span_start": None,
            "lookup_first_missed_span_end": None,
            "lookup_hit_token_end": int(hit_token_end) if isinstance(hit_token_end, int) else None,
        }

    overlapped = []
    for idx, span in enumerate(spans):
        start = int(span["start"])
        end = int(span["end"])
        if end <= object_start or start >= object_end:
            continue
        overlapped.append((idx, start, end))

    if not overlapped:
        return {
            "was_missed_in_baseline_lookup": None,
            "lookup_overlapped_span_count": 0,
            "lookup_hit_overlapped_span_count": 0,
            "lookup_first_missed_span_start": None,
            "lookup_first_missed_span_end": None,
            "lookup_hit_token_end": int(hit_token_end) if isinstance(hit_token_end, int) else None,
        }

    hit_overlapped = [item for item in overlapped if item[0] < hit_span_count]
    missed_overlapped = [item for item in overlapped if item[0] >= hit_span_count]
    first_missed = missed_overlapped[0] if missed_overlapped else None
    return {
        "was_missed_in_baseline_lookup": bool(missed_overlapped),
        "lookup_overlapped_span_count": len(overlapped),
        "lookup_hit_overlapped_span_count": len(hit_overlapped),
        "lookup_first_missed_span_start": first_missed[1] if first_missed else None,
        "lookup_first_missed_span_end": first_missed[2] if first_missed else None,
        "lookup_hit_token_end": int(hit_token_end) if isinstance(hit_token_end, int) else None,
    }


def _summary(rows: list[dict]) -> dict:
    gains = [row["marginal_gain_ms"] for row in rows]
    missed_rows = [row for row in rows if row.get("was_missed_in_baseline")]
    missed_gains = [row["marginal_gain_ms"] for row in missed_rows]
    return {
        "counterfactuals_measured": len(rows),
        "marginal_gain_ms_mean": statistics.fmean(gains) if gains else 0.0,
        "marginal_gain_ms_p50": statistics.median(gains) if gains else 0.0,
        "marginal_gain_ms_max": max(gains) if gains else 0.0,
        "missed_counterfactuals": len(missed_rows),
        "missed_marginal_gain_ms_mean": statistics.fmean(missed_gains) if missed_gains else 0.0,
        "missed_marginal_gain_ms_p50": statistics.median(missed_gains) if missed_gains else 0.0,
        "missed_marginal_gain_ms_max": max(missed_gains) if missed_gains else 0.0,
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
        "marginal_counterfactuals_empirical",
        "pilot",
        {
            "system": args.system,
            "model": args.model,
            "bundle_root": args.bundle_root,
            "request_count": len(requests),
            "max_counterfactuals_per_request": args.max_counterfactuals_per_request,
            "profile_baseline_every": args.profile_baseline_every,
            "profile_counterfactual_every": args.profile_counterfactual_every,
            "track": bundle.track,
            "repeat_count": args.repeat_count,
            "isolated": args.isolated,
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    profile_dir = output_root / "nsys"
    profile_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict] = []
    object_index = {obj.object_id: obj for obj in bundle.objects}
    request_index = {request.request_id: request for request in requests}
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
                should_profile = args.profile_baseline_every and trial_idx == 0 and idx % args.profile_baseline_every == 0
                if should_profile:
                    row = _run_profiled_case(
                        args=args,
                        request_id=request.request_id,
                        preload_ids=[],
                        history_request_ids=history_ids,
                        warmup_request_ids=warmup_request_ids,
                        out_dir=profile_dir,
                        mode="baseline_replay",
                    )
                else:
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
            baseline_rows.append(_aggregate_trial_rows(trial_rows))
    else:
        with build_adapter(args.system, adapter_config) as adapter:
            if args.warmup and requests:
                adapter.clear_state()
                adapter.prewarm([requests[0].prompt])
                adapter.clear_state()
            for idx, request in enumerate(requests):
                history_ids = [item.request_id for item in requests[:idx]]
                if args.profile_baseline_every and idx % args.profile_baseline_every == 0:
                    row = _run_profiled_case(
                        args=args,
                        request_id=request.request_id,
                        preload_ids=[],
                        history_request_ids=history_ids,
                        warmup_request_ids=warmup_request_ids,
                        out_dir=profile_dir,
                        mode="baseline_replay",
                    )
                else:
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
                baseline_rows.append(row)

    write_measurements(
        output_root / "baseline_replay_measurements.jsonl",
        [RequestMeasurement(**row) for row in baseline_rows],
    )

    counterfactual_rows: list[dict] = []
    if args.isolated:
        for idx, request in enumerate(requests):
            history_ids = [item.request_id for item in requests[:idx]]
            baseline_row = baseline_rows[idx]
            candidate_ids = candidate_object_ids(
                bundle,
                request,
                limit=args.max_counterfactuals_per_request,
            )
            for object_position, object_id in enumerate(candidate_ids):
                trial_rows: list[dict] = []
                mode = f"counterfactual__{object_id}"
                for trial_idx in range(args.repeat_count):
                    should_profile = args.profile_counterfactual_every and trial_idx == 0 and len(counterfactual_rows) % args.profile_counterfactual_every == 0
                    if should_profile:
                        cf_row = _run_profiled_case(
                            args=args,
                            request_id=request.request_id,
                            preload_ids=[object_id],
                            history_request_ids=history_ids,
                            warmup_request_ids=warmup_request_ids,
                            out_dir=profile_dir,
                            mode=mode,
                        )
                    else:
                        cf_row = asdict(
                            run_replay_case(
                                system=args.system,
                                model=args.model,
                                bundle_root=Path(args.bundle_root),
                                request_id=request.request_id,
                                history_request_ids=history_ids,
                                preload_object_ids=[object_id],
                                mode=mode,
                                max_model_len=args.max_model_len,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                warmup_request_ids=warmup_request_ids,
                                warmup_full_path=bool(warmup_request_ids),
                            )
                        )
                    trial_rows.append(cf_row)
                cf_row = _aggregate_trial_rows(trial_rows)
                baseline_cached_tokens = baseline_row.get("num_cached_tokens")
                counterfactual_cached_tokens = cf_row.get("num_cached_tokens")
                cached_token_gain = None
                if baseline_cached_tokens is not None and counterfactual_cached_tokens is not None:
                    cached_token_gain = int(counterfactual_cached_tokens) - int(baseline_cached_tokens)
                coverage = _object_coverage(baseline_row, object_id)
                lookup_status = _lookup_based_object_status(
                    baseline_row,
                    object_start=coverage["token_position_start"],
                    object_end=coverage["token_position_end"],
                )
                counterfactual_rows.append(
                    {
                        "request_id": request.request_id,
                        "track": request.track,
                        "object_id": object_id,
                        "object_type": object_index[object_id].object_type,
                        "object_size_tokens": _object_size_tokens_from_row(
                            baseline_row,
                            object_id,
                            object_index[object_id],
                        ),
                        "source_tier": object_index[object_id].source_tier,
                        "object_position": object_position,
                        "request_reusable_object_count": len(request.reusable_object_ids),
                        "request_prompt_tokens": int(
                            baseline_row.get("prompt_tokens") or request.prompt_tokens
                        ),
                        **coverage,
                        **lookup_status,
                        "baseline_prefill_ms": baseline_row.get("prefill_time_ms"),
                        "counterfactual_prefill_ms": cf_row.get("prefill_time_ms"),
                        "baseline_ttft_ms": baseline_row.get("ttft_ms"),
                        "counterfactual_ttft_ms": cf_row.get("ttft_ms"),
                        "baseline_prefill_or_wall_ms": _pick_prefill_or_wall(baseline_row),
                        "counterfactual_prefill_or_wall_ms": _pick_prefill_or_wall(cf_row),
                        "marginal_prefill_gain_ms": _pick_prefill_or_wall(baseline_row) - _pick_prefill_or_wall(cf_row),
                        "marginal_ttft_gain_ms": (
                            float(baseline_row["ttft_ms"]) - float(cf_row["ttft_ms"])
                            if baseline_row.get("ttft_ms") is not None
                            and cf_row.get("ttft_ms") is not None
                            else None
                        ),
                        "marginal_gain_ms": _pick_prefill_or_wall(baseline_row) - _pick_prefill_or_wall(cf_row),
                        "baseline_num_cached_tokens": baseline_cached_tokens,
                        "counterfactual_num_cached_tokens": counterfactual_cached_tokens,
                        "cached_token_gain": cached_token_gain,
                        "was_missed_in_baseline": (
                            lookup_status["was_missed_in_baseline_lookup"]
                            if lookup_status["was_missed_in_baseline_lookup"] is not None
                            else bool(cached_token_gain is not None and cached_token_gain > 0)
                        ),
                        "baseline_lmcache_retrieve_time_ms": _measurement_meta_value(
                            baseline_row, "lmcache_retrieve_time_ms"
                        ),
                        "counterfactual_lmcache_retrieve_time_ms": _measurement_meta_value(
                            cf_row, "lmcache_retrieve_time_ms"
                        ),
                        "baseline_lmcache_retrieved_tokens": _measurement_meta_value(
                            baseline_row, "lmcache_retrieved_tokens"
                        ),
                        "counterfactual_lmcache_retrieved_tokens": _measurement_meta_value(
                            cf_row, "lmcache_retrieved_tokens"
                        ),
                        "baseline_lmcache_remote_read_bytes": _measurement_meta_value(
                            baseline_row, "lmcache_remote_read_bytes_delta"
                        ),
                        "counterfactual_lmcache_remote_read_bytes": _measurement_meta_value(
                            cf_row, "lmcache_remote_read_bytes_delta"
                        ),
                        "history_request_count": len(history_ids),
                        "repair_expected": cf_row.get("repair_expected", False),
                        "metadata": {
                            "system": args.system,
                            "bundle_root": args.bundle_root,
                            "request_metadata": request_index[request.request_id].metadata,
                            "object_metadata": object_index[object_id].metadata,
                            "trial_count": args.repeat_count,
                        },
                    }
                )
    else:
        with build_adapter(args.system, adapter_config) as adapter:
            if args.warmup and requests:
                adapter.clear_state()
                adapter.prewarm([requests[0].prompt])
                adapter.clear_state()
            for idx, request in enumerate(requests):
                history_ids = [item.request_id for item in requests[:idx]]
                baseline_row = baseline_rows[idx]
                candidate_ids = candidate_object_ids(
                    bundle,
                    request,
                    limit=args.max_counterfactuals_per_request,
                )
                for object_position, object_id in enumerate(candidate_ids):
                    mode = f"counterfactual__{object_id}"
                    if args.profile_counterfactual_every and len(counterfactual_rows) % args.profile_counterfactual_every == 0:
                        cf_row = _run_profiled_case(
                            args=args,
                            request_id=request.request_id,
                            preload_ids=[object_id],
                            history_request_ids=history_ids,
                            warmup_request_ids=warmup_request_ids,
                            out_dir=profile_dir,
                            mode=mode,
                        )
                    else:
                        cf_row = asdict(
                            run_replay_case_with_adapter(
                                adapter=adapter,
                                bundle=bundle,
                                system=args.system,
                                bundle_root=Path(args.bundle_root),
                                request_id=request.request_id,
                                history_request_ids=history_ids,
                                preload_object_ids=[object_id],
                                mode=mode,
                            )
                        )
                    coverage = _object_coverage(baseline_row, object_id)
                    lookup_status = _lookup_based_object_status(
                        baseline_row,
                        object_start=coverage["token_position_start"],
                        object_end=coverage["token_position_end"],
                    )
                    counterfactual_rows.append(
                        {
                            "request_id": request.request_id,
                            "track": request.track,
                            "object_id": object_id,
                            "object_type": object_index[object_id].object_type,
                            "object_size_tokens": _object_size_tokens_from_row(
                                baseline_row,
                                object_id,
                                object_index[object_id],
                            ),
                            "source_tier": object_index[object_id].source_tier,
                            "object_position": object_position,
                            "request_reusable_object_count": len(request.reusable_object_ids),
                            "request_prompt_tokens": int(
                                baseline_row.get("prompt_tokens") or request.prompt_tokens
                            ),
                            **coverage,
                            **lookup_status,
                            "baseline_prefill_ms": baseline_row.get("prefill_time_ms"),
                            "counterfactual_prefill_ms": cf_row.get("prefill_time_ms"),
                            "baseline_ttft_ms": baseline_row.get("ttft_ms"),
                            "counterfactual_ttft_ms": cf_row.get("ttft_ms"),
                            "baseline_prefill_or_wall_ms": _pick_prefill_or_wall(baseline_row),
                            "counterfactual_prefill_or_wall_ms": _pick_prefill_or_wall(cf_row),
                            "marginal_prefill_gain_ms": _pick_prefill_or_wall(baseline_row) - _pick_prefill_or_wall(cf_row),
                            "marginal_ttft_gain_ms": (
                                float(baseline_row["ttft_ms"]) - float(cf_row["ttft_ms"])
                                if baseline_row.get("ttft_ms") is not None
                                and cf_row.get("ttft_ms") is not None
                                else None
                            ),
                            "marginal_gain_ms": _pick_prefill_or_wall(baseline_row) - _pick_prefill_or_wall(cf_row),
                            "baseline_num_cached_tokens": baseline_row.get("num_cached_tokens"),
                            "counterfactual_num_cached_tokens": cf_row.get("num_cached_tokens"),
                            "cached_token_gain": (
                                int(cf_row["num_cached_tokens"]) - int(baseline_row["num_cached_tokens"])
                                if cf_row.get("num_cached_tokens") is not None
                                and baseline_row.get("num_cached_tokens") is not None
                                else None
                            ),
                            "was_missed_in_baseline": (
                                lookup_status["was_missed_in_baseline_lookup"]
                                if lookup_status["was_missed_in_baseline_lookup"] is not None
                                else (
                                    int(cf_row["num_cached_tokens"]) > int(baseline_row["num_cached_tokens"])
                                    if cf_row.get("num_cached_tokens") is not None
                                    and baseline_row.get("num_cached_tokens") is not None
                                    else False
                                )
                            ),
                            "baseline_lmcache_retrieve_time_ms": _measurement_meta_value(
                                baseline_row, "lmcache_retrieve_time_ms"
                            ),
                            "counterfactual_lmcache_retrieve_time_ms": _measurement_meta_value(
                                cf_row, "lmcache_retrieve_time_ms"
                            ),
                            "baseline_lmcache_retrieved_tokens": _measurement_meta_value(
                                baseline_row, "lmcache_retrieved_tokens"
                            ),
                            "counterfactual_lmcache_retrieved_tokens": _measurement_meta_value(
                                cf_row, "lmcache_retrieved_tokens"
                            ),
                            "baseline_lmcache_remote_read_bytes": _measurement_meta_value(
                                baseline_row, "lmcache_remote_read_bytes_delta"
                            ),
                            "counterfactual_lmcache_remote_read_bytes": _measurement_meta_value(
                                cf_row, "lmcache_remote_read_bytes_delta"
                            ),
                            "history_request_count": len(history_ids),
                            "repair_expected": cf_row.get("repair_expected", False),
                            "metadata": {
                                "system": args.system,
                                "bundle_root": args.bundle_root,
                                "request_metadata": request_index[request.request_id].metadata,
                                "object_metadata": object_index[object_id].metadata,
                            },
                        }
                    )

    out_path = output_root / "marginal_counterfactuals.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in counterfactual_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "summary.json").write_text(
        json.dumps(_summary(counterfactual_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
