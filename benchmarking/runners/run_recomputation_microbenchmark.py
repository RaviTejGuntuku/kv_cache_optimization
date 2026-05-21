#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from headroom_common import ROOT, profile_payload, resolve_timestamped_output_root, write_manifest


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "vocab_size": 152064,
        "torch_dtype": "bfloat16",
    }
}

PROFILES = {
    "pilot": {
        "k_values": [1, 10, 50, 100, 200, 400, 800],
        "repetitions": 5,
        "warmup": 1,
        "estimated_runtime_hours": 0.2,
    },
    "full": {
        "k_values": [1] + list(range(10, 801, 10)),
        "repetitions": 7,
        "warmup": 2,
        "estimated_runtime_hours": 2.2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-model KV recovery crossover benchmark."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    weight = index - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def summarize(values: list[float]) -> dict[str, float]:
    trimmed = sorted(values)
    if len(trimmed) >= 5:
        trimmed = trimmed[1:-1]
    return {
        "mean": float(statistics.fmean(values)),
        "trimmed_mean": float(statistics.fmean(trimmed)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def load_model_metadata(model_path: str) -> dict[str, Any]:
    if model_path in MODEL_PRESETS:
        preset = MODEL_PRESETS[model_path].copy()
    else:
        preset = {}
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    preset.setdefault("vocab_size", int(getattr(cfg, "vocab_size")))
    preset["torch_dtype"] = str(getattr(cfg, "torch_dtype", preset.get("torch_dtype", "bfloat16")))
    preset["hidden_size"] = int(getattr(cfg, "hidden_size"))
    preset["num_hidden_layers"] = int(getattr(cfg, "num_hidden_layers"))
    preset["num_attention_heads"] = int(getattr(cfg, "num_attention_heads"))
    preset["num_key_value_heads"] = int(
        getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))
    )
    return preset


def torch_dtype_from_string(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.bfloat16)


def cache_nbytes(past_key_values) -> int:
    total = 0
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            for tensor_name in ("keys", "values"):
                tensor = getattr(layer, tensor_name, None)
                if tensor is None:
                    continue
                total += tensor.numel() * tensor.element_size()
        return total
    for layer in past_key_values:
        if layer is None:
            continue
        for tensor in layer:
            if tensor is None:
                continue
            total += tensor.numel() * tensor.element_size()
    return total


def cache_to_pinned_cpu(past_key_values):
    from transformers.cache_utils import DynamicCache

    if hasattr(past_key_values, "layers"):
        cache_data = []
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            key_cpu = key.detach().to("cpu").pin_memory() if key is not None else None
            value_cpu = value.detach().to("cpu").pin_memory() if value is not None else None
            cache_data.append((key_cpu, value_cpu))
        return DynamicCache(ddp_cache_data=cache_data)

    pinned = []
    for layer in past_key_values:
        if layer is None:
            pinned.append((None, None))
            continue
        key, value = layer[:2]
        key_cpu = key.detach().to("cpu").pin_memory() if key is not None else None
        value_cpu = value.detach().to("cpu").pin_memory() if value is not None else None
        pinned.append((key_cpu, value_cpu))
    return DynamicCache(ddp_cache_data=pinned)


def cache_to_cuda(past_key_values_cpu, device):
    from transformers.cache_utils import DynamicCache

    cache_data = []
    if hasattr(past_key_values_cpu, "layers"):
        for layer in past_key_values_cpu.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            key_gpu = key.to(device=device, non_blocking=False) if key is not None else None
            value_gpu = value.to(device=device, non_blocking=False) if value is not None else None
            cache_data.append((key_gpu, value_gpu))
    else:
        for layer in past_key_values_cpu:
            if layer is None:
                cache_data.append((None, None))
                continue
            key, value = layer[:2]
            key_gpu = key.to(device=device, non_blocking=False) if key is not None else None
            value_gpu = value.to(device=device, non_blocking=False) if value is not None else None
            cache_data.append((key_gpu, value_gpu))
    return DynamicCache(ddp_cache_data=cache_data)


def cache_to_ssd_file(past_key_values_cpu, path: Path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = []
    if hasattr(past_key_values_cpu, "layers"):
        for layer in past_key_values_cpu.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            cache_data.append((key, value))
    else:
        for layer in past_key_values_cpu:
            if layer is None:
                cache_data.append((None, None))
                continue
            key, value = layer[:2]
            cache_data.append((key, value))
    torch.save(cache_data, path)


def cache_from_ssd_to_cuda(path: Path, device):
    import torch
    from transformers.cache_utils import DynamicCache

    cache_data_cpu = torch.load(path, map_location="cpu", weights_only=False)
    cache_data = []
    for key, value in cache_data_cpu:
        key_gpu = key.to(device=device, non_blocking=False) if key is not None else None
        value_gpu = value.to(device=device, non_blocking=False) if value is not None else None
        cache_data.append((key_gpu, value_gpu))
    return DynamicCache(ddp_cache_data=cache_data)


def benchmark_method(
    *,
    torch,
    model,
    method: str,
    prefix_ids,
    next_token_ids,
    cached_gpu,
    cached_cpu,
    cached_ssd_path,
    repetitions: int,
    warmup: int,
) -> dict[str, float]:
    def do_work():
        if method == "HBM_reuse":
            return model(input_ids=next_token_ids, past_key_values=cached_gpu, use_cache=True)
        if method == "DRAM_fetch":
            restored = cache_to_cuda(cached_cpu, prefix_ids.device)
            return model(input_ids=next_token_ids, past_key_values=restored, use_cache=True)
        if method == "SSD_fetch":
            restored = cache_from_ssd_to_cuda(cached_ssd_path, prefix_ids.device)
            return model(input_ids=next_token_ids, past_key_values=restored, use_cache=True)
        if method == "recompute":
            pref = model(input_ids=prefix_ids, use_cache=True)
            return model(input_ids=next_token_ids, past_key_values=pref.past_key_values, use_cache=True)
        raise ValueError(f"Unknown method {method}")

    for _ in range(warmup):
        out = do_work()
        _ = out.logits[:, -1, :1]
        torch.cuda.synchronize()

    wall_ms: list[float] = []
    cuda_ms: list[float] = []

    for _ in range(repetitions):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        out = do_work()
        _ = out.logits[:, -1, :1]
        end_event.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter()
        wall_ms.append((wall_end - wall_start) * 1000.0)
        cuda_ms.append(float(start_event.elapsed_time(end_event)))

    wall = summarize(wall_ms)
    cuda = summarize(cuda_ms)
    return {
        "wall_mean_ms": wall["mean"],
        "wall_trimmed_mean_ms": wall["trimmed_mean"],
        "wall_p50_ms": wall["p50"],
        "wall_p95_ms": wall["p95"],
        "wall_std_ms": wall["std"],
        "cuda_mean_ms": cuda["mean"],
        "cuda_trimmed_mean_ms": cuda["trimmed_mean"],
        "cuda_p50_ms": cuda["p50"],
        "cuda_p95_ms": cuda["p95"],
        "cuda_std_ms": cuda["std"],
    }


def first_crossover(rows: list[dict], *, metric_key: str) -> dict[str, Any]:
    return first_crossover_by_methods(
        rows, left_method="DRAM_fetch", right_method="recompute", metric_key=metric_key
    )


def first_crossover_by_methods(
    rows: list[dict], *, left_method: str, right_method: str, metric_key: str
) -> dict[str, Any]:
    dram = sorted(
        [row for row in rows if row["method"] == left_method], key=lambda row: row["k_blocks"]
    )
    recomp = {row["k_blocks"]: row for row in rows if row["method"] == right_method}
    previous_diff = None
    previous_k = None
    for row in dram:
        k = row["k_blocks"]
        if k not in recomp:
            continue
        diff = float(row[metric_key]) - float(recomp[k][metric_key])
        if previous_diff is not None and diff == 0:
            return {"found": True, "lower_k": k, "upper_k": k, "estimate_k": k}
        if previous_diff is not None and previous_diff * diff < 0:
            return {
                "found": True,
                "lower_k": previous_k,
                "upper_k": k,
                "estimate_k": (previous_k + k) / 2.0,
            }
        previous_diff = diff
        previous_k = k
    return {"found": False, "lower_k": None, "upper_k": None, "estimate_k": None}


def refine_grid(lower_k: int | None, upper_k: int | None) -> list[int]:
    if lower_k is None or upper_k is None or lower_k == upper_k:
        return []
    return list(range(lower_k, upper_k + 1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    k_values = args.k_values or profile["k_values"]
    repetitions = args.repetitions or profile["repetitions"]
    output_root = resolve_timestamped_output_root(args.output_root, no_timestamp=args.no_timestamp)

    model_meta = load_model_metadata(args.model_path)
    manifest = profile_payload(
        "recomputation_microbenchmark",
        args.mode,
        {
            "invoked_at_utc": output_root.name.rsplit("__", 1)[-1] if "__" in output_root.name else None,
            "resolved_output_root": str(output_root),
            "model_path": args.model_path,
            "page_size": args.page_size,
            "device": args.device,
            "k_values": k_values,
            "repetitions": repetitions,
            "warmup": profile["warmup"],
            "dtype": model_meta["torch_dtype"],
            "estimated_runtime_hours": profile["estimated_runtime_hours"],
            "runner_mode": "real_model_recovery",
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    if args.dry_run:
        return

    import torch
    from transformers import AutoModelForCausalLM

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")

    device = torch.device("cuda")
    dtype = torch_dtype_from_string(model_meta["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()
    torch.manual_seed(1)

    vocab_size = int(model_meta["vocab_size"])
    next_token_ids = torch.full((1, 1), 42, device=device, dtype=torch.long)

    rows: list[dict[str, Any]] = []
    methods = ("HBM_reuse", "DRAM_fetch", "SSD_fetch", "recompute")
    cache_bytes_by_k: dict[int, int] = {}
    spill_dir = output_root / "spill_cache"

    with torch.inference_mode():
        for k in k_values:
            prefix_len = k * args.page_size
            prefix_ids = (torch.arange(prefix_len, device=device, dtype=torch.long) % vocab_size).unsqueeze(0)
            pref = model(input_ids=prefix_ids, use_cache=True)
            cached_gpu_obj = pref.past_key_values
            cached_cpu = cache_to_pinned_cpu(cached_gpu_obj)
            cached_ssd_path = spill_dir / f"k_{k:04d}.pt"
            cache_to_ssd_file(cached_cpu, cached_ssd_path)
            cache_bytes_by_k[k] = cache_nbytes(cached_gpu_obj)

            for method in methods:
                stats = benchmark_method(
                    torch=torch,
                    model=model,
                    method=method,
                    prefix_ids=prefix_ids,
                    next_token_ids=next_token_ids,
                    cached_gpu=cached_gpu_obj,
                    cached_cpu=cached_cpu,
                    cached_ssd_path=cached_ssd_path,
                    repetitions=repetitions,
                    warmup=profile["warmup"],
                )
                rows.append(
                    {
                        "phase": "coarse",
                        "batch_size": 1,
                        "k_blocks": k,
                        "prefix_tokens": prefix_len,
                        "cache_bytes": cache_bytes_by_k[k],
                        "method": method,
                        **stats,
                    }
                )
            del pref, cached_cpu, cached_gpu_obj, prefix_ids
            torch.cuda.empty_cache()

    metrics_dir = output_root / "metrics"
    write_csv(metrics_dir / "recovery_times.csv", rows)
    write_manifest(
        metrics_dir / "recovery_times.json",
        {
            "rows": rows,
            "cache_bytes_by_k": cache_bytes_by_k,
        },
    )
    write_manifest(
        metrics_dir / "crossover_points.json",
        {
            "dram_vs_recompute_wall_p50_ms": first_crossover(rows, metric_key="wall_p50_ms"),
            "dram_vs_recompute_wall_mean_ms": first_crossover(rows, metric_key="wall_mean_ms"),
            "dram_vs_recompute_wall_trimmed_mean_ms": first_crossover(rows, metric_key="wall_trimmed_mean_ms"),
            "dram_vs_recompute_cuda_p50_ms": first_crossover(rows, metric_key="cuda_p50_ms"),
            "dram_vs_recompute_cuda_mean_ms": first_crossover(rows, metric_key="cuda_mean_ms"),
            "dram_vs_recompute_cuda_trimmed_mean_ms": first_crossover(rows, metric_key="cuda_trimmed_mean_ms"),
            "ssd_vs_recompute_wall_p50_ms": first_crossover_by_methods(
                rows, left_method="SSD_fetch", right_method="recompute", metric_key="wall_p50_ms"
            ),
            "ssd_vs_recompute_cuda_p50_ms": first_crossover_by_methods(
                rows, left_method="SSD_fetch", right_method="recompute", metric_key="cuda_p50_ms"
            ),
        },
    )

    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarking" / "analysis_scripts" / "plot_headroom_recompute.py"),
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
                "note": "Benchmark completed, but plotting failed. Install matplotlib in the active Python environment to materialize graphs.",
            },
        )


if __name__ == "__main__":
    main()
