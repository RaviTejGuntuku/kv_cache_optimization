#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Oracle 0 headroom against baseline replay.")
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_prefill(row: dict) -> float:
    return float(row["prefill_time_ms"] if row.get("prefill_time_ms") is not None else row["wall_time_ms"])


def pick_ttft(row: dict) -> float:
    return float(row["ttft_ms"] if row.get("ttft_ms") is not None else row["wall_time_ms"])


def reusable_tokens(row: dict) -> int:
    metadata = row.get("metadata") or {}
    plan = metadata.get("execution_plan") or {}
    occurrences = plan.get("ordered_occurrences") or []
    by_object: dict[str, int] = {}
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        object_id = occ.get("object_id")
        if object_id is None:
            continue
        token_len = max(0, int(occ.get("token_end", 0)) - int(occ.get("token_start", 0)))
        by_object[object_id] = max(by_object.get(object_id, 0), token_len)
    return sum(by_object.values())


def short_label(request_id: str) -> str:
    if request_id.startswith("broad_req_"):
        return "r" + request_id.rsplit("_", 1)[-1]
    if request_id.startswith("rag_req_"):
        return "r" + request_id.rsplit("_", 1)[-1]
    if request_id.startswith("prefix_req_"):
        parts = request_id.split("_")
        if len(parts) >= 4:
            return f"g{parts[2]}b{parts[3]}"
    if "__req_" in request_id:
        return request_id.replace("group_", "g").replace("__req_", "r")
    return request_id


def paired_rows(oracle_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    baseline_index = {row["request_id"]: row for row in baseline_rows}
    pairs = []
    for oracle in oracle_rows:
        request_id = oracle["request_id"]
        baseline = baseline_index.get(request_id)
        if baseline is None:
            continue
        prompt_tokens = int(oracle.get("prompt_tokens") or baseline.get("prompt_tokens") or 0)
        reuse_tokens = reusable_tokens(oracle)
        prefill_gap = pick_prefill(baseline) - pick_prefill(oracle)
        ttft_gap = pick_ttft(baseline) - pick_ttft(oracle)
        pairs.append(
            {
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "reusable_tokens": reuse_tokens,
                "baseline_prefill_ms": pick_prefill(baseline),
                "oracle0_prefill_ms": pick_prefill(oracle),
                "prefill_saved_ms": prefill_gap,
                "prefill_saved_ms_per_1k_prompt_tokens": (
                    1000.0 * prefill_gap / prompt_tokens if prompt_tokens > 0 else 0.0
                ),
                "prefill_saved_ms_per_1k_reusable_tokens": (
                    1000.0 * prefill_gap / reuse_tokens if reuse_tokens > 0 else 0.0
                ),
                "baseline_ttft_ms": pick_ttft(baseline),
                "oracle0_ttft_ms": pick_ttft(oracle),
                "ttft_saved_ms": ttft_gap,
                "ttft_saved_ms_per_1k_prompt_tokens": (
                    1000.0 * ttft_gap / prompt_tokens if prompt_tokens > 0 else 0.0
                ),
                "ttft_saved_ms_per_1k_reusable_tokens": (
                    1000.0 * ttft_gap / reuse_tokens if reuse_tokens > 0 else 0.0
                ),
                "baseline_num_cached_tokens": baseline.get("num_cached_tokens"),
                "oracle0_num_cached_tokens": oracle.get("num_cached_tokens"),
            }
        )
    return pairs


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bar_saved(rows: list[dict], field: str, out_path: Path, *, title: str, ylabel: str) -> None:
    labels = [short_label(row["request_id"]) for row in rows]
    values = [float(row[field]) for row in rows]
    colors = ["#4c9f70" if value >= 0 else "#c44e52" for value in values]
    plt.figure(figsize=(max(10, len(rows) * 0.22), 5))
    plt.bar(range(len(rows)), values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    step = max(1, len(rows) // 24)
    plt.xticks(
        range(0, len(rows), step),
        [labels[idx] for idx in range(0, len(rows), step)],
        rotation=35,
        ha="right",
    )
    plt.xlabel("Request")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_baseline_vs_oracle(rows: list[dict], metric: str, out_path: Path, *, title: str, ylabel: str) -> None:
    labels = [short_label(row["request_id"]) for row in rows]
    baseline_field = f"baseline_{metric}_ms"
    oracle_field = f"oracle0_{metric}_ms"
    x = list(range(len(rows)))
    plt.figure(figsize=(max(10, len(rows) * 0.25), 5))
    plt.bar([idx - 0.2 for idx in x], [row[baseline_field] for row in rows], width=0.4, label="Baseline", color="#c44e52")
    plt.bar([idx + 0.2 for idx in x], [row[oracle_field] for row in rows], width=0.4, label="Oracle 0", color="#4c72b0")
    step = max(1, len(rows) // 24)
    plt.xticks(
        range(0, len(rows), step),
        [labels[idx] for idx in range(0, len(rows), step)],
        rotation=35,
        ha="right",
    )
    plt.xlabel("Request")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_saved(rows: list[dict], x_field: str, y_field: str, out_path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter([row[x_field] for row in rows], [row[y_field] for row in rows], s=40, alpha=0.75, color="#4c72b0")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary(rows: list[dict], path: Path, *, oracle_root: Path, baseline_root: Path) -> None:
    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0

    summary = {
        "request_count": len(rows),
        "oracle_root": str(oracle_root),
        "baseline_root": str(baseline_root),
        "mean_prefill_saved_ms": mean("prefill_saved_ms"),
        "mean_prefill_saved_ms_per_1k_prompt_tokens": mean("prefill_saved_ms_per_1k_prompt_tokens"),
        "mean_prefill_saved_ms_per_1k_reusable_tokens": mean("prefill_saved_ms_per_1k_reusable_tokens"),
        "mean_ttft_saved_ms": mean("ttft_saved_ms"),
        "mean_ttft_saved_ms_per_1k_prompt_tokens": mean("ttft_saved_ms_per_1k_prompt_tokens"),
        "mean_ttft_saved_ms_per_1k_reusable_tokens": mean("ttft_saved_ms_per_1k_reusable_tokens"),
        "negative_prefill_saved_rows": sum(1 for row in rows if float(row["prefill_saved_ms"]) < 0),
        "negative_ttft_saved_rows": sum(1 for row in rows if float(row["ttft_saved_ms"]) < 0),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    oracle_root = Path(args.oracle_root)
    baseline_root = Path(args.baseline_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = load_jsonl(oracle_root / "oracle0_measurements.jsonl")
    baseline_rows = load_jsonl(baseline_root / "baseline_replay_measurements.jsonl")
    oracle_manifest = load_json(oracle_root / "run_manifest.json")
    params = oracle_manifest.get("parameters") or oracle_manifest
    workload = Path(params["bundle_root"]).name
    system = params["system"]

    rows = paired_rows(oracle_rows, baseline_rows)
    write_csv(rows, output_dir / "oracle0_vs_baseline_per_request.csv")
    write_summary(rows, output_dir / "summary.json", oracle_root=oracle_root, baseline_root=baseline_root)

    title_suffix = f"{workload} | {system}"
    bar_baseline_vs_oracle(
        rows,
        "prefill",
        output_dir / "prefill_oracle0_vs_baseline_by_request.png",
        title=f"Prefill: Baseline vs Oracle 0\n{title_suffix}",
        ylabel="Prefill latency (ms)",
    )
    bar_baseline_vs_oracle(
        rows,
        "ttft",
        output_dir / "ttft_oracle0_vs_baseline_by_request.png",
        title=f"TTFT: Baseline vs Oracle 0\n{title_suffix}",
        ylabel="TTFT (ms)",
    )
    for metric in ("prefill", "ttft"):
        pretty = "Prefill" if metric == "prefill" else "TTFT"
        bar_saved(
            rows,
            f"{metric}_saved_ms",
            output_dir / f"{metric}_saved_ms_by_request.png",
            title=f"{pretty} Saved by Oracle 0 per Request\n{title_suffix}",
            ylabel="Baseline minus Oracle 0 (ms)",
        )
        bar_saved(
            rows,
            f"{metric}_saved_ms_per_1k_prompt_tokens",
            output_dir / f"{metric}_saved_ms_per_1k_prompt_tokens_by_request.png",
            title=f"{pretty} Saved per 1k Prompt Tokens\n{title_suffix}",
            ylabel="Saved ms / 1k prompt tokens",
        )
        bar_saved(
            rows,
            f"{metric}_saved_ms_per_1k_reusable_tokens",
            output_dir / f"{metric}_saved_ms_per_1k_reusable_tokens_by_request.png",
            title=f"{pretty} Saved per 1k Reusable Tokens\n{title_suffix}",
            ylabel="Saved ms / 1k reusable tokens",
        )
        scatter_saved(
            rows,
            "prompt_tokens",
            f"{metric}_saved_ms",
            output_dir / f"{metric}_saved_ms_vs_prompt_tokens.png",
            title=f"{pretty} Saved vs Prompt Tokens\n{title_suffix}",
            xlabel="Prompt tokens",
            ylabel="Baseline minus Oracle 0 (ms)",
        )
        scatter_saved(
            rows,
            "reusable_tokens",
            f"{metric}_saved_ms",
            output_dir / f"{metric}_saved_ms_vs_reusable_tokens.png",
            title=f"{pretty} Saved vs Reusable Tokens\n{title_suffix}",
            xlabel="Reusable tokens",
            ylabel="Baseline minus Oracle 0 (ms)",
        )


if __name__ == "__main__":
    main()
