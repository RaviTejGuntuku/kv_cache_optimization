#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot concurrency-tension pilot outputs.")
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_pilot_root(path: Path) -> Path:
    direct_manifest = path / "run_manifest.json"
    if direct_manifest.exists():
        return path
    candidates = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "run_manifest.json").exists()
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"Could not find concurrency pilot outputs under {path}"
    )


def _short_label(request_id: str) -> str:
    parts = request_id.split("_")
    if len(parts) >= 4 and parts[0] == "prefix" and parts[1] == "req":
        return f"g{parts[2]}"
    if len(parts) >= 3 and parts[0] == "broad" and parts[1] == "req":
        return f"r{parts[2]}"
    return request_id


def _by_level(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(int(row["concurrency"]), []).append(row)
    return out


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def plot_mean_prefill_vs_concurrency(baseline_rows: list[dict], oracle_rows: list[dict], out_path: Path) -> None:
    base = _by_level(baseline_rows)
    oracle = _by_level(oracle_rows)
    levels = sorted(set(base) & set(oracle))
    base_means = [_mean([float(row["prefill_time_ms"]) for row in base[level] if row["prefill_time_ms"] is not None]) for level in levels]
    oracle_means = [_mean([float(row["prefill_time_ms"]) for row in oracle[level] if row["prefill_time_ms"] is not None]) for level in levels]
    plt.figure(figsize=(8, 5))
    plt.plot(levels, base_means, marker="o", linewidth=2, color="#c44e52", label="Baseline")
    plt.plot(levels, oracle_means, marker="o", linewidth=2, color="#4c72b0", label="Oracle 0")
    plt.xlabel("Concurrent prefills in batch")
    plt.ylabel("Mean target-request prefill latency (ms)")
    plt.title("Target Prefill Latency vs Prefill Concurrency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_gap_vs_concurrency(baseline_rows: list[dict], oracle_rows: list[dict], out_path: Path) -> None:
    oracle_idx = {(int(row["concurrency"]), row["target_request_id"]): row for row in oracle_rows}
    grouped: dict[int, list[float]] = {}
    for base_row in baseline_rows:
        key = (int(base_row["concurrency"]), base_row["target_request_id"])
        oracle_row = oracle_idx.get(key)
        if oracle_row is None:
            continue
        if base_row["prefill_time_ms"] is None or oracle_row["prefill_time_ms"] is None:
            continue
        grouped.setdefault(key[0], []).append(float(base_row["prefill_time_ms"]) - float(oracle_row["prefill_time_ms"]))
    levels = sorted(grouped)
    means = [_mean(grouped[level]) for level in levels]
    plt.figure(figsize=(8, 5))
    plt.bar(levels, means, width=0.6, color="#55a868")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Concurrent prefills in batch")
    plt.ylabel("Mean baseline minus Oracle 0 gap (ms)")
    plt.title("Mean Oracle 0 Headroom vs Prefill Concurrency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_request_level_gaps(baseline_rows: list[dict], oracle_rows: list[dict], out_path: Path) -> None:
    oracle_idx = {(int(row["concurrency"]), row["target_request_id"]): row for row in oracle_rows}
    labels = []
    gaps = []
    colors = []
    for base_row in sorted(baseline_rows, key=lambda row: (int(row["concurrency"]), row["target_request_id"])):
        key = (int(base_row["concurrency"]), base_row["target_request_id"])
        oracle_row = oracle_idx.get(key)
        if oracle_row is None or base_row["prefill_time_ms"] is None or oracle_row["prefill_time_ms"] is None:
            continue
        labels.append(f"n={key[0]} | {_short_label(key[1])}")
        gap = float(base_row["prefill_time_ms"]) - float(oracle_row["prefill_time_ms"])
        gaps.append(gap)
        colors.append("#55a868" if gap >= 0 else "#8172b3")
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(labels)), gaps, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.xlabel("Target request under batch concurrency")
    plt.ylabel("Baseline minus Oracle 0 prefill (ms)")
    plt.title("Request-Level Oracle Gap Under Concurrent Prefill")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_marginal_gain_vs_concurrency(marginal_rows: list[dict], out_path: Path) -> None:
    grouped: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    for row in marginal_rows:
        if not row.get("was_missed_in_baseline"):
            continue
        gain = row.get("marginal_gain_ms")
        if gain is None:
            continue
        grouped[(int(row["concurrency"]), str(row["object_type"]))].append(float(gain))
    if not grouped:
        return
    levels = sorted({level for level, _ in grouped})
    object_types = sorted({obj_type for _, obj_type in grouped})
    width = 0.8 / max(1, len(object_types))
    xs = list(range(len(levels)))
    plt.figure(figsize=(9, 5))
    for idx, object_type in enumerate(object_types):
        values = [
            _mean(grouped.get((level, object_type), []))
            for level in levels
        ]
        offsets = [x + (idx - (len(object_types) - 1) / 2.0) * width for x in xs]
        plt.bar(offsets, values, width=width, label=object_type)
    plt.xticks(xs, [str(level) for level in levels])
    plt.xlabel("Concurrent prefills in batch")
    plt.ylabel("Mean marginal gain over missed rows (ms)")
    plt.title("Marginal Gain vs Prefill Concurrency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_total_marginal_gain_vs_concurrency(marginal_rows: list[dict], out_path: Path) -> None:
    grouped: dict[tuple[int, str], float] = collections.defaultdict(float)
    for row in marginal_rows:
        if not row.get("was_missed_in_baseline"):
            continue
        gain = row.get("marginal_gain_ms")
        if gain is None:
            continue
        grouped[(int(row["concurrency"]), str(row["object_type"]))] += max(0.0, float(gain))
    if not grouped:
        return
    levels = sorted({level for level, _ in grouped})
    object_types = sorted({obj_type for _, obj_type in grouped})
    width = 0.8 / max(1, len(object_types))
    xs = list(range(len(levels)))
    plt.figure(figsize=(9, 5))
    for idx, object_type in enumerate(object_types):
        values = [
            grouped.get((level, object_type), 0.0)
            for level in levels
        ]
        offsets = [x + (idx - (len(object_types) - 1) / 2.0) * width for x in xs]
        plt.bar(offsets, values, width=width, label=object_type)
    plt.xticks(xs, [str(level) for level in levels])
    plt.xlabel("Concurrent prefills in batch")
    plt.ylabel("Total marginal gain over missed rows (ms)")
    plt.title("Total Marginal Gain vs Prefill Concurrency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    pilot_root = resolve_pilot_root(Path(args.pilot_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_jsonl(pilot_root / "baseline_batch_measurements.jsonl")
    oracle_rows = load_jsonl(pilot_root / "oracle0_batch_measurements.jsonl")
    marginal_rows = load_jsonl(pilot_root / "marginal_counterfactuals_batch.jsonl") if (pilot_root / "marginal_counterfactuals_batch.jsonl").exists() else []
    manifest = load_json(pilot_root / "run_manifest.json")

    plot_mean_prefill_vs_concurrency(
        baseline_rows,
        oracle_rows,
        output_dir / "mean_target_prefill_vs_concurrency.png",
    )
    plot_mean_gap_vs_concurrency(
        baseline_rows,
        oracle_rows,
        output_dir / "mean_oracle_gap_vs_concurrency.png",
    )
    plot_request_level_gaps(
        baseline_rows,
        oracle_rows,
        output_dir / "request_level_gap_under_concurrency.png",
    )
    if marginal_rows:
        plot_mean_marginal_gain_vs_concurrency(
            marginal_rows,
            output_dir / "mean_marginal_gain_vs_concurrency.png",
        )
        plot_total_marginal_gain_vs_concurrency(
            marginal_rows,
            output_dir / "total_marginal_gain_vs_concurrency.png",
        )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Concurrency Tension Pilot",
                "",
                f"- system: `{manifest['system']}`",
                f"- workload: `{Path(manifest['bundle_root']).name}`",
                f"- concurrency levels: `{', '.join(str(item) for item in manifest['concurrency_levels'])}`",
                f"- target requests per level: `{manifest['target_count_per_level']}`",
                f"- marginal rows: `{len(marginal_rows)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
