#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def group_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def plot_lines(
    *,
    rows: list[dict],
    x_key: str,
    y_key: str,
    series_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    series_order: list[str] | None = None,
) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(7.5, 4.5))
    grouped = group_by(rows, series_key)
    order = series_order or sorted(grouped.keys())
    for series in order:
        points = grouped.get(series, [])
        if not points:
            continue
        points = sorted(points, key=lambda row: float(row[x_key]))
        plt.plot(
            [float(row[x_key]) for row in points],
            [float(row[y_key]) for row in points],
            marker="o",
            label=series,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grouped_bars(
    *,
    rows: list[dict],
    category_key: str,
    value_key: str,
    series_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    category_order: list[str] | None = None,
    series_order: list[str] | None = None,
) -> None:
    ensure_dir(output_path.parent)
    categories = category_order or sorted({str(row[category_key]) for row in rows})
    series = series_order or sorted({str(row[series_key]) for row in rows})
    width = 0.8 / max(len(series), 1)
    x_positions = list(range(len(categories)))

    plt.figure(figsize=(8.0, 4.8))
    for idx, series_name in enumerate(series):
        values = []
        for category in categories:
            match = next(
                (
                    row
                    for row in rows
                    if str(row[category_key]) == category
                    and str(row[series_key]) == series_name
                ),
                None,
            )
            values.append(float(match[value_key]) if match is not None else 0.0)
        offsets = [x + (idx - (len(series) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=series_name)

    plt.xticks(x_positions, categories)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
