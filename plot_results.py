#!/usr/bin/env python3
"""Plot progression and ranking outputs from one Isolation Join artifact run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from datasets import DATASETS


FIGURE_SIZE = (7.5, 6.0)  # 1 : 0.8, matching the paper-figure aspect ratio.
OURS_COLOR = "#c62828"
RANKING_COLORS = ["#c62828", "#f4c542", "#2e8b57", "#2563eb", "#7c3aed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-root>/plots.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def configure_axes(ax: plt.Axes, x_label: str, y_label: str) -> None:
    ax.set_xlabel(x_label, fontsize=14, labelpad=5)
    ax.set_ylabel(y_label, fontsize=14, labelpad=7)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_probe_progression(run_root: Path, output_dir: Path) -> Path:
    metrics_path = run_root / "probe" / "anti_dominance_prompting_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Probe metrics not found: {metrics_path}")

    by_decided: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(metrics_path):
        decided = int(record.get("decided_points") or 0)
        if decided > 0:
            by_decided[decided] = record
    if not by_decided:
        raise ValueError(f"No probe checkpoints found in {metrics_path}")

    checkpoints = [by_decided[key] for key in sorted(by_decided)]
    x_values = [int(record["decided_points"]) for record in checkpoints]
    y_values = [int(record["true_positive_points"]) for record in checkpoints]

    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        x_values,
        y_values,
        color=OURS_COLOR,
        linewidth=2.8,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="Isolation Join",
    )
    configure_axes(ax, "Probed Pairs", "True Positive Pairs")
    ax.legend(frameon=False, fontsize=12, loc="best")
    figure.tight_layout()
    output_path = output_dir / "true_positive_pairs_vs_probed_pairs.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def ranking_precision_inputs(run_root: Path) -> pd.DataFrame:
    pattern = re.compile(r"ranking_decided(?P<decided>\d+)_top\d+_openai")
    frames: list[pd.DataFrame] = []
    for path in sorted((run_root / "ranking").glob("ranking_decided*_top*_openai/ranking_precision.csv")):
        match = pattern.fullmatch(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        required = {"k", "ranking_precision_at_k_percent"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Ranking precision CSV has unexpected columns: {path}")
        frame = frame[["k", "ranking_precision_at_k_percent"]].copy()
        frame["decided_points"] = int(match.group("decided"))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["k", "ranking_precision_at_k_percent", "decided_points"])
    return pd.concat(frames, ignore_index=True).sort_values(["k", "decided_points"])


def plot_ranking_precision(run_root: Path, output_dir: Path) -> Path | None:
    results = ranking_precision_inputs(run_root)
    if results.empty:
        return None

    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    for index, (k_value, group) in enumerate(results.groupby("k", sort=True)):
        group = group.sort_values("decided_points")
        color = RANKING_COLORS[index % len(RANKING_COLORS)]
        ax.plot(
            group["decided_points"],
            group["ranking_precision_at_k_percent"],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=f"RP@{int(k_value)}",
        )
    configure_axes(ax, "Probed Pairs", "Ranking Precision (%)")
    ax.legend(frameon=False, fontsize=11, loc="best", ncol=2)
    figure.tight_layout()
    output_path = output_dir / "ranking_precision_vs_probed_pairs.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")
    output_dir = (args.output_dir or run_root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_plot = plot_probe_progression(run_root, output_dir)
    print(f"Wrote {probe_plot}")
    ranking_plot = plot_ranking_precision(run_root, output_dir)
    if ranking_plot is None:
        print("No ranking_precision.csv files found; skipped ranking plot.")
    else:
        print(f"Wrote {ranking_plot}")


if __name__ == "__main__":
    main()
