#!/usr/bin/env python3
"""Compute mean ranking precision@k from ranking outputs with right-table IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-json", type=Path, required=True)
    parser.add_argument("--ground-truth-json", type=Path, required=True)
    parser.add_argument("--ground-truth-key", required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--right-text-key", required=True)
    parser.add_argument("--ks", type=int, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if any(k <= 0 for k in args.ks):
        parser.error("All --ks values must be positive.")

    ground_truth = {
        int(row.get("id", index)): {normalize(value) for value in row.get(args.ground_truth_key, [])}
        for index, row in enumerate(load_rows(args.ground_truth_json))
    }
    right_texts = {
        index: normalize(row[args.right_text_key])
        for index, row in enumerate(load_rows(args.right_json))
    }
    records = {
        int(row["left_id"]): row
        for row in load_rows(args.ranked_json)
        if row.get("record_type") == "ranking_result" and row.get("success") is True
    }
    missing = set(ground_truth) - set(records)
    if missing:
        raise ValueError(f"Missing successful ranking records for {len(missing)} left records.")

    rows = []
    for k in sorted(set(args.ks)):
        values = []
        for left_id, gt in ground_truth.items():
            if not gt:
                raise ValueError(f"left_id={left_id} has no ground-truth labels.")
            ranked_ids = [int(value) for value in records[left_id].get("ranked_right_ids", [])[:k]]
            hits = sum(right_texts[right_id] in gt for right_id in ranked_ids)
            values.append(hits / min(len(gt), k))
        rows.append({
            "k": k,
            "ranking_precision_at_k": sum(values) / len(values),
            "ranking_precision_at_k_percent": 100 * sum(values) / len(values),
            "left_count": len(values),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
