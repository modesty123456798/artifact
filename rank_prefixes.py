#!/usr/bin/env python3
"""Rank LLM-positive candidates from a Full Isolation Join run at chosen prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

if __package__:
    from .datasets import DATASETS, ROOT, absolute, dataset, spec_path
else:  # Supports direct script execution from the repository root.
    from datasets import DATASETS, ROOT, absolute, dataset, spec_path

sys.path.insert(0, str(ROOT / "scripts"))
import llm_ranking_engine as engine  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decided-points", type=int, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default="", help="Falls back to OPENAI_API_KEY.")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--compact-jsonl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if any(point <= 0 for point in args.decided_points):
        parser.error("--decided-points values must be positive")
    if len(set(args.decided_points)) != len(args.decided_points):
        parser.error("--decided-points contains duplicates")
    if args.retry_limit <= 0 or args.retry_sleep_seconds < 0:
        parser.error("Invalid retry settings")
    return args


def main() -> None:
    args = parse_args()
    config = dataset(args.dataset)
    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    for path in (source_dir, absolute(config, config["left_json"]), absolute(config, config["right_json"])):
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    probe_order = engine.load_probe_order(source_dir)
    if max(args.decided_points) > len(probe_order):
        raise SystemExit(
            f"Requested prefix {max(args.decided_points):,} exceeds "
            f"{len(probe_order):,} probed pairs in {source_dir}."
        )
    left_texts = engine.load_indexed_text(absolute(config, config["left_json"]), config["left_key"])
    right_texts = engine.load_indexed_text(absolute(config, config["right_json"]), config["right_key"])
    for point in args.decided_points:
        prefix = probe_order.iloc[:point]
        positives = prefix[prefix["rpe_match"].eq(1)]
        print(
            f"prefix={point:,} | LLM-positive candidates={len(positives):,} | "
            f"lefts with positives={positives['_left_id'].nunique():,}"
        )
    if args.validate_only:
        return

    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")
    client = None
    if not args.dry_run:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if args.base_url.strip():
            client_kwargs["base_url"] = args.base_url.strip()
        client = OpenAI(**client_kwargs)

    system_path = spec_path(config, "prompts/ranking_system.txt")
    user_path = spec_path(config, "prompts/ranking_user_template.txt")
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "dataset": args.dataset,
        "source_dir": str(source_dir),
        "decided_points": args.decided_points,
        "top_k": args.top_k,
        "model": args.model,
        "temperature": 0,
        "system_prompt_file": str(system_path),
        "system_prompt_sha256": sha256_file(system_path),
        "user_prompt_file": str(user_path),
        "user_prompt_sha256": sha256_file(user_path),
        "strict_json_schema": True,
        "schema_name": config["ranking_schema"],
    }
    (output_root / "ranking_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    engine_args = SimpleNamespace(
        source_dir=source_dir,
        output_root=output_root,
        top_k=args.top_k,
        model=args.model,
        system_prompt=system_path.read_text(encoding="utf-8").rstrip("\n"),
        prompt_template=user_path.read_text(encoding="utf-8").rstrip("\n"),
        left_label=config["left_label"],
        right_label=config["right_label"],
        schema_name=config["ranking_schema"],
        compact_jsonl=args.compact_jsonl,
        dry_run=args.dry_run,
        retry_limit=args.retry_limit,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    for point in args.decided_points:
        engine.run_prefix(engine_args, probe_order, left_texts, right_texts, point, client)


if __name__ == "__main__":
    main()
