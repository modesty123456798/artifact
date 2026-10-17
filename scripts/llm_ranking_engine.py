#!/usr/bin/env python3
"""Rank LLM-positive candidates at selected probing prefixes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the openai package with `pip install openai`.") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "runs/dataset/probe"
DEFAULT_PREFIXES = [2_500, 6_250, 10_000, 12_500, 13_750]

SYSTEM_PROMPT = (
    "The user will provide one European Union legal document and multiple candidate "
    "EuroVoc concepts. Rank the candidate EuroVoc concepts by how likely each concept "
    "is described in the legal document, from most likely to least likely. Consider "
    "the document's context and semantic meaning, not only exact word overlap. Use "
    "only the provided candidate IDs, do not repeat IDs, and follow the requested "
    "JSON output format exactly."
)

PROMPT_TEMPLATE = (
    "Given the following {left_label}, rank the Candidate {right_label}s by the "
    "likelihood that each concept is described in the {left_label}, "
    "ordered from most likely to least likely."
    "\n\n"
    "{left_label}:\n"
    "«{left_text}»\n\n"
    "Candidate {right_label}s:\n"
    "{candidate_list}\n\n"
    "Return JSON with a single field ranked_list containing only Candidate IDs, "
    "ordered from most likely to least likely. Return exactly {expected_count} IDs. "
    "Use only the provided Candidate IDs and do not repeat or invent IDs."
)

INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"{path} must contain a JSON list of objects.")
    return records


def load_indexed_text(path: Path, key: str) -> dict[int, str]:
    records = load_json_list(path)
    missing = [index for index, record in enumerate(records) if key not in record]
    if missing:
        raise ValueError(f"{path} is missing key {key!r} at rows {missing[:10]}.")
    return {
        index: str(record.get(key) or "") for index, record in enumerate(records)
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def load_probe_order(source_dir: Path) -> pd.DataFrame:
    judgments = pd.read_csv(
        source_dir / "pairs_anti_dominance_llm_results.csv",
        usecols=["_left_id", "_right_id", "rpe_match"],
    )
    pair_columns = ["_left_id", "_right_id"]
    if judgments.duplicated(pair_columns).any():
        raise ValueError("Judgment checkpoint contains duplicate pairs.")
    judgments[pair_columns] = judgments[pair_columns].astype(int)
    judgments["rpe_match"] = pd.to_numeric(
        judgments["rpe_match"], errors="raise"
    ).astype(int)

    ordered_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    response_path = source_dir / "llm_anti_dominance_prompting_responses.jsonl"
    for record in load_jsonl(response_path):
        if "left_id" not in record or "right_ids" not in record:
            continue
        left_id = int(record["left_id"])
        for value in record.get("right_ids") or []:
            pair = (left_id, int(value))
            if pair in seen:
                raise ValueError(f"Duplicate response pair encountered: {pair}")
            seen.add(pair)
            ordered_pairs.append(pair)

    ordered = pd.DataFrame(ordered_pairs, columns=pair_columns)
    if len(ordered) != len(judgments):
        raise ValueError(
            f"Response JSONL has {len(ordered):,} pairs but checkpoint has "
            f"{len(judgments):,}."
        )
    ordered = ordered.merge(
        judgments, on=pair_columns, how="left", validate="one_to_one", sort=False
    )
    if ordered["rpe_match"].isna().any():
        raise ValueError("Response order contains pairs absent from the checkpoint.")
    ordered.insert(0, "probe_index", range(1, len(ordered) + 1))
    return ordered


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    records = load_json_list(path)
    return {
        int(record["left_id"]): record
        for record in records
        if record.get("record_type") == "ranking_result"
        and record.get("success") is True
    }


def usage_from_completion(completion: Any) -> dict[str, int]:
    usage = completion.model_dump().get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_prompt_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def call_ranker(
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    candidate_ids: Sequence[str],
    expected_count: int,
    schema_name: str,
) -> tuple[list[str], str, dict[str, Any], dict[str, int]]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_list": {
                            "type": "array",
                            "minItems": expected_count,
                            "maxItems": expected_count,
                            "items": {
                                "type": "string",
                                "enum": list(candidate_ids),
                            },
                        }
                    },
                    "required": ["ranked_list"],
                    "additionalProperties": False,
                },
            },
        },
    )
    response_text = completion.choices[0].message.content or ""
    parsed = json.loads(response_text)
    ranked = [str(value) for value in parsed.get("ranked_list") or []]
    if len(ranked) != expected_count:
        raise ValueError(
            f"Expected {expected_count} ranked IDs, received {len(ranked)}."
        )
    if len(set(ranked)) != len(ranked) or not set(ranked) <= set(candidate_ids):
        raise ValueError("Ranking contains duplicate or invalid candidate IDs.")
    return ranked, response_text, completion.model_dump(), usage_from_completion(
        completion
    )


def fair_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * INPUT_PRICE_PER_MILLION
        + completion_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000


def run_prefix(
    args: argparse.Namespace,
    probe_order: pd.DataFrame,
    left_texts: dict[int, str],
    right_texts: dict[int, str],
    decided_points: int,
    client: Optional[OpenAI],
) -> None:
    if decided_points > len(probe_order):
        raise ValueError(
            f"Requested prefix {decided_points:,} exceeds the available "
            f"{len(probe_order):,} judged pairs."
        )
    output_dir = args.output_root / (
        f"ranking_decided{decided_points}_top{args.top_k}_openai"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "ranked_pairs_with_matches.json"
    output_jsonl = output_dir / "ranked_pairs_with_matches.jsonl"
    summary_json = output_dir / "ranking_run_summary.json"

    prefix = probe_order.iloc[:decided_points].copy()
    positives = prefix[prefix["rpe_match"].eq(1)]
    grouped = positives.groupby("_left_id", sort=True)["_right_id"].apply(list)
    existing = load_existing(output_json)
    results = list(existing.values())
    processed = set(existing)
    prompt_tokens = sum(
        int(record.get("prompt_tokens") or 0) for record in existing.values()
    )
    cached_prompt_tokens = sum(
        int(record.get("cached_prompt_tokens") or 0)
        for record in existing.values()
    )
    completion_tokens = sum(
        int(record.get("completion_tokens") or 0) for record in existing.values()
    )
    api_calls = sum(int(record.get("api_calls") or 0) for record in existing.values())

    with output_jsonl.open("a", encoding="utf-8") as jsonl_handle:
        for left_id in sorted(left_texts):
            if left_id in processed:
                continue
            right_ids = list(dict.fromkeys(grouped.get(left_id, [])))
            candidate_ids = [str(index) for index in range(1, len(right_ids) + 1)]
            candidate_texts = [right_texts[right_id] for right_id in right_ids]
            expected_count = min(args.top_k, len(candidate_ids))
            candidate_id_to_right_id = dict(zip(candidate_ids, right_ids))
            candidate_list = "\n".join(
                f"- {candidate_id}: {candidate_text}"
                for candidate_id, candidate_text in zip(
                    candidate_ids, candidate_texts
                )
            )
            prompt = args.prompt_template.format(
                left_label=args.left_label,
                right_label=args.right_label,
                left_text=left_texts[left_id],
                candidate_list=candidate_list,
                top_k=args.top_k,
                expected_count=expected_count,
            )

            ranked_ids = candidate_ids[:expected_count]
            response_text = ""
            raw_response = None
            usage = {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
            }
            attempts_used = 0
            api_calls_for_left = 0
            success = expected_count <= 1 or args.dry_run
            error = None
            if expected_count > 1 and not args.dry_run:
                for attempt in range(1, args.retry_limit + 1):
                    attempts_used = attempt
                    api_calls_for_left += 1
                    api_calls += 1
                    try:
                        ranked_ids, response_text, raw_response, call_usage = call_ranker(
                            client,
                            args.model,
                            args.system_prompt,
                            prompt,
                            candidate_ids,
                            expected_count,
                            args.schema_name,
                        )
                        for key, value in call_usage.items():
                            usage[key] += int(value)
                        success = True
                        error = None
                        break
                    except Exception as exc:  # pragma: no cover - API failures
                        error = str(exc)
                        if attempt < args.retry_limit and args.retry_sleep_seconds > 0:
                            time.sleep(args.retry_sleep_seconds)

            ranked_right_ids = [
                candidate_id_to_right_id[candidate_id]
                for candidate_id in ranked_ids
            ]
            record = {
                "record_type": "ranking_result",
                "source_directory": str(args.source_dir),
                "decided_points_prefix": decided_points,
                "top_k": args.top_k,
                "left_id": left_id,
                "candidate_ids": candidate_ids,
                "candidate_right_ids": right_ids,
                "candidate_concepts": candidate_texts,
                "ranked_list": ranked_ids,
                "ranked_right_ids": ranked_right_ids,
                "ranked_concepts": [right_texts[value] for value in ranked_right_ids],
                "success": success,
                "error": error,
                "attempts_used": attempts_used,
                "api_calls": api_calls_for_left,
                "prompt_tokens": usage["prompt_tokens"],
                "cached_prompt_tokens": usage["cached_prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "model": args.model,
                "system_prompt": args.system_prompt,
                "prompt": prompt,
                "response_text": response_text,
                "timestamp": time.time(),
            }
            if raw_response is not None and not args.compact_jsonl:
                record["raw_response"] = raw_response
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            results.append(record)
            processed.add(left_id)
            prompt_tokens += usage["prompt_tokens"]
            cached_prompt_tokens += usage["cached_prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            atomic_write_json(output_json, results)
            print(
                f"prefix={decided_points:,} | left_id={left_id} | "
                f"candidates={len(right_ids)} | ranked={len(ranked_right_ids)} | "
                f"success={success}"
            )

    summary = {
        "source_directory": str(args.source_dir),
        "decided_points_prefix": decided_points,
        "top_k": args.top_k,
        "left_count": len(left_texts),
        "lefts_with_positive_candidates": int(grouped.index.nunique()),
        "total_positive_candidates": int(len(positives)),
        "completed_ranking_records": len(results),
        "successful_ranking_records": sum(bool(x.get("success")) for x in results),
        "api_calls": api_calls,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "fair_cost_usd": fair_cost(prompt_tokens, completion_tokens),
        "dry_run": args.dry_run,
    }
    atomic_write_json(summary_json, summary)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--decided-points",
        type=int,
        nargs="+",
        default=DEFAULT_PREFIXES,
        help="One or more probing prefixes to rank.",
    )
    parser.add_argument("--top-k", type=int, default=11)
    parser.add_argument(
        "--left-json",
        type=Path,
        default=Path("EURLEX57K/combined_data/eurlex_legal_document_sampled_250.json"),
    )
    parser.add_argument(
        "--right-json",
        type=Path,
        default=Path("EURLEX57K/combined_data/eurovoc_concept.json"),
    )
    parser.add_argument("--left-key", default="eurlex_legal_document")
    parser.add_argument("--right-key", default="eurovoc_concept")
    parser.add_argument("--left-label", default="European Union Legal Document")
    parser.add_argument("--right-label", default="EuroVoc Concept")
    parser.add_argument("--schema-name", default="eurovoc_concept_ranking")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--compact-jsonl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to the source experiment directory.",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.decided_points):
        parser.error("Every --decided-points value must be positive.")
    if len(set(args.decided_points)) != len(args.decided_points):
        parser.error("--decided-points contains duplicates.")
    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if args.retry_limit <= 0:
        parser.error("--retry-limit must be positive.")
    if args.retry_sleep_seconds < 0:
        parser.error("--retry-sleep-seconds must be non-negative.")

    args.source_dir = resolve_path(args.source_dir)
    args.left_json = resolve_path(args.left_json)
    args.right_json = resolve_path(args.right_json)
    args.output_root = (
        resolve_path(args.output_root) if args.output_root else args.source_dir
    )
    if not args.dry_run:
        api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Use --api-key or set OPENAI_API_KEY.")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if args.base_url.strip():
            client_kwargs["base_url"] = args.base_url.strip()
        args.client = OpenAI(**client_kwargs)
    else:
        args.client = None
    return args


def main() -> None:
    args = parse_args()
    for path in (args.source_dir, args.left_json, args.right_json):
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    probe_order = load_probe_order(args.source_dir)
    left_texts = load_indexed_text(args.left_json, args.left_key)
    right_texts = load_indexed_text(args.right_json, args.right_key)
    for decided_points in args.decided_points:
        run_prefix(
            args,
            probe_order,
            left_texts,
            right_texts,
            decided_points,
            args.client,
        )


if __name__ == "__main__":
    main()
