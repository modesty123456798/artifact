#!/usr/bin/env python3
"""Generate the two mapped left representations for one Full Isolation Join dataset.

This script intentionally owns only the LLM mapping stages. The companion
``run_pipeline.py`` invokes the dataset-specific embedding scorer afterwards.
Every mapping call is checkpointed by left-record ID, so interrupted runs can
resume without repeating completed calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

if __package__:
    from .datasets import DATASETS, absolute, dataset, spec_path
else:  # Supports direct script execution from the repository root.
    from datasets import DATASETS, absolute, dataset, spec_path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_rows(path: Path, text_key: str) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    invalid = [
        index
        for index, row in enumerate(rows)
        if not isinstance(row, dict) or not str(row.get(text_key) or "").strip()
    ]
    if invalid:
        raise ValueError(f"Missing {text_key!r} in {path} at rows {invalid[:10]}")
    return rows


def load_completed(path: Path, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("config") != config:
                raise ValueError(
                    f"Checkpoint config mismatch at {path}:{line_number}; "
                    "start a new output directory for a different configuration."
                )
            completed[int(record["left_id"])] = record
    return completed


def call_chat(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
    retry_limit: int,
    retry_sleep_seconds: float,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retry_limit + 1):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                request["max_tokens"] = max_output_tokens
            return client.chat.completions.create(**request)
        except Exception as exc:  # pragma: no cover - network/API failures
            last_error = exc
            if attempt >= retry_limit:
                break
            delay = retry_sleep_seconds * (2**attempt)
            print(f"Mapping request failed ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Chat Completions request exhausted retries") from last_error


def usage_record(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def read_saved_guide(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, str):
        guide = value
    elif isinstance(value, dict):
        guide = value.get("cluster_example_summary") or value.get("guide")
    else:
        guide = None
    if not isinstance(guide, str) or not guide.strip():
        raise ValueError(f"No guide text in {path}")
    return guide.strip()


def exemplar_text(path: Path, top_n: int) -> tuple[str, int]:
    clusters = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(clusters, list) or not clusters:
        raise ValueError(f"Expected a non-empty cluster list: {path}")
    values: list[str] = []
    for cluster in clusters:
        if isinstance(cluster, list):
            values.extend(str(value) for value in cluster[:top_n] if str(value).strip())
        elif isinstance(cluster, dict):
            for value in cluster.values():
                if isinstance(value, str) and value.strip():
                    values.append(value)
                    break
        elif str(cluster).strip():
            values.append(str(cluster))
    if not values:
        raise ValueError(f"No usable exemplars in {path}")
    return "\n".join(f"- {value}" for value in values), len(values)


def abt_json_to_text(raw: str, row_index: int) -> str:
    """Match ABT's historical JSON-to-target-style-text postprocessing."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ABT mapped JSON at row {row_index}: {exc}") from exc
    expected = {"name", "description", "price"}
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"Unexpected ABT mapped JSON at row {row_index}: {actual}")
    return ", ".join(
        f"{json.dumps(key)}:{json.dumps('' if value[key] is None else str(value[key]), ensure_ascii=False)}"
        for key in ("name", "description", "price")
    )


def map_rows(
    *,
    client: OpenAI,
    rows: list[dict[str, Any]],
    source_key: str,
    output_key: str,
    template: str,
    substitutions: dict[str, str],
    model: str,
    temperature: float,
    max_output_tokens: int | None,
    max_input_characters: int | None,
    output_dir: Path,
    stage: str,
    retry_limit: int,
    retry_sleep_seconds: float,
    postprocess: str | None,
) -> tuple[Path, list[dict[str, int]]]:
    checkpoint_path = output_dir / f"{stage}_checkpoint.jsonl"
    raw_path = output_dir / f"{stage}_mapped_raw.json"
    output_path = output_dir / f"{stage}_mapped.json"
    config = {
        "stage": stage,
        "source_key": source_key,
        "output_key": output_key,
        "template_sha256": sha256_bytes(template.encode("utf-8")),
        "substitution_sha256": {
            key: sha256_bytes(value.encode("utf-8")) for key, value in substitutions.items()
        },
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_input_characters": max_input_characters,
        "postprocess": postprocess,
        "message_roles": ["user"],
        "system_message": None,
    }
    completed = load_completed(checkpoint_path, config)
    print(f"{stage}: loaded {len(completed):,}/{len(rows):,} mapping checkpoints")
    usage_rows: list[dict[str, int]] = []
    usage_path = output_dir / f"{stage}_usage.jsonl"

    for left_id, row in enumerate(rows):
        source_text = str(row[source_key])
        source_hash = sha256_bytes(source_text.encode("utf-8"))
        prior = completed.get(left_id)
        if prior is not None:
            if prior.get("source_sha256") != source_hash:
                raise ValueError(f"Source text changed for {stage} row {left_id}")
            continue
        submitted = source_text
        if max_input_characters is not None and max_input_characters > 0:
            submitted = submitted[:max_input_characters]
        prompt = template.format(product_text=submitted, **substitutions)
        response = call_chat(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retry_limit=retry_limit,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        output = (response.choices[0].message.content or "").strip()
        if not output:
            raise RuntimeError(f"Empty mapping response at {stage} row {left_id}")
        record = {
            "left_id": left_id,
            "source_sha256": source_hash,
            "output": output,
            "config": config,
        }
        append_jsonl(checkpoint_path, record)
        completed[left_id] = record
        usage = usage_record(response)
        usage_rows.append(usage)
        append_jsonl(
            usage_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "left_id": left_id,
                "response_model": getattr(response, "model", None),
                **usage,
            },
        )
        print(f"{stage}: mapped {left_id + 1:,}/{len(rows):,}")

    raw_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for left_id, row in enumerate(rows):
        raw = str(completed[left_id]["output"])
        raw_row = dict(row)
        raw_row[output_key] = raw
        raw_rows.append(raw_row)
        output_row = dict(raw_row)
        if postprocess == "abt_product_json_to_text":
            output_row[output_key] = abt_json_to_text(raw, left_id)
        output_rows.append(output_row)
    atomic_json(raw_path, raw_rows)
    atomic_json(output_path, output_rows)
    return output_path, usage_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--guide-source",
        choices=["auto", "saved", "regenerate"],
        default="regenerate",
        help=(
            "Regenerate the guide from archived prompt and exemplars (default), "
            "or use the preserved historical guide. 'auto' is retained as an "
            "alias for regeneration."
        ),
    )
    parser.add_argument("--api-key", default="", help="Falls back to OPENAI_API_KEY.")
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.retry_limit < 0 or args.retry_sleep_seconds < 0:
        raise ValueError("Retry values must be non-negative")
    config = dataset(args.dataset)
    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    left_path = absolute(config, config["left_json"])
    rows = load_rows(left_path, config["left_key"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)
    spec_dir = absolute(config, config["spec_dir"])

    no_context = config["no_context"]
    no_context_template = (spec_dir / no_context["prompt"]).read_text(encoding="utf-8").rstrip("\n")
    no_context_path, no_context_usage = map_rows(
        client=client,
        rows=rows,
        source_key=config["left_key"],
        output_key=no_context["output_key"],
        template=no_context_template,
        substitutions={},
        model=no_context["model"],
        temperature=float(no_context["temperature"]),
        max_output_tokens=no_context.get("max_output_tokens"),
        max_input_characters=no_context.get("max_input_characters"),
        output_dir=args.output_dir,
        stage="no_context",
        retry_limit=args.retry_limit,
        retry_sleep_seconds=args.retry_sleep_seconds,
        postprocess=None,
    )

    target_aware = config["target_aware"]
    guide_prompt_path = spec_dir / target_aware["guide_prompt"]
    guide_template = guide_prompt_path.read_text(encoding="utf-8").rstrip("\n")
    guide_path = args.output_dir / "target_guide.json"
    saved_guide = target_aware.get("saved_guide")
    guide_source = args.guide_source
    if guide_source == "auto":
        guide_source = "regenerate"
    if guide_source == "saved":
        if not saved_guide:
            raise SystemExit(
                f"{config['display_name']} has no persisted historical guide; use --guide-source regenerate."
            )
        saved_path = absolute(config, saved_guide)
        guide = read_saved_guide(saved_path)
        guide_record = {
            "source": "saved",
            "saved_guide_path": str(saved_path.resolve()),
            "saved_guide_sha256": sha256_file(saved_path),
            "guide": guide,
        }
        atomic_json(guide_path, guide_record)
        guide_usage: list[dict[str, int]] = []
    else:
        exemplar_path = absolute(config, target_aware["right_exemplars"])
        examples, exemplar_count = exemplar_text(exemplar_path, 1)
        guide_config = {
            "source": "regenerated",
            "model": target_aware["model"],
            "temperature": target_aware["guide_temperature"],
            "guide_prompt_sha256": sha256_file(guide_prompt_path),
            "right_exemplars_path": str(exemplar_path.resolve()),
            "right_exemplars_sha256": sha256_file(exemplar_path),
            "exemplar_count": exemplar_count,
        }
        if guide_path.exists():
            guide_record = json.loads(guide_path.read_text(encoding="utf-8"))
            if {key: guide_record.get(key) for key in guide_config} != guide_config:
                raise ValueError(
                    f"Existing regenerated guide has a different configuration: {guide_path}"
                )
            guide = str(guide_record.get("guide") or "").strip()
            if not guide:
                raise ValueError(f"Existing regenerated guide is empty: {guide_path}")
            guide_usage = []
            print(f"Reusing checkpointed regenerated guide from {guide_path}")
        else:
            guide_message = guide_template.format(cluster_example_string=examples)
            response = call_chat(
                client,
                model=target_aware["model"],
                prompt=guide_message,
                temperature=float(target_aware["guide_temperature"]),
                max_output_tokens=None,
                retry_limit=args.retry_limit,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            guide = (response.choices[0].message.content or "").strip()
            if not guide:
                raise RuntimeError("Target-guide generation returned empty text")
            usage = usage_record(response)
            guide_usage = [usage]
            guide_record = {**guide_config, "guide": guide, "usage": usage}
            atomic_json(guide_path, guide_record)

    target_template_path = spec_dir / target_aware["mapping_prompt"]
    target_template = target_template_path.read_text(encoding="utf-8").rstrip("\n")
    target_path, target_usage = map_rows(
        client=client,
        rows=rows,
        source_key=config["left_key"],
        output_key=target_aware["output_key"],
        template=target_template,
        substitutions={"cluster_example_summary": guide},
        model=target_aware["model"],
        temperature=float(target_aware["mapping_temperature"]),
        max_output_tokens=None,
        max_input_characters=None,
        output_dir=args.output_dir,
        stage="target_aware",
        retry_limit=args.retry_limit,
        retry_sleep_seconds=args.retry_sleep_seconds,
        postprocess=target_aware.get("postprocess"),
    )

    metadata = {
        "dataset": args.dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "left_source": str(left_path.resolve()),
        "left_source_sha256": sha256_file(left_path),
        "record_count": len(rows),
        "guide_source": guide_source,
        "no_context_mapped": str(no_context_path.resolve()),
        "target_guide": str(guide_path.resolve()),
        "target_aware_mapped": str(target_path.resolve()),
        "usage": {
            "no_context": {
                key: sum(row[key] for row in no_context_usage)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "target_guide": {
                key: sum(row[key] for row in guide_usage)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "target_aware": {
                key: sum(row[key] for row in target_usage)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        },
    }
    atomic_json(args.output_dir / "mapping_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
