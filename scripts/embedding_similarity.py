#!/usr/bin/env python3
"""Compute three normalized all-pairs embedding similarities for any dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def safe_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_texts(path: Path, column: str) -> list[str]:
    frame = pd.read_json(path)
    if column not in frame.columns:
        raise ValueError(f"Missing column {column!r} in {path}")
    if frame[column].isna().any():
        indices = frame.index[frame[column].isna()].tolist()
        raise ValueError(f"Null {column!r} values in {path}: {indices[:10]}")
    texts = frame[column].astype(str).str.strip().tolist()
    empty = [index for index, text in enumerate(texts) if not text]
    if empty:
        raise ValueError(f"Empty {column!r} values in {path}: {empty[:10]}")
    return texts


def truncate_inputs(
    texts: Sequence[str], model: str, max_input_tokens: int
) -> tuple[list[str], list[int], dict[str, int]]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install tiktoken to use token-aware truncation") from exc

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    prepared: list[str] = []
    prepared_token_counts: list[int] = []
    truncated_count = 0
    original_tokens = 0
    submitted_tokens = 0
    longest_original = 0
    for text in texts:
        token_ids = encoding.encode(text, disallowed_special=())
        token_count = len(token_ids)
        original_tokens += token_count
        longest_original = max(longest_original, token_count)
        if token_count > max_input_tokens:
            token_ids = token_ids[:max_input_tokens]
            text = encoding.decode(token_ids)
            truncated_count += 1
        prepared.append(text)
        submitted_count = len(
            encoding.encode(text, disallowed_special=())
        )
        prepared_token_counts.append(submitted_count)
        submitted_tokens += submitted_count

    return prepared, prepared_token_counts, {
        "text_count": len(prepared),
        "truncated_text_count": truncated_count,
        "original_token_count": original_tokens,
        "submitted_token_count_estimate": submitted_tokens,
        "longest_original_text_tokens": longest_original,
    }


def embedding_batch_ranges(
    token_counts: Sequence[int],
    start: int,
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < len(token_counts):
        end = cursor
        batch_tokens = 0
        while end < len(token_counts) and end - cursor < max_batch_size:
            next_tokens = int(token_counts[end])
            if end > cursor and batch_tokens + next_tokens > max_batch_tokens:
                break
            if next_tokens > max_batch_tokens:
                raise ValueError(
                    f"Input {end} has {next_tokens} tokens, exceeding the "
                    f"per-request token budget {max_batch_tokens}"
                )
            batch_tokens += next_tokens
            end += 1
        ranges.append((cursor, end))
        cursor = end
    return ranges


def texts_fingerprint(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding response contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding response contains a zero-length vector")
    return embeddings / norms


def request_embedding_batch(
    client: Any,
    texts: Sequence[str],
    model: str,
    dimensions: int | None,
    retry_limit: int,
    retry_sleep_seconds: float,
) -> tuple[np.ndarray, int]:
    request: dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": "float",
    }
    if dimensions is not None:
        request["dimensions"] = dimensions

    last_error: Exception | None = None
    for attempt in range(retry_limit + 1):
        try:
            response = client.embeddings.create(**request)
            response_data = sorted(response.data, key=lambda item: item.index)
            if len(response_data) != len(texts):
                raise RuntimeError(
                    "OpenAI returned an unexpected embedding count: "
                    f"expected {len(texts)}, got {len(response_data)}"
                )
            embeddings = np.asarray(
                [item.embedding for item in response_data], dtype=np.float32
            )
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            return normalize_embeddings(embeddings), prompt_tokens
        except Exception as exc:  # pragma: no cover - API/runtime failures
            last_error = exc
            if attempt >= retry_limit:
                break
            delay = retry_sleep_seconds * (2**attempt)
            print(f"  Embedding request failed: {exc}; retrying in {delay:.1f}s")
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("OpenAI embedding request exhausted its retries") from last_error


def load_or_create_embeddings(
    client: Any,
    name: str,
    texts: Sequence[str],
    cache_dir: Path,
    model: str,
    dimensions: int | None,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
    retry_limit: int,
    retry_sleep_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared, token_counts, text_stats = truncate_inputs(
        texts, model, max_input_tokens
    )
    fingerprint = texts_fingerprint(prepared)
    data_path = cache_dir / f"{name}.npy"
    metadata_path = cache_dir / f"{name}.json"
    expected = {
        "name": name,
        "model": model,
        "requested_dimensions": dimensions,
        "max_input_tokens": max_input_tokens,
        "text_count": len(prepared),
        "texts_sha256": fingerprint,
    }

    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Embedding cache metadata mismatch for {name}: {mismatches}"
            )
        if not data_path.exists():
            raise FileNotFoundError(f"Missing embedding cache array: {data_path}")
        completed = int(metadata.get("completed", 0))
        if completed == len(prepared):
            print(f"Reusing complete embedding cache for {name}: {data_path}")
            return np.load(data_path, mmap_mode="r"), {
                **metadata,
                "new_api_prompt_tokens": 0,
                "reused_complete_cache": True,
                "cache_enabled": True,
            }
    else:
        completed = 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    array: np.memmap | None = None
    if metadata is not None:
        array = np.lib.format.open_memmap(data_path, mode="r+")

    new_api_prompt_tokens = 0
    for start, end in embedding_batch_ranges(
        token_counts,
        completed,
        batch_size,
        max_batch_tokens,
    ):
        batch_embeddings, prompt_tokens = request_embedding_batch(
            client,
            prepared[start:end],
            model,
            dimensions,
            retry_limit,
            retry_sleep_seconds,
        )
        if array is None:
            array = np.lib.format.open_memmap(
                data_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(prepared), batch_embeddings.shape[1]),
            )
            metadata = {
                **expected,
                **text_stats,
                "embedding_dimensions": int(batch_embeddings.shape[1]),
                "completed": 0,
                "accounted_api_prompt_tokens": 0,
            }
        elif batch_embeddings.shape[1] != array.shape[1]:
            raise ValueError(
                f"Embedding dimension changed for {name}: "
                f"{batch_embeddings.shape[1]} != {array.shape[1]}"
            )

        array[start:end] = batch_embeddings
        array.flush()
        new_api_prompt_tokens += prompt_tokens
        metadata["completed"] = end
        metadata["accounted_api_prompt_tokens"] = int(
            metadata.get("accounted_api_prompt_tokens", 0)
        ) + prompt_tokens
        safe_write_json(metadata_path, metadata)
        print(f"  {name}: embedded {end:,}/{len(prepared):,}")

    if array is None or metadata is None:
        raise RuntimeError(f"Failed to create embeddings for {name}")
    return np.load(data_path, mmap_mode="r"), {
        **metadata,
        "new_api_prompt_tokens": new_api_prompt_tokens,
        "reused_complete_cache": False,
        "cache_enabled": True,
    }


def create_embeddings_without_cache(
    client: Any,
    name: str,
    texts: Sequence[str],
    model: str,
    dimensions: int | None,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
    retry_limit: int,
    retry_sleep_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared, token_counts, text_stats = truncate_inputs(
        texts, model, max_input_tokens
    )
    batches: list[np.ndarray] = []
    prompt_tokens = 0
    for start, end in embedding_batch_ranges(
        token_counts,
        0,
        batch_size,
        max_batch_tokens,
    ):
        batch_embeddings, batch_prompt_tokens = request_embedding_batch(
            client,
            prepared[start:end],
            model,
            dimensions,
            retry_limit,
            retry_sleep_seconds,
        )
        batches.append(batch_embeddings)
        prompt_tokens += batch_prompt_tokens
        print(f"  {name}: embedded {end:,}/{len(prepared):,}")

    embeddings = np.concatenate(batches, axis=0)
    return embeddings, {
        "name": name,
        "model": model,
        "requested_dimensions": dimensions,
        "embedding_dimensions": int(embeddings.shape[1]),
        "max_input_tokens": max_input_tokens,
        "texts_sha256": texts_fingerprint(prepared),
        **text_stats,
        "completed": len(prepared),
        "accounted_api_prompt_tokens": prompt_tokens,
        "new_api_prompt_tokens": prompt_tokens,
        "reused_complete_cache": False,
        "cache_enabled": False,
    }


def get_embeddings(
    client: Any,
    name: str,
    texts: Sequence[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.cache_embeddings:
        return load_or_create_embeddings(
            client,
            name,
            texts,
            args.cache_dir,
            args.model,
            args.openai_dimensions,
            args.batch_size,
            args.max_batch_tokens,
            args.max_input_tokens,
            args.retry_limit,
            args.retry_sleep_seconds,
        )
    return create_embeddings_without_cache(
        client,
        name,
        texts,
        args.model,
        args.openai_dimensions,
        args.batch_size,
        args.max_batch_tokens,
        args.max_input_tokens,
        args.retry_limit,
        args.retry_sleep_seconds,
    )


def validate_pairs(
    pairs: pd.DataFrame, left_count: int, right_count: int
) -> tuple[np.ndarray, np.ndarray]:
    required = {"_left_id", "_right_id"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    if pairs.empty:
        raise ValueError("Pairs CSV is empty")
    left_ids = pairs["_left_id"].to_numpy(dtype=np.int64)
    right_ids = pairs["_right_id"].to_numpy(dtype=np.int64)
    if left_ids.min() < 0 or left_ids.max() >= left_count:
        raise IndexError("Pair _left_id values are incompatible with left texts")
    if right_ids.min() < 0 or right_ids.max() >= right_count:
        raise IndexError("Pair _right_id values are incompatible with right texts")
    return left_ids, right_ids


def indexed_cosine_scores(
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(len(left_ids), dtype=np.float32)
    for start in range(0, len(left_ids), batch_size):
        end = min(start + batch_size, len(left_ids))
        scores[start:end] = np.einsum(
            "ij,ij->i",
            left_embeddings[left_ids[start:end]],
            right_embeddings[right_ids[start:end]],
            optimize=True,
        )
    return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--right-text-key", required=True)
    parser.add_argument("--direct-left-json", type=Path, required=True)
    parser.add_argument("--direct-text-key", required=True)
    parser.add_argument("--no-context-left-json", type=Path, required=True)
    parser.add_argument("--no-context-text-key", required=True)
    parser.add_argument("--target-aware-left-json", type=Path, required=True)
    parser.add_argument("--target-aware-text-key", required=True)
    parser.add_argument("--direct-score-column", default="score_direct")
    parser.add_argument("--no-context-score-column", required=True)
    parser.add_argument("--target-aware-score-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist and resume embedding arrays. Disabled by default.",
    )
    parser.add_argument(
        "--write-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a JSON run-metadata sidecar. Enabled by default.",
    )
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--openai-dimensions", type=int, default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-batch-tokens", type=int, default=250_000)
    parser.add_argument("--score-batch-size", type=int, default=100_000)
    parser.add_argument("--max-input-tokens", type=int, default=8191)
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--normalize-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 2048:
        raise ValueError("--batch-size must be between 1 and 2048")
    if not 1 <= args.max_batch_tokens <= 300_000:
        raise ValueError("--max-batch-tokens must be between 1 and 300000")
    if args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive")
    if args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive")
    if args.retry_limit < 0 or args.retry_sleep_seconds < 0:
        raise ValueError("Retry settings must be non-negative")
    if args.openai_dimensions is not None and args.openai_dimensions < 1:
        raise ValueError("--openai-dimensions must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package") from exc
    client = OpenAI(api_key=api_key)

    right_texts = load_texts(args.right_json, args.right_text_key)
    source_options = {
        "direct": (
            load_texts(args.direct_left_json, args.direct_text_key),
            args.direct_score_column,
        ),
        "no_context": (
            load_texts(args.no_context_left_json, args.no_context_text_key),
            args.no_context_score_column,
        ),
        "target_aware": (
            load_texts(args.target_aware_left_json, args.target_aware_text_key),
            args.target_aware_score_column,
        ),
    }
    left_sources = source_options
    left_counts = {len(texts) for texts, _ in left_sources.values()}
    if len(left_counts) != 1:
        raise ValueError(f"Left source counts differ: {sorted(left_counts)}")
    left_count = left_counts.pop()

    pairs = pd.read_csv(args.pairs)
    left_ids, right_ids = validate_pairs(pairs, left_count, len(right_texts))
    output_columns = ["_left_id", "_right_id"]
    if "label" in pairs.columns:
        output_columns.append("label")
    output = pairs[output_columns].copy()

    if args.cache_embeddings:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_metadata: dict[str, Any] = {}
    right_embeddings, right_metadata = get_embeddings(
        client,
        "right",
        right_texts,
        args,
    )
    embedding_metadata["right"] = right_metadata

    for cache_name, (texts, score_column) in left_sources.items():
        left_embeddings, left_metadata = get_embeddings(
            client,
            cache_name,
            texts,
            args,
        )
        if left_embeddings.shape[1] != right_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch for {score_column}: "
                f"{left_embeddings.shape[1]} != {right_embeddings.shape[1]}"
            )
        scores = indexed_cosine_scores(
            left_embeddings,
            right_embeddings,
            left_ids,
            right_ids,
            args.score_batch_size,
        )
        raw_min = float(scores.min())
        raw_max = float(scores.max())
        if args.normalize_scores:
            scores = min_max_normalize(scores)
        output[score_column] = scores
        embedding_metadata[cache_name] = left_metadata
        embedding_metadata[cache_name]["score_column"] = score_column
        embedding_metadata[cache_name]["raw_score_min"] = raw_min
        embedding_metadata[cache_name]["raw_score_max"] = raw_max
        embedding_metadata[cache_name]["output_score_min"] = float(scores.min())
        embedding_metadata[cache_name]["output_score_max"] = float(scores.max())
        print(
            f"{score_column}: raw=[{raw_min:.6f}, {raw_max:.6f}], "
            f"output=[{scores.min():.6f}, {scores.max():.6f}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    output.to_csv(temporary_output, index=False)
    temporary_output.replace(args.output)

    accounted_tokens = sum(
        int(metadata.get("accounted_api_prompt_tokens", 0))
        for metadata in embedding_metadata.values()
    )
    new_tokens = sum(
        int(metadata.get("new_api_prompt_tokens", 0))
        for metadata in embedding_metadata.values()
    )
    run_metadata = {
        "model": args.model,
        "requested_dimensions": args.openai_dimensions,
        "pairs": str(args.pairs.resolve()),
        "right_json": str(args.right_json.resolve()),
        "direct_left_json": str(args.direct_left_json.resolve()),
        "no_context_left_json": str(args.no_context_left_json.resolve()),
        "target_aware_left_json": str(args.target_aware_left_json.resolve()),
        "direct_score_column": args.direct_score_column,
        "no_context_score_column": args.no_context_score_column,
        "target_aware_score_column": args.target_aware_score_column,
        "output": str(args.output.resolve()),
        "pair_count": len(output),
        "normalized": args.normalize_scores,
        "accounted_api_prompt_tokens": accounted_tokens,
        "new_api_prompt_tokens_this_run": new_tokens,
        "embedding_sets": embedding_metadata,
    }
    print(f"Wrote {len(output):,} pairs to {args.output}")
    if args.write_metadata:
        metadata_path = args.output.with_suffix(".metadata.json")
        safe_write_json(metadata_path, run_metadata)
        print(f"Wrote run metadata to {metadata_path}")
    print(
        f"Embedding prompt tokens: accounted={accounted_tokens:,}, "
        f"new_this_run={new_tokens:,}"
    )


if __name__ == "__main__":
    main()
