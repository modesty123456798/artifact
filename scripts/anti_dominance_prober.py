"""Generic anti-dominance, round-robin LLM probing runner."""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The openai package is required. Install it with `pip install openai`."
    ) from exc


DEFAULT_MATCH_PROMPT_TEMPLATE = (
    "Decide which Candidate {right_label}s is described in the {left_label}.\n\n "
    "{left_label}:\n{left_text}\n\n"
    "Candidate {right_label}s ({right_count}):\n"
    "{right_items}\n\n"
    "Respond with a JSON object like:\n"
    "{output_format}\n"
    "Return the ids of the described candidates from the list above. "
    "If no candidates match, return {empty_format}.\n"
    "Do not include any additional keys."
)

DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT = (
    "The user will provide one medical article and multiple candidate reactions.\n"
    "For every candidate reaction, determine whether that reaction is described in\n"
    "the medical article. Consider the context and semantic meaning, not only exact "
    "word overlap."
)

DEFAULT_LEFT_ANCHORED_TEMPLATE = (
    "{left_label}:\n"
    "«{left_text}»\n\n"
    "Candidate {right_label}s ({right_count}):\n"
    "{right_items}\n\n"
    "Determine whether each candidate reaction is described in the medical article. "
    "For each candidate reaction ID, set match to true if it is described and "
    "false otherwise."
)

DEFAULT_SINGLE_PAIR_YES_NO_TEMPLATE = (
    "Given one {left_label} and one {right_label}, decide whether the {left_label} "
    "describes the {right_label}.\n\n"
    "{left_label}:\n"
    "{left_text}\n\n"
    "{right_label}:\n"
    "{right_text}\n\n"
    "Answer with exactly YES or NO. Do not include any other text."
)


# #####DBPEDIA
# DEFAULT_LEFT_ANCHORED_TEMPLATE = (
#     "Given one {left_label} and multiple {right_label} candidates, decide for each "
#     "{right_label} whether the {left_label} belongs to it.\n\n"
#     "{left_label}:\n"
#     "{left_text}\n\n"
#     "{right_label} candidates ({right_count}):\n"
#     "{right_items}\n\n"
#     "Respond with a JSON object like:\n"
#     "{output_format}\n"
#     "Include an entry for every candidate_id (use match=true when the {left_label} belongs to "
#     "the {right_label}; use match=false when it does not).\n"
#     "Do not include any additional keys."
# )
# #####DBPEDIA

# #####AMAZON531
# DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT = (
#     "The user will provide one Amazon product review and multiple candidate "
#     "product types.\n"
#     "For every candidate product type, determine whether the product being "
#     "reviewed belongs to that product type. Consider the review context and "
#     "semantic meaning, not only exact word overlap."
# )

# DEFAULT_LEFT_ANCHORED_TEMPLATE = (
#     "{left_label}:\n"
#     "«{left_text}»\n\n"
#     "Candidate {right_label}s ({right_count}):\n"
#     "{right_items}\n\n"
#     "Determine whether the product being reviewed belongs to each candidate "
#     "product type. For each candidate product type ID, set match to true if "
#     "the product belongs to that type and false otherwise."
# )
# #####AMAZON531


def load_blocking_results(path: Path, dimension_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "_left_id" not in df.columns and df.index.name == "_left_id":
        df = df.reset_index()
    required = {"_left_id", "_right_id", *dimension_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )
    for column in dimension_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def ensure_api_key(provided_key: Optional[str], dry_run: bool) -> Optional[str]:
    api_key = (
        provided_key.strip() if provided_key and provided_key.strip() else None
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key and not dry_run:
        raise SystemExit(
            "No API key provided. Use --api-key or set OPENAI_API_KEY."
        )
    return api_key


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def sum_response_tokens(path: Path) -> Tuple[int, int]:
    completion_tokens = 0
    prompt_tokens = 0
    if not path.exists():
        return completion_tokens, prompt_tokens
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            completion_tokens += int(record.get("completion_tokens") or 0)
            prompt_tokens += int(record.get("prompt_tokens") or 0)
    return completion_tokens, prompt_tokens


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_minimal_results(df: pd.DataFrame, path: Path) -> None:
    columns_to_store = ["_left_id", "_right_id", "rpe_match"]
    missing = [col for col in columns_to_store if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for persistence: {', '.join(missing)}")
    subset = df[columns_to_store].copy()
    safe_write_csv(subset, path)


def append_minimal_results(df: pd.DataFrame, indices: Sequence[int], path: Path) -> None:
    if not indices:
        return
    columns_to_store = ["_left_id", "_right_id", "rpe_match"]
    missing = [col for col in columns_to_store if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for persistence: {', '.join(missing)}")
    subset = df.loc[list(indices), columns_to_store].copy()
    header = not path.exists() or path.stat().st_size == 0
    subset.to_csv(path, mode="a", header=header, index=False)


def write_judged_pairs_snapshot(
    df: pd.DataFrame,
    indices: Sequence[int],
    path: Path,
) -> None:
    columns_to_store = ["_left_id", "_right_id", "rpe_match"]
    if indices:
        subset = df.loc[list(indices), columns_to_store].copy()
    else:
        subset = pd.DataFrame(columns=columns_to_store)
    safe_write_csv(subset, path)


def load_round_robin_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    return state if isinstance(state, dict) else {}


def write_round_robin_state(
    path: Path,
    iteration: int,
    l_by_left: Dict[int, int],
    k_by_left: Dict[int, int],
    period_stats: Optional[Dict[int, Dict[str, int]]] = None,
) -> None:
    state = {
        "iteration": int(iteration),
        "l_by_left": {str(key): int(value) for key, value in l_by_left.items()},
        "k_by_left": {str(key): int(value) for key, value in k_by_left.items()},
        "period_stats": {
            str(key): {
                "processed": int(value.get("processed", 0)),
                "positives": int(value.get("positives", 0)),
            }
            for key, value in (period_stats or {}).items()
        },
        "timestamp": time.time(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, default=_json_default)
    tmp_path.replace(path)


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def load_indexed_text(path: Path, text_key: str, id_key: Optional[str] = None) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of records.")
    mapping: Dict[int, str] = {}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object record at index {idx}.")
        value = record.get(text_key, "")
        key_id = record.get(id_key) if id_key else idx
        if key_id is None:
            raise ValueError(f"{path} record at index {idx} is missing id key {id_key}.")
        mapping[int(key_id)] = "" if value is None else str(value)
    return mapping


def extract_json_block(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(raw_text[start : end + 1])


def invoke_llm(
    client: Optional[OpenAI],
    model: str,
    user_prompt: str,
    retry_limit: int,
    sleep_seconds: float,
    batch_mode: str,
    dry_run: bool,
    system_prompt: Optional[str] = None,
    structured_candidate_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    response_text = ""
    parsed_json: Optional[Dict[str, Any]] = None
    reasoning = ""
    matches: List[str] = []
    pair_results: List[Dict[str, Any]] = []
    candidate_results: List[Dict[str, Any]] = []
    attempts_used = 0
    duration = 0.0
    parse_error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    completion_tokens = 0
    prompt_tokens = 0

    if dry_run or client is None:
        reasoning = "dry run: no LLM call executed."
        return {
            "matches": matches,
            "pair_results": pair_results,
            "candidate_results": candidate_results,
            "reasoning": reasoning,
            "response_text": response_text,
            "raw_response": raw_response,
            "parse_error": parse_error,
            "attempts_used": attempts_used,
            "duration_seconds": duration,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        }

    max_attempts = max(1, retry_limit + 1)
    for attempt in range(max_attempts):
        start = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            request: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if structured_candidate_ids is not None:
                candidate_ids = [int(value) for value in structured_candidate_ids]
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "candidate_match_results",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "candidate_results": {
                                    "type": "array",
                                    "minItems": len(candidate_ids),
                                    "maxItems": len(candidate_ids),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "candidate_id": {
                                                "type": "integer",
                                                "enum": candidate_ids,
                                            },
                                            "match": {"type": "boolean"},
                                        },
                                        "required": ["candidate_id", "match"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["candidate_results"],
                            "additionalProperties": False,
                        },
                    },
                }
            completion = client.chat.completions.create(**request)
            duration = time.time() - start
            attempts_used = attempt + 1
            response_text = completion.choices[0].message.content or ""
            # import pdb;pdb.set_trace()
            raw_response = completion.model_dump()
            usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
            if isinstance(usage, dict):
                observed_completion = int(usage.get("completion_tokens") or 0)
                observed_prompt = int(usage.get("prompt_tokens") or 0)
                completion_tokens += observed_completion
                prompt_tokens += observed_prompt
            if batch_mode != "single_pair_yes_no":
                parsed_json = extract_json_block(response_text)
            # import pdb;pdb.set_trace()
        except Exception as exc:  # pragma: no cover - network/LLM errors at runtime
            # import pdb;pdb.set_trace()
            duration = time.time() - start
            attempts_used = attempt + 1
            parse_error = str(exc)
            if attempt < max_attempts - 1 and sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        try:
            if batch_mode == "single_pair_yes_no":
                answer_match = re.match(
                    r"^\s*(YES|NO)\b", response_text, flags=re.IGNORECASE
                )
                if answer_match is None:
                    raise ValueError("response must start with YES or NO")
                match_flag = answer_match.group(1).upper() == "YES"
                candidate_results = [
                    {"candidate_id": 1, "match": match_flag}
                ]
                matches = [1] if match_flag else []
                pair_results = []
                reasoning = ""
            elif batch_mode == "pair_multi":
                raw_results = parsed_json.get("pair_results")
                if raw_results is None:
                    raw_results = parsed_json.get("pairs")
                if not isinstance(raw_results, list):
                    raise ValueError("`pair_results` missing or not a list")
                raw_reasoning = parsed_json.get("reasoning", "")
                if not isinstance(raw_reasoning, str):
                    raw_reasoning = ""
                pair_results = []
                matches = []
                for item in raw_results:
                    if not isinstance(item, dict):
                        raise ValueError("`pair_results` items must be objects")
                    pair_id = item.get("pair_id", item.get("id"))
                    match_value = item.get("match")
                    if not isinstance(pair_id, (int, np.integer)):
                        raise ValueError("`pair_results` missing integer pair_id")
                    if isinstance(match_value, bool):
                        match_flag = match_value
                    elif isinstance(match_value, (int, float, np.integer)):
                        match_flag = bool(int(match_value))
                    elif isinstance(match_value, str):
                        match_flag = match_value.strip().lower() in {"1", "true", "yes"}
                    else:
                        raise ValueError("`pair_results` missing boolean match")
                    normalized = {
                        "pair_id": int(pair_id),
                        "match": bool(match_flag),
                    }
                    if "reasoning" in item and isinstance(item.get("reasoning"), str):
                        normalized["reasoning"] = item.get("reasoning")
                    pair_results.append(normalized)
                    if match_flag:
                        matches.append(int(pair_id))
                reasoning = raw_reasoning
                candidate_results = []
            elif batch_mode == "left_anchored_multi_right":
                raw_results = parsed_json.get("candidate_results")
                if raw_results is None:
                    raw_results = parsed_json.get("pair_results")
                if raw_results is None:
                    raw_results = parsed_json.get("pairs")
                if not isinstance(raw_results, list):
                    raise ValueError("`candidate_results` missing or not a list")
                raw_reasoning = parsed_json.get("reasoning", "")
                if not isinstance(raw_reasoning, str):
                    raw_reasoning = ""
                candidate_results = []
                matches = []
                pair_results = []
                for item in raw_results:
                    if not isinstance(item, dict):
                        raise ValueError("`candidate_results` items must be objects")
                    candidate_id = (
                        item.get("candidate_id")
                        if "candidate_id" in item
                        else item.get("id", item.get("pair_id"))
                    )
                    match_value = item.get("match")
                    if not isinstance(candidate_id, (int, np.integer)):
                        raise ValueError("`candidate_results` missing integer candidate_id")
                    if isinstance(match_value, bool):
                        match_flag = match_value
                    elif isinstance(match_value, (int, float, np.integer)):
                        match_flag = bool(int(match_value))
                    elif isinstance(match_value, str):
                        normalized = match_value.strip().lower()
                        if normalized in {"1", "true", "yes"}:
                            match_flag = True
                        elif normalized in {"0", "false", "no"}:
                            match_flag = False
                        else:
                            raise ValueError("`candidate_results` missing boolean match")
                    else:
                        raise ValueError("`candidate_results` missing boolean match")
                    normalized_item = {
                        "candidate_id": int(candidate_id),
                        "match": bool(match_flag),
                    }
                    if "reasoning" in item and isinstance(item.get("reasoning"), str):
                        normalized_item["reasoning"] = item.get("reasoning")
                    candidate_results.append(normalized_item)
                if structured_candidate_ids is not None:
                    expected_ids = [int(value) for value in structured_candidate_ids]
                    returned_ids = [
                        item["candidate_id"] for item in candidate_results
                    ]
                    if (
                        len(returned_ids) != len(set(returned_ids))
                        or set(returned_ids) != set(expected_ids)
                    ):
                        raise ValueError(
                            "`candidate_results` must contain every expected "
                            "candidate_id exactly once"
                        )
                reasoning = raw_reasoning
            else:
                raw_matches = parsed_json.get("matches")
                raw_reasoning = parsed_json.get("reasoning", "")
                if not isinstance(raw_matches, list):
                    raise ValueError("`matches` missing or not a list")
                if not isinstance(raw_reasoning, str):
                    raw_reasoning = ""
                matches = []
                # import pdb;pdb.set_trace()
                for item in raw_matches:
                    # if isinstance(item, (int, float, np.integer)):
                    # import pdb;pdb.set_trace()
                    if isinstance(item, (int, np.integer)):
                        matches.append(int(item))
                    # elif isinstance(item, str):
                    #     matches.append(item)
                    else:
                        raise ValueError("`matches` missing int element")
                reasoning = raw_reasoning
                pair_results = []
            parse_error = None
            break
        except Exception as exc:
            parse_error = f"invalid fields: {exc}"
            if attempt < max_attempts - 1 and sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue
    # import pdb;pdb.set_trace()
    if parse_error and parsed_json is None:
        reasoning = f"parse_error: {parse_error}"
    return {
        "matches": matches,
        "pair_results": pair_results,
        "candidate_results": candidate_results,
        "reasoning": reasoning,
        "response_text": response_text,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "attempts_used": attempts_used,
        "duration_seconds": duration,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }


def build_match_prompt(
    left_text: str,
    right_texts: Sequence[str],
    prompt_template: str,
    left_label: str,
    right_label: str,
) -> str:
    right_items = "\n".join(
        f"{idx}. {right_text}" for idx, right_text in enumerate(right_texts, start=1)
    )
    output_format = '{"matches": [1, 3], "reasoning": "brief explanation"}'
    empty_format = '{"matches": [], "reasoning": "..."}'
    # output_format = '{"matches": [1, 3]}'
    # empty_format = '{"matches": []}'
    return (
        prompt_template.format(
            left_label=left_label,
            right_label=right_label,
            left_text=left_text,
            right_items=right_items,
            right_count=len(right_texts),
            output_format=output_format,
            empty_format=empty_format,
        )
    )


def build_left_anchored_prompt(
    left_text: str,
    right_texts: Sequence[str],
    prompt_template: str,
    left_label: str,
    right_label: str,
) -> str:
    right_items = "\n".join(
        f"{idx}. {right_label}: «{right_text}»"
        for idx, right_text in enumerate(right_texts, start=1)
    )
    return prompt_template.format(
        left_label=left_label,
        right_label=right_label,
        left_text=left_text,
        right_items=right_items,
        pair_items=right_items,
        right_count=len(right_texts),
        pair_count=len(right_texts),
    )


def build_single_pair_yes_no_prompt(
    left_text: str,
    right_text: str,
    prompt_template: str,
    left_label: str,
    right_label: str,
) -> str:
    return prompt_template.format(
        left_label=left_label,
        right_label=right_label,
        left_text=left_text,
        right_text=right_text,
    )


def build_pair_prompt(
    left_texts: Sequence[str],
    right_texts: Sequence[str],
    prompt_template: str,
    left_label: str,
    right_label: str,
) -> str:
    pair_items = "\n".join(
        f"{idx}. {left_label}: {left_text}\n   {right_label}: {right_text}"
        for idx, (left_text, right_text) in enumerate(
            zip(left_texts, right_texts), start=1
        )
    )
    output_format = (
        '{"pair_results": [{"pair_id": 1, "match": true}, '
        '{"pair_id": 2, "match": false}], "reasoning": "brief explanation"}'
    )
    return (
        prompt_template.format(
            left_label=left_label,
            right_label=right_label,
            pair_items=pair_items,
            pair_count=len(right_texts),
            output_format=output_format,
        )
    )


def mark_match(
    df: pd.DataFrame,
    indices: Sequence[int],
    candidate_texts: Sequence[str],
    matches: Sequence[Any],
    reasoning: str,
) -> List[str]:
    matched_positions: set[int] = set()
    valid_matches: List[str] = []
    # import pdb;pdb.set_trace()
    for match in matches:
        if isinstance(match, (int, float, np.integer)):
            pos = int(match)
        elif isinstance(match, str):
            if not match.strip().isdigit():
                continue
            pos = int(match.strip())
        else:
            continue
        if 1 <= pos <= len(candidate_texts):
            matched_positions.add(pos - 1)
            valid_matches.append(candidate_texts[pos - 1])
    # import pdb;pdb.set_trace()
    for pos, idx in enumerate(indices):
        flag = 1 if pos in matched_positions else 0
        df.at[idx, "rpe_match"] = flag
    # import pdb;pdb.set_trace()
    if reasoning:
        for idx in indices:
            prev = str(df.at[idx, "llm_reasoning"] or "").strip()
            df.at[idx, "llm_reasoning"] = reasoning if not prev else f"{prev}\n{reasoning}"
    return valid_matches


def mark_pair_results(
    df: pd.DataFrame,
    indices: Sequence[int],
    candidate_texts: Sequence[str],
    pair_results: Sequence[Dict[str, Any]],
    reasoning: str,
) -> List[str]:
    match_map: Dict[int, bool] = {}
    reasoning_map: Dict[int, str] = {}
    for item in pair_results:
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        if not isinstance(pair_id, (int, np.integer)):
            continue
        match_map[int(pair_id)] = bool(item.get("match"))
        item_reasoning = item.get("reasoning")
        if isinstance(item_reasoning, str):
            reasoning_map[int(pair_id)] = item_reasoning

    matched_positions: set[int] = set()
    valid_matches: List[str] = []
    for pos, idx in enumerate(indices):
        pair_id = pos + 1
        if match_map.get(pair_id, False):
            matched_positions.add(pos)
            valid_matches.append(candidate_texts[pos])
        df.at[idx, "rpe_match"] = 1 if pos in matched_positions else 0
        item_reasoning = reasoning_map.get(pair_id, reasoning)
        if item_reasoning:
            prev = str(df.at[idx, "llm_reasoning"] or "").strip()
            df.at[idx, "llm_reasoning"] = (
                item_reasoning if not prev else f"{prev}\n{item_reasoning}"
            )
    return valid_matches


def mark_candidate_results(
    df: pd.DataFrame,
    indices: Sequence[int],
    candidate_texts: Sequence[str],
    candidate_results: Sequence[Dict[str, Any]],
    reasoning: str,
) -> List[str]:
    match_map: Dict[int, bool] = {}
    reasoning_map: Dict[int, str] = {}
    # import pdb;pdb.set_trace()
    for item in candidate_results:
        # import pdb;pdb.set_trace()
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, (int, np.integer)):
            continue
        match_map[int(candidate_id)] = bool(item.get("match"))
        item_reasoning = item.get("reasoning")
        if isinstance(item_reasoning, str):
            reasoning_map[int(candidate_id)] = item_reasoning

    matched_positions: set[int] = set()
    valid_matches: List[str] = []
    for pos, idx in enumerate(indices):
        candidate_id = pos + 1
        if match_map.get(candidate_id, False):
            matched_positions.add(pos)
            valid_matches.append(candidate_texts[pos])
        df.at[idx, "rpe_match"] = 1 if pos in matched_positions else 0
        item_reasoning = reasoning_map.get(candidate_id, reasoning)
        if item_reasoning:
            prev = str(df.at[idx, "llm_reasoning"] or "").strip()
            df.at[idx, "llm_reasoning"] = (
                item_reasoning if not prev else f"{prev}\n{item_reasoning}"
            )
    return valid_matches


def chunked(iterable: Sequence[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(iterable), size):
        yield list(iterable[start : start + size])


def interleave_groups(grouped_indices: Sequence[Sequence[int]]) -> List[int]:
    interleaved: List[int] = []
    queues = [list(indices) for indices in grouped_indices if indices]
    cursor = 0
    while queues:
        if cursor >= len(queues):
            cursor = 0
        current = queues[cursor]
        interleaved.append(current.pop(0))
        if not current:
            queues.pop(cursor)
        else:
            cursor += 1
    return interleaved


def compute_k_skyband_indices(
    subset: pd.DataFrame,
    dimension_columns: Sequence[str],
    k_value: int,
    dominated_counts_cache: Optional[Dict[int, pd.Series]] = None,
) -> List[int]:
    if subset.empty:
        return []
    if not dimension_columns:
        return subset.index.tolist()
    if k_value <= 0:
        return subset.index.tolist()

    left_id = int(subset["_left_id"].iloc[0]) if "_left_id" in subset.columns else None
    cached_counts = (
        dominated_counts_cache.get(left_id)
        if dominated_counts_cache is not None and left_id is not None
        else None
    )
    if cached_counts is not None:
        dominated_counts = cached_counts.loc[subset.index].to_numpy()
    else:
        values = subset[dimension_columns].to_numpy(dtype=float)
        diff_ge = values[:, None, :] >= values[None, :, :]
        diff_gt = values[:, None, :] > values[None, :, :]
        dominance = np.all(diff_ge, axis=2) & np.any(diff_gt, axis=2)
        np.fill_diagonal(dominance, False)
        dominated_counts = dominance.sum(axis=0)
        if dominated_counts_cache is not None and left_id is not None:
            dominated_counts_cache[left_id] = pd.Series(
                dominated_counts, index=subset.index
            )
    mask = dominated_counts < k_value
    if not mask.any():
        return subset.index.tolist()
    return subset.index[mask].tolist()


def merge_existing_results(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return df

    existing = pd.read_csv(path)
    key_columns = ["_left_id", "_right_id"]
    if not set(key_columns).issubset(existing.columns):
        return df
    if "rpe_match" not in existing.columns:
        return df

    existing_subset = (
        existing[key_columns + ["rpe_match"]]
        .dropna(subset=key_columns)
        .copy()
    )
    existing_subset = existing_subset.drop_duplicates(subset=key_columns, keep="last")
    existing_subset["_left_id"] = existing_subset["_left_id"].astype(int)
    existing_subset["_right_id"] = existing_subset["_right_id"].astype(int)
    existing_subset = existing_subset.rename(
        columns={"rpe_match": "_existing_rpe_match"}
    )
    merged = df[key_columns].merge(existing_subset, on=key_columns, how="left")
    existing_mask = merged["_existing_rpe_match"].notna()
    if existing_mask.any():
        df.loc[existing_mask.to_numpy(), "rpe_match"] = merged.loc[
            existing_mask, "_existing_rpe_match"
        ].to_numpy()
    return df


def load_ground_truth(
    path: Path,
    truth_key: str,
    id_key: Optional[str] = None,
) -> Dict[int, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    truth: Dict[int, List[str]] = {}
    for idx, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object record at index {idx}.")
        key_id = entry.get(id_key) if id_key else idx
        if key_id is None:
            raise ValueError(f"{path} record at index {idx} is missing id key {id_key}.")
        matches = entry.get(truth_key, []) or []
        truth[int(key_id)] = [normalize_text(item) for item in matches if isinstance(item, str)]
    return truth


def compute_progress_metrics(
    df: pd.DataFrame,
    truth: Dict[int, List[str]],
    right_text_column: str,
) -> Dict[str, int]:
    decided_mask = df["rpe_match"].notna()
    positive_mask = df["rpe_match"] == 1
    decided = int(decided_mask.sum())
    positives = int(positive_mask.sum())
    true_positive = 0
    positive_rows = df.loc[positive_mask, ["_left_id", right_text_column]]
    for _, row in positive_rows.iterrows():
        left_id = int(row["_left_id"])
        reactions = truth.get(left_id, [])
        if normalize_text(row[right_text_column]) in reactions:
            true_positive += 1
    return {
        "decided_points": decided,
        "positive_points": positives,
        "true_positive_points": true_positive,
    }


def count_true_positives_for_indices(
    df: pd.DataFrame,
    indices: Sequence[int],
    truth: Dict[int, List[str]],
    right_text_column: str,
) -> int:
    true_positive = 0
    for idx in indices:
        if df.at[idx, "rpe_match"] != 1:
            continue
        left_id = int(df.at[idx, "_left_id"])
        if normalize_text(df.at[idx, right_text_column]) in truth.get(left_id, []):
            true_positive += 1
    return true_positive


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def remaining_decision_budget(group: pd.DataFrame, budget_per_left: int) -> Optional[int]:
    if budget_per_left <= 0:
        return None
    decided_count = int(group["rpe_match"].notna().sum())
    return max(0, budget_per_left - decided_count)


def process_group(
    df: pd.DataFrame,
    group_indices: Sequence[int],
    left_text: str,
    k_value: int,
    required_matches: int,
    args: argparse.Namespace,
    client: Optional[OpenAI],
    output_csv: Path,
    output_jsonl: Path,
    iteration: int,
) -> Tuple[int, int, int, List[int]]:
    group_df = df.loc[group_indices]
    if required_matches > 0:
        matched_count = int((group_df["rpe_match"] == 1).sum())
        if matched_count >= required_matches:
            return 0, 0, 0, []
    budget_remaining = remaining_decision_budget(group_df, args.budget_per_left)
    if budget_remaining == 0:
        return 0, 0, 0, []
    pending_mask = group_df["rpe_match"].isna()
    pending_count = int(pending_mask.sum())
    if pending_count == 0:
        return 0, 0, 0, []

    candidates = compute_k_skyband_indices(
        group_df,
        args.dimension_columns,
        k_value,
        args.skyband_dominated_count_cache,
    )
    new_indices = [idx for idx in candidates if pd.isna(df.at[idx, "rpe_match"])]
    if budget_remaining is not None:
        new_indices = new_indices[:budget_remaining]
    print(f"{k_value}_skyband candidates: {len(candidates)}")
    print(f"{k_value}_skyband new increased candidates: {len(new_indices)}")
    if not new_indices:
        return 0, 0, 0, []

    no_more_pending = pending_count == len(new_indices)
    budget_will_be_exhausted = (
        budget_remaining is not None and len(new_indices) >= budget_remaining
    )
    allow_partial = no_more_pending or budget_will_be_exhausted
    if allow_partial:
        processable_indices = new_indices
    else:
        full_batch_count = (len(new_indices) // args.batch_size) * args.batch_size
        if full_batch_count == 0:
            print(
                f"Waiting for at least {args.batch_size} new candidates "
                "before calling the LLM."
            )
            return 0, 0, 0, []
        processable_indices = new_indices[:full_batch_count]

    processed = 0
    completion_tokens_used = 0
    prompt_tokens_used = 0
    judged_indices: List[int] = []
    for batch_indices in chunked(processable_indices, args.batch_size):
        candidate_texts = [str(df.at[idx, args.right_text_column]) for idx in batch_indices]
        if args.single_pair_yes_no:
            prompt = build_single_pair_yes_no_prompt(
                left_text=left_text,
                right_text=candidate_texts[0],
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode="single_pair_yes_no",
                dry_run=args.dry_run,
            )
            valid_matches = mark_candidate_results(
                df,
                batch_indices,
                candidate_texts,
                llm_result["candidate_results"],
                f"[iter {iteration} | skyband] {llm_result['reasoning']}",
            )
        elif args.batch_mode == "left_anchored_multi_right":
            prompt = build_left_anchored_prompt(
                left_text=left_text,
                right_texts=candidate_texts,
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode=args.batch_mode,
                dry_run=args.dry_run,
                system_prompt=DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT,
                structured_candidate_ids=range(1, len(candidate_texts) + 1),
            )
            valid_matches = mark_candidate_results(
                df,
                batch_indices,
                candidate_texts,
                llm_result["candidate_results"],
                f"[iter {iteration} | skyband] {llm_result['reasoning']}",
            )
            # import pdb;pdb.set_trace()
        else:
            prompt = build_match_prompt(
                left_text=left_text,
                right_texts=candidate_texts,
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode=args.batch_mode,
                dry_run=args.dry_run,
            )
            valid_matches = mark_match(
                df,
                batch_indices,
                candidate_texts,
                llm_result["matches"],
                f"[iter {iteration} | skyband] {llm_result['reasoning']}",
            )
        # import pdb; pdb.set_trace()
        batch_completion_tokens = int(llm_result.get("completion_tokens", 0) or 0)
        batch_prompt_tokens = int(llm_result.get("prompt_tokens", 0) or 0)
        completion_tokens_used += batch_completion_tokens
        prompt_tokens_used += batch_prompt_tokens

        record = {
            "call_type": "skyband_batch",
            "iteration": iteration,
            "k_value": k_value,
            "row_indices": batch_indices,
            "left_id": int(df.at[batch_indices[0], "_left_id"]),
            "right_ids": [int(df.at[idx, "_right_id"]) for idx in batch_indices],
            "candidate_right_texts": candidate_texts,
            "left_text": left_text,
            "system_prompt": (
                DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT
                if args.batch_mode == "left_anchored_multi_right"
                and not args.single_pair_yes_no
                else None
            ),
            "prompt": prompt,
            "matches": valid_matches,
            "raw_matches": llm_result["matches"],
            "candidate_results": llm_result["candidate_results"],
            "pair_results": llm_result["pair_results"],
            "reasoning": llm_result["reasoning"],
            "response_text": llm_result["response_text"],
            "parse_error": llm_result["parse_error"],
            "attempts_used": llm_result["attempts_used"],
            "duration_seconds": llm_result["duration_seconds"],
            "completion_tokens": batch_completion_tokens,
            "prompt_tokens": batch_prompt_tokens,
            "model": args.model,
            "batch_mode": args.batch_mode,
            "single_pair_yes_no": args.single_pair_yes_no,
            "openai_structured_output": (
                args.batch_mode == "left_anchored_multi_right"
                and not args.single_pair_yes_no
            ),
            "timestamp": time.time(),
            "dry_run": args.dry_run,
        }
        # import pdb;pdb.set_trace()
        if llm_result["raw_response"] is not None:
            record["raw_response"] = llm_result["raw_response"]
        if args.compact_jsonl:
            record.pop("left_text", None)
            record.pop("candidate_right_texts", None)
        append_jsonl(output_jsonl, record)
        append_minimal_results(df, batch_indices, output_csv)
        processed += len(batch_indices)
        judged_indices.extend(batch_indices)
    return processed, completion_tokens_used, prompt_tokens_used, judged_indices


def process_explicit_indices(
    df: pd.DataFrame,
    batch_indices_to_process: Sequence[int],
    left_text: str,
    k_value: int,
    args: argparse.Namespace,
    client: Optional[OpenAI],
    output_csv: Path,
    output_jsonl: Path,
    iteration: int,
) -> Tuple[int, int, int, List[int]]:
    if not batch_indices_to_process:
        return 0, 0, 0, []

    processed = 0
    completion_tokens_used = 0
    prompt_tokens_used = 0
    judged_indices: List[int] = []

    for batch_indices in chunked(list(batch_indices_to_process), args.batch_size):
        candidate_texts = [str(df.at[idx, args.right_text_column]) for idx in batch_indices]
        if args.single_pair_yes_no:
            prompt = build_single_pair_yes_no_prompt(
                left_text=left_text,
                right_text=candidate_texts[0],
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode="single_pair_yes_no",
                dry_run=args.dry_run,
            )
            valid_matches = mark_candidate_results(
                df,
                batch_indices,
                candidate_texts,
                llm_result["candidate_results"],
                f"[iter {iteration} | round_robin] {llm_result['reasoning']}",
            )
        elif args.batch_mode == "left_anchored_multi_right":
            prompt = build_left_anchored_prompt(
                left_text=left_text,
                right_texts=candidate_texts,
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode=args.batch_mode,
                dry_run=args.dry_run,
                system_prompt=DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT,
                structured_candidate_ids=range(1, len(candidate_texts) + 1),
            )
            valid_matches = mark_candidate_results(
                df,
                batch_indices,
                candidate_texts,
                llm_result["candidate_results"],
                f"[iter {iteration} | round_robin] {llm_result['reasoning']}",
            )
        else:
            prompt = build_match_prompt(
                left_text=left_text,
                right_texts=candidate_texts,
                prompt_template=args.prompt_template,
                left_label=args.left_label,
                right_label=args.right_label,
            )
            llm_result = invoke_llm(
                client=client,
                model=args.model,
                user_prompt=prompt,
                retry_limit=args.retry_limit,
                sleep_seconds=args.sleep_seconds,
                batch_mode=args.batch_mode,
                dry_run=args.dry_run,
            )
            valid_matches = mark_match(
                df,
                batch_indices,
                candidate_texts,
                llm_result["matches"],
                f"[iter {iteration} | round_robin] {llm_result['reasoning']}",
            )

        batch_completion_tokens = int(llm_result.get("completion_tokens", 0) or 0)
        batch_prompt_tokens = int(llm_result.get("prompt_tokens", 0) or 0)
        completion_tokens_used += batch_completion_tokens
        prompt_tokens_used += batch_prompt_tokens

        record = {
            "call_type": "round_robin_batch",
            "iteration": iteration,
            "k_value": k_value,
            "row_indices": batch_indices,
            "left_id": int(df.at[batch_indices[0], "_left_id"]),
            "right_ids": [int(df.at[idx, "_right_id"]) for idx in batch_indices],
            "candidate_right_texts": candidate_texts,
            "left_text": left_text,
            "system_prompt": (
                DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT
                if args.batch_mode == "left_anchored_multi_right"
                and not args.single_pair_yes_no
                else None
            ),
            "prompt": prompt,
            "matches": valid_matches,
            "raw_matches": llm_result["matches"],
            "candidate_results": llm_result["candidate_results"],
            "pair_results": llm_result["pair_results"],
            "reasoning": llm_result["reasoning"],
            "response_text": llm_result["response_text"],
            "parse_error": llm_result["parse_error"],
            "attempts_used": llm_result["attempts_used"],
            "duration_seconds": llm_result["duration_seconds"],
            "completion_tokens": batch_completion_tokens,
            "prompt_tokens": batch_prompt_tokens,
            "model": args.model,
            "batch_mode": args.batch_mode,
            "single_pair_yes_no": args.single_pair_yes_no,
            "openai_structured_output": (
                args.batch_mode == "left_anchored_multi_right"
                and not args.single_pair_yes_no
            ),
            "round_robin_l_value": int(args.round_robin_l_by_left[int(df.at[batch_indices[0], "_left_id"])]),
            "timestamp": time.time(),
            "dry_run": args.dry_run,
        }
        if llm_result["raw_response"] is not None:
            record["raw_response"] = llm_result["raw_response"]
        if args.compact_jsonl:
            record.pop("left_text", None)
            record.pop("candidate_right_texts", None)
        append_jsonl(output_jsonl, record)
        append_minimal_results(df, batch_indices, output_csv)
        processed += len(batch_indices)
        judged_indices.extend(batch_indices)

    return processed, completion_tokens_used, prompt_tokens_used, judged_indices


def collect_round_robin_indices(
    df: pd.DataFrame,
    group_indices: Sequence[int],
    left_id: int,
    quota: int,
    args: argparse.Namespace,
) -> Tuple[List[int], int]:
    if quota <= 0:
        return [], int(args.round_robin_k_by_left.get(left_id, args.initial_k))

    collected: List[int] = []
    collected_set: set[int] = set()
    group_size = len(group_indices)
    max_k = max(1, group_size)

    while len(collected) < quota:
        group_df = df.loc[group_indices]
        pending_count = int(group_df["rpe_match"].isna().sum())
        if pending_count == 0:
            break

        current_k = int(args.round_robin_k_by_left.get(left_id, args.initial_k))
        candidates = compute_k_skyband_indices(
            group_df,
            args.dimension_columns,
            current_k,
            args.skyband_dominated_count_cache,
        )
        new_indices = [
            idx
            for idx in candidates
            if pd.isna(df.at[idx, "rpe_match"]) and idx not in collected_set
        ]

        if new_indices:
            remaining_quota = quota - len(collected)
            selected = new_indices[:remaining_quota]
            collected.extend(selected)
            collected_set.update(selected)
            if len(selected) < len(new_indices):
                break

        next_k = min(max_k, current_k + max(1, args.k_increment))
        if next_k == current_k:
            break
        args.round_robin_k_by_left[left_id] = next_k

    return collected, int(args.round_robin_k_by_left.get(left_id, args.initial_k))


def main() -> None:
    global DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT
    parser = argparse.ArgumentParser(
        description="Iteratively expand k-skyband groups per left item and validate matches via batch prompting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dataset/probe"),
        help="Directory to store all generated outputs (CSV, JSONL, metrics, snapshots).",
    )
    parser.add_argument(
        "--blocking-path",
        type=Path,
        default=Path("runs/dataset/dimensions/pairs_indexed_scored_full_isolation_join.csv"),
        help="Path to the candidate pair CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to store the augmented CSV with rpe_match results.",
    )
    parser.add_argument(
        "--iteration-output-dir",
        type=Path,
        default=None,
        help="Directory to store per-iteration CSV snapshots.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=1,
        help="Write an iteration snapshot every N iterations (1 = every iteration, 0 = disable snapshots).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Where to store raw LLM responses (JSONL).",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="Where to log iteration-level metrics.",
    )
    parser.add_argument(
        "--subproblem-metrics-jsonl",
        type=Path,
        default=None,
        help="Where to log cumulative metrics after each left subproblem probe.",
    )
    parser.add_argument(
        "--round-robin-state-json",
        type=Path,
        default=None,
        help="Where to persist round-robin l_r and per-left skyband depth for resume.",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=Path("biodex/combined_data/biodex_ground_truth_adjusted.json"),
        help="Path to ground_truth.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="meta-llama/llama-3.3-70b-instruct",
        default="gpt-4o-mini",
        help="Model identifier to query via the OpenAI-compatible chat completions API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Explicit API key; falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Override the API base URL if needed.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=5,
        help="Number of retries after the initial attempt when parsing fails.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Max right-side candidates to send in a single batch prompt.",
    )
    parser.add_argument(
        "--num-batch-per-unit",
        type=int,
        default=10,
        help="Minimum number of batch-size units to probe per left item before applying l_r.",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=1,
        help="Initial k value for the k-skyband per _left_id group.",
    )
    parser.add_argument(
        "--k-increment",
        type=int,
        default=1,
        help="Additive increment applied to k after each pass.",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=0,
        help="Stop processing a group once it has at least this many matched reactions (0 disables).",
    )
    parser.add_argument(
        "--budget-per-left",
        "--buget-per-left",
        dest="budget_per_left",
        type=int,
        default=10000,
        help="Stop processing a left item once this many candidates have been judged (0 disables).",
    )
    parser.add_argument(
        "--global-target-positive",
        type=int,
        default=0,
        help="Stop the whole run once this many positive points have been found (0 disables).",
    )
    parser.add_argument(
        "--global-target-decided",
        type=int,
        default=5000,
        help="Stop the whole run once this many points have been judged (0 disables).",
    )
    parser.add_argument(
        "--round-robin-update-every",
        type=int,
        default=2,
        help="Update each left-side l_r value every N full round-robin sweeps.",
    )
    parser.add_argument(
        "--round-robin-initial-l",
        type=int,
        default=1,
        help="Initial l_r value for every left-side subproblem.",
    )
    parser.add_argument(
        "--round-robin-max-l",
        type=int,
        default=0,
        help="Optional cap for l_r after doubling (0 disables the cap).",
    )
    parser.add_argument(
        "--dimension-columns",
        nargs="*",
        default=["score_summary", "score_target_aware"],
        help="Score columns used to compute skybands. Leave empty to process all candidates without skyband filtering.",
    )
    parser.add_argument(
        "--left-json",
        type=Path,
        default=Path("biodex/combined_data/biodex_sample_adjusted.json"),
        help="Path to left_data.json containing left-side records.",
    )
    parser.add_argument(
        "--right-json",
        type=Path,
        default=Path("biodex/combined_data/biodex_terms.json"),
        help="Path to right_data.json containing right-side records.",
    )
    parser.add_argument(
        "--left-text-key",
        type=str,
        default="medical_article",
        help="Key in left_data.json for the text to match.",
    )
    parser.add_argument(
        "--right-text-key",
        type=str,
        default="reaction",
        help="Key in right_data.json for the text to match.",
    )
    parser.add_argument(
        "--left-id-key",
        type=str,
        default=None,
        help="Optional id key in left_data.json if _left_id is not the index.",
    )
    parser.add_argument(
        "--right-id-key",
        type=str,
        default=None,
        help="Optional id key in right_data.json if _right_id is not the index.",
    )
    parser.add_argument(
        "--ground-truth-key",
        type=str,
        default="ground_truth_reactions",
        help="Key in ground_truth.json containing matching right-side strings.",
    )
    parser.add_argument(
        "--ground-truth-id-key",
        type=str,
        default="id",
        help="Optional key in ground_truth.json that maps to _left_id.",
    )
    parser.add_argument(
        "--left-label",
        type=str,
        default="Medical Article",
        help="Label used in the prompt for the left item.",
    )
    parser.add_argument(
        "--right-label",
        type=str,
        default="Reaction",
        help="Label used in the prompt for right-side candidates.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        required=True,
        help="Plain-text system prompt used for every LLM judgment request.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Prompt template for left_multi_right or left_anchored_multi_right mode. "
            "When --single-pair-yes-no is enabled, the available placeholders are "
            "{left_label}, {right_label}, {left_text}, and {right_text}."
        ),
    )
    parser.add_argument(
        "--prompt-template-path",
        type=Path,
        default=None,
        help="Optional path to a prompt template file (overrides --prompt-template).",
    )
    parser.add_argument(
        "--single-pair-yes-no",
        action="store_true",
        help=(
            "With --batch-size 1, compare one left-right pair per prompt and require "
            "an exact YES or NO response."
        ),
    )
    parser.add_argument(
        "--pair-prompt-template",
        type=str,
        default=(
            "For each {left_label}-{right_label} pair, decide if they refer to the same real-world entity.\n\n"
            "Pairs ({pair_count}):\n"
            "{pair_items}\n\n"
            "Respond with a JSON object like:\n"
            "{output_format}\n"
            "Include an entry for every pair_id (use match=false when they do not match).\n"
            "Do not include any additional keys."
        ),
        help="Prompt template for pair_multi mode with placeholders: "
        "{left_label}, {right_label}, {pair_items}, {pair_count}, {output_format}.",
    )
    parser.add_argument(
        "--pair-prompt-template-path",
        type=Path,
        default=None,
        help="Optional path to a pair prompt template file (overrides --pair-prompt-template).",
    )
    parser.add_argument(
        "--batch-mode",
        type=str,
        default="left_anchored_multi_right",
        choices=["left_multi_right", "left_anchored_multi_right", "pair_multi"],
        help="Batching strategy for prompts: left_multi_right or pair_multi.",
    )
    parser.add_argument(
        "--compact-jsonl",
        action="store_true",
        help="Omit duplicated source text fields from LLM response JSONL while keeping prompt/response/raw_response.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts and outputs without calling the API.",
    )

    args = parser.parse_args()
    if args.left_id_key == "":
        args.left_id_key = None
    if args.right_id_key == "":
        args.right_id_key = None
    if args.ground_truth_id_key == "":
        args.ground_truth_id_key = None

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.single_pair_yes_no and args.batch_size != 1:
        raise SystemExit("--single-pair-yes-no requires --batch-size 1.")
    if args.single_pair_yes_no and args.batch_mode == "pair_multi":
        raise SystemExit("--single-pair-yes-no cannot be combined with --batch-mode pair_multi.")
    if args.num_batch_per_unit <= 0:
        raise SystemExit("--num-batch-per-unit must be positive.")
    if args.initial_k <= 0:
        raise SystemExit("--initial-k must be positive.")
    if args.k_increment <= 0:
        raise SystemExit("--k-increment must be positive.")
    if args.l < 0:
        raise SystemExit("--l must be non-negative.")
    if args.budget_per_left < 0:
        raise SystemExit("--budget-per-left must be non-negative.")
    if args.global_target_positive < 0:
        raise SystemExit("--global-target-positive must be non-negative.")
    if args.global_target_decided < 0:
        raise SystemExit("--global-target-decided must be non-negative.")
    if args.round_robin_update_every <= 0:
        raise SystemExit("--round-robin-update-every must be positive.")
    if args.round_robin_initial_l <= 0:
        raise SystemExit("--round-robin-initial-l must be positive.")
    if args.round_robin_max_l < 0:
        raise SystemExit("--round-robin-max-l must be non-negative.")
    if args.snapshot_every < 0:
        raise SystemExit("--snapshot-every must be non-negative.")

    base_dir = Path(__file__).resolve().parent
    system_prompt_path = resolve_path(base_dir, args.system_prompt_path)
    if not system_prompt_path.exists():
        raise SystemExit(f"System prompt file not found: {system_prompt_path}")
    DEFAULT_LEFT_ANCHORED_SYSTEM_PROMPT = system_prompt_path.read_text(
        encoding="utf-8"
    ).rstrip("\n")
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocking_path = resolve_path(base_dir, args.blocking_path)
    output_csv = (
        resolve_path(base_dir, args.output_csv)
        if args.output_csv
        else output_dir / "pairs_skyband_llm_results.csv"
    )
    output_jsonl = (
        resolve_path(base_dir, args.output_jsonl)
        if args.output_jsonl
        else output_dir / "llm_skyband_prompting_responses.jsonl"
    )
    iteration_output_dir = (
        resolve_path(base_dir, args.iteration_output_dir)
        if args.iteration_output_dir
        else output_dir / "skyband_iteration_snapshots"
    )
    metrics_jsonl = (
        resolve_path(base_dir, args.metrics_jsonl)
        if args.metrics_jsonl
        else output_dir / "skyband_prompting_metrics.jsonl"
    )
    subproblem_metrics_jsonl = (
        resolve_path(base_dir, args.subproblem_metrics_jsonl)
        if args.subproblem_metrics_jsonl
        else output_dir / "skyband_subproblem_metrics.jsonl"
    )
    round_robin_state_json = (
        resolve_path(base_dir, args.round_robin_state_json)
        if args.round_robin_state_json
        else output_dir / "round_robin_state.json"
    )
    ground_truth_path = resolve_path(base_dir, args.ground_truth_path)
    left_json_path = resolve_path(base_dir, args.left_json)
    right_json_path = resolve_path(base_dir, args.right_json)

    if not blocking_path.exists():
        raise SystemExit(f"Blocking result file not found: {blocking_path}")
    if not ground_truth_path.exists():
        raise SystemExit(f"Ground truth file not found: {ground_truth_path}")
    if not left_json_path.exists():
        raise SystemExit(f"Left JSON file not found: {left_json_path}")
    if not right_json_path.exists():
        raise SystemExit(f"Right JSON file not found: {right_json_path}")
    if args.prompt_template_path:
        prompt_template_path = resolve_path(base_dir, args.prompt_template_path)
        if not prompt_template_path.exists():
            raise SystemExit(f"Prompt template file not found: {prompt_template_path}")
        args.prompt_template = prompt_template_path.read_text(encoding="utf-8")
    if args.prompt_template is None:
        if args.single_pair_yes_no:
            args.prompt_template = DEFAULT_SINGLE_PAIR_YES_NO_TEMPLATE
        elif args.batch_mode == "left_anchored_multi_right":
            args.prompt_template = DEFAULT_LEFT_ANCHORED_TEMPLATE
        else:
            args.prompt_template = DEFAULT_MATCH_PROMPT_TEMPLATE
    if args.pair_prompt_template_path:
        pair_prompt_template_path = resolve_path(base_dir, args.pair_prompt_template_path)
        if not pair_prompt_template_path.exists():
            raise SystemExit(
                f"Pair prompt template file not found: {pair_prompt_template_path}"
            )
        args.pair_prompt_template = pair_prompt_template_path.read_text(encoding="utf-8")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    subproblem_metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    round_robin_state_json.parent.mkdir(parents=True, exist_ok=True)
    iteration_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_blocking_results(blocking_path, args.dimension_columns).reset_index(drop=True)
    left_map = load_indexed_text(left_json_path, args.left_text_key, args.left_id_key)
    right_map = load_indexed_text(right_json_path, args.right_text_key, args.right_id_key)
    args.left_text_column = "left_text"
    args.right_text_column = "right_text"
    if args.left_text_column not in df.columns:
        df[args.left_text_column] = ""
    if args.right_text_column not in df.columns:
        df[args.right_text_column] = ""
    df["_left_id"] = df["_left_id"].astype(int)
    df["_right_id"] = df["_right_id"].astype(int)
    df[args.left_text_column] = df["_left_id"].map(left_map).fillna("").astype(str)
    df[args.right_text_column] = df["_right_id"].map(right_map).fillna("").astype(str)
    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(int)
    if "rpe_match" not in df.columns:
        df["rpe_match"] = pd.NA
    if "llm_reasoning" not in df.columns:
        df["llm_reasoning"] = ""

    df = merge_existing_results(df, output_csv)
    truth_map = load_ground_truth(
        ground_truth_path,
        args.ground_truth_key,
        args.ground_truth_id_key,
    )
    api_key = ensure_api_key(args.api_key, args.dry_run)
    client: Optional[OpenAI] = None
    if not args.dry_run:
        if args.base_url and args.base_url.strip():
            client = OpenAI(base_url=args.base_url.strip(), api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)

    iteration = 0
    k_value = args.initial_k
    args.skyband_dominated_count_cache = {}
    left_ids_in_order = [int(left_id) for left_id, _ in df.groupby("_left_id", sort=False)]
    args.round_robin_l_by_left = {
        left_id: int(args.round_robin_initial_l) for left_id in left_ids_in_order
    }
    args.round_robin_k_by_left = {
        left_id: int(args.initial_k) for left_id in left_ids_in_order
    }
    round_robin_period_stats: Dict[int, Dict[str, int]] = {
        left_id: {"processed": 0, "positives": 0} for left_id in left_ids_in_order
    }
    round_robin_state = load_round_robin_state(round_robin_state_json)
    if round_robin_state:
        state_l_by_left = round_robin_state.get("l_by_left", {})
        state_k_by_left = round_robin_state.get("k_by_left", {})
        state_period_stats = round_robin_state.get("period_stats", {})
        if isinstance(state_l_by_left, dict):
            for left_id_text, value in state_l_by_left.items():
                left_id = int(left_id_text)
                if left_id in args.round_robin_l_by_left:
                    args.round_robin_l_by_left[left_id] = max(1, int(value))
        if isinstance(state_k_by_left, dict):
            for left_id_text, value in state_k_by_left.items():
                left_id = int(left_id_text)
                if left_id in args.round_robin_k_by_left:
                    args.round_robin_k_by_left[left_id] = max(args.initial_k, int(value))
        if isinstance(state_period_stats, dict):
            for left_id_text, value in state_period_stats.items():
                left_id = int(left_id_text)
                if left_id not in round_robin_period_stats or not isinstance(value, dict):
                    continue
                round_robin_period_stats[left_id] = {
                    "processed": max(0, int(value.get("processed", 0))),
                    "positives": max(0, int(value.get("positives", 0))),
                }
        iteration = int(round_robin_state.get("iteration", 0) or 0)
        print(
            f"Loaded round-robin state from {round_robin_state_json} "
            f"at iteration {iteration}."
        )

    def remaining_pending_points() -> int:
        remaining = 0
        for _, group in df.groupby("_left_id", sort=False):
            matched_count = int((group["rpe_match"] == 1).sum())
            if args.l > 0 and matched_count >= args.l:
                continue
            pending_count = int(group["rpe_match"].isna().sum())
            budget_remaining = remaining_decision_budget(group, args.budget_per_left)
            if budget_remaining == 0:
                continue
            if budget_remaining is None:
                remaining += pending_count
            else:
                remaining += min(pending_count, budget_remaining)
        return remaining

    total_pending = remaining_pending_points()
    total_completion_tokens, total_prompt_tokens = sum_response_tokens(output_jsonl)
    if total_completion_tokens or total_prompt_tokens:
        print(
            "Loaded historical token usage from response JSONL: "
            f"prompt={total_prompt_tokens}, completion={total_completion_tokens}."
        )
    running_metrics = compute_progress_metrics(
        df, truth_map, args.right_text_column
    )
    subproblem_step = count_nonempty_lines(subproblem_metrics_jsonl)

    while total_pending > 0:
        current_metrics = running_metrics
        if (
            args.global_target_positive > 0
            and current_metrics["positive_points"] >= args.global_target_positive
        ):
            print(
                "Global target positive count reached before starting the next round: "
                f"{current_metrics['positive_points']} >= {args.global_target_positive}"
            )
            break
        if (
            args.global_target_decided > 0
            and current_metrics["decided_points"] >= args.global_target_decided
        ):
            print(
                "Global target decided count reached before starting the next round: "
                f"{current_metrics['decided_points']} >= {args.global_target_decided}"
            )
            break
        # import pdb; pdb.set_trace()
        print(f"total_pending: {total_pending}")
        iteration += 1
        processed_this_iter = 0
        iteration_completion_tokens = 0
        iteration_prompt_tokens = 0
        iteration_judged_indices: List[int] = []
        round_robin_stats: Dict[int, Dict[str, int]] = {}
        subproblem_position = 0

        groups = list(df.groupby("_left_id", sort=False))
        active_groups = []
        for left_id, group in groups:
            matched_count = int((group["rpe_match"] == 1).sum())
            if args.l > 0 and matched_count >= args.l:
                continue
            budget_remaining = remaining_decision_budget(group, args.budget_per_left)
            if budget_remaining == 0:
                continue
            active_groups.append((left_id, group))

        if args.batch_mode == "pair_multi":
            per_group_candidates: List[List[int]] = []
            for left_id, group in active_groups:
                group_indices = group.index.tolist()
                group_df = df.loc[group_indices]
                pending_mask = group_df["rpe_match"].isna()
                pending_count = int(pending_mask.sum())
                if pending_count == 0:
                    continue
                candidates = compute_k_skyband_indices(
                    group_df,
                    args.dimension_columns,
                    k_value,
                    args.skyband_dominated_count_cache,
                )
                new_indices = [
                    idx for idx in candidates if pd.isna(df.at[idx, "rpe_match"])
                ]
                budget_remaining = remaining_decision_budget(group_df, args.budget_per_left)
                if budget_remaining is not None:
                    new_indices = new_indices[:budget_remaining]
                if not new_indices:
                    continue
                per_group_candidates.append(new_indices)

            candidate_indices = interleave_groups(per_group_candidates)

            for batch_indices in chunked(candidate_indices, args.batch_size):
                left_texts = [str(df.at[idx, args.left_text_column]) for idx in batch_indices]
                right_texts = [str(df.at[idx, args.right_text_column]) for idx in batch_indices]
                prompt = build_pair_prompt(
                    left_texts=left_texts,
                    right_texts=right_texts,
                    prompt_template=args.pair_prompt_template,
                    left_label=args.left_label,
                    right_label=args.right_label,
                )
                llm_result = invoke_llm(
                    client=client,
                    model=args.model,
                    user_prompt=prompt,
                    retry_limit=args.retry_limit,
                    sleep_seconds=args.sleep_seconds,
                    batch_mode=args.batch_mode,
                    dry_run=args.dry_run,
                )
                valid_matches = mark_pair_results(
                    df,
                    batch_indices,
                    right_texts,
                    llm_result["pair_results"],
                    f"[iter {iteration} | pair] {llm_result['reasoning']}",
                )
                batch_completion_tokens = int(llm_result.get("completion_tokens", 0) or 0)
                batch_prompt_tokens = int(llm_result.get("prompt_tokens", 0) or 0)
                iteration_completion_tokens += batch_completion_tokens
                iteration_prompt_tokens += batch_prompt_tokens

                record = {
                    "call_type": "pair_batch",
                    "iteration": iteration,
                    "k_value": k_value,
                    "row_indices": batch_indices,
                    "left_ids": [int(df.at[idx, "_left_id"]) for idx in batch_indices],
                    "right_ids": [int(df.at[idx, "_right_id"]) for idx in batch_indices],
                    "left_texts": left_texts,
                    "right_texts": right_texts,
                    "prompt": prompt,
                    "matches": valid_matches,
                    "raw_matches": llm_result["matches"],
                    "pair_results": llm_result["pair_results"],
                    "reasoning": llm_result["reasoning"],
                    "response_text": llm_result["response_text"],
                    "parse_error": llm_result["parse_error"],
                    "attempts_used": llm_result["attempts_used"],
                    "duration_seconds": llm_result["duration_seconds"],
                    "completion_tokens": batch_completion_tokens,
                    "prompt_tokens": batch_prompt_tokens,
                    "model": args.model,
                    "batch_mode": args.batch_mode,
                    "timestamp": time.time(),
                    "dry_run": args.dry_run,
                }
                if llm_result["raw_response"] is not None:
                    record["raw_response"] = llm_result["raw_response"]
                if args.compact_jsonl:
                    record.pop("left_texts", None)
                    record.pop("right_texts", None)
                append_jsonl(output_jsonl, record)
                append_minimal_results(df, batch_indices, output_csv)
                processed_this_iter += len(batch_indices)
                iteration_judged_indices.extend(batch_indices)
        else:
            for left_id, group in active_groups:
                group_indices = group.index.tolist()
                group_df = df.loc[group_indices]
                budget_remaining = remaining_decision_budget(group_df, args.budget_per_left)
                if budget_remaining == 0:
                    continue
                pending_count = int(group_df["rpe_match"].isna().sum())
                if pending_count == 0:
                    continue

                left_id_int = int(left_id)
                l_value = int(args.round_robin_l_by_left.get(left_id_int, args.round_robin_initial_l))
                quota = l_value * args.batch_size * args.num_batch_per_unit
                if budget_remaining is not None:
                    quota = min(quota, budget_remaining)
                quota = min(quota, pending_count)
                if quota <= 0:
                    continue

                candidate_indices, current_left_k = collect_round_robin_indices(
                    df=df,
                    group_indices=group_indices,
                    left_id=left_id_int,
                    quota=quota,
                    args=args,
                )
                if not candidate_indices:
                    continue

                left_text = str(group.iloc[0][args.left_text_column])
                print(
                    f"Processing _left_id={left_id} | l_r={l_value} | "
                    f"quota={quota} | selected={len(candidate_indices)} | k_until={current_left_k}"
                )
                (
                    processed,
                    completion_tokens_used,
                    prompt_tokens_used,
                    judged_indices,
                ) = process_explicit_indices(
                    df=df,
                    batch_indices_to_process=candidate_indices,
                    left_text=left_text,
                    k_value=current_left_k,
                    args=args,
                    client=client,
                    output_csv=output_csv,
                    output_jsonl=output_jsonl,
                    iteration=iteration,
                )
                processed_this_iter += processed
                iteration_completion_tokens += completion_tokens_used
                iteration_prompt_tokens += prompt_tokens_used
                iteration_judged_indices.extend(judged_indices)
                positive_this_left = int((df.loc[judged_indices, "rpe_match"] == 1).sum())
                true_positive_this_left = count_true_positives_for_indices(
                    df,
                    judged_indices,
                    truth_map,
                    args.right_text_column,
                )
                running_metrics["decided_points"] += int(processed)
                running_metrics["positive_points"] += positive_this_left
                running_metrics["true_positive_points"] += true_positive_this_left
                subproblem_step += 1
                subproblem_position += 1
                append_jsonl(
                    subproblem_metrics_jsonl,
                    {
                        "subproblem_step": subproblem_step,
                        "iteration": iteration,
                        "subproblem_position": subproblem_position,
                        "left_id": left_id_int,
                        "l_value": l_value,
                        "quota": quota,
                        "processed_this_subproblem": int(processed),
                        "positive_this_subproblem": positive_this_left,
                        "true_positive_this_subproblem": true_positive_this_left,
                        "k_value": current_left_k,
                        "decided_points": running_metrics["decided_points"],
                        "positive_points": running_metrics["positive_points"],
                        "true_positive_points": running_metrics["true_positive_points"],
                        "timestamp": time.time(),
                    },
                )
                period_stat = round_robin_period_stats.setdefault(
                    left_id_int, {"processed": 0, "positives": 0}
                )
                period_stat["processed"] += int(processed)
                period_stat["positives"] += positive_this_left
                round_robin_stats[left_id_int] = {
                    "l_value": l_value,
                    "quota": quota,
                    "processed": processed,
                    "positives": positive_this_left,
                    "k_until": current_left_k,
                }

        if args.batch_mode != "pair_multi" and args.round_robin_k_by_left:
            k_value = max(args.round_robin_k_by_left.values())

        metrics = compute_progress_metrics(df, truth_map, args.right_text_column)
        running_metrics = metrics.copy()
        total_completion_tokens += iteration_completion_tokens
        total_prompt_tokens += iteration_prompt_tokens
        metrics_record = {
            "iteration": iteration,
            "k_value": k_value,
            "decided_points": metrics["decided_points"],
            "positive_points": metrics["positive_points"],
            "true_positive_points": metrics["true_positive_points"],
            "timestamp": time.time(),
            "processed_this_iteration": processed_this_iter,
            "budget_per_left": args.budget_per_left,
            "batch_size": args.batch_size,
            "num_batch_per_unit": args.num_batch_per_unit,
            "single_pair_yes_no": args.single_pair_yes_no,
            "openai_structured_output": (
                args.batch_mode == "left_anchored_multi_right"
                and not args.single_pair_yes_no
            ),
            "global_target_positive": args.global_target_positive,
            "global_target_decided": args.global_target_decided,
            "snapshot_every": args.snapshot_every,
            "round_robin_update_every": args.round_robin_update_every,
            "round_robin_l_min": min(args.round_robin_l_by_left.values()) if args.round_robin_l_by_left else 0,
            "round_robin_l_max": max(args.round_robin_l_by_left.values()) if args.round_robin_l_by_left else 0,
            "round_robin_l_sum": sum(args.round_robin_l_by_left.values()) if args.round_robin_l_by_left else 0,
            "iteration_completion_tokens": iteration_completion_tokens,
            "iteration_prompt_tokens": iteration_prompt_tokens,
            "cumulative_completion_tokens": total_completion_tokens,
            "cumulative_prompt_tokens": total_prompt_tokens,
        }
        append_jsonl(metrics_jsonl, metrics_record)
        if args.snapshot_every > 0 and iteration % args.snapshot_every == 0:
            iteration_snapshot_path = iteration_output_dir / f"iteration_{iteration:03d}.csv"
            write_judged_pairs_snapshot(df, iteration_judged_indices, iteration_snapshot_path)
        print(
            f"Iteration {iteration} | k={k_value} | processed={processed_this_iter} | "
            f"decided={metrics['decided_points']} | positives={metrics['positive_points']} | "
            f"true_positives={metrics['true_positive_points']} | "
            f"iter_tokens(prompt={iteration_prompt_tokens}, completion={iteration_completion_tokens}) | "
            f"total_tokens(prompt={total_prompt_tokens}, completion={total_completion_tokens})"
        )

        if args.batch_mode != "pair_multi" and iteration % args.round_robin_update_every == 0:
            increases = 0
            decreases = 0
            unchanged = 0
            ratio_rows: List[Tuple[float, int, int, int]] = []
            for left_id, _ in active_groups:
                left_id_int = int(left_id)
                stats = round_robin_period_stats.get(
                    left_id_int, {"processed": 0, "positives": 0}
                )
                processed = int(stats.get("processed", 0))
                positives = int(stats.get("positives", 0))
                ratio = (positives / processed) if processed > 0 else 0.0
                ratio_rows.append((ratio, positives, processed, left_id_int))

            ratio_rows.sort(key=lambda item: (-item[0], -item[1], item[3]))
            update_count = math.ceil(len(ratio_rows) * 0.25) if ratio_rows else 0
            top_left_ids = {item[3] for item in ratio_rows[:update_count]}
            bottom_left_ids = {
                item[3] for item in ratio_rows[-update_count:]
            } - top_left_ids

            for _, _, _, left_id_int in ratio_rows:
                old_l = int(args.round_robin_l_by_left.get(left_id_int, args.round_robin_initial_l))
                if left_id_int in top_left_ids:
                    new_l = old_l * 2
                    if args.round_robin_max_l > 0:
                        new_l = min(new_l, args.round_robin_max_l)
                elif left_id_int in bottom_left_ids:
                    new_l = max(1, old_l // 2)
                else:
                    new_l = old_l

                if new_l > old_l:
                    increases += 1
                elif new_l < old_l:
                    decreases += 1
                else:
                    unchanged += 1
                args.round_robin_l_by_left[left_id_int] = new_l
            print(
                f"Round-robin l_r update at iteration {iteration}: "
                f"top_25_percent={len(top_left_ids)}, bottom_25_percent={len(bottom_left_ids)}, "
                f"increased={increases}, decreased={decreases}, unchanged={unchanged}"
            )
            for left_id in left_ids_in_order:
                round_robin_period_stats[left_id] = {"processed": 0, "positives": 0}

        if args.batch_mode != "pair_multi":
            write_round_robin_state(
                round_robin_state_json,
                iteration,
                args.round_robin_l_by_left,
                args.round_robin_k_by_left,
                round_robin_period_stats,
            )

        total_pending = remaining_pending_points()
        if (
            args.global_target_positive > 0
            and metrics["positive_points"] >= args.global_target_positive
        ):
            print(
                "Global target positive count reached: "
                f"{metrics['positive_points']} >= {args.global_target_positive}"
            )
            break
        if (
            args.global_target_decided > 0
            and metrics["decided_points"] >= args.global_target_decided
        ):
            print(
                "Global target decided count reached: "
                f"{metrics['decided_points']} >= {args.global_target_decided}"
            )
            break
        if total_pending == 0:
            break
        # if processed_this_iter == 0:
        #     # import pdb; pdb.set_trace()
        #     if not active_groups:
        #         break
        #     increment = max(1, args.k_increment)
        #     k_value = k_value + increment
        #     # import pdb; pdb.set_trace()
        #     continue
        # import pdb; pdb.set_trace()
        if args.batch_mode == "pair_multi":
            group_sizes = [len(group) for _, group in active_groups]
            max_group_size = max(group_sizes) if group_sizes else k_value
            increment = max(1, args.k_increment)
            k_value = min(max_group_size, k_value + increment)
        # import pdb; pdb.set_trace()

    final_output_csv = output_csv.with_suffix(output_csv.suffix + ".final.csv")
    write_minimal_results(df, final_output_csv)
    print(
        f"Completed k-skyband prompting. Incremental checkpoint saved to {output_csv}, "
        f"final full results saved to {final_output_csv}, responses in {output_jsonl}."
    )


if __name__ == "__main__":
    main()
