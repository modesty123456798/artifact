#!/usr/bin/env python3
"""Run the three-stage Full Isolation Join reproduction pipeline for one dataset.

Stages are deliberately separate and resumable:

1. ``dimensions`` generates both LLM-mapped left representations and their
   three embedding similarities.
2. ``probe`` runs the reported two-dimension anti-dominance strategy.
3. ``rank`` ranks the LLM-positive candidates at selected probing prefixes.
4. ``metrics`` computes ranking precision from the ranking outputs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

if __package__:
    from .datasets import DATASETS, ROOT, absolute, dataset, spec_path
else:  # Supports direct script execution from the repository root.
    from datasets import DATASETS, ROOT, absolute, dataset, spec_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--stage",
        choices=["dimensions", "probe", "rank", "metrics", "all"],
        default="all",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Defaults to artifact_runs/full_isolation_join/<dataset>.",
    )
    parser.add_argument("--api-key", default="", help="Falls back to OPENAI_API_KEY.")
    parser.add_argument(
        "--guide-source",
        choices=["auto", "saved", "regenerate"],
        default="regenerate",
        help="Regenerate target guides end-to-end by default; use saved only for historical-guide replay.",
    )
    parser.add_argument("--target-decided", type=int, default=None)
    parser.add_argument("--decided-points", type=int, nargs="+")
    parser.add_argument("--ranking-top-k", type=int, default=None)
    parser.add_argument("--overwrite-scores", action="store_true")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Show the fully resolved commands without executing them.",
    )
    return parser.parse_args()


def api_arguments(api_key: str) -> list[str]:
    return ["--api-key", api_key] if api_key else []


def redacted(command: Sequence[str], api_key: str) -> str:
    shown = list(command)
    if api_key:
        for index, value in enumerate(shown[:-1]):
            if value == "--api-key":
                shown[index + 1] = "<redacted>"
    return shlex.join(shown)


def run(command: list[str], *, api_key: str, print_only: bool) -> None:
    print("$", redacted(command, api_key))
    if not print_only:
        subprocess.run(command, cwd=ROOT, check=True)


def verify_probing_prompts(config: dict[str, object]) -> None:
    runner = absolute(config, str(config["probe_runner"]))
    archived_system = spec_path(config, "prompts/join_judgment_system.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
    archived_user = spec_path(config, "prompts/join_judgment_user_template.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
    if not archived_system or not archived_user:
        raise RuntimeError(
            f"Archived probing prompt is empty for {config['name']}."
        )
    if "--system-prompt-path" not in runner.read_text(encoding="utf-8"):
        raise RuntimeError(
            f"{runner} does not support the archived system prompt argument."
        )
def dimensions_command(
    config: dict[str, object], run_root: Path, api_key: str, guide_source: str, overwrite: bool
) -> list[list[str]]:
    dimensions_dir = run_root / "dimensions"
    maps_dir = dimensions_dir / "maps"
    mapping_command = [
        sys.executable,
        str(ROOT / "generate_mappings.py"),
        "--dataset",
        str(config["name"]),
        "--output-dir",
        str(maps_dir),
        "--guide-source",
        guide_source,
        *api_arguments(api_key),
    ]

    args = config["similarity_args"]
    if not isinstance(args, dict):  # pragma: no cover - configuration invariant
        raise TypeError("similarity_args must be a mapping")
    no_context = config["no_context"]
    target_aware = config["target_aware"]
    score_path = dimensions_dir / "pairs_indexed_scored_full_isolation_join.csv"
    score_command = [
        sys.executable,
        str(ROOT / str(config["similarity_script"])),
        "--pairs",
        str(absolute(config, str(config["pairs"]))),
        "--right-json",
        str(absolute(config, str(config["right_json"]))),
        "--right-text-key",
        str(config["right_key"]),
        "--direct-left-json",
        str(absolute(config, str(config["left_json"]))),
        "--direct-text-key",
        str(config["left_key"]),
        "--no-context-left-json",
        str(maps_dir / "no_context_mapped.json"),
        "--no-context-text-key",
        str(no_context["output_key"]),
        "--target-aware-left-json",
        str(maps_dir / "target_aware_mapped.json"),
        "--target-aware-text-key",
        str(target_aware["output_key"]),
        "--direct-score-column",
        str(args["direct_score_column"]),
        "--no-context-score-column",
        str(args["no_context_score_column"]),
        "--target-aware-score-column",
        str(args["target_aware_score_column"]),
        "--output",
        str(score_path),
        "--cache-dir",
        str(dimensions_dir / "embedding_cache"),
        "--cache-embeddings",
        "--model",
        "text-embedding-3-small",
        "--max-input-tokens",
        "8191",
        "--normalize-scores",
        "--write-metadata",
        *api_arguments(api_key),
    ]
    if overwrite:
        score_command.append("--overwrite")
    return [mapping_command, score_command]


def probe_command(
    config: dict[str, object], run_root: Path, api_key: str, target_decided: int
) -> list[str]:
    score_path = run_root / "dimensions/pairs_indexed_scored_full_isolation_join.csv"
    probe_dir = run_root / "probe"
    command = [
        sys.executable,
        str(ROOT / str(config["probe_runner"])),
        "--blocking-path",
        str(score_path),
        "--dimension-columns",
        *(str(value) for value in config["probe_dimensions"]),
        "--ground-truth-path",
        str(absolute(config, str(config["ground_truth_json"]))),
        "--left-json",
        str(absolute(config, str(config["left_json"]))),
        "--right-json",
        str(absolute(config, str(config["right_json"]))),
        "--left-text-key",
        str(config["left_key"]),
        "--right-text-key",
        str(config["right_key"]),
        "--ground-truth-key",
        str(config["ground_truth_key"]),
        "--ground-truth-id-key",
        str(config["ground_truth_id_key"]),
        "--left-label",
        str(config["left_label"]),
        "--right-label",
        str(config["right_label"]),
        "--system-prompt-path",
        str(spec_path(config, "prompts/join_judgment_system.txt")),
        "--prompt-template-path",
        str(spec_path(config, "prompts/join_judgment_user_template.txt")),
        "--batch-mode",
        "left_anchored_multi_right",
        "--model",
        "gpt-4o-mini",
        "--batch-size",
        str(config["batch_size"]),
        "--num-batch-per-unit",
        str(config["num_batch_per_unit"]),
        "--round-robin-update-every",
        str(config["round_robin_update_every"]),
        "--initial-k",
        "1",
        "--k-increment",
        "1",
        "--round-robin-initial-l",
        "1",
        "--round-robin-max-l",
        "0",
        "--budget-per-left",
        "10000",
        "--global-target-positive",
        "0",
        "--global-target-decided",
        str(target_decided),
        "--compact-jsonl",
        "--output-dir",
        str(probe_dir),
        *api_arguments(api_key),
    ]
    return command


def rank_command(
    config: dict[str, object], run_root: Path, api_key: str, decided_points: Sequence[int], top_k: int
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "rank_prefixes.py"),
        "--dataset",
        str(config["name"]),
        "--source-dir",
        str(run_root / "probe"),
        "--output-root",
        str(run_root / "ranking"),
        "--decided-points",
        *(str(value) for value in decided_points),
        "--top-k",
        str(top_k),
        *api_arguments(api_key),
        "--compact-jsonl",
    ]


def metric_commands(config: dict[str, object], run_root: Path, decided_points: Sequence[int], top_k: int) -> list[list[str]]:
    commands: list[list[str]] = []
    for point in decided_points:
        rank_dir = run_root / "ranking" / f"ranking_decided{point}_top{top_k}_openai"
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts/compute_ranking_precision_from_ranked_ids.py"),
                "--ranked-json",
                str(rank_dir / "ranked_pairs_with_matches.json"),
                "--ground-truth-json",
                str(absolute(config, str(config["ground_truth_json"]))),
                "--ground-truth-key",
                str(config["ground_truth_key"]),
                "--right-json",
                str(absolute(config, str(config["right_json"]))),
                "--right-text-key",
                str(config["right_key"]),
                "--ks",
                *(str(value) for value in config["ranking_ks"]),
                "--output-csv",
                str(rank_dir / "ranking_precision.csv"),
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    config = dict(dataset(args.dataset))
    config["name"] = args.dataset
    run_root = args.run_root or ROOT / "artifact_runs/full_isolation_join" / args.dataset
    run_root = run_root.resolve()
    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    target_decided = args.target_decided or int(config["default_target_decided"])
    if target_decided <= 0:
        raise SystemExit("--target-decided must be positive")
    decided_points = args.decided_points or list(config["ranking_prefixes"])
    top_k = args.ranking_top_k or int(config["ranking_top_k"])
    if any(point <= 0 for point in decided_points) or len(set(decided_points)) != len(decided_points):
        raise SystemExit("--decided-points must be unique positive integers")
    if max(decided_points) > target_decided and args.stage in {"all", "probe", "rank", "metrics"}:
        raise SystemExit(
            f"Largest ranking prefix ({max(decided_points):,}) exceeds target decided "
            f"({target_decided:,}); increase --target-decided or pass smaller --decided-points."
        )
    if args.stage in {"dimensions", "probe", "rank", "all"} and not api_key and not args.print_commands:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    # Command inspection must be side-effect free. Each concrete stage creates its
    # own output directory when it actually runs.
    if not args.print_commands:
        run_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"dimensions", "all"}:
        for command in dimensions_command(config, run_root, api_key, args.guide_source, args.overwrite_scores):
            run(command, api_key=api_key, print_only=args.print_commands)
    if args.stage in {"probe", "all"}:
        verify_probing_prompts(config)
        run(
            probe_command(config, run_root, api_key, target_decided),
            api_key=api_key,
            print_only=args.print_commands,
        )
    if args.stage in {"rank", "all"}:
        run(
            rank_command(config, run_root, api_key, decided_points, top_k),
            api_key=api_key,
            print_only=args.print_commands,
        )
    if args.stage in {"metrics", "all"}:
        for command in metric_commands(config, run_root, decided_points, top_k):
            run(command, api_key="", print_only=args.print_commands)


if __name__ == "__main__":
    main()
