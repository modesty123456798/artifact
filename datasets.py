"""Dataset presets for the reported two-dimension anti-dominance experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Keep every submitted resource relative to this bundle, rather than to the
# larger development workspace.
ROOT = Path(__file__).resolve().parent


def root_path(relative: str) -> Path:
    return ROOT / relative


DATASETS: dict[str, dict[str, Any]] = {
    "biodex": {
        "display_name": "BioDEX",
        "spec_dir": "artifact/specs/biodex",
        "pairs": "data/biodex/pairs_indexed.csv",
        "left_json": "data/biodex/biodex_sample_adjusted.json",
        "right_json": "data/biodex/biodex_terms.json",
        "ground_truth_json": "data/biodex/biodex_ground_truth_adjusted.json",
        "left_key": "medical_article",
        "right_key": "reaction",
        "ground_truth_key": "ground_truth_reactions",
        "ground_truth_id_key": "id",
        "left_label": "Medical Article",
        "right_label": "Reaction",
        "no_context": {
            "prompt": "prompts/summary_mapping_user.txt",
            "output_key": "medical_article_summary",
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.2,
        },
        "target_aware": {
            "guide_prompt": "prompts/target_guide_user.txt",
            "mapping_prompt": "prompts/target_aware_mapping_user.txt",
            "saved_guide": "artifact/specs/biodex/guides/historical_target_guide.json",
            "right_exemplars": "data/biodex/biodex_terms_cluster_exemplars_190_200_196.json",
            "output_key": "medical_article_summary",
            "model": "gpt-4o-mini",
            "guide_temperature": 0.2,
            "mapping_temperature": 0.2,
        },
        "similarity_script": "scripts/embedding_similarity.py",
        "similarity_args": {
            "direct_score_column": "score_direct",
            "no_context_score_column": "score_summary",
            "target_aware_score_column": "score_target_aware",
        },
        "probe_runner": "scripts/anti_dominance_prober.py",
        "probe_dimensions": ["score_summary", "score_target_aware"],
        "batch_size": 10,
        "num_batch_per_unit": 1,
        "round_robin_update_every": 2,
        "default_target_decided": 22540,
        "ranking_top_k": 25,
        "ranking_prefixes": [5000, 11260, 15100, 22500],
        "ranking_ks": [5, 10, 15, 20, 25],
        "ranking_schema": "reaction_ranking",
    },
    "eurlex57k": {
        "display_name": "EURLEX57K",
        "spec_dir": "artifact/specs/eurlex57k",
        "pairs": "data/eurlex57k/pairs_indexed.csv",
        "left_json": "data/eurlex57k/eurlex_legal_document_sampled_250.json",
        "right_json": "data/eurlex57k/eurovoc_concept.json",
        "ground_truth_json": "data/eurlex57k/ground_truth_sampled_250.json",
        "left_key": "eurlex_legal_document",
        "right_key": "eurovoc_concept",
        "ground_truth_key": "ground_truth_eurovoc_concepts",
        "ground_truth_id_key": "id",
        "left_label": "European Union Legal Document",
        "right_label": "EuroVoc Concept",
        "no_context": {
            "prompt": "prompts/summary_mapping_user.txt",
            "output_key": "_eurlex_legal_document",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_output_tokens": 256,
            "max_input_characters": 16000,
        },
        "target_aware": {
            "guide_prompt": "prompts/target_guide_user.txt",
            "mapping_prompt": "prompts/target_aware_mapping_user.txt",
            "saved_guide": "artifact/specs/eurlex57k/guides/historical_target_guide.json",
            "right_exemplars": "data/eurlex57k/eurovoc_concept_cluster_exemplars.json",
            "output_key": "eurlex_legal_document_summary",
            "model": "gpt-4o-mini",
            "guide_temperature": 0.2,
            "mapping_temperature": 0.2,
        },
        "similarity_script": "scripts/embedding_similarity.py",
        "similarity_args": {
            "direct_score_column": "score_direct",
            "no_context_score_column": "score_summary",
            "target_aware_score_column": "score_target_aware",
        },
        "probe_runner": "scripts/anti_dominance_prober.py",
        "probe_dimensions": ["score_summary", "score_target_aware"],
        "batch_size": 10,
        "num_batch_per_unit": 1,
        "round_robin_update_every": 2,
        "default_target_decided": 45000,
        "ranking_top_k": 11,
        "ranking_prefixes": [2500, 6250, 10000, 12500, 13750],
        "ranking_ks": [1, 2, 5, 8, 10, 11],
        "ranking_schema": "eurovoc_concept_ranking",
    },
    "amazon": {
        "display_name": "Amazon-531",
        "spec_dir": "artifact/specs/amazon",
        "pairs": "data/amazon/pairs_indexed.csv",
        "left_json": "data/amazon/amazon_product_review_sampled_250.json",
        "right_json": "data/amazon/product_type.json",
        "ground_truth_json": "data/amazon/ground_truth_sampled_250.json",
        "left_key": "amazon_product_review",
        "right_key": "product_type",
        "ground_truth_key": "ground_truth_product_type",
        "ground_truth_id_key": "id",
        "left_label": "Amazon Product Review",
        "right_label": "Product Type",
        "no_context": {
            "prompt": "prompts/summary_mapping_user.txt",
            "output_key": "amazon_product_review_summary",
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "target_aware": {
            "guide_prompt": "prompts/target_guide_user.txt",
            "mapping_prompt": "prompts/target_aware_mapping_user.txt",
            "saved_guide": "artifact/specs/amazon/guides/historical_target_guide.json",
            "right_exemplars": "data/amazon/product_type_cluster_exemplars.json",
            "output_key": "amazon_product_review_summary",
            "model": "gpt-4o-mini",
            "guide_temperature": 0.2,
            "mapping_temperature": 0.2,
        },
        "similarity_script": "scripts/embedding_similarity.py",
        "similarity_args": {
            "direct_score_column": "score_direct",
            "no_context_score_column": "score_summary",
            "target_aware_score_column": "score_target_aware",
        },
        "probe_runner": "scripts/anti_dominance_prober.py",
        "probe_dimensions": ["score_summary", "score_target_aware"],
        "batch_size": 4,
        "num_batch_per_unit": 2,
        "round_robin_update_every": 2,
        "default_target_decided": 12000,
        "ranking_top_k": 6,
        "ranking_prefixes": [2000, 4000, 9000, 12000],
        "ranking_ks": [1, 2, 3, 4, 5, 6],
        "ranking_schema": "product_type_ranking",
    },
    "dbpedia": {
        "display_name": "DBpedia-298",
        "spec_dir": "artifact/specs/dbpedia",
        "pairs": "data/dbpedia/pairs_indexed.csv",
        "left_json": "data/dbpedia/wikipedia_article_sampled_250.json",
        "right_json": "data/dbpedia/article_category.json",
        "ground_truth_json": "data/dbpedia/ground_truth_sampled_250.json",
        "left_key": "wikipedia_article",
        "right_key": "article_category",
        "ground_truth_key": "ground_truth_article_categories",
        "ground_truth_id_key": "id",
        "left_label": "Wikipedia Article",
        "right_label": "Article Category",
        "no_context": {
            "prompt": "prompts/summary_mapping_user.txt",
            "output_key": "wikipedia_article_summary",
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "target_aware": {
            "guide_prompt": "prompts/target_guide_user.txt",
            "mapping_prompt": "prompts/target_aware_mapping_user.txt",
            "saved_guide": None,
            "right_exemplars": "data/dbpedia/article_category_cluster_exemplars.json",
            "output_key": "wikipedia_article_summary",
            "model": "gpt-4o-mini",
            "guide_temperature": 0.0,
            "mapping_temperature": 0.0,
        },
        "similarity_script": "scripts/embedding_similarity.py",
        "similarity_args": {
            "direct_score_column": "score_direct",
            "no_context_score_column": "score_summary",
            "target_aware_score_column": "score_target_aware",
        },
        "probe_runner": "scripts/anti_dominance_prober.py",
        "probe_dimensions": ["score_summary", "score_target_aware"],
        "batch_size": 4,
        "num_batch_per_unit": 2,
        "round_robin_update_every": 2,
        "default_target_decided": 22000,
        "ranking_top_k": 6,
        "ranking_prefixes": [2000, 4000, 9000, 12000],
        "ranking_ks": [1, 2, 3, 4, 5, 6],
        "ranking_schema": "article_category_ranking",
    },
    "abt": {
        "display_name": "ABT",
        "spec_dir": "artifact/specs/abt",
        "pairs": "data/abt/pairs_indexed.csv",
        "left_json": "data/abt/left_data.json",
        "right_json": "data/abt/right_data.json",
        "ground_truth_json": "data/abt/ground_truth.json",
        "left_key": "left_product",
        "right_key": "right_product",
        "ground_truth_key": "ground_truth_right_product",
        "ground_truth_id_key": "id",
        "left_label": "Source Product Record",
        "right_label": "Product Record",
        "no_context": {
            "prompt": "prompts/summary_mapping_user.txt",
            "output_key": "left_product_summary",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
        },
        "target_aware": {
            "guide_prompt": "prompts/target_guide_user.txt",
            "mapping_prompt": "prompts/target_aware_mapping_user.txt",
            "saved_guide": "artifact/specs/abt/guides/historical_target_guide.json",
            "right_exemplars": "data/abt/right_product_cluster_exemplars.json",
            "output_key": "left_product_summary",
            "model": "gpt-4o-mini",
            "guide_temperature": 0.0,
            "mapping_temperature": 0.0,
            "postprocess": "abt_product_json_to_text",
        },
        "similarity_script": "scripts/embedding_similarity.py",
        "similarity_args": {
            "direct_score_column": "score_direct",
            "no_context_score_column": "score_summary",
            "target_aware_score_column": "score_target_aware",
        },
        "probe_runner": "scripts/anti_dominance_prober.py",
        "probe_dimensions": ["score_summary", "score_target_aware"],
        "batch_size": 4,
        "num_batch_per_unit": 1,
        "round_robin_update_every": 2,
        "default_target_decided": 12000,
        "ranking_top_k": 2,
        "ranking_prefixes": [1250, 2500, 5000, 7500, 10000, 12000],
        "ranking_ks": [1, 2],
        "ranking_schema": "product_record_ranking",
    },
}


def dataset(name: str) -> dict[str, Any]:
    try:
        return DATASETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}") from exc


def absolute(dataset_config: dict[str, Any], value: str) -> Path:
    return root_path(value)


def spec_path(dataset_config: dict[str, Any], relative: str) -> Path:
    return absolute(dataset_config, dataset_config["spec_dir"]) / relative
