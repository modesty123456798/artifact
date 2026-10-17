# Amazon-531 artifact experiment specification

Snapshot date: 2026-10-17

This directory records the confirmed Amazon-531 configuration for the current
artifact. It covers:

1. construction of `score_dim1`, `score_dim4`, and `score_dim7`;
2. the LLM prompt used to judge candidate pairs during probing;
3. the LLM prompt used to rank LLM-positive candidates;
4. the canonical score file, prompt-reproduction evidence, and source checksums.

No API key or credential is stored in this directory.

## Canonical score file

`amazon_531/combined_data/pairs_indexed_scored_dim1_dim4_dim7_openai_text_embedding_3_small_normalized.csv`

The file contains 132,750 pairs (250 reviews by 531 product types):

| Artifact name | CSV column | Meaning | Left representation |
| --- | --- | --- | --- |
| `score_dim1` | `score_dim1` | `sim(r, s)` | Original Amazon product review |
| `score_dim4` | `score_dim4` | `sim(summary(r), s)` | No-context product-type summary |
| `score_dim7` | `score_dim7` | `sim(t-summary(r), s)` | Target-guide-aware hierarchical product-type mapping |

All dimensions use `text-embedding-3-small`, no E5 query/passage prefixes,
1,536-dimensional L2-normalized embeddings, cosine similarity by dot product,
and independent min-max normalization to `[0, 1]`.

## Prompt files

- `prompts/dim4_mapping_user.txt`: no-context review-to-product-type mapping.
- `prompts/dim7_cluster_summary_user.txt`: builds the reusable product-type
  guide from one representative per right-table cluster.
- `prompts/dim7_mapping_user.txt`: maps reviews with the saved guide.
- `prompts/dim7_saved_product_type_guide.txt`: saved guide used by the
  historical mapped input.
- `prompts/join_judgment_system.txt`: system message for batch probing.
- `prompts/join_judgment_user_template.txt`: user template for batch probing.
- `prompts/ranking_system.txt`: system message for ranking LLM-positive pairs.
- `prompts/ranking_user_template.txt`: user template for ranking.

`experiment_spec.json` is the machine-readable specification.

## Important distinctions

- `score_dim1` has no map stage: the original review and product type are
  embedded directly.
- Both map stages use `gpt-4o-mini`, temperature `0.2`, a user message only,
  and no system message.
- `score_dim7` is a two-stage map. It summarizes the first exemplar from each
  of 70 product-type clusters, then supplies that guide when mapping reviews.
- The exact confirmed historical dim7 guide prompt intentionally contains
  `PRODUCT-TYPE STYLE.Treat` without a separating space. This historical string
  is preserved rather than silently corrected.
- Real probing uses strict structured output and batch size 4. The probing API
  request does not set temperature explicitly.
- Ranking is a separate stage using temperature 0 and a strict `ranked_list`
  schema. The historical Amazon progression run requested six IDs per review.
- Ground-truth simulation uses no LLM; it substitutes the canonical pair
  file's `label` value for each probing judgment.

## Prompt confirmation

The accepted prompts were rerun by
`artifact_preparation/run_amazon_dim4_dim7_prompt_reproduction.py`. Against the
stored historical scores, the reproduction report records Spearman/Pearson
correlations of 0.9207/0.9315 for dim4 and 0.8576/0.8778 for dim7. Exact values
are pinned in `outputs_amazon_dim4_dim7_prompt_reproduction/amazon_dim4_dim7_comparison.json`.

The probing prompts were additionally checked against a persisted request in
`outputs_amazon_531_round_robin_dim74_openai_text_embedding_3_small_anchor_openaiformat_batch4_unit2_update2_targetdecided12000/llm_skyband_prompting_responses.jsonl`.
The ranking prompts were checked against a persisted ranking result in that
experiment's `ranking_decided27926_top6_openai` subdirectory.
