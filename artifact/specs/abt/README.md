# ABT artifact experiment specification

Snapshot date: 2026-10-17

This directory records the confirmed ABT configuration for the current
artifact. It covers score construction, the real LLM probing prompt, and a
canonical LLM ranking prompt. No API key or credential is stored here.

## Canonical score file

`abt/combined_data/pairs_indexed_scored_dim1_dim3_dim4_openai_text_embedding_3_small_normalized.csv`

The file contains 1,167,480 pairs (1,081 source products by 1,080 target
products):

| Column | Meaning | Left representation |
| --- | --- | --- |
| `score_dim1` | `sim(r, s)` | Original source product record |
| `score_dim4` | `sim(summary(r), s)` | No-context canonical product representation |
| `score_dim3` | `sim(t-summary(r), s)` | Product representation conditioned on a target-catalog guide |

All three dimensions use OpenAI `text-embedding-3-small`, 1,536-dimensional
L2-normalized embeddings, no query/passage prefix, cosine similarity by dot
product, and independent min-max normalization to `[0, 1]`.

## Prompt files

- `prompts/dim4_mapping_user.txt`: no-context source-product mapping.
- `prompts/dim3_cluster_summary_user.txt`: creates a reusable target-catalog
  guide from one representative per target-product cluster.
- `prompts/dim3_mapping_user.txt`: maps a source product using that guide.
- `prompts/dim3_saved_product_catalog_guide.txt`: exact guide saved by the
  historical map run.
- `prompts/join_judgment_system.txt`: verified system message for batch
  probing.
- `prompts/join_judgment_user_template.txt`: verified user template for batch
  probing.
- `prompts/ranking_user_template.txt`: historical ranking user template,
  recovered from nonempty persisted prompts.
- `prompts/ranking_system.txt`: canonical artifact system message for future
  ranking runs; the historical ranking system message was not persisted.

`experiment_spec.json` is the machine-readable specification.

## Dimension details

Dim1 embeds the unmodified left and right product records directly. Dim4 maps
each source product without target context. Dim3 first summarizes the first
representative from each of 34 target-product clusters, then supplies the saved
guide while mapping each source product.

The historical dim3 mapping output was JSON-like text. Before embedding, it
was deterministically converted to plain text by
`blocking/convert_abt_mapped_summary_json_to_text.py`; this postprocessing is
part of the dimension definition.

Historical map generation used `gpt-4o-mini`, temperature 0.2, a user message
only, and no system message. The artifact reproduction used temperature 0 for
both guide generation and product mapping to reduce avoidable variation.

## Prompt confirmation

The accepted map prompts were rerun by
`artifact_preparation/run_abt_dim4_dim3_prompt_reproduction.py`. Against the
stored historical scores, the temperature-0 reproduction obtained:

| Dimension | Pearson | Spearman | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| dim4 | 0.995305 | 0.992238 | 0.008144 | 0.011391 |
| dim3 | 0.984382 | 0.972494 | 0.013277 | 0.018725 |

The exact report is
`outputs_abt_dim4_dim3_prompt_reproduction_temperature0/abt_dim4_dim3_comparison.json`.
These strong correlations validate the archived prompts while avoiding a claim
of bitwise reproduction across stochastic map calls or model snapshots.

## Real probing

The red-line experiment uses `score_dim3 score_dim4`, `gpt-4o-mini`, strict
structured output, left-anchored batches of four candidates, one batch per
probe unit, and a round-robin update interval of two. The API request does not
set temperature explicitly. The prompt text is independently verified by the
persisted requests in:

`outputs_abt_round_robin_dim34_anchor_openaiformat_batch4_unit1_update2_targetdecided12000/llm_skyband_prompting_responses.jsonl`

Candidate lines are rendered as
`{candidate_id}. Product Record: «{right_text}»`.

## Ranking provenance

The older ranking artifact `abt/output/ranked_pairs_with_matches.json` contains
1,022 records from `meta-llama/llama-3.3-70b-instruct`. Of these, 79 preserve a
nonempty user prompt and 943 have empty prompt fields. All 79 nonempty prompts
instantiate the template archived in `prompts/ranking_user_template.txt`; they
request top-2 ranking with labels `Left Product` and `Right Product`.

The historical records do not preserve the request's system message or
temperature. Their response text can contain prose and fenced JSON, so they
must not be described as evidence of the modern strict structured-output
protocol. `prompts/ranking_system.txt` is therefore a canonical system message
for future artifact runs, not a recovered historical system message. For a
new reproducible run, use `gpt-4o-mini`, temperature 0, and a strict JSON object
with one `ranked_list` field.
