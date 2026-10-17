# DBpedia-298 artifact experiment specification

Snapshot date: 2026-10-17

This directory records the confirmed DBpedia-298 prompts and experiment
settings for score construction, batch probing, ground-truth simulation, and
LLM ranking. No API key or credential is stored here.

## Dimensions

The canonical historical score file is:

`dbpedia_298/combined_data/pairs_indexed_scored_dim1_dim3_dim4_openai_text_embedding_3_small_normalized.csv`

It contains 74,500 pairs (250 Wikipedia articles by 298 article categories).

| Column | Meaning | Left representation |
| --- | --- | --- |
| `score_dim1` | `sim(r, s)` | Original Wikipedia article |
| `score_dim4` | `sim(summary(r), s)` | No-context category-oriented mapping |
| `score_dim3` | `sim(t-summary(r), s)` | Mapping conditioned on a reusable target-category guide |

All dimensions use OpenAI `text-embedding-3-small`, 1,536-dimensional
embeddings, no query/passage prefix, L2 normalization, cosine similarity by dot
product, and independent min-max normalization to `[0, 1]`.

## Dim3 provenance

Dim3 is produced in two map stages. First, one representative from each of 41
right-table clusters is summarized into a reusable category guide. Second, that
guide is supplied while mapping each Wikipedia article. Both calls use a user
message only and no system message.

The exact guide generated for the original 2026-02-14 run was not persisted.
The historical mapped articles and final scores were persisted and remain the
authoritative inputs for the original reported experiment. The prompt strings
were recovered from a bytecode file created on the same date and are archived
verbatim here. In particular, `dim3_mapping_user.txt` intentionally preserves
the historical `Categoriessummarized` typo.

Historical generation used `gpt-4o-mini` with temperature 0.2 for both stages.
For artifact regeneration, temperature 0 is recommended for both stages to
reduce avoidable run-to-run variation. Temperature 0 improves repeatability but
does not guarantee bitwise-identical output across model snapshots.

Two independent temperature-0.2 reproductions produced Pearson correlations of
0.8380 and 0.8826 against the historical score. Their mutual Pearson
correlation was 0.9516, confirming some stochastic variation while also showing
that the missing original guide or model-version drift matters.

A third reproduction used temperature 0 for both guide generation and article
mapping. Its Pearson/Spearman correlations against the historical score were
0.8431/0.8021. It was very close to the first reproduction
(Pearson/Spearman 0.9860/0.9815), but it was not closer to the historical score
than the best temperature-0.2 run. Temperature 0 is therefore retained as an
artifact repeatability setting, not as a claim that it reconstructs the missing
historical guide.

## Prompt files

- `prompts/dim3_cluster_summary_user.txt`: builds the right-category guide.
- `prompts/dim3_mapping_user.txt`: maps an article using that guide.
- `prompts/dim4_mapping_user.txt`: no-context article mapping.
- `prompts/join_judgment_system.txt`: system prompt for real batch probing.
- `prompts/join_judgment_user_template.txt`: probing user template.
- `prompts/ranking_system.txt`: system prompt for ranking LLM-positive pairs.
- `prompts/ranking_user_template.txt`: ranking user template.

The probing prompts were verified against persisted API requests in the
original dim3+dim4 experiment. The ranking prompts were verified against the
persisted 23,997-probe, top-6 ranking run.

## Canonical settings

Real probing uses `gpt-4o-mini`, strict structured output, left-anchored batches
of four candidates, two batches per probe unit, and round-robin updates every
two units. The probing request does not explicitly set temperature.

Ground-truth simulation uses `simulate_round_robin_with_groundtruth.py` and
substitutes the pair file's `label` for the LLM judgment. Ranking uses
`gpt-4o-mini`, temperature 0, a strict `ranked_list` schema, and six returned
category IDs for the historical DBpedia experiment.

See `experiment_spec.json` for machine-readable settings.
