# BioDEX artifact experiment specification

Snapshot date: 2026-10-16

This directory records the confirmed BioDEX configuration for the current
artifact. It covers:

1. construction of `score_dim1`, the revised `score_dim8`, and `score_dim6`;
2. the LLM prompt used to judge candidate pairs during probing;
3. the LLM prompt used to rank positive candidates;
4. the canonical score file and source-file checksums.

No API key or credential is stored in this directory.

## Canonical score file

`biodex/combined_data/pairs_indexed_scored_dim1_dim8articleend_dim6_openai_text_embedding_3_small_normalized.csv`

The file contains 6,078,250 pairs and these score columns:

| Column | Meaning | Left representation |
| --- | --- | --- |
| `score_dim1` | `sim(r, s)` | Original medical article |
| `score_dim8` | `sim(summary(r), s)` | Revised article-at-end ADR mapping |
| `score_dim6` | `sim(t-summary(r), s)` | ADR target-vocabulary-aware mapping |

All three dimensions use `text-embedding-3-small`, no E5-style query or
passage prefix, L2-normalized embeddings, cosine similarity by dot product,
and independent min-max normalization to `[0, 1]`.

## Prompt files

- `prompts/dim8_mapping_user.txt`: revised no-context mapping for `score_dim8`.
- `prompts/dim6_cluster_summary_user.txt`: builds the reusable target
  vocabulary guide for `score_dim6`.
- `prompts/dim6_mapping_user.txt`: maps each article using that guide.
- `prompts/join_judgment_system.txt`: system message for batch probing.
- `prompts/join_judgment_user_template.txt`: user template for batch probing.
- `prompts/ranking_system.txt`: system message for ranking LLM-positive
  candidates.
- `prompts/ranking_user_template.txt`: user template for ranking.

`experiment_spec.json` is the machine-readable form of the same configuration.

## Important distinctions

- `score_dim1` uses no mapping prompt. The original article and reaction are
  embedded directly.
- The revised `score_dim8` uses one user message and no system message.
- `score_dim6` is a two-stage mapping: summarize selected target-vocabulary
  exemplars once, then use that guide when mapping every article.
- The ground-truth simulation uses no LLM and therefore no judgment prompt. It
  substitutes the CSV `label` for the LLM output while retaining the probing
  strategy.
- Real probing uses the join-judgment prompt. Ranking is a separate downstream
  stage and uses the ranking prompt.

## Source-of-truth notes

The revised dim8 metadata directly records its prompt, model, temperature, and
message roles. The historical dim6 mapped JSON does not embed its generation
prompt; the dim6 prompt pair recorded here is the confirmed canonical
ADR-focused configuration selected for the artifact and matches the historical
BioDEX prompt block and the saved target-guide metadata.

