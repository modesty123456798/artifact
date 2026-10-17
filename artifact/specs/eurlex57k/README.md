# EURLEX57K artifact experiment specification

Snapshot date: 2026-10-16

This directory records the confirmed EURLEX57K configuration for the current
artifact. It covers:

1. construction of `score_dim1`, the revised `score_dim4_version2`, and
   `score_dim3`;
2. the LLM prompt used to judge candidate pairs during probing;
3. the LLM prompt used to rank positive candidates;
4. the canonical score file and source-file checksums.

No API key or credential is stored in this directory.

## Canonical score file

`EURLEX57K/combined_data/pairs_indexed_scored_dim1_dim3_dim4_lotus_map_key_themes_key_themes_guide_dim4_version2_article_end_normalized.csv`

The file contains 1,067,750 pairs and the artifact uses these score columns:

| Artifact name | CSV column | Meaning | Left representation |
| --- | --- | --- | --- |
| `score_dim1` | `score_dim1` | `sim(r, s)` | Original EU legal document |
| `score_dim4_version2` | `score_dim4_version2_article_end` | `sim(summary(r), s)` | Revised article-at-end key-concepts mapping |
| `score_dim3` | `score_dim3` | `sim(t-summary(r), s)` | EuroVoc target-guide-aware mapping |

All three dimensions use `text-embedding-3-small`, no E5-style query or
passage prefix, L2-normalized embeddings, cosine similarity by dot product,
and independent min-max normalization to `[0, 1]`.

The canonical artifact name `score_dim4_version2` refers to the revised CSV
column `score_dim4_version2_article_end`; the older column
`score_dim4_version2` is retained in the CSV only for comparison and is not the
artifact dimension.

## Prompt files

- `prompts/dim4v2_mapping_user.txt`: revised no-context mapping for the
  artifact `score_dim4_version2`.
- `prompts/dim3_cluster_summary_user.txt`: constructs the reusable EuroVoc
  labeling guide from right-table cluster exemplars.
- `prompts/dim3_mapping_user.txt`: maps each legal document using that guide.
- `prompts/dim3_saved_eurovoc_guide.txt`: the canonical saved EuroVoc guide
  explicitly reused by the repeated-sample dim3 runs. See the provenance note
  below for the original-data run.
- `prompts/join_judgment_system.txt`: system message for batch probing.
- `prompts/join_judgment_user_template.txt`: user template for batch probing.
- `prompts/ranking_system.txt`: system message for ranking LLM-positive
  candidates.
- `prompts/ranking_user_template.txt`: user template for ranking.

`experiment_spec.json` is the machine-readable version of this configuration.

## Important distinctions

- `score_dim1` uses no mapping prompt. The original legal document and EuroVoc
  concept are embedded directly.
- The revised `score_dim4_version2` uses one user message, no system message,
  and places the complete legal document after the instruction.
- `score_dim3` is a two-stage mapping. It first summarizes one exemplar from
  each right-table cluster into a EuroVoc labeling guide and then includes that
  guide when mapping every legal document.
- `USER_TEMPLATE_WITH_EXAMPLE_SUMMARY_EURLEX57K_KEY_THEMES` and the alternative
  dense concept-vocabulary prompt are not used for the artifact dim3.
- Ground-truth simulation uses no LLM. It substitutes the pair CSV's `label`
  column for the probing judgment while retaining the same probing strategy.
- Real probing and downstream ranking are separate LLM stages with separate
  prompts and strict JSON schemas.

## Provenance note for dim3

The historical original-data dim3 mapped JSON does not embed its full
generation prompt or guide. The default EuroVoc labeling-guide prompt pair
recorded here is the confirmed canonical prompt family. The repeated-sample
commands explicitly selected `--example-summary-template-variant default` and
loaded `eurovoc_concept_cluster_example_summary.json`. A controlled
reproduction on the original data also supports this choice:
the historical mapped outputs average 302.9 characters, the default-prompt
reproduction averages 305.8 characters and has mean sequence similarity
0.6131, while the key-themes alternative averages 782.0 characters and has
mean sequence similarity 0.1202.

The current saved guide file has a modification time later than the original
mapped JSON. It is therefore archived as the canonical guide used by the
repeated runs, not claimed as a byte-for-byte snapshot of the guide supplied to
the original 2026-08-20 mapping call. This distinction does not change which
cluster-summary and document-mapping prompt templates define dim3.

## Dim3 reproduction check

`artifact_preparation/run_eurlex_dim3_prompt_reproduction.py` reads the two
prompt files in this directory directly, regenerates the EuroVoc guide and all
250 mapped legal documents, and records resumable checkpoints and token usage.
`artifact_preparation/compare_eurlex_dim3_reproduction.py` compares the newly
embedded score against the stored `score_dim3`, including correlations, errors,
top-N pair overlap, and GT-positive counts.

