# Bundled Execution Scripts

This directory contains the local implementations invoked by `run_pipeline.py`:

- `embedding_similarity.py`: one parameterized OpenAI embedding scorer for all
  five datasets;
- `anti_dominance_prober.py`: one parameterized anti-dominance / round-robin
  LLM probing runner for all five datasets;
- `llm_ranking_engine.py`: the strict-JSON LLM ranking engine;
  and
- `compute_ranking_precision_from_ranked_ids.py`: ranking-precision metric
  computation.

Dataset-specific differences are declarative: `datasets.py` supplies schema,
score-column, batch, and stopping settings, while `artifact/specs/<dataset>/`
supplies the archived prompts. The generic prober receives both archived
probing prompts explicitly at runtime.

The pipeline verifies the active probing system and user prompts against the
archived prompt files before submitting any probe request. Do not edit these
scripts when reproducing reported settings; create a separate branch or copy
for exploratory variants.
