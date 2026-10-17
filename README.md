# Isolation Join reproduction

This directory reproduces our Isolation Join method:

`Ours: {sim(summary(r), s), sim(t-summary(r), s)}`.

## External baselines

The paper compares Full Isolation Join with three external systems. They are
not vendored into this artifact bundle; their reported results were obtained
using the corresponding baseline implementations and experiment
configurations.

- [DocETL](https://github.com/ucbepic/docetl)
- [LOTUS](https://github.com/lotus-data/lotus)
- [FDJ (Featurized-Decomposition Join)](https://github.com/szeighami/FDJ)

Consult the paper and the accompanying experiment-result records for the exact
baseline settings, prompts, budgets, and version/commit information used for
each comparison.

The pipeline has three operational stages:

1. `dimensions`: generate the no-context and target-aware maps, then embed the
   original and mapped left records with `text-embedding-3-small` to produce
   all three score columns.
2. `probe`: run the anti-dominance / per-left k-skyband, round-robin LLM
   probing strategy on the two Full Isolation Join score columns.
3. `rank`: rank the LLM-positive candidates at chosen probing prefixes.
   `metrics` then computes mean ranking precision at the dataset's reported
   k values.

No credentials are stored in this directory. The fixed validation data inputs
for all five datasets are included under `data/`; mapping outputs, embedding
caches, LLM judgments, and API keys are intentionally excluded. Run from this
artifact bundle's root in the `lotus_original` environment:

```bash
conda activate lotus_original
export OPENAI_API_KEY='...'
```

For a standalone environment, run `conda env create -f environment.yml` and
then `conda activate full-isolation-join`. See `DATASET_PROVENANCE.md` before
redistributing the included validation snapshots.

## Dataset presets

| Dataset | Full Isolation Join dimensions | Batch size | Batches / probe unit | Round-robin update | Default probe target |
| --- | --- | ---: | ---: | ---: | ---: |
| BioDEX | `score_summary`, `score_target_aware` | 10 | 1 | 2 | 22,540 |
| EURLEX57K | `score_summary`, `score_target_aware` | 10 | 1 | 2 | 45,000 |
| Amazon-531 | `score_summary`, `score_target_aware` | 4 | 2 | 2 | 12,000 |
| DBpedia-298 | `score_summary`, `score_target_aware` | 4 | 2 | 2 | 22,000 |
| ABT | `score_summary`, `score_target_aware` | 4 | 1 | 2 | 12,000 |

The prompt files, map models, mapping temperatures, source data paths, and
ground-truth keys are all pinned in `datasets.py`. The self-contained
`artifact/specs/<dataset>/`, `data/<dataset>/`, and `scripts/` directories
contain every local input and executable needed by the pipeline.

## Run commands

First inspect a fully resolved command sequence without API calls:

```bash
python run_pipeline.py \
  --dataset biodex --stage all --print-commands
```

Run each stage independently. This is generally preferable for a large run,
because all output directories are resumable.

```bash
# 1. Maps and all three normalized embedding dimensions.
python run_pipeline.py \
  --dataset biodex --stage dimensions

# 2. Two-dimensional anti-dominance probing.
python run_pipeline.py \
  --dataset biodex --stage probe

# 3. Ranking and RP@k computation at selected probing prefixes.
python run_pipeline.py \
  --dataset biodex --stage rank --decided-points 5000 11260 15100 22500
python run_pipeline.py \
  --dataset biodex --stage metrics --decided-points 5000 11260 15100 22500
```

Replace `biodex` with `eurlex57k`, `amazon`, `dbpedia`, or `abt`. Override a
budget expressed as a number of judged pairs with `--target-decided N`.
Use `--run-root PATH` to place a run outside the default
`artifact_runs/full_isolation_join/<dataset>` directory.

## Dimension construction

`generate_mappings.py` writes its outputs under
`<run-root>/dimensions/maps`:

- `no_context_mapped.json`
- `target_guide.json`
- `target_aware_mapped_raw.json`
- `target_aware_mapped.json`
- mapping checkpoints, per-call usage, and `mapping_metadata.json`

The default `--guide-source regenerate` performs the complete target-aware
stage: it builds a target guide from the archived guide prompt and one exemplar
per right-table cluster, then maps every left record with that newly generated
guide. This is the end-to-end dimension-generation path for all five datasets.

BioDEX, EURLEX57K, Amazon-531, and ABT also retain historical guide JSON files
under their artifact specification directories. Use
`--guide-source saved` only to replay the historical-guide condition when
comparing against stored scores. DBpedia has no preserved historical guide.
The legacy `auto` option is an alias for `regenerate`.

The output score file is:

`<run-root>/dimensions/pairs_indexed_scored_full_isolation_join.csv`

It has the original `label`, pair IDs, direct score, no-context score, and
target-aware score. The canonical columns are `score_direct` for `sim(r, s)`,
`score_summary` for `sim(summary(r), s)`, and `score_target_aware` for
`sim(t-summary(r), s)`. Each existing dataset scorer is invoked with
`text-embedding-3-small`, an 8,191-token input limit, L2-normalized embeddings,
cosine/dot-product scoring, independent min-max score normalization, and
embedding caching. The scorer metadata records token counts, input truncation,
normalization, and cache use.

## Probing prompts and settings

`run_pipeline.py --stage probe` reads each dataset's archived system prompt
and user-template files and passes both explicitly to the generic probing
runner. Thus, no dataset-specific prompt is taken from a runner default. The
pipeline also checks that the archived prompt files are nonempty and that the
runner supports the explicit system-prompt argument before it begins.

The pipeline also passes the algorithmic defaults explicitly: `initial-k=1`,
`k-increment=1`, `round-robin-initial-l=1`, no `l_r` cap, no positive-target
stop, and a per-left decision budget of 10,000. These match the historical
Full Isolation Join runs; the reported batch, unit, update cadence, and global
judged-pair target are dataset-specific and are shown above.

The probe directory contains the ordered response JSONL, judged pair CSV,
round-robin state, iteration metrics, and snapshots. Re-running the same
command resumes from the existing judged pairs.

## Ranking

`rank_prefixes.py` is a generic five-dataset wrapper around the existing strict
JSON ranking engine. It ranks only pairs judged positive in the selected probe
prefix, records prompt/token usage, and writes one subdirectory per prefix.
`metrics` invokes `scripts/compute_ranking_precision_from_ranked_ids.py` and writes
`ranking_precision.csv` beside each ranked result.

The ABT ranking system message is a canonical artifact prompt for new runs;
the ABT historical ranking system message was not persisted. This provenance
distinction is recorded in `artifact/specs/abt/README.md` and
the ABT prompt files in that directory.

## Stochasticity

Mapping calls are LLM calls. The archived temperature, model, and prompt are
used, and all map/guide calls are checkpointed, but a regenerated map need not
be bit-identical to the historical map. The existing per-dataset artifact specs
contain prompt-reproduction comparisons against historical scores. Use a fresh
`--run-root` for independent repetitions.
