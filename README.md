# Isolation Join reproduction

This artifact reproduces our Isolation Join pipeline on BioDEX, EURLEX57K,
Amazon-531, DBpedia-298, and ABT.

## Setup

```bash
conda env create -f environment.yml
conda activate full-isolation-join
export OPENAI_API_KEY='...'
```

The bundled evaluation inputs are under `data/`. See
`DATASET_PROVENANCE.md` before redistributing them.

## Run

Inspect the resolved commands without making API calls:

```bash
python run_pipeline.py --dataset biodex --stage all --print-commands
```

Run the three stages separately:

```bash
# Generate mapped representations and embedding similarities.
python run_pipeline.py --dataset biodex --stage dimensions

# Run anti-dominance LLM probing.
python run_pipeline.py --dataset biodex --stage probe

# Rank LLM-positive candidates and compute RP@k.
python run_pipeline.py --dataset biodex --stage rank --decided-points 5000 11260 15100 22500
python run_pipeline.py --dataset biodex --stage metrics --decided-points 5000 11260 15100 22500
```

Replace `biodex` with `eurlex57k`, `amazon`, `dbpedia`, or `abt`. Use
`--target-decided N` to set a probing budget and `--run-root PATH` to choose
an output directory.

## Pipeline

1. `dimensions`: map left records, then compute normalized embedding scores.
2. `probe`: apply two-dimensional anti-dominance, round-robin LLM probing.
3. `rank` and `metrics`: rank LLM-positive candidates at selected prefixes and
   compute ranking precision.

The three score columns are `score_direct = sim(r, s)`,
`score_summary = sim(summary(r), s)`, and
`score_target_aware = sim(t-summary(r), s)`. Reported Isolation Join runs use
the latter two columns. Dataset prompts and settings are declared in
`datasets.py`; prompt files are in `artifact/specs/<dataset>/`.

## External baselines

- [DocETL](https://github.com/ucbepic/docetl)
- [LOTUS](https://github.com/lotus-data/lotus)
- [FDJ (Featurized-Decomposition Join)](https://github.com/szeighami/FDJ)
