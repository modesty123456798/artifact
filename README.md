# Isolation Join reproduction

This artifact reproduces Isolation Join on BioDEX, EURLEX57K, Amazon-531,
DBpedia-298, and ABT.

## Setup

```bash
conda env create -f environment.yml
conda activate full-isolation-join
export OPENAI_API_KEY='...'
```

## Run

Run the three stages separately:

```bash
python run_pipeline.py --dataset biodex --stage dimensions
python run_pipeline.py --dataset biodex --stage probe
python run_pipeline.py --dataset biodex --stage rank --decided-points 5000 11260 15100 20000
python run_pipeline.py --dataset biodex --stage metrics --decided-points 5000 11260 15100 20000
```

Replace `biodex` with `eurlex57k`, `amazon`, `dbpedia`, or `abt`. Use
`--target-decided N` to set a probing budget and `--run-root PATH` to choose
an output directory.

## External baselines

- [DocETL](https://github.com/ucbepic/docetl)
- [LOTUS](https://github.com/lotus-data/lotus)
- [FDJ (Featurized-Decomposition Join)](https://github.com/szeighami/FDJ)
