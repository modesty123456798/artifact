# Validation Data Snapshot

This directory contains the fixed inputs used by the reported validation runs:

- left records (250 sampled records for BioDEX, EURLEX57K, Amazon-531, and
  DBpedia-298; the complete ABT left table);
- right-table records and ground-truth labels;
- the ordered `pairs_indexed.csv` used to preserve the original all-pairs
  enumeration order; and
- one-exemplar-per-cluster files used to regenerate each target-aware guide.

Each dataset is self-contained in `data/<dataset>/`. The artifact does not
include LLM-generated maps, embedding caches, LLM response logs, or API keys;
those are generated in the caller-selected run directory.

These are research validation snapshots derived from the corresponding public
source datasets. Any redistribution must comply with the original dataset
terms. See each dataset specification under `artifact/specs/<dataset>/` for
the schema and experiment configuration.
