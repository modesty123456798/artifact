# Dataset Provenance and Redistribution

`data/` is a fixed validation snapshot, not a new source dataset release. It
contains the inputs used by this artifact: 250 left records for BioDEX,
EURLEX57K, Amazon-531, and DBpedia-298; the corresponding complete right
tables and labels; and the complete ABT benchmark tables.

| Artifact dataset | Local validation source | Included inputs |
| --- | --- | --- |
| BioDEX | `biodex/combined_data/` original 250-record validation snapshot | 250 medical articles, reaction vocabulary, labels, pair order, and cluster exemplars |
| EURLEX57K | `EURLEX57K/combined_data/` original 250-record validation snapshot | 250 EU legal documents, EuroVoc concepts, labels, pair order, and cluster exemplars |
| Amazon-531 | `amazon_531/combined_data/` original 250-record validation snapshot | 250 reviews, product types, labels, pair order, and cluster exemplars |
| DBpedia-298 | `dbpedia_298/combined_data/` original 250-record validation snapshot | 250 Wikipedia articles, categories, labels, pair order, and cluster exemplars |
| ABT | `abt/combined_data/` benchmark snapshot | complete left/right records, labels, pair order, and cluster exemplars |

The four 250-record datasets above are the original validation inputs used by
the primary experiments. They are not any of the later
`*_repeated_test_samples_9x250_.../group_*` resamples.

Before making a public GitHub release, confirm that each original dataset's
license permits redistribution of these snapshots and add the required
citations and license notices here. If any source terms prohibit redistribution,
replace that dataset directory with a deterministic downloader/preprocessor and
keep only its checksums and preprocessing instructions in the repository.

The historical guide JSON files under `artifact/specs/` retain their guide text,
model, temperature, and exemplar-count metadata. Their descriptive exemplar
paths have been rewritten to corresponding bundle-local paths; the pipeline
uses only bundle-local inputs defined in `datasets.py`.
