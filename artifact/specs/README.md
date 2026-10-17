# Dataset prompt specifications

Every dataset uses the same runtime interface. `datasets.py` selects the
dataset-specific prompt text and settings while preserving these semantic roles:

| Resource | Purpose |
| --- | --- |
| `prompts/summary_mapping_user.txt` | Map a left record without target-guide context for `score_summary = sim(summary(r), s)`. |
| `prompts/target_guide_user.txt` | Produce a target-vocabulary guide from right-table cluster exemplars. |
| `prompts/target_aware_mapping_user.txt` | Map a left record using that guide for `score_target_aware = sim(t-summary(r), s)`. |
| `prompts/join_judgment_*` | System and user prompts for anti-dominance LLM probing. |
| `prompts/ranking_*` | System and user prompts for ranking LLM-positive candidates. |

`score_direct = sim(r, s)` uses the original left and right text and therefore
has no mapping prompt. Historical target guides are optional inputs for
`--guide-source saved`; normal artifact runs use `--guide-source regenerate`.

Historical dimension identifiers from the development experiments are kept only
in the external provenance archive, not in this runtime interface.
