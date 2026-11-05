#!/usr/bin/env bash
# Journalism segments: the items exactly as judged by the participants of the
# lab course segment experiment (headline + subheadline + 2-paragraph segment).
# Lens-model comparison of humans, base and instruct on identical items.
#
# Confidential texts: everything runs locally; nothing is printed or committed.
# Usage: bash run_journalism_seg.sh [preset]      (default gemma3-4b)
set -euo pipefail
cd "$(dirname "$0")"
PRESET="${1:-gemma3-4b}"
PY=.venv/bin/python                                   # torch / transformers
PY_FEAT=/path/to/spacy-env/bin/python   # spaCy de_core_news_lg

$PY prepare_texts.py --corpus journalism_seg                   # -> data/texts_journalism_seg.parquet
Rscript export_human_items.R journalism_seg                    # -> data/human_items_journalism_seg.csv
$PY_FEAT segment_features.py data/texts_journalism_seg.parquet # -> outputs/journalism_seg/features_*.csv
$PY judge_logits.py --texts data/texts_journalism_seg.parquet --preset "$PRESET"
$PY sdt_summary.py outputs/journalism_seg
$PY probe_layers.py outputs/journalism_seg --n-splits 7      # 14 sources -> 7 folds of 2
$PY lens_model.py outputs/journalism_seg \
    --features outputs/journalism_seg/features_journalism_seg.csv \
    --human data/human_items_journalism_seg.csv
