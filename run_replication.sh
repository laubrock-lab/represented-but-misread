#!/usr/bin/env bash
# Replicate the full analysis with another model preset (default gemma3-12b) on
# all four item sets, most important first. Expects the parquet files from
# prepare_texts.py in data/. Results go next to the 4B results; probe files get
# the suffix _<preset>, SDT and lens summaries contain all presets.
#
# Usage: bash run_replication.sh [preset]
set -uo pipefail          # no -e: one failing corpus should not stop the others
cd "$(dirname "$0")"
PRESET="${1:-gemma3-12b}"
PY=.venv/bin/python
CONDS="${PRESET}_base_raw_mk ${PRESET}_instruct_raw_mk ${PRESET}_instruct_chat_mk_neutral"
LABCOURSE_FEATS=/path/to/labcourse/experiment_v1/python/output/chat/detection_scores.csv
LABCOURSE_RAW=/path/to/labcourse/experiment_v1/python/output/chat/features_raw.csv

run_corpus () {   # $1 = corpus stem, $2 = outer CV folds, $3.. = extra probe args
  local corpus=$1 folds=$2; shift 2
  echo "##### $PRESET | $corpus  ($(date +%H:%M))"
  $PY judge_logits.py --texts "data/texts_${corpus}.parquet" --preset "$PRESET" &&
  $PY sdt_summary.py "outputs/${corpus}" &&
  $PY probe_layers.py "outputs/${corpus}" --conditions $CONDS --n-splits "$folds" \
      --suffix "_${PRESET}" "$@"
}

run_corpus journalism_seg 7
$PY lens_model.py outputs/journalism_seg \
    --features outputs/journalism_seg/features_journalism_seg.csv \
    --human data/human_items_journalism_seg.csv
run_corpus chat_tw25 10
run_corpus journalism_win80 10
run_corpus chat 10 --baseline-csv "$LABCOURSE_FEATS"
$PY lens_model.py outputs/chat --features "$LABCOURSE_RAW"
echo "##### done ($(date +%H:%M))"
