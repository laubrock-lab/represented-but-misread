#!/usr/bin/env bash
# Within-corpus controls and prompt-type stratification.
#
#   1. author-grouped and source-grouped CV, generator-held-out transfer, and
#      naive-vs-optimized transfer (probe_controls.py), for all four item sets;
#   2. transfer from chat to journalism with the source probe trained on all,
#      only naive, or only style-matched AI texts (probe_transfer.py).
#
# The layer is always the full-corpus layer from probe_layers*.csv. Permutation
# nulls are already in transfer_selected.csv; the stratified runs only add the
# target-stratum d' columns, so --n-perm 0 is intentional.
#
# Usage: bash run_controls.sh [python]
set -uo pipefail
cd "$(dirname "$0")"
PY="${1:-.venv/bin/python}"

for corpus in chat chat_tw25 journalism_seg journalism_win80; do
  echo "##### controls | $corpus  ($(date +%H:%M))"
  $PY probe_controls.py --corpus "$corpus" --n-boot 1000 --n-jobs 8 || exit 1
done

for stratum in all naive optimized; do
  echo "##### transfer | train stratum $stratum  ($(date +%H:%M))"
  $PY probe_transfer.py --train-stratum "$stratum" --tag "_strata_${stratum}" \
      --pairs chat:journalism_seg chat:journalism_win80 \
              chat_tw25:journalism_seg chat_tw25:journalism_win80 \
      --n-perm 0 --n-boot 500 --n-jobs 8 || exit 1
done

echo "##### done ($(date +%H:%M))"
