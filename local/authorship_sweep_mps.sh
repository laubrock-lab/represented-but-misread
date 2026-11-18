#!/bin/bash
# Authorship patching across layers on a local Apple-silicon machine (18 Nov 2026), because the
# cluster GPUs were not available: textbook summaries (all 238 texts), OLMo 2 7B base vs. SFT,
# both directions, position sets none / text / post, float16 (about a third faster than bfloat16
# on MPS; the run is compared only with its own unpatched conditions).
# Layer groups in order of information value (layers 8 and 16 exist from the cluster runs); each
# group is run in both directions before the next, and every layer is written when it is done.
# Output: outputs/patch_sweep_local/textbook/  (never mixed with the cluster runs).
# Usage: bash local/authorship_sweep_mps.sh ["12" "14 10" ...]
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
GROUPS_=("$@"); [ ${#GROUPS_[@]} -eq 0 ] && GROUPS_=("12" "14 10")
LOG=outputs/patch_sweep_local/run.log
mkdir -p outputs/patch_sweep_local
for G in "${GROUPS_[@]}"; do
  for dir in "base sft" "sft base"; do
    set -- $dir
    echo "$(date) start layers $G recipient $1 donor $2" >> "$LOG"
    python -W ignore patch_positions.py --corpus textbook --preset olmo2-7b --recipient "$1" --donor "$2" \
        --layers $G --sets none text post --device mps --dtype float16 --out-root patch_sweep_local \
        --skip-existing 2>&1 | grep -v "Loading weights" >> "$LOG"
    echo "$(date) done layers $G recipient $1 donor $2 EXIT=${PIPESTATUS[0]}" >> "$LOG"
  done
done
echo "$(date) ALL DONE" >> "$LOG"
