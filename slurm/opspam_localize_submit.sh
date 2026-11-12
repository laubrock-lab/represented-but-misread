#!/bin/bash
# Localization of the post-training change in DECEPTION judgments (opspam, Ott et al.):
# one family that LOSES sensitivity (Gemma 3 4B: d' 0.45 -> 0.08) and one that GAINS
# (Qwen3 8B: 0.06 -> 0.50); base vs. final checkpoint (no intermediate stages released).
# OLMo 2 / Apertus differ by only 0.1-0.2 d_z on opspam -> not localizable.
#
#   stitching: base lower + final upper and vice versa, all 800 pairs
#   patching:  final <- base and base <- final at ~1/4 and ~1/2 depth, 400 pairs
#
# Run ON THE LOGIN NODE from a clean shell:
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/opspam_localize_submit.sh"'
set -euo pipefail
DS=truth:opspam
submit() {   # preset, splits, patch layers
  local P=$1 SPLITS=$2 PLAYERS=$3
  for S in $SPLITS; do
    sbatch -J ostitch --time=06:00:00 slurm/stitch.sh "$DS" "$P" base instruct "$S" --n-ppl 40
    sbatch -J ostitch --time=06:00:00 slurm/stitch.sh "$DS" "$P" instruct base "$S" --n-ppl 40
  done
  for L in $PLAYERS; do
    sbatch -J opatch --time=10:00:00 slurm/patch.sh "$DS" "$P" instruct base "$L" --limit 400
    sbatch -J opatch --time=10:00:00 slurm/patch.sh "$DS" "$P" base instruct "$L" --limit 400
  done
}
submit gemma3-4b "2 4 6 8 12 17 23 28" "8 17"     # 34 layers
submit qwen3-8b  "2 4 6 9 12 18 24 30" "9 18"     # 36 layers
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
