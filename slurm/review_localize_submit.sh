#!/bin/bash
# Response to review (16 Nov 2026): (1) stitching control task, (2) localization on a second
# authorship corpus.
#   (1) counterfact statements, where base and SFT judge alike: do stitched hybrids leave the
#       range of their parents (overshoot = stitching artifact)? Same splits as for the
#       textbook authorship stitching, both directions, plus both parents.
#   (2) short chats (chat_tw25): position-specific patching at layers 8 and 16, both
#       directions, and stitching at 4/8/16/24 layers, both directions, plus both parents.
# Run ON THE LOGIN NODE from a clean shell:
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/review_localize_submit.sh"'
set -euo pipefail
for P in olmo2-7b apertus15-8b; do
  NL=32
  # (1) stitching control: counterfact
  sbatch -J cstitch slurm/stitch.sh truth:counterfact "$P" base sft 0 --limit 300 --n-ppl 40
  sbatch -J cstitch slurm/stitch.sh truth:counterfact "$P" base sft "$NL" --limit 300 --n-ppl 40
  for S in 1 2 4 6 8 16 24; do
    sbatch -J cstitch slurm/stitch.sh truth:counterfact "$P" base sft "$S" --limit 300 --n-ppl 40
    sbatch -J cstitch slurm/stitch.sh truth:counterfact "$P" sft base "$S" --limit 300 --n-ppl 40
  done
  # (2) chats: patching and stitching
  for L in 8 16; do
    sbatch -J cpatch slurm/patch.sh chat_tw25 "$P" sft base "$L"
    sbatch -J cpatch slurm/patch.sh chat_tw25 "$P" base sft "$L"
  done
  sbatch -J chstitch slurm/stitch.sh chat_tw25 "$P" base sft 0 --n-ppl 40
  sbatch -J chstitch slurm/stitch.sh chat_tw25 "$P" base sft "$NL" --n-ppl 40
  for S in 4 8 16 24; do
    sbatch -J chstitch slurm/stitch.sh chat_tw25 "$P" base sft "$S" --n-ppl 40
    sbatch -J chstitch slurm/stitch.sh chat_tw25 "$P" sft base "$S" --n-ppl 40
  done
done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
