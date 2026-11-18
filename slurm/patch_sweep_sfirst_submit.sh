#!/bin/bash
# Layer x position sweep of activation patching with the statement-first truth prompt
# (patch_positions.py --prompt-variant sfirst), replicating the default-prompt sweep in
# outputs/patch_sweep/truth_neg_cities/: 100 pairs, layers 2 ... 32, both directions, all
# position sets. Under sfirst the question line stands after the statement, so the set
# "question" is non-empty and "post" = question + answer cue. Output files get the suffix
# _sfirst and therefore never overwrite the default-prompt sweep.
#
# Run only for the models whose base-vs-SFT answer d' differ by >= 1 under sfirst
# (outputs/truth_neg_cities/sfirst_answers_summary.csv, truth_variant_answers.py).
# OLMo 2 7B on a V100, 3 layers per 4 h job; Apertus 1.5 8B on the A100 node (>16 GB VRAM),
# 4 layers per job — submit the Apertus part only behind the other A100 jobs in the queue.
#
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && \
#       bash slurm/patch_sweep_sfirst_submit.sh [olmo|apertus|both]"'
set -euo pipefail
WHICH=${1:-both}

if [[ $WHICH == olmo || $WHICH == both ]]; then
  for D in "base sft" "sft base"; do
    for G in "2 4 6" "8 10 12" "14 16 20" "24 28 32"; do
      sbatch --time=04:00:00 --gres=gpu:tesla_v100:1 -J vpsf slurm/patch_sweep_v100.sh \
        truth:neg_cities olmo2-7b $D "$G" --limit 100 --out-root patch_sweep \
        --prompt-variant sfirst --skip-existing
    done
  done
fi

if [[ $WHICH == apertus || $WHICH == both ]]; then
  for D in "base sft" "sft base"; do
    for G in "2 4 6 8" "10 12 14 16" "20 24 28 32"; do
      sbatch --time=04:00:00 -J apsf slurm/patch_sweep_a100.sh \
        truth:neg_cities apertus15-8b $D "$G" --limit 100 --out-root patch_sweep \
        --prompt-variant sfirst --skip-existing
    done
  done
fi
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
