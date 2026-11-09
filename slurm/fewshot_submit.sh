#!/bin/bash
# Few-shot experiment (belief/prior vs. fixed readout): k = 8 labelled
# demonstrations (4 AI, 4 human, from other sources/authors), 4 seeds (last
# label alternates), both answer orders, raw + chat, no activations.
# Run on <cluster-login-host> from /work/$USER/latent-detection in a CLEAN shell:
#   bash slurm/fewshot_submit.sh
# Analysis (locally, after copying outputs/fewshot/):
#   python fewshot_summary.py --corpora chat_tw25 textbook
set -eo pipefail
K=8; SEEDS="0 1 2 3"
for c in chat_tw25 textbook; do
  common="--fewshot $K --fewshot-seeds $SEEDS --acts-orders --out-dir outputs/fewshot/$c"
  sbatch --parsable slurm/stages.sh gemma3-4b $c --variants base instruct $common
  sbatch --parsable slurm/stages.sh qwen3-8b  $c --variants base instruct $common
  sbatch --parsable slurm/stages.sh olmo2-7b  $c --variants base sft instruct $common
done
