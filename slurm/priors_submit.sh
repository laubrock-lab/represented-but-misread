#!/bin/bash
# ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/priors_submit.sh"'
set -euo pipefail
for C in chat_tw25 journalism_seg textbook; do
  for P in gemma3-4b olmo2-7b; do sbatch slurm/priors_v100.sh "$P" "$C"; done
  for P in gemma3-12b qwen3-8b apertus15-8b olmo3-7b-think; do sbatch slurm/priors.sh "$P" "$C"; done
done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
