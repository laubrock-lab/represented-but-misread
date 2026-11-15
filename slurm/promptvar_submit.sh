#!/bin/bash
# Submit the prompt-robustness runs (answers) and the question-first patching.
# ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/promptvar_submit.sh"'
set -euo pipefail
for PV in qfirst para; do
  for C in chat_tw25 journalism_seg textbook; do
    for P in gemma3-4b olmo2-7b; do sbatch slurm/promptvar_v100.sh "$P" "$C" "$PV"; done
    for P in gemma3-12b qwen3-8b apertus15-8b olmo3-7b-think; do sbatch slurm/promptvar.sh "$P" "$C" "$PV"; done
  done
done
# question-first patching (does the change move into the text positions when the task is known?)
for P in olmo2-7b apertus15-8b; do
  sbatch -J qpatch slurm/patch.sh textbook "$P" sft base 16 --prompt-variant qfirst
  sbatch -J qpatch slurm/patch.sh textbook "$P" base sft 16 --prompt-variant qfirst
done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
