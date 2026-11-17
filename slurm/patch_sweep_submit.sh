#!/bin/bash
# Submit the layer x position sweep of activation patching (OLMo 2 7B, both directions).
# Sweep A: truth:neg_cities (public statements, --limit 300), layers 1, 2, 4, ..., 32.
# Sweep B: textbook (238 texts, longer prompts), layers 2, 4, 8, ..., 32.
# One job per direction x layer group; the donor is cached once per job and each layer's CSV
# is written as soon as that layer is done, so a wall-clock kill keeps the finished layers.
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/patch_sweep_submit.sh [A|B] [dependency-jobid]"'
set -euo pipefail
WHICH=${1:-A}
DEP=${2:-}
DEPARG=()
[[ -n $DEP ]] && DEPARG=(--dependency="afterok:$DEP")

if [[ $WHICH == A ]]; then
  for D in "base sft" "sft base"; do
    for G in "1 2 4 6 8 10" "12 14 16 18 20 22" "24 26 28 30 32"; do
      sbatch "${DEPARG[@]}" --time=08:00:00 -J vpA slurm/patch_sweep_v100.sh \
        truth:neg_cities olmo2-7b $D "$G" --limit 300 --out-root patch_sweep
    done
  done
else
  for D in "base sft" "sft base"; do
    for G in "2 4 8" "12 16 20" "24 28 32"; do
      sbatch "${DEPARG[@]}" --time=12:00:00 --mem=128G -J vpB slurm/patch_sweep_v100.sh \
        textbook olmo2-7b $D "$G" --out-root patch_sweep
    done
  done
fi
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
