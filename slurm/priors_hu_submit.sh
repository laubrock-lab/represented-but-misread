#!/bin/bash
# Rerun of the base-rate conditions with the other mention order of the base-rate sentence
# (human share named first, --prior-orders hu); 18 Nov 2026.
# One dependency chain per preset (3 corpora in sequence), so at most three V100 jobs of
# this batch run at the same time and other users' jobs still fit under the per-user cap.
# ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/priors_hu_submit.sh"'
set -euo pipefail
PO=${1:-hu}
for P in gemma3-4b olmo2-7b olmo3-7b-think; do
  DEP=""
  for C in chat_tw25 journalism_seg textbook; do
    if [[ -z $DEP ]]; then
      JID=$(sbatch --parsable slurm/priors_v100.sh "$P" "$C" "$PO")
    else
      JID=$(sbatch --parsable --dependency=afterany:"$DEP" slurm/priors_v100.sh "$P" "$C" "$PO")
    fi
    echo "submitted $JID  $P $C prior_order=$PO"
    DEP=$JID
  done
done
squeue -u "$USER" -o "%i %j %T %R"
