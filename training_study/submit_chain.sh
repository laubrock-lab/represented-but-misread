#!/bin/bash
# Training study: queue the whole chain with SLURM dependencies.
#   prep (after generation) -> 12 training runs (C0-C3 x seeds 1-3) -> one evaluation per run
#   + reference evaluations (OLMo 2 base and official SFT) on the held-out GNAD corpus
# Usage (login node, clean shell):  bash training_study/submit_chain.sh <generation job ids...>
set -euo pipefail
cd /work/$USER/latent-detection
GEN=$(IFS=:; echo "$*")
PREP=$(sbatch --parsable ${GEN:+--dependency=afterok:$GEN} training_study/prep.sh)
echo "prep $PREP"
sbatch --dependency=afterok:$PREP -J tsref slurm/stages.sh olmo2-7b gnad_heldout --formats raw >/dev/null
for SEED in 1 2 3; do
  for COND in C0 C1 C2 C3; do
    T=$(sbatch --parsable --dependency=afterok:$PREP training_study/train.sh $COND $SEED)
    E=$(sbatch --parsable --dependency=afterok:$T training_study/eval.sh $COND $SEED)
    echo "$COND s$SEED: train $T, eval $E"
  done
done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
