#!/bin/bash
# Submit the layer x position sweep for Apertus 1.5 8B on the A100 node.
# Replicates the OLMo 2 sweep (truth:neg_cities, layers 2-32, both directions, 100 pairs)
# to test whether the text-state inverted-U generalises beyond OLMo 2.
# Apertus 8B needs >16 GB VRAM (OOMs on V100), so this uses the A100 node.
#
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/patch_sweep_apertus_submit.sh [dependency-jobid]"'
set -euo pipefail
DEP=${1:-}
DEPARG=()
[[ -n $DEP ]] && DEPARG=(--dependency="afterok:$DEP")

# 4 layers per job (Apertus is slower than OLMo 2 — keep jobs under 8h).
# Both directions: base-from-sft and sft-from-base.
for D in "base sft" "sft base"; do
  for G in "2 4 6 8" "10 12 14 16" "20 24 28 32"; do
    sbatch "${DEPARG[@]}" slurm/patch_sweep_a100.sh \
      truth:neg_cities apertus15-8b $D "$G" --limit 100 --out-root patch_sweep --skip-existing
  done
done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
