#!/bin/bash
# Localization of the post-training change in TRUE/FALSE judgments (neg_cities), the
# counterpart of the textbook stitching/patching for authorship. On neg_cities the base
# models represent truth perfectly (probe AUC 1.00) but answer poorly (OLMo 2 d' 0.80,
# Apertus -0.88); SFT repairs the answer (3.36 / 4.54). Where does the repair arise?
#
#   stitching: base lower + SFT upper, and SFT lower + base upper, at several splits
#   patching:  SFT <- base and base <- SFT at layers 8 and 16, all position sets
#
# 300 pairs (600 statements) per run. Run ON THE LOGIN NODE from a clean shell:
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/truth_localize_submit.sh"'
# (zsh does not split the loop words; never after `source slurm/cluster_env.sh`.)
set -euo pipefail
DS=truth:neg_cities
N=300
for P in olmo2-7b apertus15-8b; do
  for S in 2 4 6 8 12 16 24; do
    sbatch -J tstitch slurm/stitch.sh "$DS" "$P" base sft "$S" --limit "$N" --n-ppl 40
    sbatch -J tstitch slurm/stitch.sh "$DS" "$P" sft base "$S" --limit "$N" --n-ppl 40
  done
  for L in 8 16; do
    sbatch -J tpatch slurm/patch.sh "$DS" "$P" sft base "$L" --limit "$N"
    sbatch -J tpatch slurm/patch.sh "$DS" "$P" base sft "$L" --limit "$N"
  done
done
squeue -u "$USER" -o "%i %j %T %M" | head -40
