#!/bin/bash
#SBATCH -J prior
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -o slurm/prior_%j.out
#SBATCH -e slurm/prior_%j.err
# Bias manipulation for the SDT invariance test (17 Nov 2026): stated AI base rate of 20, 50,
# or 80% in the raw prompt (one wording of the base-rate sentence), both answer orders, base
# and final model; answers only. Existing conditions are skipped.
# Usage: sbatch slurm/priors.sh <preset> <corpus>
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PRESET=$1; CORPUS=$2
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $PRESET $CORPUS"
python -W ignore judge_logits.py --texts "data/texts_${CORPUS}.parquet" --preset "$PRESET" \
    --variants base instruct --formats raw --priors 20 50 80 --prior-orders ai --acts-orders \
    --out-dir "outputs/${CORPUS}"
echo "$(date) done"
