#!/bin/bash
#SBATCH -J pvar
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o slurm/pvar_%j.out
#SBATCH -e slurm/pvar_%j.err
# Prompt robustness (response to review, 16 Nov 2026): answers of base and final model with a
# raw template variant (qfirst = question before text; para = paraphrase), no activations.
# Usage: sbatch slurm/promptvar.sh <preset> <corpus> <qfirst|para>
# Output: outputs/promptvar_<variant>/<corpus>/<preset>_<variant>_raw_<order>.csv
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PRESET=$1; CORPUS=$2; PV=$3
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $PRESET $CORPUS $PV"
python -W ignore judge_logits.py --texts "data/texts_${CORPUS}.parquet" --preset "$PRESET" \
    --variants base instruct --formats raw --acts-orders --prompt-variant "$PV" \
    --out-dir "outputs/promptvar_${PV}/${CORPUS}"
echo "$(date) done"
