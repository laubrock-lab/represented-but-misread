#!/bin/bash
#SBATCH -J steerT
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -o slurm/steerT_%j.out
#SBATCH -e slurm/steerT_%j.err
# Steering on true/false-type judgments (truth sets, opspam): class direction at ~67 % depth,
# last prompt token, with strict controls. Usage:
#   sbatch slurm/steer_truth.sh <dataset> <preset> <variant> [extra steer_logits args]
# Submit from a clean shell.
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DS=$1; PRESET=$2; VARIANT=$3; shift 3
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $DS $PRESET $VARIANT $*"
python -W ignore steer_logits.py "outputs/truth_${DS}" --preset "$PRESET" --variant "$VARIANT" \
    --add -6 -3 0 3 6 --scale --unit sd --controls decision random pcrandom split --n-random 3 \
    --limit 300 "$@"
echo "$(date) done"
