#!/bin/bash
#SBATCH -J steer-12b
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/steer_12b_%j.out
#SBATCH -e slurm/steer_12b_%j.err

source slurm/cluster_env.sh
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "$(date) — start on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Usage: sbatch slurm/steer_12b.sh <corpus> <layer>
# e.g.:  sbatch slurm/steer_12b.sh journalism_seg 31
#        sbatch slurm/steer_12b.sh chat_tw25 20
CORPUS="${1:?Usage: sbatch steer_12b.sh <corpus> <layer>}"
LAYER="${2:?Usage: sbatch steer_12b.sh <corpus> <layer>}"

echo "$(date) — ${CORPUS} layer ${LAYER}"
python steer_logits.py "outputs/${CORPUS}" \
    --preset gemma3-12b --layer "$LAYER" \
    --add -6 -3 0 3 6 --scale 0 0.5 1 3 6 \
    --unit sd --n-random 5

echo "$(date) — done, exit $?"
