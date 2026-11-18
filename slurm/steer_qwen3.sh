#!/bin/bash
#SBATCH -J steer-qwen3
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/steer_qwen3_%j.out
#SBATCH -e slurm/steer_qwen3_%j.err

source slurm/cluster_env.sh
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "$(date) — start on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Qwen3-8B steering, mirroring steer_12b.sh (same design: 5 alphas, 5 gains,
# 5 random nulls, decision positive control) so the corpus x model steering
# matrix is directly comparable across model families.
# Usage:  sbatch slurm/steer_qwen3.sh <corpus> <layer> [extra steer_logits.py args]
#   e.g.:  sbatch slurm/steer_qwen3.sh chat_tw25 24
# Needs the condition's acts.npz + judgment CSVs in outputs/<corpus>/
# (synced from the local machine; the steering directions are cross-fitted
# from those activations).
CORPUS="${1:?Usage: sbatch steer_qwen3.sh <corpus> <layer>}"
LAYER="${2:?Usage: sbatch steer_qwen3.sh <corpus> <layer>}"
shift 2

echo "$(date) — ${CORPUS} layer ${LAYER} $*"
python steer_logits.py "outputs/${CORPUS}" \
    --preset qwen3-8b --layer "$LAYER" \
    --add -6 -3 0 3 6 --scale 0 0.5 1 3 6 \
    --unit sd --n-random 5 --controls random decision "$@"

echo "$(date) — done, exit $?"
