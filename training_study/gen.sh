#!/bin/bash
#SBATCH -J tsgen
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/tsgen_%j.out
#SBATCH -e slurm/tsgen_%j.err
# Training study, step 2: generate AI counterparts. Usage: sbatch training_study/gen.sh <qwen3-8b|gemma3-12b>
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date) start on $(hostname): $1"
python -W ignore training_study/generate_ai.py --generator "$1"
echo "$(date) done"
