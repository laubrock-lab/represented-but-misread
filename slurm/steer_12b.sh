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

echo "$(date) — start on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 1) Steering on journalism segments (the primary corpus)
#    Layers 20 and 31 are the 12B equivalents of 4B L14 (pre-flip) and L22 (post-flip)
python steer_logits.py outputs/journalism_seg \
    --preset gemma3-12b --layers 20 31 \
    --steps -6 -3 0 3 6 --unit sd --n-random 5

echo "$(date) — journalism_seg steering done"

# 2) Steering on chat_tw25 (length-controlled chat corpus)
python steer_logits.py outputs/chat_tw25 \
    --preset gemma3-12b --layers 20 31 \
    --steps -6 -3 0 3 6 --unit sd --n-random 5

echo "$(date) — chat_tw25 steering done"
echo "$(date) — all done, exit $?"
