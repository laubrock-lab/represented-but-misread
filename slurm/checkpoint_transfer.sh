#!/bin/bash
#SBATCH -J ckpttr
#SBATCH -p all
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -o slurm/ckpttr_%j.out
#SBATCH -e slurm/ckpttr_%j.err
# Layer-wise probe transfer between checkpoints (CPU). Usage: sbatch slurm/checkpoint_transfer.sh [--sets ...]
# Submit from a clean shell.
source slurm/cluster_env.sh
echo "$(date) start on $(hostname): $*"
python -W ignore checkpoint_transfer.py "$@"
echo "$(date) done"
