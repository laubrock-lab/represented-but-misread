#!/bin/bash
#SBATCH -J steerNA
#SBATCH -p all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm/steerNA_%j.out
#SBATCH -e slurm/steerNA_%j.err
# In-range check, null test and natural-scale comparison for all finished _fixed_nulls runs.
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -W ignore steer_nulls_analysis.py
