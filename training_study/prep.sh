#!/bin/bash
#SBATCH -J tsprep
#SBATCH -p all
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o slurm/tsprep_%j.out
#SBATCH -e slurm/tsprep_%j.err
# Training study: check labels and write the held-out GNAD corpus (after generation).
source slurm/cluster_env.sh
python -W ignore training_study/judgment_examples.py
