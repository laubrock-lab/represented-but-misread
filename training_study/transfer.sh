#!/bin/bash
#SBATCH -J tstrans
#SBATCH -p all
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o slurm/tstrans_%j.out
#SBATCH -e slurm/tstrans_%j.err
# Training study: probes and probe transfer base vs. adapters (CPU).
source slurm/cluster_env.sh
echo "$(date) start on $(hostname)"
python -W ignore training_study/transfer_ts.py "$@"
echo "$(date) done"
