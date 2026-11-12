#!/bin/bash
#SBATCH -J truthana
#SBATCH -p all
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o slurm/truthana_%j.out
#SBATCH -e slurm/truthana_%j.err
# Stage analysis of the truth / opspam sets (CPU; reads the 800 MB activation files).
# Usage: sbatch slurm/truth_analyze.sh [dataset ...]   (default: all sets in truth_stages.py)
# Submit from a clean shell.
source slurm/cluster_env.sh
echo "$(date) start on $(hostname): $*"
python -W ignore truth_stages.py ${1:+--datasets "$@"} --n-boot 1000
echo "$(date) done"
