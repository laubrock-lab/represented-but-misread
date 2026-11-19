#!/bin/bash
#SBATCH -J ngramctl
#SBATCH -p all
#SBATCH --qos=short
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o slurm/ngramctl_%j.out
#SBATCH -e slurm/ngramctl_%j.err
# Reviewer control for P1: n-gram classifiers fitted on the same news as the news probe
# (training_study/prediction_ngram_control.py). CPU only; reads the OLMo 2 base
# activations of gnad_heldout and chat_tw25 and the two text parquets, and writes
# outputs/training_study/prediction_ngram_control.csv (numbers and IDs only).
# Usage: sbatch slurm/prediction_ngram_control.sh [prediction_ngram_control.py args]
# Submit from a clean shell (do not source cluster_env.sh first).
source slurm/cluster_env.sh
echo "$(date) start on $(hostname): $*"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -W ignore training_study/prediction_ngram_control.py \
    --n-jobs 2 --n-boot 1000 "$@"
echo "$(date) done"
