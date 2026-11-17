#!/bin/bash
#SBATCH -J predtest
#SBATCH -p all
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -o slurm/predtest_%j.out
#SBATCH -e slurm/predtest_%j.err
# Training study: the misleading-news-cue account as a prediction (training_study/prediction_test.py).
# CPU only; reads the OLMo 2 base and ts* activation files (0.1-0.9 GB each) and writes
# outputs/training_study/prediction_test.csv (numbers and IDs only).
# Usage: sbatch slurm/prediction_test.sh [prediction_test.py args]
# Submit from a clean shell (do not source cluster_env.sh first).
source slurm/cluster_env.sh
echo "$(date) start on $(hostname): $*"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -W ignore training_study/prediction_test.py \
    --n-jobs 8 --n-boot 1000 --n-rand 1000 \
    --answers-summary outputs/training_study/answers_summary.csv "$@"
echo "$(date) done"
