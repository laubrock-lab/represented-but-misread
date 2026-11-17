#!/bin/bash
#SBATCH -J fmlLab
#SBATCH -p all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o repro_jobs/fmlLab_%j.out
#SBATCH -e repro_jobs/fmlLab_%j.err
# Reproducibility run (17 Nov 2026): the confounds of the C2 stereotype labels
# (Results 3.7). Needs data/training_study/ai_*.parquet, which exist only here.
# -> outputs/training_study/formality_label_confounds.csv
set -euo pipefail
export WORK=/work/$USER/latent-detection
cd "$WORK"
module load lang/Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate latent-detection
export HF_HOME="$WORK/.hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u -W ignore training_study/formality_check.py --labels --no-validity
