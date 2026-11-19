#!/bin/bash
#SBATCH -J steerNA2
#SBATCH -p all
#SBATCH --qos=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o repro_jobs/steerNA2_%j.out
#SBATCH -e repro_jobs/steerNA2_%j.err
# Reproducibility run: steer_nulls_analysis.py with the new columns
# cos_class_decision (Results 3.3 / 3.6) and dv_diff_lo/hi + dv_diff_clear +
# sign_agree (the explicit criterion behind "all 20 cells in which this difference
# was clearly different from zero"). Needs the *_acts.npz of all 25 cells.
# -> outputs/steer_nulls_summary.csv, outputs/steer_cosines.csv
set -euo pipefail
export WORK=/work/$USER/latent-detection
cd "$WORK"
module load lang/Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate latent-detection
export HF_HOME="$WORK/.hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u -W ignore steer_nulls_analysis.py
