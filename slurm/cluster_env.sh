#!/bin/bash
# Shared environment setup for all SLURM jobs.
# Source this at the top of every sbatch script.

set -euo pipefail

export WORK=/work/$USER/latent-detection
cd "$WORK"

module load lang/Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate latent-detection

# HuggingFace cache on /work (not quota-limited $HOME)
export HF_HOME="$WORK/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"
