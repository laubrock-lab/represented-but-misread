#!/bin/bash
# Shared environment setup for all SLURM jobs.
# Source this at the top of every sbatch script.

set -euo pipefail

export WORK=/work/$USER/latent-detection
cd "$WORK"

# Activate conda env (adjust path if miniconda is elsewhere)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate latent-detection

# HuggingFace cache on /work (not quota-limited $HOME)
export HF_HOME="$WORK/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"
