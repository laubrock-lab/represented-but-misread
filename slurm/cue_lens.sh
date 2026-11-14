#!/bin/bash
#SBATCH -J cuelens
#SBATCH -p all
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm/cuelens_%j.out
#SBATCH -e slurm/cuelens_%j.err
# Cue-level lens analysis of the chat judgments (CPU), incl. probe judges from activations.
# Needs outputs/chat_cues/cues_chat_tw25_all.csv (computed locally: cue_lens_chats.py features).
# Usage: sbatch slurm/cue_lens.sh     (submit from a clean shell)
source slurm/cluster_env.sh
echo "$(date) start on $(hostname)"
python -W ignore cue_lens_chats.py analyze --probes
echo "$(date) done"
