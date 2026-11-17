#!/bin/bash
#SBATCH -J vpatch
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -o slurm/psweep_%j.out
#SBATCH -e slurm/psweep_%j.err
# Layer sweep of position-specific patching on a V100 (18 Nov 2026): one donor pass caches the
# residual stream for all requested layers, the recipient is loaded once and looped over layers
# x position sets; one CSV per layer, written as soon as the layer is done.
# Usage: sbatch slurm/patch_sweep_v100.sh <corpus> <preset> <recipient> <donor> "<layers>" [extra args]
#   e.g. sbatch slurm/patch_sweep_v100.sh truth:neg_cities olmo2-7b base sft "1 2 4 6" --limit 300 \
#            --out-root patch_sweep
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CORPUS=$1; PRESET=$2; REC=$3; DON=$4; LAYERS=$5; shift 5
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $CORPUS $PRESET $REC from $DON | layers $LAYERS | $*"
python -W ignore patch_positions.py --corpus "$CORPUS" --preset "$PRESET" --recipient "$REC" \
    --donor "$DON" --layers $LAYERS "$@"
echo "$(date) done"
