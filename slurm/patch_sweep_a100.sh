#!/bin/bash
#SBATCH -J apatch
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -o slurm/psweep_%j.out
#SBATCH -e slurm/psweep_%j.err
# Layer sweep of position-specific patching on the A100 node (Apertus 8B needs >16 GB VRAM).
# Usage: sbatch slurm/patch_sweep_a100.sh <corpus> <preset> <recipient> <donor> "<layers>" [extra args]
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CORPUS=$1; PRESET=$2; REC=$3; DON=$4; LAYERS=$5; shift 5
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $CORPUS $PRESET $REC from $DON | layers $LAYERS | $*"
python -W ignore patch_positions.py --corpus "$CORPUS" --preset "$PRESET" --recipient "$REC" \
    --donor "$DON" --layers $LAYERS "$@"
echo "$(date) done"
