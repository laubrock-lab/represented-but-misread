#!/bin/bash
#SBATCH -J patch
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o slurm/patch_%j.out
#SBATCH -e slurm/patch_%j.err
# Position-specific patching. Usage: sbatch slurm/patch.sh <corpus> <preset> <recipient> <donor> <layer> [--limit N]
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $2 == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $*"
python patch_positions.py --corpus "$1" --preset "$2" --recipient "$3" --donor "$4" --layer "$5" "${@:6}"
echo "$(date) done"
