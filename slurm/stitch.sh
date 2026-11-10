#!/bin/bash
#SBATCH -J stitch
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o slurm/stitch_%j.out
#SBATCH -e slurm/stitch_%j.err
# Model stitching. Usage: sbatch slurm/stitch.sh <corpus> <preset> <lower> <upper> <split> [--limit N]
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $2 == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $*"
python stitch.py --corpus "$1" --preset "$2" --lower "$3" --upper "$4" --split "$5" "${@:6}"
echo "$(date) done"
