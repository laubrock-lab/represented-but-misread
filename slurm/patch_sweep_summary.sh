#!/bin/bash
#SBATCH -J psum
#SBATCH -p all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH -o slurm/psum_%j.out
#SBATCH -e slurm/psum_%j.err
# CPU job: d' per corpus x direction x layer x position set from the patching sweep.
# Usage: sbatch slurm/patch_sweep_summary.sh [extra args for patch_sweep_summary.py]
source slurm/cluster_env.sh
echo "$(date) start on $(hostname): $*"
python -W ignore patch_sweep_summary.py "$@"
echo "$(date) done"
