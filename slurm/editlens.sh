#!/bin/bash
#SBATCH -J editlens
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o slurm/editlens_%j.out
#SBATCH -e slurm/editlens_%j.err
# EditLens scores for one corpus. Needs access to meta-llama/Llama-3.2-3B and
# pangram/editlens_Llama-3.2-3B, both pre-downloaded on <cluster-login-host>.
# Usage: sbatch slurm/editlens.sh <corpus> [--limit N]   (from a clean shell)
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date) start on $(hostname): $*"
python editlens_score.py --corpus "$@"
echo "$(date) done"
