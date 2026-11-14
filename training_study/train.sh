#!/bin/bash
#SBATCH -J tstrain
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -o slurm/tstrain_%j.out
#SBATCH -e slurm/tstrain_%j.err
# Training study: LoRA SFT. Usage: sbatch training_study/train.sh <C0|C1|C2|C3> <seed> [extra args]
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COND=$1; SEED=$2; shift 2
echo "$(date) start on $(hostname): $COND seed $SEED $*"
python -W ignore training_study/sft_readout.py --condition "$COND" --seed "$SEED" \
    --out-dir "adapters/ts/olmo2_${COND}_s${SEED}" "$@"
echo "$(date) done"
