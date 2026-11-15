#!/bin/bash
#SBATCH -J steerN
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -o slurm/steerN_%j.out
#SBATCH -e slurm/steerN_%j.err
# Wider steering nulls (response to review, 15 Nov 2026): 50 pcrandom + 50 split control
# directions, each drawn ONCE for all texts (--fixed-controls), at +-2 SD; class and decision
# directions with a dose curve (+-1, 2, 4, 6 SD). KV-cached batched scoring (--batched).
# Usage: sbatch slurm/steer_nulls.sh <outdir> <preset> <variant> <fmt> <layer> [extra args]
#   e.g. sbatch slurm/steer_nulls.sh chat_tw25 qwen3-8b instruct raw 24
#        sbatch slurm/steer_nulls.sh truth_neg_cities olmo2-7b base raw 21 --limit 300
# Submit from a clean shell.
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$1; PRESET=$2; VARIANT=$3; FMT=$4; LAYER=$5; shift 5
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $OUT $PRESET $VARIANT $FMT L$LAYER $*"
python -W ignore steer_logits.py "outputs/${OUT}" --preset "$PRESET" --variant "$VARIANT" --fmt "$FMT" \
    --layer "$LAYER" --add -6 -4 -2 -1 0 1 2 4 6 --scale --unit sd \
    --controls decision pcrandom split --n-random 50 --fixed-controls --control-add -2 2 \
    --batched --suffix _nulls "$@"
echo "$(date) done"
