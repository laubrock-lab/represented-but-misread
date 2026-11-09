#!/bin/bash
#SBATCH -J steer
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/steer_%j.out
#SBATCH -e slurm/steer_%j.err
# Causal steering for any preset and post-training stage.
# Usage: sbatch slurm/steer.sh <preset> <corpus> <variant> <fmt> <layer> [extra steer_logits.py args]
# e.g.:  sbatch slurm/steer.sh apertus15-8b textbook sft raw 21
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
set -eo pipefail   # no -u: conda activation scripts use unset variables
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PRESET=$1; CORPUS=$2; VARIANT=$3; FMT=$4; LAYER=$5; shift 5
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $PRESET $VARIANT $FMT $CORPUS L$LAYER $*"
python steer_logits.py "outputs/${CORPUS}" --preset "$PRESET" --variant "$VARIANT" --fmt "$FMT" \
    --layer "$LAYER" --add -6 0 6 --scale 0 6 --unit sd \
    --controls random decision --n-random 3 "$@"
echo "$(date) done"
