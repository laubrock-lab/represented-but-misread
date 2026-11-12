#!/bin/bash
#SBATCH -J truth
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o slurm/truth_%j.out
#SBATCH -e slurm/truth_%j.err
# True/false judgments across post-training stages (truth-judgment replication).
# Usage: sbatch slurm/factuality_stages.sh <dataset> <preset> <variant> [<variant> ...]
#   dataset: cities | neg_cities | counterfact  (truth/<dataset>_statements.py)
# Submit from a clean shell.
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATASET=$1; PRESET=$2; shift 2
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $DATASET $PRESET $*"
python factuality.py --statements truth/${DATASET}_statements.py --out-dir outputs/truth_${DATASET} \
    --presets "$PRESET" --variants "$@" --no-analyze
echo "$(date) done"
