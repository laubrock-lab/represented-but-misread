#!/bin/bash
#SBATCH -J truthpv
#SBATCH -p gpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o slurm/truthpv_%j.out
#SBATCH -e slurm/truthpv_%j.err
# True/false judgments with a raw prompt variant (factuality.py --prompt-variant).
# Same as slurm/factuality_stages.sh, but the prompt layout is the first argument and
# the GPU is requested on the command line (V100 for 7B, A100 for Apertus 8B):
#   sbatch --gres=gpu:tesla_v100:1 slurm/factuality_variant.sh sfirst neg_cities olmo2-7b base sft
#   sbatch --nodelist=<gpu-node> --gres=gpu:a100_40gb:1 \
#       slurm/factuality_variant.sh sfirst neg_cities apertus15-8b base sft
# Output: outputs/truth_<dataset>/<preset>_<variant>_raw_{tf,ft}_<prompt-variant>.csv
# Submit from a clean shell.
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PV=$1; DATASET=$2; PRESET=$3; shift 3
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
echo "$(date) start on $(hostname): $DATASET $PRESET $* | prompt-variant $PV"
python factuality.py --statements truth/${DATASET}_statements.py --out-dir outputs/truth_${DATASET} \
    --presets "$PRESET" --variants "$@" --prompt-variant "$PV" --no-analyze
echo "$(date) done"
