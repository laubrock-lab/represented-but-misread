#!/bin/bash
#SBATCH -J stages
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/stages_%j.out
#SBATCH -e slurm/stages_%j.err

# Judgments + activations for all post-training stages of one family.
# Usage:  sbatch slurm/stages.sh <preset> <corpus> [extra judge_logits args]
#   e.g.  sbatch slurm/stages.sh olmo2-7b chat_tw25
#         sbatch slurm/stages.sh apertus15-8b journalism_seg
#         sbatch slurm/stages.sh olmo2-7b chat_tw25 --limit 10 --orders mk --out-dir outputs/smoke_stages
# Apertus 1.5 needs the swiss-ai transformers fork (model card), installed in an
# isolated folder so that the main environment stays unchanged:
#   pip install --no-deps --target /work/$USER/tf_apertus \
#     "transformers @ git+https://github.com/swiss-ai/transformers.git@0c5e12f18a5ff5c93cb1ef755aebbd557e26e884" \
#     "huggingface-hub>=1.31,<2" httpx httpcore certifi sniffio
set -euo pipefail
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PRESET=$1; CORPUS=$2; shift 2
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
python -c "import transformers; print('transformers', transformers.__version__)"
echo "$(date) start on $(hostname): $PRESET $CORPUS $*"
python judge_logits.py --texts "data/texts_${CORPUS}.parquet" --preset "$PRESET" \
    --variants base sft dpo instruct "$@"
echo "$(date) done"
