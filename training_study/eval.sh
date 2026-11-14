#!/bin/bash
#SBATCH -J tseval
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/tseval_%j.out
#SBATCH -e slurm/tseval_%j.err
# Training study: evaluate the final adapter of a run on the authorship corpora and the
# held-out GNAD items (raw prompt, both orders, activations for probes).
# Usage: sbatch training_study/eval.sh <C0|C1|C2|C3> <seed>
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COND=$1; SEED=$2
RUN=adapters/ts/olmo2_${COND}_s${SEED}
CKPT=$(ls -d $RUN/ckpt-* | sort | tail -1)
echo "$(date) start on $(hostname): $CKPT"
for C in chat_tw25 journalism_seg textbook gnad_heldout; do
  python -W ignore judge_logits.py --texts data/texts_${C}.parquet --preset olmo2-7b --variants base \
      --formats raw --adapter "$CKPT" --tag "ts${COND}s${SEED}" --out-dir outputs/training_study/$C
done
echo "$(date) done"
