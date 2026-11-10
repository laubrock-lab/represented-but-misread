#!/bin/bash
#SBATCH -J grad
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o slurm/grad_%j.out
#SBATCH -e slurm/grad_%j.err
# Answer gradients (gradient_readout.py) for several conditions of one preset.
# Usage: sbatch slurm/grad.sh <preset> <corpus> <variant:fmt> [<variant:fmt> ...]
# e.g.:  sbatch slurm/grad.sh olmo2-7b textbook base:raw sft:raw sft:chat instruct:raw
# Extra args via env, e.g.: GRAD_EXTRA=--checkpointing sbatch --export=ALL,GRAD_EXTRA=--checkpointing ...
# Submit from a clean shell (not after `source slurm/cluster_env.sh`).
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PRESET=$1; CORPUS=$2; shift 2
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
for vf in "$@"; do
  v=${vf%%:*}; f=${vf##*:}
  echo "$(date) $PRESET $v $f $CORPUS"
  python gradient_readout.py "outputs/$CORPUS" --preset "$PRESET" --variant "$v" --fmt "$f" $GRAD_EXTRA
done
echo "$(date) done"
