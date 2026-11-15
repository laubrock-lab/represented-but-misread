#!/bin/bash
#SBATCH -J steerC
#SBATCH -p gpu
#SBATCH --nodelist=<gpu-node>
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o slurm/steerC_%j.out
#SBATCH -e slurm/steerC_%j.err
# Equivalence check: batched KV-cached scoring vs. the original per-setting scoring.
# Usage: sbatch slurm/steer_check.sh <outdir> <preset> <variant> <layer>
source slurm/cluster_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$1; PRESET=$2; VARIANT=$3; LAYER=$4
if [[ $PRESET == apertus* ]]; then export PYTHONPATH=/work/$USER/tf_apertus; fi
COMMON=(--preset "$PRESET" --variant "$VARIANT" --layer "$LAYER" --add -6 -2 0 2 6 --scale --unit sd
        --controls decision pcrandom split --n-random 2 --fixed-controls --control-add -2 2 --limit 3 --n-boot 10)
time python -W ignore steer_logits.py "outputs/$OUT" "${COMMON[@]}" --suffix _chkA
time python -W ignore steer_logits.py "outputs/$OUT" "${COMMON[@]}" --batched --suffix _chkB
python - "$OUT" <<'PY'
import sys, glob, pandas as pd
d = f"outputs/{sys.argv[1]}"
a = pd.read_csv([f for f in glob.glob(f"{d}/steer_*_chkA.csv") if "summary" not in f and "decomp" not in f][0])
b = pd.read_csv([f for f in glob.glob(f"{d}/steer_*_chkB.csv") if "summary" not in f and "decomp" not in f][0])
k = ["text_id", "order", "direction", "mode", "alpha"]
m = a.merge(b, on=k, suffixes=("_a", "_b"))
print(len(a), len(b), len(m))
print("max |dv diff|", (m.dv_a - m.dv_b).abs().max(), " max |p_mass diff|", (m.p_mass_a - m.p_mass_b).abs().max())
print("corr", m.dv_a.corr(m.dv_b), " dv sd", m.dv_a.std())
print(m.groupby("direction").apply(lambda g: (g.dv_a - g.dv_b).abs().max()).to_string())
PY
