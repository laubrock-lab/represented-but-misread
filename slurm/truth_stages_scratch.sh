#!/bin/bash
#SBATCH -J tstages
#SBATCH -p all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o slurm/tstages_%j.out
#SBATCH -e slurm/tstages_%j.err
# Answer d', probe d' and the cities -> negated transfer probe for a subset of presets,
# WITHOUT touching the paper's outputs/truth_<ds>/stages.csv, item_scores.csv and
# outputs/truth_stages_contrasts.csv.
#
# truth_stages.py has no output option: it takes its root from the location of config.py.
# It is therefore run in a throw-away copy of the repository whose outputs/ is a directory
# of symlinks to the judgment files of the requested presets only — the mechanism that
# reproduce/negation.sh documents under OUT=. Nothing of the repository is modified.
#
# Usage: sbatch slurm/truth_stages_scratch.sh <preset> [<preset> ...]
#   e.g. sbatch slurm/truth_stages_scratch.sh qwen25-7b
# Output: outputs/truth_stages_scratch/out/truth_{cities,neg_cities}/stages.csv,
#         outputs/truth_stages_scratch/out/truth_stages_contrasts.csv
source slurm/cluster_env.sh
PRESETS="$*"
[[ -n $PRESETS ]] || { echo "give at least one preset" >&2; exit 2; }
SCR=$WORK/outputs/truth_stages_scratch
rm -rf "$SCR"
mkdir -p "$SCR/repo/truth" "$SCR/repo/data" "$SCR/out"
cp "$WORK"/*.py "$SCR/repo/"
cp "$WORK"/truth/*.py "$SCR/repo/truth/"
ln -s "$WORK/data/external" "$SCR/repo/data/external"
ln -s "$SCR/out" "$SCR/repo/outputs"
for ds in cities neg_cities; do
    mkdir -p "$SCR/out/truth_$ds"
    for p in $PRESETS; do
        n=$(ls "$WORK/outputs/truth_$ds/${p}"_* 2>/dev/null | wc -l)
        echo "truth_$ds: $n files for $p"
        [[ $n -gt 0 ]] && ln -s "$WORK/outputs/truth_$ds/${p}"_* "$SCR/out/truth_$ds/"
    done
done
cd "$SCR/repo"
echo "$(date) truth_stages.py --presets $PRESETS --datasets cities neg_cities"
python -W ignore truth_stages.py --presets $PRESETS --datasets cities neg_cities --n-boot 1000
echo "-> $SCR/out/truth_{cities,neg_cities}/stages.csv, $SCR/out/truth_stages_contrasts.csv"
