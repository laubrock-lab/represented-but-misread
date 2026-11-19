#!/usr/bin/env bash
# Reproduce the negated-statements analysis of Section 3.1 from public data.
#
#   bash reproduce/negation.sh                 # paper settings, OLMo 2 7B, steps a-e
#   bash reproduce/negation.sh --limit 12      # smoke test (few pairs, same code path)
#   OUT=/scratch/neg bash reproduce/negation.sh --steps a,b
#
# Steps (see reproduce/README.md for the mapping to the numbers in the manuscript):
#   a  factuality.py       answers + activations for every checkpoint (cities, neg_cities)
#   b  truth_stages.py     answer d', probe d', transfer probe -> truth_*/stages.csv
#   c  stitch.py           model stitching base/SFT, 7 splits, both directions, 300 pairs
#   d  patch_positions.py  activation patching at layers 8 and 16, both directions
#   e  localization_summary.py  -> patch/patch_summary.csv, stitch/stitch_neg_cities.csv
#   f  steer_logits.py + steer_nulls_analysis.py   (optional; steering at layer 21)
#   g  checkpoint_transfer.py                      (optional; probe transfer base <-> post)
#   sweep  patch_positions.py --layers ...  -> outputs/patch_sweep/ (if --layers exists)
#
# Everything this script runs is public: the statement sets come from Marks & Tegmark
# (2024) and the checkpoints from Hugging Face. The authorship corpora of the other
# sections are confidential and are NOT touched here.

set -euo pipefail

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    cat <<'EOF'

Options
  --preset NAME       model family (default olmo2-7b; apertus15-8b needs the swiss-ai
                      transformers fork, see slurm/stages.sh and slurm/stitch.sh)
  --post VARIANT      post-trained checkpoint used for stitching/patching
                      (default: sft if the preset has one, else instruct)
  --variants "..."    checkpoints scored in step a (default: base, --post, instruct)
  --limit N           N statement pairs per run instead of 748 (a, b) / 300 (c, d);
                      smoke test only, the results are NOT the paper's numbers
  --steps a,b,c,d,e   subset of the steps, comma-separated (default a,b,c,d,e)
  --sweep             also run the layer x position sweep (needs patch_positions.py --layers)
  --splits "2 4 ..."  stitching splits (default: 2 4 6 8 12 16 24, as in the paper)
  --layers "8 16"     patching layers (default: 8 16, as in the paper)
  --sweep-layers "..."  sweep layers (default: 1 2 4 6 ... 32, as in slurm/patch_sweep_submit.sh)
  --with-counterfact  also score the counterfact set (the "diverse facts" rows of Table 2)
  --n-boot N          bootstrap resamples in steps b and e (default 1000, as in the paper)
  --device DEV        auto | cuda | mps | cpu (default auto)
  --fetch-data        download the four public statement CSVs into data/external/
  --dry-run           print the commands without running them
  -h, --help          this text

Output root
  Everything is written below outputs/ of the repository. The scripts of steps b, c, e, f
  and g take their output path from config.OUTPUT_DIR (= <repo>/outputs) and have no
  option for a different root, so OUT is honoured by running them in a throw-away copy of
  the repository whose outputs/ is a symlink to $OUT. No file of the repository is
  modified by this; only the copy is. Set OUT for a trial run, a smoke test, or whenever
  the existing outputs/ must stay untouched (steps b and e rewrite aggregate files such as
  outputs/truth_stages_contrasts.csv and outputs/patch/patch_summary.csv with the subset
  of conditions that are present).
EOF
}

PRESET=olmo2-7b
POST=""
VARIANTS=""
LIMIT=""
STEPS=a,b,c,d,e
SWEEP=0
SPLITS="2 4 6 8 12 16 24"
LAYERS="8 16"
SWEEP_LAYERS="1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
COUNTERFACT=0
NBOOT=1000
DEVICE=""
FETCH=0
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET=$2; shift 2 ;;
        --post) POST=$2; shift 2 ;;
        --variants) VARIANTS=$2; shift 2 ;;
        --limit) LIMIT=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        --sweep) SWEEP=1; shift ;;
        --splits) SPLITS=$2; shift 2 ;;
        --layers) LAYERS=$2; shift 2 ;;
        --sweep-layers) SWEEP_LAYERS=$2; shift 2 ;;
        --with-counterfact) COUNTERFACT=1; shift ;;
        --n-boot) NBOOT=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --fetch-data) FETCH=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
# the steps run in $WORK, so a relative interpreter path has to be resolved first
[[ $PY != /* && -x $PWD/$PY ]] && PY=$PWD/$PY
OUT=${OUT:-"$REPO/outputs"}
has_step() { [[ ",$STEPS," == *",$1,"* ]]; }

# --------------------------------------------------------------------------- work dir
# The scripts resolve their output root from the location of config.py. Running them from
# a copy of the repository whose outputs/ points at $OUT therefore redirects every write,
# without changing a single line of the analysis code.
if [[ "$(cd "$OUT" 2>/dev/null && pwd || echo "$OUT")" == "$REPO/outputs" ]]; then
    WORK=$REPO
    OUT=$REPO/outputs
else
    mkdir -p "$OUT"
    OUT=$(cd "$OUT" && pwd)
    WORK=${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/negation_repo.XXXXXX")}
    mkdir -p "$WORK/truth" "$WORK/data"
    cp "$REPO"/*.py "$WORK/"
    cp "$REPO"/truth/*.py "$WORK/truth/"
    [[ -e "$WORK/outputs" ]] || ln -s "$OUT" "$WORK/outputs"
    [[ -e "$WORK/data/external" ]] || ln -s "$REPO/data/external" "$WORK/data/external"
    echo "# work dir (copy of the repository): $WORK"
fi
echo "# repository: $REPO"
echo "# outputs:    $OUT"

# ------------------------------------------------------------------------- model names
variant_ids() { "$PY" - "$PRESET" <<'EOF'
import sys
sys.path.insert(0, ".")
import config
print(" ".join(sorted(config.MODEL_PRESETS[sys.argv[1]])))
EOF
}
cd "$WORK"
AVAILABLE=$(variant_ids)
if [[ -z $POST ]]; then
    POST=sft
    [[ " $AVAILABLE " == *" sft "* ]] || POST=instruct
fi
[[ " $AVAILABLE " == *" $POST "* ]] || { echo "preset $PRESET has no variant '$POST' (has: $AVAILABLE)" >&2; exit 2; }
if [[ -z $VARIANTS ]]; then
    VARIANTS="base $POST"
    [[ " $AVAILABLE " == *" instruct "* && $POST != instruct ]] && VARIANTS="$VARIANTS instruct"
fi
if [[ $PRESET == apertus* ]]; then
    cat >&2 <<'EOF'
# NOTE: Apertus 1.5 8B needs the developers' fork of transformers (swiss-ai), which the
#       paper's runs reached with PYTHONPATH=/work/$USER/tf_apertus (slurm/stages.sh,
#       slurm/stitch.sh, slurm/patch.sh). With the released transformers the checkpoints
#       do not load. Everything else below is identical to OLMo 2.
EOF
fi

DEV_ARG=()
[[ -n $DEVICE ]] && DEV_ARG=(--device "$DEVICE")
LIM_ARG=()
[[ -n $LIMIT ]] && LIM_ARG=(--limit "$LIMIT")
# the paper ran stitching and patching on the first 300 pairs
PAIRS=${LIMIT:-300}

run() {
    echo "+ $*"
    [[ $DRY -eq 1 ]] || "$@"
}

# --------------------------------------------------------------- 0. public statement sets
SETS=(cities neg_cities)
[[ $COUNTERFACT -eq 1 ]] && SETS+=(counterfact)
GOT_CSV="$REPO/data/external/geometry_of_truth"
if [[ $FETCH -eq 1 ]]; then
    mkdir -p "$GOT_CSV"
    for f in cities neg_cities counterfact_true_false common_claim_true_false; do
        [[ -f "$GOT_CSV/$f.csv" ]] && continue
        run curl -fsSL -o "$GOT_CSV/$f.csv" \
            "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/$f.csv"
    done
fi
for ds in "${SETS[@]}"; do
    csv=$GOT_CSV/$ds.csv
    [[ $ds == counterfact ]] && csv=$GOT_CSV/counterfact_true_false.csv
    if [[ ! -f $csv && $DRY -eq 1 ]]; then
        echo "# (dry run) $csv is not present; --fetch-data would download it"
    elif [[ ! -f $csv ]]; then
        echo "missing: $csv" >&2
        echo "The statement sets are public but are not part of the repository." >&2
        echo "Run with --fetch-data, or copy datasets/*.csv of" >&2
        echo "https://github.com/saprmarks/geometry-of-truth into data/external/geometry_of_truth/." >&2
        exit 1
    fi
done

# ------------------------------------------------------- a. answers + activations (GPU)
if has_step a; then
    echo "## step a: factuality.py ($PRESET: $VARIANTS)"
    for ds in "${SETS[@]}"; do
        dir=outputs/truth_$ds
        run "$PY" factuality.py --statements "truth/${ds}_statements.py" --out-dir "$dir" \
            --presets "$PRESET" --variants $VARIANTS --no-analyze ${LIM_ARG[@]+"${LIM_ARG[@]}"} ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        # factuality.py puts a --limit run into <out-dir>/smoke/; move it up so that the
        # downstream scripts, which read outputs/truth_<set>/, find it.
        if [[ -n $LIMIT && -d $WORK/$dir/smoke && $DRY -eq 0 ]]; then
            mv "$WORK/$dir"/smoke/* "$WORK/$dir"/ && rmdir "$WORK/$dir/smoke"
        fi
    done
fi

# ------------------------------- b. answer d', probe d', transfer probe (CPU, needs a)
if has_step b; then
    echo "## step b: truth_stages.py"
    run "$PY" -W ignore truth_stages.py --n-boot "$NBOOT" --presets "$PRESET" \
        --datasets "${SETS[@]}"
fi

# ----------------------------------------------------------- c. model stitching (GPU)
if has_step c; then
    echo "## step c: stitch.py (truth:neg_cities, $PAIRS pairs)"
    for s in $SPLITS; do
        run "$PY" stitch.py --corpus truth:neg_cities --preset "$PRESET" \
            --lower base --upper "$POST" --split "$s" --limit "$PAIRS" --n-ppl 40 ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        run "$PY" stitch.py --corpus truth:neg_cities --preset "$PRESET" \
            --lower "$POST" --upper base --split "$s" --limit "$PAIRS" --n-ppl 40 ${DEV_ARG[@]+"${DEV_ARG[@]}"}
    done
fi

# ------------------------------------------------------- d. activation patching (GPU)
if has_step d; then
    echo "## step d: patch_positions.py (layers $LAYERS, $PAIRS pairs)"
    for l in $LAYERS; do
        run "$PY" patch_positions.py --corpus truth:neg_cities --preset "$PRESET" \
            --recipient "$POST" --donor base --layer "$l" --limit "$PAIRS" ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        run "$PY" patch_positions.py --corpus truth:neg_cities --preset "$PRESET" \
            --recipient base --donor "$POST" --layer "$l" --limit "$PAIRS" ${DEV_ARG[@]+"${DEV_ARG[@]}"}
    done
fi

# --------------------------------------------------- e. summary tables (CPU, needs c+d)
if has_step e; then
    echo "## step e: localization_summary.py"
    run "$PY" -W ignore localization_summary.py --n-boot "$NBOOT"
fi

# ------------------------------------------------- sweep. layer x position (GPU, slow)
if [[ $SWEEP -eq 1 ]]; then
    if "$PY" patch_positions.py --help 2>/dev/null | grep -q -- "--layers"; then
        echo "## sweep: patch_positions.py --layers $SWEEP_LAYERS -> outputs/patch_sweep/"
        run "$PY" patch_positions.py --corpus truth:neg_cities --preset "$PRESET" \
            --recipient "$POST" --donor base --layers $SWEEP_LAYERS --limit "$PAIRS" \
            --out-root patch_sweep --skip-existing ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        run "$PY" patch_positions.py --corpus truth:neg_cities --preset "$PRESET" \
            --recipient base --donor "$POST" --layers $SWEEP_LAYERS --limit "$PAIRS" \
            --out-root patch_sweep --skip-existing ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        if [[ -f $WORK/patch_sweep_summary.py ]]; then
            run "$PY" -W ignore patch_sweep_summary.py --n-boot "$NBOOT"
        else
            echo "# patch_sweep_summary.py not present — sweep CSVs written, not summarized"
        fi
    else
        echo "# patch_positions.py has no --layers option in this checkout — sweep skipped" >&2
    fi
fi

# ------------------------------------------------------------- f. steering (GPU, slow)
# Arguments as in slurm/steer_nulls.sh and slurm/steer_nulls_own_submit.sh: 50 pcrandom +
# 50 split control directions, each drawn once for all texts, at +-2 SD (--fixed-controls),
# first norm-matched (suffix _nulls), then scale-matched (--control-unit own, _ownsd).
# Needs the FULL step a (the directions are fitted on all 748 pairs); --limit 300 selects
# the first 300 statements that are scored, as in the paper.
if has_step f; then
    echo "## step f: steer_logits.py (layer 21) + steer_nulls_analysis.py"
    [[ -n $LIMIT ]] && echo "# WARNING: step f after a --limit run does not give the paper's numbers" >&2
    BS=()
    [[ $PRESET == olmo2-7b ]] && BS=(--batch-size 16)
    for v in base "$POST"; do
        run "$PY" -W ignore steer_logits.py outputs/truth_neg_cities --preset "$PRESET" \
            --variant "$v" --fmt raw --layer 21 --add -6 -4 -2 -1 0 1 2 4 6 --scale --unit sd \
            --controls decision pcrandom split --n-random 50 --fixed-controls \
            --control-add -2 2 --batched --suffix _nulls --limit 300 ${BS[@]+"${BS[@]}"} ${DEV_ARG[@]+"${DEV_ARG[@]}"}
        run "$PY" -W ignore steer_logits.py outputs/truth_neg_cities --preset "$PRESET" \
            --variant "$v" --fmt raw --layer 21 --add -6 -4 -2 -1 0 1 2 4 6 --scale --unit sd \
            --n-random 50 --fixed-controls --control-add -2 2 --batched --limit 300 \
            --only-controls --controls pcrandom split --control-unit own --suffix _ownsd \
            ${BS[@]+"${BS[@]}"} ${DEV_ARG[@]+"${DEV_ARG[@]}"}
    done
    # aggregates every steering cell it finds in outputs/ -> outputs/steer_nulls_summary.csv
    run "$PY" -W ignore steer_nulls_analysis.py
fi

# ------------------------------------------- g. probe transfer between checkpoints (CPU)
# The paper's file outputs/checkpoint_transfer_truth.csv was produced for all three
# further-judgment sets at once; restricted to the negated statements here.
if has_step g; then
    echo "## step g: checkpoint_transfer.py (truth_neg_cities)"
    run "$PY" -W ignore checkpoint_transfer.py --sets truth_neg_cities \
        --presets "$PRESET" --out checkpoint_transfer_truth_neg_cities.csv
fi

echo "## done. Compare with the paper's values:"
echo "   $PY reproduce/check_negation.py --outputs $OUT"
