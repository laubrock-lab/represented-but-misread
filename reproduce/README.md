# Reproducing Section 3.1 (negated statements)

One entry point, `reproduce/negation.sh`, regenerates the negated-statements analysis of
Section 3.1 from public data: the statement sets of Marks and Tegmark (2024) and the
open-weight checkpoints listed in Methods Table 1. Nothing in this pipeline touches the
three authorship corpora, which are confidential (see "What cannot be reproduced
publicly").

```bash
git clone <repo> && cd latent-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login                                   # Gemma 3 is gated; OLMo and Qwen are not

bash reproduce/negation.sh --fetch-data         # OLMo 2 7B, steps a-e, paper settings
python reproduce/check_negation.py              # compare with the paper's values
```

A smoke test on a few statement pairs, writing to a scratch directory:

```bash
OUT=/tmp/neg_smoke bash reproduce/negation.sh --limit 12 --splits "8 16" --layers 16
```

`bash reproduce/negation.sh --help` lists all options.

## The steps

| Step | Script | What it does | Writes |
|---|---|---|---|
| a | `factuality.py` | scores every checkpoint on the statement sets (both answer orders, raw and chat prompt) and stores the residual stream at the last prompt position | `outputs/truth_cities/`, `outputs/truth_neg_cities/` (`<preset>_<variant>_raw_{tf,ft}.csv`, `*_raw_tf_acts.npz`) |
| b | `truth_stages.py` | answer d′, probe d′ (last position, 2/3 depth, 10-fold grouped CV), transfer probe (trained on `cities`, applied to `neg_cities`), stage contrasts | `outputs/truth_neg_cities/stages.csv`, `item_scores.csv`, `outputs/truth_stages_contrasts.csv` |
| c | `stitch.py` | model stitching base/SFT at splits 2, 4, 6, 8, 12, 16, 24 in both directions, 300 pairs, perplexity check on 40 statements | `outputs/stitch/truth_neg_cities/*.csv`, `perplexity.csv` |
| d | `patch_positions.py` | activation patching at layers 8 and 16, both directions, all position sets, 300 pairs | `outputs/patch/truth_neg_cities/<preset>_<recipient>-from-<donor>_L<k>_n300.csv` |
| e | `localization_summary.py` | d′ with grouped-bootstrap CIs per patching run and per stitch | `outputs/patch/patch_summary.csv`, `outputs/stitch/stitch_neg_cities.csv` |
| sweep | `patch_positions.py --layers`, `patch_sweep_summary.py` | layer × position sweep (`--sweep`), layers 1, 2, 4, …, 32 | `outputs/patch_sweep/truth_neg_cities/`, its summary |
| f | `steer_logits.py`, `steer_nulls_analysis.py` | steering at layer 21 with 100 fixed control directions, norm-matched (`_nulls`) and scale-matched (`_ownsd`) | `outputs/truth_neg_cities/steer_*`, `outputs/steer_nulls_summary.csv` |
| g | `checkpoint_transfer.py` | layer-wise probe transfer base ↔ post-trained | `outputs/checkpoint_transfer_truth_neg_cities.csv` |

Steps a–e are the default (`--steps a,b,c,d,e`). The arguments are those of the paper's
runs, recorded in `slurm/factuality_stages.sh`, `slurm/truth_analyze.sh`,
`slurm/truth_localize_submit.sh` (which submits `slurm/stitch.sh` and `slurm/patch.sh`),
`slurm/patch_sweep_submit.sh`, `slurm/steer_nulls.sh` with
`slurm/steer_nulls_own_submit.sh`, and `slurm/checkpoint_transfer.sh`. The SLURM scripts
remain the authority; `negation.sh` only runs the same Python commands without a scheduler.

Steps f and g are optional and off by default:

* **f (steering)** needs the full step a: the steering direction is fitted on all 748
  pairs (`--limit 300` only selects the statements that are then scored, as in the paper).
  `steer_nulls_analysis.py` aggregates every steering cell it finds under `outputs/`, so
  in a fresh output tree it writes a `steer_nulls_summary.csv` that contains the
  negated-statement cells only — the other rows of Appendix Table D1 come from the
  authorship and review cells, which are not part of this pipeline.
* **g (probe transfer)** is not quoted in Section 3.1 (it is used in Section 3.6). The
  paper's `outputs/checkpoint_transfer_truth.csv` was produced for all three
  further-judgment sets in one call; the step here restricts it to the negated statements
  and writes a differently named file, so that the paper's file is not overwritten with a
  subset.

## Which number comes from where

Every value below is stored in `reproduce/reference_negation.csv` with the exact filter
(`file`, `filter`, `column`) and the number as printed in the manuscript; `id` is the key
in that file and in the output of `check_negation.py`.

| Manuscript | Step | File | Column | Reference ids |
|---|---|---|---|---|
| Table 2, negated statements, "Probe base"/"Probe final" (6 models) | b | `truth_neg_cities/stages.csv` | `probe_d` | `T2_probe_*` |
| Table 2, negated statements, "Answer base"/"SFT"/"Final" (6 models) | b | `truth_neg_cities/stages.csv` | `ans_d` | `T2_ans_*` |
| 3.1 "criterion c = 2.6 and 3.0" | b | `truth_neg_cities/stages.csv` | `ans_c` | `c_olmo2_base`, `c_apertus_base` |
| 3.1 transfer probe (−1.28, −1.43, +0.69, +3.05) | b | `truth_neg_cities/stages.csv` | `transfer_d` | `tr_*` |
| Table 2 footnote a (18%, 25%) | b | `truth_neg_cities/stages.csv` | `p_mass` | `pmass_*` |
| 3.1 stitching (OLMo 2: 3.76, 0.38; parents 0.63, 3.20; Apertus: 4.15, 1.33, −0.94, 1.21, 4.15) | e | `stitch/stitch_neg_cities.csv` | `d`, `d_base`, `d_sft` | `st_*` |
| 3.1 patching at layer 16 (0.63 → 3.26, 1.51; 3.20 → 1.07; −0.87 → 4.15; 4.15 → −1.09; Apertus `last` 4.15) | e | `patch/patch_summary.csv` | `d` | `pa_*` |
| 3.1 / Table D1 steering (+0.36, +1.07, −0.14, +0.09 and their p-values) | f | `steer_nulls_summary.csv` | `class_E2`, `all_p`, `own_all_p` | `se_*` |
| Table 2, diverse facts (counterfact) rows | a + b with `--with-counterfact` | `truth_counterfact/stages.csv` | `probe_d`, `ans_d` | not in the reference file |
| 3.1 "answers 1.61 → 1.34; probes 1.89 → 1.64" (OLMo 3, diverse facts) | a + b with `--with-counterfact` | `truth_counterfact/stages.csv` | `ans_d`, `probe_d` | not in the reference file |
| 3.1 "TODO(sweep): layer-by-position patching for OLMo 2" | `--sweep` | `outputs/patch_sweep/` | — | open in the manuscript |

Two statements of Section 3.1 are **not** backed by a value in the current outputs:

* "in these models the truth direction coincided with the decision direction (cosine ≈
  1.0)". `steer_nulls_analysis.py` gained the columns `cos_class_decision` /
  `cos_class_decision_scored` on 17 Nov 2026, but the `outputs/steer_nulls_summary.csv`
  on disk predates that and has no such column. Rerunning step f regenerates it; until
  then the cosine exists only in the steering job logs.
* "The maximum d′ (AUC = 1) is 4.54 for the 748 statement pairs" is a property of the
  estimator (√2 · z of the clipped AUC), not a measured value; the corresponding ceiling
  for the 300-pair stitching and patching runs is 4.15.

The remaining prose of 3.1 paraphrases the values above (e.g. "repaired the readout",
"abolished the repair"), and the directional statements in Table D1 (`class_E2`) and the
share of the donor-recipient difference that a patch transfers (`share_transferred` in
`patch_summary.csv`) are produced by the same steps.

## Checking a rerun

```bash
python reproduce/check_negation.py --outputs outputs                # all 63 values
python reproduce/check_negation.py --steps b --only olmo2-7b        # one step, one family
python reproduce/check_negation.py --outputs /tmp/neg_smoke --allow-missing
```

For every reference row the script prints the stored value, the regenerated value, their
difference, the tolerance (0.05 for d′ and nats, 0.02 for p-values and answer mass), and
a `ms` column that states whether the stored value rounds to the number printed in the
manuscript. It exits non-zero if anything fails; cells that are absent count as failures
unless `--allow-missing` is given. Against the outputs that the manuscript was written
from, all 63 rows pass and all 63 match the printed numbers.

A smoke test (`--limit`) produces different numbers by construction — it checks that the
pipeline runs, not that it reproduces the paper. The 12-pair Gemma 3 4B run above gives
`FAIL 5, MISSING 58`: the five Gemma 3 4B cells of Table 2 exist but differ (with 12
pairs the d′ ceiling is 2.45 instead of 4.54), and everything else belongs to models the
run did not touch.

## Output root, and why `$OUT` works the way it does

Only `factuality.py` (`--out-dir`) and `patch_positions.py` (`--out-root`, relative to
`outputs/`) can be told where to write. `truth_stages.py`, `stitch.py`,
`localization_summary.py`, `steer_logits.py`, `steer_nulls_analysis.py` and
`checkpoint_transfer.py` derive their output path from `config.OUTPUT_DIR`, which is
`<repo>/outputs` and has no option or environment override. Rather than change those
scripts, `negation.sh` honours `OUT` by copying the repository's Python files to a
throw-away directory whose `outputs/` is a symlink to `$OUT` and running the steps there
(`WORKDIR=` chooses the directory, otherwise `mktemp -d`). The analysis code is used
unmodified; only the copy is written to. The copy links in `data/external` (the public
statement CSVs) and nothing else of `data/`, so a run with `OUT` set cannot reach the
confidential corpora even by accident.

Use `OUT` whenever the existing `outputs/` must stay as it is. Three traps in a run
against the repository's own `outputs/`:

* `truth_stages.py` rewrites `outputs/truth_stages_contrasts.csv` from scratch with only
  the datasets and presets of the current call, and `item_scores.csv` per set likewise.
* `localization_summary.py` rebuilds `outputs/patch/patch_summary.csv` and
  `outputs/stitch/stitch_neg_cities.csv` from whatever patching and stitching files are
  present, for all corpora.
* `stitch.py` does not put the `--limit` into its file names (unlike `patch_positions.py`,
  which writes `…_n300.csv`), so a run with a different `--limit` silently replaces the
  paper's stitching files, and `perplexity.csv` is appended to.

`factuality.py --limit N` writes into `<out-dir>/smoke/`; `negation.sh` moves those files
one level up afterwards, because the downstream scripts read `outputs/truth_<set>/`.

## Hardware and time

**Measured** (Apple silicon, MPS, 64 GB, bfloat16, Gemma 3 4B base/instruct, 12 statement
pairs per set = 24 statements, splits 8 and 16, patching at layer 16 only; 17 Nov 2026):

```bash
OUT=/tmp/neg_smoke bash reproduce/negation.sh \
    --preset gemma3-4b --limit 12 --splits "8 16" --layers 16 --device mps
```

| Step | Work done | Measured wall clock |
|---|---|---|
| a | 2 sets × (base raw, instruct raw + chat) × 2 answer orders | 2.5 min (10 conditions, 8–12 s per condition of 24 statements ≈ 0.4 s per statement) |
| b | probes, transfer probe, 1,000 bootstrap resamples, 2 sets | 10 s |
| c | 4 stitched models (2 splits × 2 directions), incl. perplexity | 2.5 min (≈ 38 s per stitch, half of it loading two checkpoints) |
| d | 1 layer × 2 directions × 7 position sets | 8.5 min (318 s and 185 s for the recipient pass, 16 s for the donor cache) |
| e | summary with 1,000 bootstrap resamples | 20 s |
| total | | **14.5 min** |

The rate that carries over to other configurations: one scored statement (one prompt, one
answer order, five forward passes) takes ≈ 0.4 s for a 4 B model on MPS, and a patched
one ≈ 0.6–0.9 s.

**Estimated** for the paper's configuration (OLMo 2 7B, 748 pairs for steps a and b, 300
pairs for c–e). These are estimates, not measurements: the paper's runs were submitted to
A100 40 GB nodes and no timing logs are kept in the repository. The SLURM wall-clock
limits that the jobs were submitted with are an upper bound: 4 h per `factuality.py` job
(`slurm/factuality_stages.sh`), 2 h per stitch (`slurm/stitch.sh`), 4 h per patching run
(`slurm/patch.sh`), 3 h for the CPU analysis (`slurm/truth_analyze.sh`), 8 h per steering
cell (`slurm/steer_nulls.sh`), 8–12 h per sweep job (`slurm/patch_sweep_submit.sh`), 12 h
for the probe transfer (`slurm/checkpoint_transfer.sh`).

| Step | Jobs | Order of magnitude on one A100 40 GB (estimate) |
|---|---|---|
| a | 2 sets × 3 checkpoints | ~1–3 h each (1,496 statements × 2 orders × ~5 forward passes per statement; the post-trained checkpoints are scored in the raw and the chat format, i.e. twice) |
| b | 1 CPU job | ~0.5–2 h, dominated by the 1,000 bootstrap resamples and the probe cross-validation; needs ~10 GB RAM for the activation files (~800 MB each) |
| c | 14 stitched models | ~0.5–1 h each (two checkpoints loaded per stitch, 600 statements × 2 orders) |
| d | 4 runs (2 layers × 2 directions) | ~1–2 h each (donor pass plus 6 position sets × 600 statements × 2 orders) |
| e | 1 CPU job | minutes |
| sweep | 17 layers × 2 directions | the paper's submission splits this into 6 jobs of 8 h |
| f | 4 cells × 2 control families | several hours per cell; `--batched` is what makes 100 control directions feasible |

On MPS everything runs, but a 7–8 B model in bfloat16 needs roughly 16 GB for one
checkpoint and twice that for stitching and patching, which load two checkpoints at once.
Scaled from the measured rate (0.4 s per scored statement for a 4 B model, and a 7 B model
is roughly twice as slow), step a alone would be some 6–8 h per statement set on this
laptop, step c about 7 h and step d about 10 h. A local MPS run of the full paper
configuration is therefore possible in principle but takes days; a GPU is the sensible
choice.

Disk: step a writes one activation file per checkpoint and set (`float32`, items × layers
× hidden size; ~800 MB for a 7 B model and 1,496 statements), so steps a–b for one family
and two sets need ~5 GB.

## Models

Hugging Face identifiers, as in Methods Table 1 and `config.MODEL_PRESETS`:

| Preset | base | post-training checkpoints |
|---|---|---|
| `olmo2-7b` | `allenai/OLMo-2-1124-7B` | `-SFT`, `-DPO`, `-Instruct` |
| `apertus15-8b` | `swiss-ai/Apertus-v1.5-8B-base` | `-SFT`, `-RL`, `-DPO`, `swiss-ai/Apertus-v1.5-8B` |
| `olmo3-7b-think` | `allenai/Olmo-3-1025-7B` | `allenai/Olmo-3-7B-Think-SFT`, `-Think-DPO`, `-Think` |
| `gemma3-4b` | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` |
| `gemma3-12b` | `google/gemma-3-12b-pt` | `google/gemma-3-12b-it` |
| `qwen3-8b` | `Qwen/Qwen3-8B-Base` | `Qwen/Qwen3-8B` |

`negation.sh --preset olmo2-7b` (the default) scores `base`, `sft` and `instruct` in step
a — the three columns of Table 2 — and uses `base` and `sft` for stitching and patching,
as in `slurm/truth_localize_submit.sh`. `--variants "base sft dpo instruct"` adds the
intermediate DPO stage, which fills the remaining rows of
`outputs/truth_neg_cities/stages.csv` and `outputs/truth_stages_contrasts.csv` (the
stage-by-stage comparisons; Table 2 reports only base, SFT and the final model);
`--post` chooses a different post-trained checkpoint for steps c and d. The Gemma 3 weights are gated: accept the
licence on the model page and run `hf auth login` once.

**Apertus 1.5 8B** (`--preset apertus15-8b`) does not load with the released
`transformers`; the paper's runs used the developers' fork, which the SLURM scripts put on
`PYTHONPATH` (`PYTHONPATH=/work/$USER/tf_apertus` in `slurm/factuality_stages.sh`,
`slurm/stitch.sh`, `slurm/patch.sh`, `slurm/steer_nulls.sh`). `negation.sh` prints a
reminder and otherwise runs the identical commands. Everything in Table 2 and Section 3.1
for Apertus therefore requires that fork; the OLMo 2, OLMo 3, Qwen3 and Gemma 3 rows do
not.

Precision is bfloat16 throughout; Methods 2.10 records `transformers` 5.22.2 (5.18 fork
for Apertus) and PyTorch 2.6.

## The statement sets

`truth/cities_statements.py`, `truth/neg_cities_statements.py` and
`truth/counterfact_statements.py` read three CSVs that the repository does not contain
(`data/` is git-ignored):

| Set | File | Source, as recorded in the module header |
|---|---|---|
| cities | `data/external/geometry_of_truth/cities.csv` | Marks & Tegmark (2024), *The Geometry of Truth*, https://github.com/saprmarks/geometry-of-truth, `datasets/cities.csv`; 748 matched pairs "The city of X is in Y." |
| neg_cities | `data/external/geometry_of_truth/neg_cities.csv` | same repository, `datasets/neg_cities.csv`; the negated statements "The city of X is not in Y.", 748 matched pairs |
| counterfact | `data/external/geometry_of_truth/counterfact_true_false.csv` | same repository, `datasets/counterfact_true_false.csv`, derived from CounterFact (Meng et al.); 748 pairs drawn with a fixed seed (`random_state=1`) |

`negation.sh --fetch-data` downloads them from the raw URLs of that repository into
`data/external/geometry_of_truth/`; the pairing, the fixed seed and the item ids are
produced by the `truth/*.py` modules, so the sets are fully determined by the CSVs. The
copies used for the paper are byte-identical to the files that
`raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/` served on
17 Nov 2026 (SHA-256):

```
f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e  cities.csv
66c5c84de98ca15570138c205ab0e3907b57750083fac72a4eacafc3d9ebcea7  neg_cities.csv
5af51456d6dbd3a53d8e2ee6e4aadaca7eb02ff9136ad825866458e7283b5273  counterfact_true_false.csv
```

**License:** TODO: provenance to be confirmed by the author. Neither the module headers
nor this repository state under which license the Geometry of Truth CSVs (and the
CounterFact data they derive from) may be redistributed; only the source URL and the
citation are recorded, and the upstream GitHub repository carried no LICENSE file when
this was checked (17 Nov 2026). The manuscript cites Marks and Tegmark (2024) and
describes the data as publicly available (Methods 2.8, 2.11). `--fetch-data` therefore
downloads the CSVs from the original repository instead of this one redistributing them.

## What cannot be reproduced publicly

The three authorship corpora of Sections 3.2–3.6 cannot be shared, as Methods 2.11
states: the newspaper texts are unpublished and confidential, the human chat messages
come from participants who have not consented to publication, and the human textbook
summaries are protected by copyright. They live in `data/` (git-ignored) and are
processed only on university machines. Nothing in this pipeline reads them: steps a–g use
`truth/*_statements.py` and the public CSVs above, and the corpus argument is always
`truth:neg_cities` or `truth:cities`.

The consequence for Section 3.1 is none — it is the section the manuscript describes as
reproducible in full. What a public rerun cannot reproduce are the authorship results of
Sections 3.2–3.6 (and the lens-model analyses of Appendix B that use those texts). The
per-text model outputs without text content and all derived results are to be released
with the repository (Methods 2.11), so the authorship analyses can be re-derived from the
stored judgments, but not re-run from the texts.

## Things that will bite

* **Nothing in `outputs/` is in version control.** A fresh clone starts empty, and step e
  can only summarize what steps c and d have produced in the same output root.
* `factuality.py` skips a condition whose CSV already exists, which makes restarts cheap
  but means that a changed prompt or a changed statement set needs the old files removed.
* `localization_summary.py`'s stitch table looks up the parent models under the names
  `base` and `sft` (`d_base`, `d_sft`); with a preset whose post-trained variant is called
  `instruct` (Gemma 3, Qwen3), those two columns stay empty while the stitched d′ is
  computed normally.
* `patch_positions.py` has gained a seventh position set since the paper's runs: the
  files in `outputs/patch/truth_neg_cities/` contain `none, text, pre, post, last, all`,
  a rerun with today's default adds `question` (the question line only, wherever it
  stands). The values Section 3.1 quotes come from `post` and are unaffected, but the
  regenerated CSVs have more rows than the stored ones. On the truth prompts that set is
  empty — the position finder looks for the German "Frage:" of the authorship prompt — so
  it reproduces `none` exactly and carries no information here. Add
  `--sets none text pre post last all` to the two `patch_positions.py` calls in
  `negation.sh` to get files with the paper's layout.
* `steer_nulls_analysis.py` takes no arguments and always writes
  `outputs/steer_nulls_summary.csv` for all cells it finds.
* The `--sweep` option is guarded: it checks `patch_positions.py --help` for `--layers`
  and skips the sweep if this checkout does not have it yet.
* `slurm/steer_nulls_own_submit.sh` refers to `slurm/steer_nulls_v100.sh`, which is not in
  the repository; the cells it submits with that script (among them OLMo 2 on the negated
  statements) were run on a different node type. `negation.sh` runs them with the
  arguments of `slurm/steer_nulls.sh` plus `--batch-size 16`, which is what the submission
  script passes.
