# Represented but misread

## About this repository

Code, derived results and manuscript sources for the paper *Represented but misread:
Post-training changes how language models read out what they represent*
(arXiv:2611.26584, https://arxiv.org/abs/2611.26584).

**The three authorship corpora are not part of this repository and cannot be shared.**
The newspaper texts are unpublished and confidential, the human chat messages were written
by students who have not yet been asked for consent to publication, and the human textbook
summaries are protected by copyright. `data/` and `outputs/` have never been part of any
revision of this repository; `results/` holds the aggregate tables from which the numbers
in the paper are taken.

**What can be reproduced here.** The truth judgments use public data. `reproduce/README.md`
describes the entry point: `reproduce/negation.sh` regenerates the analysis of the negated
statements from the public statement sets and the open-weight checkpoints, and
`reproduce/check_negation.py` compares the regenerated numbers with the values reported in
the paper.

**License.** The MIT license (`LICENSE`) applies to the software in this repository. The
manuscript text and figures in `paper/` are the author's and are not covered by it. No
third-party data are included; the public statement sets are downloaded from their source.

The history of the public repository was filtered to remove unrelated internal work,
internal planning documents and private paths; commit dates, order, authorship and
messages are otherwise unchanged.

## Data policy

The texts of the three authorship corpora are not part of this repository:

- `data/` and `outputs/` are not tracked. Texts, per-text judgments and activations stay on
  the machines on which they were processed.
- The newspaper corpus is confidential and was processed only locally, with open-weight
  models and without calls to external services.
- Scripts do not print text content.

## Code overview

The manuscript sources are in `paper/draft/` (build: `cd paper && bash build_draft.sh`).
Every number in the manuscript is written by one of the scripts below into `outputs/`
(not part of the repository, because the per-text files of the authorship corpora carry
identifiers of unpublished texts). The aggregate tables are copied to `results/` by
`tools/export_results.py`. The public part, the negated statements, can be rerun
with `reproduce/negation.sh`; see `reproduce/README.md`.

| Step | Script | What it does | Manuscript |
|---|---|---|---|
| Answers | `judge_logits.py`, `config.py` | Prompts, log-odds of the two answer words (dv), activations at the last position and averaged over the text positions | 2.3 |
| Pilot (Introduction) | `pilot_llm_judge_summary.py` | Commercial models as judges of the short chats: AUC of the confidence ratings per judge model and prompt version, from judgments collected in the lab course | 1 |
| Signal detection | `sdt_summary.py`, `contrasts.py`, `meta_multilevel.R`, `meta_prompts.R`, `promptvar_summary.py` | d′ from the area under the ROC curve, criterion, grouped bootstrap, contrasts between checkpoints, family-level tests | 2.4, 3.1, 3.2 |
| Base rate | `sdt_invariance.py`, `base_rate_shift.py`, `base_rate_scale.py` | Invariance of d′ and c under bias manipulations; shift of dv with the stated share of AI texts | 3.3, Appendix F |
| Readout probes | `probe_layers.py` (`_make_probe`, `_fit_probe`, `cv_logits`) | The linear probe: standardization, principal component rotation, L2-regularized logistic regression with the penalty chosen by inner grouped cross-validation; out-of-fold scores with whole source groups held out; label-shuffle control | 2.5, 3.1 |
| Probe transfer | `probe_transfer.py` (between corpora), `checkpoint_transfer.py` (between base and post-trained model), `checkpoint_transfer_last_summary.py` | A probe fitted on one corpus or checkpoint and tested on another | 3.1, 3.5 |
| Baselines | `baselines.py`, `editlens_score.py` | n-gram and surface classifiers, input-embedding probe, external detector | 2.5, 3.1 |
| Few-shot | `fewshot_summary.py` | Labeled examples in the prompt | 2.6, 3.2 |
| Lens model | `lens_model.py`, `cue_lens_chats.py`, `segment_features.py`, `lens_opspam.py` | Lens model equation on interpretable text properties. For the authorship corpora, `segment_features.py` and `human_cues.py` import a feature extractor from the lab course repository, which is not public; `lens_opspam.py` (public reviews) computes its cues itself | Appendix B |
| Stitching | `stitch.py`, `stitch_summary.py` | Lower layers of one checkpoint under the upper layers of another | 2.7, 3.4 |
| Patching | `patch_positions.py`, `localization_summary.py`, `patch_sweep_summary.py` | Exchange of internal states between checkpoints at sets of prompt positions, at one layer or across layers | 2.7, 3.4, 3.6 |
| Steering | `steer_logits.py`, `steer_nulls_analysis.py`, `steer_cache_check.py` | Shift of the state at the last position along the class direction, with control directions | Appendix D |
| Similarity | `cka.py` | Linear CKA between checkpoints | 3.4 |
| Truth and review judgments | `factuality.py`, `truth_stages.py`, `truth_variant_answers.py`, `truth/*.py`, `review_convergence.py` | Answers and probes (logistic regression, C = 0.01, last position, two thirds of the depth) for the public statement and review sets | 2.8, 3.6, Appendix C |
| Fine-tuning data | `sft_check.py` | Pattern search in the published fine-tuning mixtures | 3.6 |
| Training study | `training_study/` | Data preparation, LoRA fine-tuning, evaluation, contrasts, formality check, prediction test | 2.9, 3.7 |
| Figures | `plots/` | All figures from tidy tables without text content | |
| Cluster jobs | `slurm/`, `local/` | Job scripts that record the arguments of every run | |

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login          # once; accept the Gemma / Llama licences on huggingface.co
```

## Usage

The public part of the study can be run as described in `reproduce/README.md`. The commands
below show the general pattern for the authorship corpora, which requires the texts
(`prepare_texts.py` expects the materials of the lab course in the folder given by
`LABCOURSE_DIR`; these are not public).

```bash
# answers (log-odds of the two answer words) and activations for one model family
python judge_logits.py --texts data/texts_chat_tw25.parquet --preset olmo2-7b

# signal detection summary with grouped-bootstrap confidence intervals
python sdt_summary.py outputs/chat_tw25

# layer-wise probes, cross-corpus and cross-checkpoint probe transfer
python probe_layers.py outputs/chat_tw25
python probe_transfer.py
python checkpoint_transfer.py

# stated base rate, both answer orders
python judge_logits.py --texts data/texts_textbook.parquet --preset olmo2-7b --priors 20 50 80
python base_rate_shift.py

# model stitching and activation patching between two checkpoints
python stitch.py --corpus textbook --preset olmo2-7b --lower base --upper sft --split 8
python patch_positions.py --corpus textbook --preset olmo2-7b --recipient base --donor sft --layers 8 12 16
python patch_sweep_summary.py
```

Model presets (base model and post-training checkpoints per family) are defined in
`config.py`. The job scripts in `slurm/` record the arguments of every run reported in the
paper.

## Notes on the outputs

### Word probabilities

The probability of the answer *word* includes the end-of-word correction of
Pimentel & Meister (2024): P(tokens of "Mensch") × P(the next token starts a new
word), so continuations such as "Menschen" or "KI-generiert" don't count. The
beginning-of-word term is the same for both answers and cancels in `dv`. The
uncorrected value is kept as `dv_prefix`.

### Reading the output

- `p_mass` = P(Mensch) + P(KI) for the next token. Low values (base models often
  predict a newline first) don't bias `dv`, which is the log-odds *conditional*
  on answering at all. Values near 0 mean the prompt isn't working for that model.
- Small models show strong **answer-order bias**: they pick whichever option is
  mentioned first or last. Use the `avg` rows, which average `dv` over both orders.
- `c0 > 0` means a conservative bias toward "Mensch", the sign convention of the
  lab course analyses.
- Probes: `probe_layers.csv` has one row per condition × site × layer. Read the
  whole layer curve, not just the best layer: picking the maximum over ~35–50
  layers inflates it. Sanity check: at layer 0 the `last` site is the same token
  for every text, so its d′ must be ≈ 0. `auc_shuffled` (labels permuted within
  source groups) should be ≈ 0.5.
- `probe_oof.csv` holds the out-of-fold probe score per text, which allows item-level
  comparison with human detection rates.
- joblib may print `ResourceTracker … No child processes` at exit. This is harmless.
