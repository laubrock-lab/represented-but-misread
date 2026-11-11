# Figure captions (draft, 11 Nov 2026)

Figures: `paper/figures/` (PDF for the manuscript, PNG for review). Code:
`plots/fig_data.py` (tidy data, no text content) → `plots/figures.R`,
`plots/fig_schematic.R`. Colour roles in all figures: blue = base model / base
checkpoint, orange = post-trained model / SFT checkpoint, grey = baselines and
references.

**Figure 1 (methods schematic, `fig0_methods_schematic`).** How the interventions work.
Each grid shows the positions of the prompt (instruction, text, question, answer
position) from left to right and the layers of the network from bottom to top. (A) The
answer is produced at the last position; it can access the text only through the
positions that follow it, i.e., the question. (B) Steering adds a step along the
direction that separates AI from human texts to the internal state at the answer
position. (C) Model stitching combines the lower layers of one checkpoint (here the
base model, blue) with the upper layers of another (here the SFT model, orange). (D)
Activation patching replaces the internal states of one model (here SFT) at selected
positions and one layer with those of the other model (here the base model's states at
the question positions).

**Figure 2 (`fig1_information_vs_answers`).** The authorship information is available but
not used. Sensitivity (d′) for separating AI from human texts in the three corpora.
Upper rows: classifiers on word and character n-grams and on surface statistics,
probes on the average input embedding and on the internal activations at two thirds of
the network depth, and an externally trained detector (EditLens). Lines show 95%
bootstrap confidence intervals, except for the two probe rows, where they show the
range across models. Lower rows: sensitivity of the models' own answers (raw prompt),
one small dot per model family, large dot = median. The dashed line marks the maximum
d′ that can be estimated with the given number of texts. TODO: news baselines were
computed on the article body without headline; rerun on the exact model input.

**Figure 3 (`fig2_answers_and_fewshot`).** Post-training lowers the sensitivity of the
answers, and labelled examples help base models more. (A) Sensitivity of the base
model (blue) and of the final post-trained model with the identical raw prompt (filled
orange) and with its chat template (open orange), per model family and corpus.
Asterisks mark decreases that survive Holm correction across all tested contrasts.
(B) Sensitivity as a function of the number of labelled examples placed before each
text (raw prompt; base vs. final post-trained model). Vertical lines show the range
across four example sets. On the textbook summaries, only 0, 8, and 16 examples were
tested (OLMo 2: 0 and 8).

**Figure 4 (`fig3_stitching_patching`).** Post-training changes how the question reads
the text. (A) Model stitching between base and SFT checkpoints (textbook summaries).
The x-axis gives the number of lower layers taken from the other checkpoint. Blue: base
layers below, SFT layers above; orange: SFT layers below, base layers above. Dashed and
dotted lines mark the sensitivity of the unmodified base and SFT models; bands are 95%
bootstrap confidence intervals. (B) Activation patching at layer 16: sensitivity after
exchanging the internal states at the indicated positions. Blue: base model's states
inserted into the SFT model; orange: SFT model's states inserted into the base model.
"Unpatched" shows the recipient model without any exchange. Error bars are 95%
bootstrap confidence intervals.

Numbering note: the schematic is planned as Figure 1 (Method), so the data figures
become Figures 2–4; the results draft still refers to them as Figures 1–3 (TODO: update
when assembling the manuscript).
