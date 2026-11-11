# Figure captions (draft, 11 Nov 2026)

Figures: `paper/figures/` (PDF for the manuscript, PNG for review). Code:
`plots/fig_data.py` (tidy data, no text content) → `plots/figures.R`,
`plots/fig_schematic.R`. Colour roles in all figures: blue = base model / base
checkpoint, orange = post-trained model / SFT checkpoint, grey = baselines and
references.

**Figure 1 (methods schematic, `fig0_methods_schematic`).** How the interventions work.
In panels A, C, and D, the horizontal axis shows the positions of the prompt in reading
order (the instruction, the text to be judged, the question, and the position at which
the answer is produced), and the vertical axis shows the layers of the network. (A) Information
flow: through attention, later positions read information from earlier ones. The answer
is produced at the last position, so information from the text can reach it only via
the positions in between, i.e., the question (dark tiles). (B) Steering, sketched in a two-dimensional
version of the high-dimensional representation space: human texts (open circles) and AI
texts (triangles) differ along an AI–human direction (arrow between the class means).
Steering moves the state of one text along this direction (diamonds) and measures how
the model's answer changes; the dashed line marks where the answer switches. If the
model read the direction correctly, a shift toward the AI texts would move the answer
toward "KI". (C) Model stitching combines the lower layers of one checkpoint (here the
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
d′ that can be estimated with the given number of texts.

**Figure 3 (`fig2_answers_and_fewshot`).** Post-training mostly lowers the sensitivity
of the answers; labelled examples help almost all models. (A) Sensitivity of the base
model (blue) and of the final post-trained model with the identical raw prompt (filled
orange) and with its chat template (open orange), per model family and corpus.
Asterisks mark differences between the base and the final model on the identical raw
prompt that survive Holm correction within the family of 18 final-vs-base comparisons;
all of them are decreases. The increases of OLMo 2 on the short chats and of OLMo 3
Think on the textbook summaries are significant only without correction or under
false-discovery-rate control (the latter in a condition with low answer probability,
≈ .30). Random-effects summary across all 18 comparisons: Δd′ = −0.33 [−0.50, −0.17]. (B) Sensitivity as a function of the number of
labelled examples placed before each text (raw prompt; base vs. final post-trained
model). Vertical lines show the range across four example sets. Whether base models
profit more than post-trained models (difference of gains, 0 vs. 16 examples) is tested
in the text: significantly so in three of five comparisons, reversed for Qwen3 on the
textbook summaries. On the textbook summaries, only 0, 8, and 16 examples were
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
bootstrap confidence intervals. Patching at layer 8 (more mixed; text positions
contribute somewhat) is shown in Supplementary Figure S1.

Numbering note: the schematic is planned as Figure 1 (Method), so the data figures
become Figures 2–4; the results draft still refers to them as Figures 1–3 (TODO: update
when assembling the manuscript).

**Figure S1 (`figS1_patching_layers`).** Activation patching at layers 8 and 16
(textbook summaries). As Figure 4B, with layer 8 added. At layer 8 the picture is
mixed: in Apertus, exchanging the states of the text positions still has a partial
effect; in OLMo 2, even exchanging all positions transfers only part of the difference
(the change is not yet complete at layer 8; cf. the stitching curve in Figure 4A). At
layer 16, the whole difference between base and SFT model is carried by the question
positions. Error bars
are 95% bootstrap confidence intervals.
