# Figure captions

Figures: `paper/figures/` (PDF for the manuscript, PNG for review). Code:
`plots/fig_data.py` (tidy data, no text content) → `plots/figures.R`,
`plots/fig_schematic.R`. Colour roles in all figures: blue = base model / base
checkpoint, orange = post-trained model / SFT checkpoint, grey = baselines and
references.

**Figure 1 (lens model with an internal lens, `fig_lens_two_stage`).** Brunswik's lens
model, applied to a language model as judge and extended by an internal lens. A state of
the world (Brunswik's distal criterion; the ground truth, here whether a text was written
by an AI, whether a statement is true, or whether a review is genuine) is reflected in
observable cues of the text, some of which are valid and some of which are not (dashed).
In the first lens (encoding), the model integrates these cues into its internal
representation; whether the ground truth can be decoded from this representation is
measured with linear probes, and whether two checkpoints represent it in the same way
with probe transfer. In the second lens (readout), the question reads the representation
out into an answer by weighting internal features. Only one of these features was
measured directly: the class direction, i.e., the direction that separates the two
classes in the representation; its relation to the answer was examined with
steering at the answer position (Appendix D). The other features are schematic. Achievement, i.e., the agreement of the
answer with the state of the world, requires both lenses. Post-training changes the
second lens (orange); model stitching and activation patching locate the change in depth
and in the prompt. Splitting the judge side of the lens model has precedents in
personality judgment [detection vs. utilization of cues; @funder1995accuracy] and in
neuroscience [information encoded in a neural population vs. read out for behavior;
@panzeri2017cracking]. Note: for the authorship corpora, probes were fitted both on the
text positions and on the last prompt position; for the truth and review sets, only on
the last prompt position.

**Figure 2 (methods schematic, `fig0_methods_schematic`).** How the interventions work.
In panels A, C, and D, the horizontal axis shows the positions of the prompt in reading
order (the instruction, the text to be judged, the question, and the position at which
the answer is produced), and the vertical axis shows the layers of the network. (A) Information
flow: through attention, later positions read information from earlier ones. The answer
is produced at the last position, so information from the text can reach it only via
the positions in between, i.e., the question (dark tiles). (B) Steering, sketched in a two-dimensional
version of the high-dimensional representation space: human texts (open circles) and AI
texts (triangles) differ along an AI–human direction (arrow between the class means).
Steering moves the state of one text along this direction (diamonds) and measures how
the model's answer changes; the dashed line marks where the answer switches. The two
need not be orthogonal: the angle between the AI–human direction and the model's
decision boundary determines how strongly a shift along the direction changes the
answer. If the
model read the direction correctly, a shift toward the AI texts would move the answer
toward "KI". (C) Model stitching combines the lower layers of one checkpoint (here the
base model, blue) with the upper layers of another (here the SFT model, orange). (D)
Activation patching replaces the internal states of one model (here SFT) at selected
positions and one layer with those of the other model (here the base model's states at
the question positions).

**Figure 3 (`fig1_information_vs_answers`).** The authorship information is available but
not used. Sensitivity (d′) for separating AI from human texts in the three corpora.
Upper rows: classifiers on word and character n-grams and on surface statistics,
probes on the average input embedding and on the internal activations at two thirds of
the network depth, and an externally trained detector (EditLens). Lines show 95%
bootstrap confidence intervals, except for the two probe rows, where they show the
range across models. Lower rows: sensitivity of the models' own answers (raw prompt),
one small dot per model, large dot = median. The dashed line marks the maximum
d′ that can be estimated with the given number of texts.

**Figure 4 (`fig2_answers_and_fewshot`).** Post-training mostly lowers the sensitivity
of the answers; with 16 labelled examples, most models improve. (A) Sensitivity of the base
model (blue) and of the final post-trained model with the identical raw prompt (filled
orange) and with its chat template (open orange), per model and corpus.
Asterisks mark differences between the base and the final model on the identical raw
prompt that survive Holm correction within the family of 18 final-vs-base comparisons;
all of them are decreases. The increases of OLMo 2 on the short chats and of OLMo 3
Think on the textbook summaries are significant only without correction or under
false-discovery-rate control (the latter in a condition with low answer probability,
≈ .30). Multilevel random-effects summary across all 18 comparisons: Δd′ = −0.33
[−0.53, −0.13]. (B) Sensitivity as a function of the number of
labelled examples placed before each text (raw prompt; base vs. final post-trained
model). Points show the mean of d′ computed separately for each of four example sets;
vertical lines show their range. Whether base models profit more than post-trained
models (difference of gains, 0 vs. 16 examples) is tested in the supplement: the base
model gained more in four of five comparisons, reliably only for OLMo 2 on the chats; for
Qwen3 on the textbook summaries, the post-trained model gained more. On the textbook summaries, only 0, 8, and 16 examples were
tested; for OLMo 2 only 0 and 8, because 16 textbook summaries exceed its context window
of 4,096 tokens.

**Figure 5 (`fig3_stitching_patching`).** Authorship: model stitching and activation
patching at layer 16. (A) Model stitching between base and SFT checkpoints (textbook
summaries; the numbers of layers that were tested differ between the two directions and
between the models).
The x-axis gives the number of lower layers taken from the other checkpoint. Blue: base
layers below, SFT layers above; orange: SFT layers below, base layers above. Dashed and
dotted lines mark the sensitivity of the unmodified base and SFT models; bands are 95%
bootstrap confidence intervals. (B) Activation patching at layer 16: sensitivity after
exchanging the internal states at the indicated positions. Blue: base model's states
inserted into the SFT model; orange: SFT model's states inserted into the base model.
"Unpatched" shows the recipient model without any exchange. Error bars are 95%
bootstrap confidence intervals. Patching at layer 8 (more mixed; text positions
contribute somewhat) is shown in Supplementary Figure S1.

**Figure 6 (`fig_authorship_sweep`).** Authorship: activation patching across layers
(textbook summaries, 238 texts; error bars: 95% grouped-bootstrap confidence intervals).
Sensitivity (d′) of the recipient checkpoint after its residual stream was replaced by the
donor's at one set of positions and one layer: the tokens of the text, the question and
answer cue after the text, the final position only, or all positions. Horizontal lines show
the unpatched recipient (dotted) and the donor (dashed). In OLMo 2, the text states had no
effect at any layer; in Apertus, they carried part of the difference between the
checkpoints up to layer 12. From layer 14 onward, the states after the text carried it in
both models. Hybrids can lie outside the range of the two checkpoints.

**Figure 7 (`fig_negation`).** Negated statements: where supervised fine-tuning repairs the
readout (300 statement pairs; error bars: 95% grouped-bootstrap confidence intervals). (A)
Model stitching: sensitivity (d′) of the answers of hybrids that take the first ℓ layers
from one checkpoint and all layers above from the other; horizontal lines show the
unstitched base (dotted) and SFT (dashed) checkpoints. (B) Activation patching at layers 8
and 16: d′ of the recipient checkpoint after its residual stream was replaced by the
donor's at one set of positions (the statement, the instruction before it, the question
and answer cue after it, the final position only, or all positions); dotted lines show the
unpatched recipients. Note that in the truth prompt the question precedes the statement,
so that the positions after the statement contain only the answer instruction and cue. (C) The same
exchange at 12 layers (100 statement pairs), for base and SFT model of OLMo 2 and Apertus
and for base and final post-trained model of Gemma 3 4B and 12B: share of the difference
between the two checkpoints that the exchange transferred (values below 0 and above 1 are
clipped). Upper row: the base model receives the states of the post-trained model; lower
row: the post-trained model receives the states of the base model.

**Figure S1 (`figS1_patching_layers`).** Activation patching at layers 8 and 16
(textbook summaries). As Figure 5B, with layer 8 added. At layer 8 the picture is
mixed: in Apertus, exchanging the states of the text positions still has a partial
effect; in OLMo 2, even exchanging all positions transfers only part of the difference
(the change is not yet complete at layer 8; cf. the stitching curve in Figure 5A). At
layer 16, the whole difference between base and SFT model is carried by the question
positions. Error bars
are 95% bootstrap confidence intervals.

**Figure S2 (`figS2_checkpoint_transfer`).** Post-training changes the text positions
little and reorganizes the answer position. For each layer, a linear probe was
trained on one checkpoint and tested on the other (base vs. final post-trained model,
both directions averaged); the transfer deficit is the loss in d′ relative to the probe
trained on the checkpoint itself. A deficit of 0 means that the same direction separates
the classes in both checkpoints. Dark lines: OLMo 2 7B (solid) and Apertus 1.5 8B
(dashed), the two models included in all six sets; thin grey lines: the other
models (OLMo 3 Think and Instruct, which share a base model, counted separately); thick
light line: mean over all models at the text positions (authorship
corpora only). At the text positions, probes transferred without loss at every depth,
also where they were well below ceiling (chats, d′ ≈ 2). At the last prompt position, at
which the answer is produced, the deficit grew with depth in all models and tasks,
although probes trained on each checkpoint itself still decoded the information. For the
negated statements, the deficit drops to zero at mid-depth because both checkpoints are
at ceiling there (AUC = 1); OLMo 2 shows no deficit in the upper layers for this set.
