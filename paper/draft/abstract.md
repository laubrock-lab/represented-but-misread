# Title and abstract (draft, 17 Nov 2026; negation first, authorship second, training study third)


## Title options

1. Post-training changes how the question reads the text: Representation and readout
   in language-model judgments
2. Same knowledge, different reading: How post-training reshapes the readout of
   judgments in language models
3. Represented but misread: Authorship, truth, and deception judgments in language
   models before and after post-training

(Option 1 keeps the earlier title's core and widens the scope; option 2 states the
finding in plain words; option 3 names the three judgments. The previous,
authorship-only title: "Post-training changes how the question reads the text:
Authorship judgments as a test case for representation versus readout in language
models".)

## Abstract

Does post-training change what a language model represents, or how this representation
is read out into an answer? We compared linear probes with the models' own answers,
analyzed with signal detection theory, in six open-weight base models and their
post-trained versions. For negated statements from a public set, probes decoded truth
perfectly in every model, but the answers of several base models used this information
poorly or ordered the statements inversely; supervised fine-tuning repaired the readout
(e.g., d′ from −0.88 to the maximum of 4.54) while the probes stayed at ceiling. For
judgments of whether German texts from three corpora were written by a human or an AI,
probes and word-frequency classifiers separated the classes well, but the answers did not,
and post-training lowered their sensitivity on average (Δd′ = −0.34 across model families;
−0.29 across three prompt wordings), with considerable variation between models and
prompts. Post-training also changed the use of a stated base rate from underweighting to
overweighting relative to an ideal observer. In two models with stage-wise checkpoints,
combining layers of base and fine-tuned checkpoints (model stitching) implicated the middle
layers for negation and the early layers for authorship, and exchanging internal states at
mid-depth (activation patching) transferred the change with the states at the positions
after the text, not with those of the text. In a training study with one model, adding
2,000 correctly labeled examples to otherwise identical fine-tuning data shifted the
answers strongly, improving them on two other corpora and inverting them on a third, with
little change of the probes at the text positions. Post-training thus changed how the
question reads the text, for better or worse, and a small training signal can set the
direction of this change.
