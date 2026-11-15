# Title and abstract (draft, 14 Nov 2026; option B: authorship + two further judgments)


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

Does post-training change what a language model represents, or only how this
representation is read out into an answer? We used judgments with known ground truth as
test cases, primarily whether a text was written by a human or by an AI. Across six
open-weight base models and their post-trained versions and three corpora of German
texts, linear probes and simple word-frequency classifiers separated AI from human texts
well, but the models' answers, analyzed with signal detection theory, used this
information poorly and often ordered the texts inversely. Post-training lowered their
sensitivity on average (Δd′ = −0.33), mostly during supervised fine-tuning, and in
several models steering showed that the represented authorship direction pushed the
answer the wrong way. Combining layers of base and fine-tuned checkpoints (model
stitching) implicated the early layers, and exchanging internal states at selected
prompt positions (activation patching) located the change at the positions of the
question rather than of the text. The pattern generalized to two judgments with public
data. For negated statements, several base models represented truth perfectly but
answered poorly, and post-training repaired their readout; for deceptive reviews, all
models answered near chance despite accurate probes. Probes transferred between base and
post-trained checkpoints at the text positions, but only partially at the answer
position. In a training study, adding 2,000 correctly labeled examples to otherwise identical
fine-tuning data shifted the readout strongly in the implied direction, improving it on
other corpora and inverting it on one, with little change at the text positions. Post-training thus mainly changes
how the question reads the text, in a direction that depends on whether the training
data specify a valid readout.
