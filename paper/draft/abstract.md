# Title and abstract (draft, 12 Nov 2026)


## Title options

1. Post-training changes how the question reads the text: Authorship judgments as a
   test case for representation versus readout in language models
2. Knowing but not saying: Post-training changes the readout, not the representation,
   of authorship in language models
3. Language models represent who wrote a text but answer wrongly: A causal test of
   representation versus readout under post-training

(Option 1 is closest to the framing of the introduction; option 2 is the most
accessible but "knowing but not saying" is used in several recent titles.)

## Abstract

Does post-training change what a language model represents, or only how this
representation is read out into an answer? The superficial alignment hypothesis
suggests the latter, but direct causal tests are scarce. We used the judgment whether a
text was written by a human or by an AI as a test case, because the correct answer is
known, the relevant information is contained in the input, and its availability can be
established independently of the model. Across eight families of open-weight language
models and three corpora of German texts, including unpublished newspaper articles, the
information that distinguishes AI from human texts was readily available: Linear probes
on the models' activations and simple word-frequency classifiers separated the two
classes well. The models' own answers, analyzed with signal detection theory, used this
information poorly and often with the wrong sign, and post-training lowered their
sensitivity on average (Δd′ = −0.33). In two model families with checkpoints after each
stage of post-training, the decrease arose during supervised fine-tuning. Three causal
methods converged on where it takes effect. Shifting the internal state toward "AI" along the
direction that separates AI from human texts moved the answers of post-trained models
toward "human". Combining
the lower layers of base and fine-tuned checkpoints (model stitching) showed that the
change originates in the early layers, and exchanging internal states between
checkpoints at selected prompt positions (activation patching) showed that it resides
in the positions of the question, not in those of the text. Post-training thus left the
representation of the text intact but changed how the question reads it out, which
supports the superficial alignment hypothesis in one respect and qualifies it in
another.
