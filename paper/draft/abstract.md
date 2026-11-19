# Represented but misread: Post-training changes how language models read out what they represent

## Abstract

Does post-training change what a language model represents, or how this representation
is read out into an answer? We compared linear probes with the models' own answers,
analyzed with signal detection theory, in six open-weight base models and their
post-trained versions. For judgments of whether German texts were written by a human or an
AI, probes and word-frequency classifiers separated the classes well in three corpora, but
the answers did not, and often ordered the texts inversely. Post-training lowered the
sensitivity of the answers on average (Δd′ = −0.34 across model families; −0.29 across three
prompt wordings). It also changed the use of a stated base rate: Base models shifted their
answers less than an ideal observer would, most post-trained models more. In two models with
stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching
implicated the early layers, and from mid-depth onward the internal states at the positions
after the text, not those of the text, transferred the change between checkpoints. Because our corpora cannot be
shared, we add a case reproducible from public data, the truth of negated
statements. Here, post-training acted in the opposite direction: Probes decoded truth
perfectly in every model, the answers of several base models did not use it, and post-training
repaired them; by mid-depth the repair resided at the positions after the statement, and
below it the representation of the statement contributed in three of four models. In a training study, adding 2,000 correctly labeled authorship
judgments to otherwise identical fine-tuning data improved the answers on two other corpora
and inverted them on a third, where they followed a probe trained on news. Post-training
thus changed how a judgment is read out of what a model represents, for better or worse,
and a small training signal can set the direction of this change.
