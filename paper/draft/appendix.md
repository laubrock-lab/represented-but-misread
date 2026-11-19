# Appendix

## A Terminology across disciplines

This study combines concepts from the psychology of judgment with methods from machine
learning and interpretability research. The two fields often describe the same idea with
different words, and they sometimes use the same word for different ideas. Table A1 lists
the correspondences that are relevant here, ordered by the two lenses of Figure 1
(terms marked † are used in this paper); Table A2 lists terms that are used in both fields with different meanings and
states how we use them in this paper.

**Table A1. Correspondences between the terminology of psychology and statistics and that
of machine learning and interpretability research.**

| Psychology / statistics | Machine learning / interpretability | Meaning in this study |
|---|---|---|
| *The state of the world* | | |
| state of the world, distal criterion (Brunswik)† | ground truth, label, target | whether a text was written by an AI, whether a statement is true, whether a review is genuine |
| proximal cue† | input feature | an observable property of the text, e.g., word choice, first-person pronouns, length |
| ecological (cue) validity† | feature–label correlation | correlation of a text property with the ground truth |
| environmental predictability (R~e~) | performance of a (linear) model on the features | how well the cues jointly predict the ground truth |
| representative design (Brunswik) | natural, in-distribution test data | real texts rather than constructed stimuli |
| *Signal detection* | | |
| sensitivity d′† | separability; AUC (d′ = √2 · z(AUC)) | how well the answers separate the two classes |
| response criterion c, response bias† | decision threshold; prior or calibration offset | tendency to give one answer rather than the other, e.g., "human" |
| hit rate, false-alarm rate | true-positive rate (recall), false-positive rate | as in signal detection theory |
| ROC curve† | ROC curve | identical |
| *Lens 1: encoding (internal representation)* | | |
| internal representation | hidden states, activations, residual stream† | the model's internal vectors at a given layer and position |
| decoding analysis, multivariate pattern analysis (neuroscience) | linear probe† | logistic regression that predicts the ground truth from activations |
| representational similarity analysis | centered kernel alignment (CKA)† | similarity of the representations of two checkpoints |
| cross-decoding, generalization across conditions | probe transfer† | a probe fitted on one checkpoint (or set) and tested on another |
| *Lens 2: readout (utilization and answer)* | | |
| task set, instruction | prompt; system prompt; question† | the question that asks for the judgment |
| cue utilization (r~s~, β~s~)† | feature attribution; feature importance | how strongly a cue influences the answer |
| policy capturing, judgment analysis | behavioral surrogate model | regression of the answers on the cues |
| matching index G (lens model equation)† | cosine similarity of weight vectors; agreement of decision rules | whether the answer weights the cues as the environment does |
| achievement r~a~† | accuracy; correlation of predictions with labels | how well the answers track the ground truth |
| practice trials, worked examples | few-shot examples, in-context learning† | labeled examples given in the prompt |
| *Causal tests and learning history* | | |
| experimental manipulation; microstimulation (neurophysiology) | activation steering† | shifting the internal state along a direction and observing the answer |
| transplantation, lesion and rescue | model stitching†; activation patching† | combining layers or internal states of two checkpoints |
| developmental or training stages; longitudinal measurement | checkpoints†; pretraining, supervised fine-tuning, preference optimization† | the published states of a model after each stage of training |
| *General statistics* | | |
| fitting, estimation | training, learning | |
| predictor, independent variable | feature, input | |
| outcome, dependent variable | target, label | |
| regression coefficient; intercept | weight; bias term | |
| ridge (penalized) regression | L2 regularization, weight decay | used in the probes |
| clustered (grouped) resampling | grouped cross-validation; avoiding leakage† | texts from the same source never in training and test, or resampled together |
| random-effects meta-analysis† | — | summary of the post-training contrasts across models and corpora |

**Table A2. False friends: terms with different meanings in the two fields, and how we use
them.**

| Term | Psychology / statistics | Machine learning / interpretability | Use in this paper |
|---|---|---|---|
| criterion | response criterion c (signal detection); distal criterion (Brunswik) | loss function (e.g., `criterion` in PyTorch code) | only the response criterion; the distal variable is called the state of the world (ground truth) |
| bias | response bias (signal detection); social bias | bias term (intercept) of a layer; social bias, fairness | "response bias" for the signal-detection sense |
| sensitivity | d′, discrimination ability; also "sensitivity analysis" (robustness check of an analysis) | true-positive rate (recall) | d′; "sensitivity analysis" only in the statistical sense, where unambiguous |
| validity | ecological, construct, or predictive validity | validation set | cue validity: correlation of a cue with the ground truth |
| alignment | — (lens model: matching, G) | AI alignment (post-training toward human preferences); representational alignment | "post-training" for the training stages; "matching" for G; fixed names kept (superficial alignment hypothesis, centered kernel alignment) |
| attention | selective attention | attention mechanism of the transformer | only the transformer mechanism; "selection" for the psychological sense |
| representation | mental representation | vector in activation space | activations, read with probes |
| probe | probe stimulus, probe trial | linear classifier on activations | the classifier |
| policy | judgment policy (policy capturing) | policy π of a reinforcement-learning agent | "cue weighting" or "judgment policy", defined where used |
| model | statistical or computational model of a judge | the neural network itself | "model" for the language model; statistical models named explicitly |
| feature | stimulus feature | input feature; learned feature (e.g., of a sparse autoencoder) | text properties are "cues" |
| readout | read-out of a neural population by a downstream area | output layer; decoding from activations | how the question turns the representation into an answer |

## B Lens-model analyses

*Method.* For each judge, i.e., the answers of a checkpoint or the out-of-fold scores of
a probe on its representation, we computed the components of the lens model equation
[@tucker1964lme; @karelaia2008lens] from linear regressions of the ground truth and
of the judgments on a set of interpretable cues: the environmental predictability R~e~,
the consistency of the judge R~s~, the matching index G (the correlation of the two
predictions), the unmodeled agreement C, and the achievement r~a~, which decomposes as
r~a~ = G R~e~ R~s~ + C √(1 − R_e²) √(1 − R_s²). Because the regressions are fitted in
sample, we also report cross-validated R~e~, R~s~ and G (10-fold, grouped by source). For
the short chats, the cues were 34 text properties (word and sentence length, lexical
diversity, part-of-speech shares, readability, punctuation, and properties that the
students had named as typical of AI text), computed with a feature extractor written for the
lab course (based on spaCy, textstat, and wordfreq); for the reviews, 15 cues motivated by @ott2011finding, such as the rates of first-person pronouns, of references to family and to the
hotel or city, of spatial details and of emotion words, and the length of the review and
of its words and sentences, all centered within the positive and the negative reviews.

*Short chats.* On the 300 messages for which probes were available, the cues predicted
authorship moderately well (R~e~ = .56; cross-validated .40). The answers of most
post-trained models weighted the cues in the direction opposite to their validity
(cross-validated G: Gemma 3 4B −.49, Gemma 3 12B −.58, Qwen3 −.61, Apertus after SFT −.57);
for Qwen3, this was already the case in the base model (−.53), whereas the other base models
were closer to zero (−.28 to −.04). The AI messages in this corpus were richer in adverbs
and verbs, lexically more diverse, and contained longer sentences than the human
messages (r = .16 to .23 with authorship), whereas the human messages contained more
adjectives, proper nouns, and long words; the post-trained models of Gemma 3 12B, Qwen3,
and Apertus took long words, adjectives, nouns, and low readability as signs of AI
authorship (e.g., Gemma 3 12B, mean word length: r = −.04 before and +.24 after
post-training; adjectives: −.06 and +.27). OLMo 2 showed the opposite weighting after SFT
(G = +.37). Probes weighted the cues in line with their validity in every checkpoint
(cross-validated G = .69–.78); because probes are trained on the ground truth, this is
expected by construction and only shows that the cues were represented. In-sample values
and the 422-message set including the style-matched messages gave the same picture.

*Deceptive reviews.* Fifteen cues predicted the ground truth moderately well (R~e~ = .56;
cross-validated .55). Probes again matched the environment closely in all 13 checkpoints
of the four models analyzed (G = .97–.99), as expected by construction. The answers were
poorly matched (G = .15–.83), and post-training changed this matching in opposite
directions: from .83 to .26 in Gemma 3 4B and from .17 to .53 in Qwen3. After
post-training, all models treated long, detailed reviews as truthful, a cue without
validity (r = .03 with the ground truth), and ignored first-person pronouns, the most valid
single cue (r = −.33).

## C Deceptive reviews

> **Table C1.** Deceptive reviews: sensitivity (d′) of the probes (last prompt position, two
> thirds of the network depth) and of the answers (plain prompt). SFT: answers after
> supervised fine-tuning; —: no intermediate checkpoint released.
>
> | Judgment | Model | Probe base | Probe final | Answer base | SFT | Final |
> |:---|:---|---:|---:|---:|---:|---:|
> | Authenticity | OLMo 2 7B | 2.93 | 2.79 | 0.16 | 0.20 | 0.17 |
> | (hotel reviews) | Apertus 1.5 8B | 2.62 | 2.49 | 0.14 | 0.32 | 0.23 |
> | | OLMo 3 7B Think | 2.80 | 2.69 | 0.13 | −0.01 | 0.00 |
> | | Gemma 3 4B | 2.82 | 2.51 | 0.45 | — | 0.08 |
> | | Gemma 3 12B | 3.01 | 2.57 | 0.52 | — | 0.86 |
> | | Qwen3 8B | 2.56 | 2.59 | 0.06 | — | 0.50 |

*Answers.* All models judged the reviews close to chance (d′ between 0.00 and
0.86), although probes separated truthful from deceptive reviews well (d′ = 2.5–3.0), at
least as well as a classifier on word frequencies (d′ = 2.44). Post-training changed the
probes little and the answers by small amounts in a direction that depended on the model:
Gemma 3 4B lost sensitivity (d′ 0.45 → 0.08), Qwen3 (0.06 → 0.50) and Gemma 3 12B (0.52 →
0.86) gained, and the others changed little (Table C1). What post-training changed
consistently was a shared heuristic. The base models judged positive reviews as more
truthful than negative ones, and review polarity accounted for 41–90% of the variance of
their judgments; post-training largely removed this heuristic (2–32%), except in OLMo 3
Think (76%), and all final models showed a strong bias toward "Truthful" (83–92% of the
answers). The localization and steering analyses were less conclusive than for the other
two judgments: Patching located the change at the question positions in two of four
cases, and steering did not explain it (see below). The reviews are thus a case in which
post-training reshaped the readout without improving it.

*Convergence.* Within each review polarity, the item-level judgments of the five models
post-trained mainly for conversation became more similar: The mean correlation between
models rose from .36 (range .16–.50) for the base models to .52 (.45–.58) for the final
models; OLMo 3 Think did not join this convergence (.18–.33 with the other final models).
Without separating the polarities, the correlations fell (from .75 to .48), because the
shared polarity heuristic of the base models disappeared (review_convergence.py).

*Localization.* Stitching and patching were run for Gemma 3 4B (which lost sensitivity)
and Qwen3 (which gained). In Qwen3, the gain arose gradually over layers 2–18 in stitching;
at layer 18, the base model's question states removed the gain from the final model (0.53
→ 0.17; text states: 0.29), whereas the final model's question states raised the base model
only partly (0.08 → 0.22; all positions: 0.49). In Gemma 3 4B, the final model's question
states lowered the base model (0.53 → 0.23; text states: no effect), but the base model's
states did not restore the final model at any set of positions (≤ 0.25), and stitching
produced non-additive effects in the middle and upper layers with strongly varying answer
probability. Thus, the question positions carried the change in two of the four
directions tested.

*Steering.* In none of the four cells did the review direction move the answer more than
the control directions (Table D1).

These results are less conclusive than those for authorship and negated statements: The
effects of post-training were small (|Δd′| ≤ 0.44), their sign differed between models, and
neither localization nor steering explained them fully. We can only speculate about the
reason. Although judging whether a review is genuine resembles judging whether a text was
written by an AI on the surface, the two differ in what the models bring to them: The cues
that separate truthful from deceptive reviews are subtle and specific to this corpus (human
readers, too, judge these reviews near chance [@ott2011finding]), the models' answers were
near chance before and after post-training, and we found no examples of this judgment in
the fine-tuning data. It is conceivable that post-training is largely irrelevant to a
judgment for which neither the base model nor the training data provide a usable mapping
from the represented information to the answer; what it changed here was a general
heuristic (review polarity) and the response bias, not the use of the valid cues.

## D Steering at the answer position

The probes and the answers could dissociate either because the answers ignore the
authorship information or because they use it with the wrong sign. As a check on the
second possibility, we shifted a model's internal state at the position where the
answer is produced along the direction that separates AI from human texts (estimated on
held-out groups; Section 2.7), and measured how the answer changed. The primary test used
shifts of ±2 within-class standard deviations, which kept 48–77% of the shifted texts
within the range of the observed texts (±1 SD: 60–94%). We compared the effect with 100
control directions per cell, each drawn once and applied to all texts: 50 random
directions within the high-variance subspace of the activations and 50 differences between
random halves of the texts within each class. The controls took either steps of the same
length as the authorship direction (norm-matched) or steps of two standard deviations along
their own direction (scale-matched), which kept them within the range of the texts as
often as the authorship direction (49–69%). As a positive control, we shifted along the
direction that the model's own answers follow (the decision direction). In 21 of 25 cells
(model × corpus × layer, including the truth judgments of Section 3.6 and the review judgments of Appendix C), the
decision direction moved the answer more than the norm-matched controls (rank-based
p ≤ .05), so that the test can detect a direction that the answer reads; against the
scale-matched controls, this was the case in 16 of 25 cells. Where the positive control
failed, a null result for the authorship direction is not informative.

In six of the 17 authorship cells, the authorship direction moved the answer more than the
controls (rank-based p ≤ .06 against both control families; with 100 controls, the smallest
possible p is .01, so that no cell would survive a correction for 17 tests), and in all six
it moved it the wrong way: Shifting a text toward "AI" made the
model answer "human" more often. This was the case for the short chats in Gemma 3 12B
(layer 31: −7.38 nats; rank-based p = .01 against both control families), Gemma 3 4B
(layer 17: −1.74; p = .04 norm-matched, .06 scale-matched), and Qwen3 (layers 24 and 30:
−1.18 and −2.06; p ≤ .03), and for the newspaper segments in Qwen3 (−0.43 and −0.85;
p ≤ .04). In these cells, the authorship direction pointed against the decision direction
(cosine −0.59 to −0.92), and the answers were clearly below chance (Table 2). Note that
these are the cells with the most negative d′, and that this is expected: The authorship
direction is the difference between the class means at the position from which the answer is
read, so that, if the answer depends approximately linearly on this state, shifting along it
must reproduce the observed difference between the classes. Across the 25 cells, the effect
of a shift by one class distance correlated with the observed difference (r = .91;
outputs/steer_nulls_summary.csv). The test therefore does not provide evidence that is independent of the answers'
d′; it shows that the inverse ordering is carried by the class-mean difference at the answer
position, more than by typical directions. In the
remaining cells, the authorship direction did not move the answer more than arbitrary
directions: Gemma 3 4B on the newspaper segments (−1.04 and −2.00 at layers 22 and 28;
p = .26–.42), Apertus and OLMo 3 on the textbook summaries before and after SFT, and Gemma
3 12B on the newspaper segments, on which its answers were at chance (+0.03). In some of
these cells, the positive control did not exceed the scale-matched controls either (Gemma 3
12B on the newspaper segments, Gemma 3 4B at layer 28, Apertus at layers 21 and 27 and
OLMo 3 after SFT; Table D1), so that their null results are weaker evidence. In the
strongest cell (Gemma 3 12B on the chats), the decision direction itself did not clearly
exceed the norm-matched controls (p = .069; scale-matched: p = .01), and only 48% of the shifted
texts stayed within the observed range. The norm-matched controls in the subspace of
leading components moved the state far outside the observed range in the Gemma 3 and Qwen3
cells, so that the scale-matched controls are the fairer comparison. In summary, where the
answers ordered the texts inversely, the class-mean difference at the answer position
carried this inverse ordering; steering at earlier layers or at the text positions, which
would be an independent test, was not carried out.

A complementary, descriptive comparison concerns the size of the effects. Shifting a
text by one class-mean distance along the authorship direction, i.e., from the average
human text to the average AI text, changed the answer in the same direction as the
observed difference between AI and human texts in all 16 cells in which the bootstrap
interval of this difference excluded zero, and in 23 of the 25 cells overall (e.g., Qwen3 on the newspaper segments: −0.50 and −0.66
nats vs. an observed difference of −0.59). As stated above, this agreement is expected if the answer depends approximately linearly
on the internal state; it is consistent with the patching results (Section 3.4).

Steering did not, however, reveal what post-training changes for authorship. In the
Apertus base model, shifting toward "AI" moved the answer in the correct direction,
increasingly so with depth (+0.08, +0.21, and +0.35 nats at layers 16, 21, and 27), but
not more than the control directions (p ≥ .15), and after SFT the effect was close to zero
(−0.04 to +0.04). In OLMo 3 Think, neither the base nor the SFT model used the direction
(−0.01 and +0.09).[^olmo3] The change of the answers during SFT is therefore localized by
the stitching and patching analyses (Section 3.4), not by steering.

[^olmo3]: An earlier comparison, in which the SFT model received its chat template and the
base model the plain prompt, had suggested a reversed use after SFT (−0.27); with the same
plain prompt for both checkpoints, this was not the case.

> **Table D1.** Steering at the last prompt position (L: layer). Dist.: distance between
> the class means along the class direction, in within-class SD. In range: share of
> steered texts (±2 SD) within the 1st–99th percentile of the observed projections.
> Effect: directional effect on dv (nats) of ±2 SD along the class direction (authorship:
> toward "AI"; truth and reviews: toward "true"). p: rank among 100 fixed control
> directions (norm-matched and scale-matched; minimum .01). Decision: effect of the
> decision direction (positive control). Per dist.: effect of one class distance,
> extrapolated from ±1 SD. Observed: difference of the unsteered dv between the classes
> (AI − human, true − false). fin.: final post-trained model.

| Judgment | Model | L | Dist. | In range | Effect | p (norm) | p (scale) | Decision | Per dist. | Observed |
|:--------|:----------------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Chats | Gemma 3 12B (fin.) | 31 | 0.9 | 0.48 | −7.38 | .01 | .01 | +5.77 | −2.96 | −4.70 |
| Chats | Gemma 3 4B (fin.) | 17 | 0.8 | 0.62 | −1.74 | .04 | .06 | +3.08 | −0.65 | −3.33 |
| Chats | Qwen3 8B (fin.) | 24 | 1.2 | 0.63 | −1.18 | .02 | .01 | +1.52 | −0.72 | −1.23 |
| Chats | Qwen3 8B (fin.) | 30 | 0.9 | 0.60 | −2.06 | .03 | .02 | +2.30 | −0.96 | −1.23 |
| News | Gemma 3 12B (fin.) | 31 | 0.7 | 0.54 | +0.03 | .97 | .94 | +1.36 | +0.01 | +0.14 |
| News | Gemma 3 4B (fin.) | 22 | 0.8 | 0.60 | −1.04 | .42 | .26 | +3.83 | −0.40 | −0.40 |
| News | Gemma 3 4B (fin.) | 28 | 0.5 | 0.51 | −2.00 | .37 | .26 | +3.60 | −0.49 | −0.40 |
| News | Qwen3 8B (fin.) | 24 | 2.3 | 0.75 | −0.43 | .04 | .03 | +0.67 | −0.50 | −0.59 |
| News | Qwen3 8B (fin.) | 30 | 1.6 | 0.65 | −0.85 | .03 | .03 | +1.05 | −0.66 | −0.59 |
| Textbook | Apertus 1.5 8B (base) | 16 | 3.2 | 0.70 | +0.08 | .48 | .45 | +0.37 | +0.11 | +0.15 |
| Textbook | Apertus 1.5 8B (base) | 21 | 2.2 | 0.67 | +0.21 | .37 | .24 | +0.57 | +0.23 | +0.15 |
| Textbook | Apertus 1.5 8B (base) | 27 | 1.6 | 0.62 | +0.35 | .22 | .15 | +0.68 | +0.28 | +0.15 |
| Textbook | Apertus 1.5 8B (SFT) | 16 | 3.0 | 0.74 | +0.04 | .67 | .73 | +0.32 | +0.05 | −0.00 |
| Textbook | Apertus 1.5 8B (SFT) | 21 | 2.8 | 0.72 | −0.03 | .83 | .87 | +0.37 | −0.04 | −0.00 |
| Textbook | Apertus 1.5 8B (SFT) | 27 | 2.5 | 0.77 | −0.04 | .74 | .80 | +0.36 | −0.05 | −0.00 |
| Textbook | OLMo 3 7B Think (base) | 21 | 1.6 | 0.71 | −0.01 | .95 | .93 | +0.42 | −0.01 | −0.02 |
| Textbook | OLMo 3 7B Think (SFT) | 21 | 2.4 | 0.75 | +0.09 | .28 | .58 | +0.15 | +0.12 | +0.14 |
| Negated statements | Apertus 1.5 8B (base) | 21 | 7.9 | 0.64 | −0.14 | .13 | .33 | +0.27 | −0.54 | −0.31 |
| Negated statements | Apertus 1.5 8B (SFT) | 21 | 14.2 | 0.58 | +0.36 | .02 | .02 | +0.36 | +2.51 | +3.08 |
| Negated statements | OLMo 2 7B (base) | 21 | 9.5 | 0.57 | +0.09 | .58 | .54 | +0.42 | +0.43 | +0.22 |
| Negated statements | OLMo 2 7B (SFT) | 21 | 7.7 | 0.59 | +1.07 | .01 | .04 | +1.14 | +4.13 | +3.64 |
| Reviews | Gemma 3 4B (base) | 23 | 0.8 | 0.66 | +0.21 | .68 | .35 | +1.03 | +0.09 | +0.20 |
| Reviews | Gemma 3 4B (fin.) | 23 | 0.7 | 0.60 | +0.85 | .66 | .53 | +4.45 | +0.31 | +0.55 |
| Reviews | Qwen3 8B (base) | 24 | 1.3 | 0.70 | −0.01 | .98 | .96 | +0.53 | −0.01 | +0.06 |
| Reviews | Qwen3 8B (fin.) | 24 | 1.2 | 0.68 | +0.50 | .16 | .20 | +1.08 | +0.30 | +1.08 |

## E Few-shot gains

> **Table E1.** Gain in d′ from 0 to 16 labeled examples (mean of the per-set d′ minus
> zero-shot d′), for base and final post-trained models on the same texts, and the
> difference of the gains with a grouped-bootstrap CI (fewshot_summary.py).
>
> | Corpus | Model | Gain base | Gain post-trained | Difference [95% CI] |
> |:---|:---|---:|---:|---:|
> | Chats | Gemma 3 4B | +0.69 | +0.45 | +0.24 [−0.09, +0.60] |
> | Chats | OLMo 2 7B | +0.55 | +0.01 | +0.54 [+0.27, +0.83] |
> | Chats | Qwen3 8B | +0.83 | +0.63 | +0.19 [−0.06, +0.45] |
> | Textbook | Gemma 3 4B | +0.50 | +0.33 | +0.18 [−0.06, +0.41] |
> | Textbook | Qwen3 8B | −0.13 | +0.18 | −0.31 [−0.55, −0.09] |

## F Signal detection invariance

> **Table F1.** Mean absolute change of the criterion c and of the sensitivity d′ under three
> manipulations of the plain prompt, averaged over corpora and answer orders: stated AI share
> 20 → 80% (from the slope over 20, 50, and 80%), answer order ("KI oder Mensch" vs. "Mensch
> oder KI", without base-rate sentence), and presence of a base-rate sentence (50% vs. none).
> The changes of d′ had no consistent direction (mean signed change ≤ 0.03). In 56% of the
> conditions, the model gave the same answer to at least 98% of the texts, so that c could
> hardly move (sdt_invariance.py; outputs/sdt_invariance_conditions.csv, sdt_invariance_tests.csv).

| Model | Checkpoint | Base rate 20→80%: c | Base rate 20→80%: d′ | Answer order: c | Answer order: d′ | Base-rate sentence: c | Base-rate sentence: d′ |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Apertus 1.5 8B | base | 0.63 | 0.24 | 1.09 | 0.30 | 0.37 | 0.17 |
| Apertus 1.5 8B | final | 0.97 | 0.18 | 0.08 | 0.16 | 0.14 | 0.05 |
| Gemma 3 12B | base | 1.67 | 0.09 | 1.27 | 0.22 | 0.40 | 0.18 |
| Gemma 3 12B | final | 2.93 | 0.10 | 0.26 | 0.08 | 0.61 | 0.19 |
| Gemma 3 4B | base | 0.42 | 0.07 | 0.65 | 0.10 | 0.29 | 0.05 |
| Gemma 3 4B | final | 3.28 | 0.15 | 0.53 | 0.10 | 0.65 | 0.18 |
| OLMo 2 7B | base | 1.09 | 0.05 | 3.11 | 0.29 | 0.40 | 0.12 |
| OLMo 2 7B | final | 0.47 | 0.13 | 0.16 | 0.12 | 0.28 | 0.09 |
| OLMo 3 7B Think | base | 0.93 | 0.30 | 0.90 | 0.21 | 0.32 | 0.13 |
| OLMo 3 7B Think | final | 0.10 | 0.09 | 0.68 | 0.13 | 0.34 | 0.07 |
| Qwen3 8B | base | 1.73 | 0.04 | 0.54 | 0.12 | 1.19 | 0.23 |
| Qwen3 8B | final | 0.07 | 0.24 | 0.06 | 0.12 | 0.03 | 0.10 |
| All | | 1.19 | 0.14 | 0.78 | 0.16 | 0.42 | 0.13 |

> **Table F2.** Use of the stated base rate: mean shift of dv (nats) between a stated share of
> 20% and 80% AI texts, averaged over three corpora and both answer orders (six cells; range
> over cells in brackets), its ratio to the shift of the prior log-odds (ln 16 = 2.77), and the
> mean between-text SD of the shift within a cell. Wording: share named first in the base-rate
> sentence (base_rate_shift.py; outputs/base_rate_shift_models.csv).

| Model | Checkpoint | Wording | Shift [range] | Ratio to ln 16 | Between-text SD |
|:---|:---|:---|---:|---:|---:|
| Apertus 1.5 8B | base | AI first | 0.41 [0.33, 0.47] | 0.15 | 0.14 |
| Apertus 1.5 8B | final | AI first | 6.25 [3.06, 8.85] | 2.26 | 2.01 |
| Gemma 3 12B | base | AI first | 0.50 [0.42, 0.55] | 0.18 | 0.11 |
| Gemma 3 12B | final | AI first | 10.93 [6.48, 14.83] | 3.94 | 2.34 |
| Gemma 3 4B | base | AI first | 0.18 [0.14, 0.21] | 0.06 | 0.08 |
| Gemma 3 4B | base | human first | 0.16 [0.12, 0.20] | 0.06 | 0.08 |
| Gemma 3 4B | final | AI first | 10.65 [7.84, 12.78] | 3.84 | 1.54 |
| Gemma 3 4B | final | human first | 5.61 [4.08, 8.53] | 2.03 | 1.80 |
| OLMo 2 7B | base | AI first | 1.00 [0.54, 1.49] | 0.36 | 0.18 |
| OLMo 2 7B | base | human first | 1.35 [0.70, 2.13] | 0.49 | 0.23 |
| OLMo 2 7B | final | AI first | 5.39 [4.25, 6.54] | 1.94 | 0.89 |
| OLMo 2 7B | final | human first | 4.37 [3.75, 4.88] | 1.58 | 0.88 |
| OLMo 3 7B Think | base | AI first | 1.17 [0.56, 1.85] | 0.42 | 0.24 |
| OLMo 3 7B Think | base | human first | 1.07 [0.41, 1.89] | 0.38 | 0.21 |
| OLMo 3 7B Think | final | AI first | 1.21 [0.81, 1.44] | 0.44 | 0.19 |
| OLMo 3 7B Think | final | human first | 1.48 [1.00, 2.07] | 0.53 | 0.22 |
| Qwen3 8B | base | AI first | 1.40 [1.06, 1.68] | 0.51 | 0.27 |
| Qwen3 8B | final | AI first | 3.67 [3.28, 4.43] | 1.32 | 0.46 |

## G Individual post-training contrasts

Among the individual comparisons, six decreases survived Holm correction within this
family of 18 tests (all p_Holm ≤ .013): Gemma 3 12B (Δd′ = −0.95 [−1.28, −0.66]),
Qwen3 (−0.57 [−0.79, −0.36]), Gemma 3 4B (−0.50 [−0.72, −0.29]), Apertus (−0.47
[−0.71, −0.25]) and OLMo 2 (−0.32 [−0.52, −0.12]) on the textbook summaries, and Qwen3
on the newspaper segments (−0.65 [−0.93, −0.42]). On the short chats, three decreases of
similar size (Gemma 3 4B: −0.51; Apertus: −0.47; Gemma 3 12B: −0.45; uncorrected
p ≤ .008) did not survive Holm correction (p_Holm = .059–.074), because the chat estimates
were less precise (61 independent originals compared with 119 pairs for the textbook
summaries); they remained significant under false-discovery-rate control. Two increases
were significant only without correction or under false-discovery-rate control: OLMo 2
on the short chats (+0.35 [0.06, 0.63]) and OLMo 3 Think on the textbook summaries, the
latter in a condition with low answer probability. Note that we defined the family of
tests post hoc as the comparisons that address our hypothesis; Holm correction across
all 111 contrasts we computed, including comparisons between adjacent post-training
stages and across prompt formats, left five of the six decreases significant. Hence,
post-training lowered the sensitivity of the answers on average, but neither uniformly
nor in a single direction across models and text types.

## H Localization: further analyses

*Sublayer types.* Both sublayer types contributed. Exchanging only the attention blocks or only the
feed-forward (MLP) blocks of the first 8 Apertus layers had partial effects in both
directions (e.g., restoring the base model's blocks under SFT: attention +0.17
[−0.06, 0.40], MLP +0.18 [0.03, 0.37], both together +0.59). In OLMo 2, the base
model's MLP blocks of the first 16 layers restored discrimination almost completely
(−0.09 [−0.29, 0.11]), whereas its attention blocks did so only partly (−0.35
[−0.55, −0.19]). Within the limits just described, the change thus appears to be
distributed over the early layers and to involve both attention and feed-forward
blocks.

*Question placed before the text.* See Section 3.4 for the interpretation.

> **Table H1.** Activation patching at layer 16 with the question placed before the text
> (textbook summaries): d′ of the recipient after receiving the donor's states at one set of
> positions; change relative to the unpatched recipient with 95% CI (grouped bootstrap) for
> the three sets of interest (localization_summary.py; outputs/patch/patch_summary.csv).
>
> | Recipient ← donor | Unpatched | Donor | Text | Question (before text) | After text | All positions |
> |:---|---:|---:|---:|---:|---:|---:|
> | Apertus base ← SFT | +0.36 | −0.19 | +0.31 (−0.05 [−0.15, 0.05]) | +0.55 (+0.19 [0.06, 0.32]) | +0.13 (−0.23 [−0.46, −0.02]) | −0.17 |
> | Apertus SFT ← base | −0.19 | +0.36 | −0.07 (+0.12 [0.04, 0.21]) | +0.10 (+0.29 [0.21, 0.38]) | +0.44 (+0.63 [0.43, 0.86]) | +0.50 |
> | OLMo 2 base ← SFT | +0.09 | −0.42 | +0.06 (−0.04 [−0.13, 0.07]) | −0.08 (−0.18 [−0.30, −0.04]) | −0.16 (−0.25 [−0.39, −0.10]) | −0.43 |
> | OLMo 2 SFT ← base | −0.42 | +0.09 | −0.40 (+0.01 [−0.06, 0.08]) | −0.33 (+0.09 [−0.03, 0.21]) | −0.17 (+0.24 [0.09, 0.40]) | +0.08 |
