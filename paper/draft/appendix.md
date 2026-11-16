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
(Tucker, 1964; Karelaia & Hogarth, 2008) from linear regressions of the ground truth and
of the judgments on a set of interpretable cues: the environmental predictability R~e~,
the consistency of the judge R~s~, the matching index G (the correlation of the two
predictions), the unmodeled agreement C, and the achievement r~a~, which decomposes as
r~a~ = G R~e~ R~s~ + C √(1 − R_e²) √(1 − R_s²). Because the regressions are fitted in
sample, we also report cross-validated R~e~, R~s~ and G (10-fold, grouped by source). For
the short chats, the cues were 34 text properties (word and sentence length, lexical
diversity, part-of-speech shares, readability, punctuation, and properties that the
students had named as typical of AI text); for the reviews, 15 cues motivated by Ott et
al. (2011), such as the rates of first-person pronouns, of references to family and to the
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

## C Deceptive reviews: full results

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
neither localization nor steering explained them fully.

## D Steering: all cells

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
