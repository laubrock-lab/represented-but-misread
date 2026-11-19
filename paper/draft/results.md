# 3 Results

We first report the authorship judgments, for which we used our own corpora: whether the
information is available and whether the answers use it (Section 3.1), how post-training
changed the answers (Section 3.2), how the models used a stated base rate (Section 3.3),
and where in the network the change acts (Sections 3.4 and 3.5). Because these corpora
cannot be shared, Section 3.6 reports a judgment on public data that can be reproduced in
full, the truth of negated statements, for which post-training changed the answers in the
opposite direction. Section 3.7 reports a training study that tests what sets the direction
of the change. A further judgment, whether a hotel review is genuine, gave less conclusive
results and is reported in Appendix C. Unless stated otherwise, sensitivity is expressed as
d′ estimated from the area under the ROC curve, with 95% confidence intervals from a grouped
bootstrap.

## 3.1 Authorship: the information is available, but the answers do not use it

We used three corpora
of German texts that each pair human-written texts
with AI-generated counterparts: short chat messages truncated to 25 words (300 texts
from 61 human originals), segments of local newspaper articles (103 segments from 14
articles), and textbook summaries (238 texts, 119 matched human–AI pairs; Rosin & Laubrock,
in preparation). In all
analyses, texts derived from the same human original form one group, so that a model,
a probe, or a bootstrap resample never profits from recognizing a topic. Positive values
of d′ indicate that AI
texts are judged as more AI-like than human texts, negative values indicate the reverse.

Is the information that separates AI from human texts available to the models at all?
We first answered this question with linear probes, i.e., simple classifiers that are
trained on a model's internal activations to predict authorship and are evaluated on
held-out texts. At a fixed depth of two thirds of the network, probes separated AI from
human texts well in every model and corpus, with a median d′ of 1.35 for the short chats
(range across model conditions 0.95–1.80), 2.66 for the newspaper segments (1.65–3.14),
and 3.70 for the textbook summaries (2.36–3.73). When the authorship labels were
shuffled within groups, probe performance dropped to chance (mean AUC = .50), which
rules out that the probes merely memorized the training texts.

However, this information is not specific to the internal machinery of language models.
Classifiers trained on simple word frequencies reached the same or higher sensitivity
than the probes (short chats: d′ = 1.73 [1.44, 2.06]; newspaper segments and textbook
summaries: AUC ≥ .99[^auc]), and a probe on the average input embedding of the text, i.e.,
before any processing by the network, performed about as well as probes on deep layers
(short chats: d′ = 1.48). Holding out whole authors instead of single originals did not
change this picture (word frequencies: d′ = 1.79). A detector that was trained by
others to recognize AI-edited English text [EditLens; @thai2026editlens] transferred to
our German material in a graded manner, reaching an AUC of .99 for the textbook
summaries, .73 for the newspaper segments, and .57 for the short chats (Figure 3). Thus,
the cues that distinguish AI from human texts are present and readily accessible, both
in the texts themselves and in the models' representations; they consist largely of
ordinary lexical and stylistic regularities. In terms of Brunswik's lens model, the
environment offers valid cues.

These cues, however, are specific to the corpus. When a probe trained on the newspaper
segments was applied unchanged to the short chats, it separated them with the wrong
sign (Gemma 3 4B and 12B, six model conditions: median d′ = −0.46 at the last prompt
position and −0.48 for the averaged text positions; Qwen3: −0.48 and −0.60), and the
authorship direction of the news pointed slightly against that of the chats (cosine
about −0.2). In the opposite direction, probes trained on the chats transferred weakly
but with the correct sign (median d′ = 0.37–0.87). Hence, there is no single authorship
direction that holds across text types: Properties that mark AI texts in one corpus can
mark human texts in another. This is important for reading the answers below, because a
judge that applies one fixed policy to all texts can be above chance on one corpus and
below chance on another.

[^auc]: When nearly all pairs of texts are ordered correctly, the AUC-based d′ approaches
a maximum that depends on the number of texts (d′ = 3.73 for the textbook summaries and
3.14 for the newspaper segments); for such near-perfect classifiers, we therefore report
the AUC.

When the models were asked directly whether a text had been written by a human or
by an AI, their answers did not reflect this information. Table 2 summarizes the
order-averaged sensitivity of the answers for the pretrained base models and for the
final post-trained models. Many cells are close to chance, and a considerable number are
clearly below it: Qwen3-8B after post-training judged the newspaper segments with
d′ = −1.01 (raw prompt) and −1.30 (chat prompt), and Gemma 3 12B judged the short chats
with d′ = −0.81 and −1.09. Several models thus ordered the texts inversely: What
distinguished AI from human texts moved their answers toward the wrong class. Note that
these values were computed from the graded
preference of the model, i.e., the log-odds of "AI" versus "human" for each text, which
plays the role of a confidence rating in a rating-scale ROC analysis. The binary answer
itself often did not vary: In 23 of the 36 conditions with the plain prompt (Table 2,
base and final models), the model gave the same answer to more than 95% of the texts,
and in several base models the share of "AI" answers depended strongly on the order in
which the two options were named (e.g., OLMo 2 on the short chats: 89% vs. 0%). A
negative d′ therefore does not mean that the models gave many wrong binary answers.
It means that the graded evidence on which the answer is based was ordered the wrong
way round: human texts received a stronger preference for "AI" than AI texts did. Such
an inverse ordering cannot result from a response bias, which only shifts the
preference for all texts alike; properties typical of human texts pushed the answer
toward "AI", and properties typical of AI texts toward "human".

> **Table 2.** Sensitivity (d′) of the models' answers. Base models were queried with a
> plain completion prompt ("raw"); post-trained models with the same prompt and with
> their chat template ("chat"). Conditions in which a model placed less than 30% of its
> probability on the two answer words are omitted (—).
>
> | Base model | Corpus | Base (raw) | Final (raw) | Final (chat) |
> |---|---|---|---|---|
> | Gemma 3 4B | Chats | −0.21 | −0.72 | −0.66 |
> | | News | +0.48 | −0.34 | −0.28 |
> | | Textbook | +0.01 | −0.49 | −0.34 |
> | Gemma 3 12B | Chats | −0.36 | −0.81 | −1.09 |
> | | News | −0.02 | +0.07 | −0.01 |
> | | Textbook | +1.64 | +0.68 | +0.27 |
> | Qwen3 8B | Chats | −0.68 | −0.80 | −0.76 |
> | | News | −0.36 | −1.01 | −1.30 |
> | | Textbook | +0.55 | −0.02 | −0.27 |
> | OLMo 2 7B | Chats | −0.07 | +0.28 | +0.03 |
> | | News | −0.14 | −0.45 | −0.34 |
> | | Textbook | −0.12 | −0.44 | −0.16 |
> | Apertus 1.5 8B | Chats | −0.23 | −0.70 | −0.54 |
> | | News | +0.07 | −0.07 | −0.26 |
> | | Textbook | +0.43 | −0.04 | +0.03 |
> | OLMo 3 7B (Think) | Chats | −0.05 | −0.05 | −0.06 |
> | | News | +0.60 | +0.04 | −0.23 |
> | | Textbook | +0.01 | (+0.39)ᵃ | −0.32 |
>
> ᵃ Answer probability ≈ .30; interpret with caution.

The comparison between the columns of Table 2 already shows the pattern that we test in
the next section: On the identical plain prompt, the post-trained model was less sensitive
than its base model in 14 of the 18 comparisons.

## 3.2 Post-training lowers the sensitivity of authorship judgments

Base models were not uniformly better, but several of them discriminated clearly above
chance where their post-trained counterparts did not (e.g., Gemma 3 12B on the textbook
summaries, d′ = +1.64; OLMo 3 on the newspaper segments, +0.60; Qwen3 on the textbook
summaries, +0.55). To test the effect of post-training without confounding it with the
prompt format, we compared each final model with its base model on the identical plain
prompt. Of these 18 comparisons, 14 showed a decrease. Because each model family
contributes three corpora and each corpus six families, the 18 comparisons are not
independent. Our primary test therefore treats the model family as the unit: A t-test over
the six family means yielded a mean change of Δd′ = −0.34 [−0.56, −0.11] (t(5) = −3.82,
p = .012); all six families showed a decrease on average, two of them only slightly
(OLMo 2: −0.10; OLMo 3 Think: −0.06). A multilevel random-effects model with random effects
for model family, corpus, and cell gave the same estimate (−0.33 [−0.53, −0.13]); its
nominal test (t(17) = −3.54, p = .003) uses residual degrees of freedom and is likely too
liberal with six families. Heterogeneity between cells was considerable (I² = 83%). Restricted to the nine
comparisons in which both models placed at least half of their probability on the two
answer words, the estimate was similar but less precise (−0.31 [−0.64, 0.01], p = .057;
four families). In the three comparisons in which the
binary answers varied in both the base and the final model (Gemma 3 12B on the short
chats and the textbook summaries, Gemma 3 4B on the short chats), post-training lowered
d′ in all three cases (−0.45 to −0.95). Per corpus, with six families each, the decrease
was reliable for the newspaper segments (−0.42 [−0.75, −0.08], p = .024), but not for the
textbook summaries (−0.41 [−0.85, 0.04], p = .066) or the short chats (−0.19 [−0.55, 0.16],
p = .22).

Did the decline depend on the wording of the prompt? We repeated all comparisons with a
paraphrased instruction and question and with the question placed before the text. Across
the three prompts, the mean change over the six family means (each the mean of nine
comparisons) was Δd′ = −0.29 [−0.54, −0.03] (t(5) = −2.88, p = .035), with a decrease in
five of six families and in 39 of the 54 comparisons. A multilevel model with additional
random effects for the prompt gave the same estimate (−0.29), with variance components near
zero for prompt and corpus, but it is not informative as a test, because the prompt and the
corpus have only three levels each (t(2) = −2.81, p = .11). With the paraphrased prompt
alone, the mean change was −0.23 (13 of 18 comparisons negative; t(5) = −3.63, p = .015),
and the changes of the individual cells correlated with those for the original prompt
(r = .81). With the question placed before the text, the mean change was similar (−0.30;
12 of 18 negative), but two families changed direction (OLMo 3 Think: +0.38; Qwen3: +0.04),
so that the family-level test was not significant (p = .16; r = .63 with the original
prompt). Note that the wording moved the individual comparisons considerably: Between the
original prompt and the question-first prompt, Δd′ of a cell changed by 0.30 on average,
compared with a mean absolute effect of post-training of 0.43, and only 8 of the 18 cells
showed a decrease with all three prompts (5 of them reliably each time). The decline is
therefore a property of the average over models and prompts, not of every single cell. The
below-chance judgments of the short chats by Gemma 3 and Qwen3 persisted with all three
prompts.

Among the individual comparisons, six decreases survived Holm correction within this
family of 18 tests, five of them on the textbook summaries (Appendix G). Hence,
post-training lowered the sensitivity of the answers on average, but neither uniformly
nor in a single direction across models and text types.

For OLMo 2 and Apertus 1.5, the developers released checkpoints after each stage of
post-training, which allowed us to ask when the change occurs. In both models it
occurred at the first, supervised stage (supervised fine-tuning, SFT), in which the
model learns to imitate example answers. For OLMo 2 on the textbook summaries, d′ fell
from −0.12 (base) to −0.50 after SFT and then remained stable (−0.46 after preference
optimization, −0.44 after the final reinforcement-learning stage). For Apertus, d′ fell
from +0.43 to −0.04 after SFT and stayed near zero through the subsequent stages (0.00,
−0.03, −0.04). The later, preference-based stages thus added little to a change that
was already complete after imitation learning.

Could the post-trained models simply hold a wrong belief about what AI texts look like,
a belief that a few correctly labeled examples would correct? We tested this by placing
2, 8, or 16 labeled examples from other sources before each text (four different sets of
examples; d′ computed per set and averaged; Figure 4B). With 16 examples, d′ rose reliably
in 8 of 11 model conditions on the short chats and in 4 of 6 on the textbook summaries,
whereas two examples hardly helped. The gains of base and post-trained models did not differ
consistently: The difference was reliable in two of five comparisons, with opposite signs
(OLMo 2 on the chats, where the base model gained more, +0.54 [0.27, 0.83]; Qwen3 on the
textbook summaries, where the post-trained model gained more, −0.31 [−0.55, −0.09]; Table
E1). The gap between them persisted:
After 16 examples, the post-trained model was still less sensitive than its base model in
all five comparisons (e.g., Gemma 3 4B on the short chats: +0.48 vs. −0.27; Qwen3: +0.15
vs. −0.17), and the post-trained models of Gemma 3 4B and Qwen3 remained below chance on
the chats (Appendix E). Labeled examples do not
give a model a new ability; they only show which properties go with which label in the
material at hand. That they raised d′ in base and post-trained models alike therefore
indicates that part of the failure lies in which properties the models take as signs of AI
authorship, i.e., in a mapping that information can correct, and not in an inability to
tell the texts apart. Because the examples did not remove the difference between base and
post-trained models, however, such a correctable mapping does not account for the deficit
that post-training introduced.

Which properties of the texts did the answers follow? A lens-model analysis of the short
chats (Appendix B) showed that the answers of most post-trained models weighted 34 text
properties in the direction opposite to their validity (cross-validated matching index G:
Gemma 3 4B −.49, Gemma 3 12B −.58, Qwen3 −.61, Apertus after SFT −.57). They took long
words, adjectives, nouns, and low readability as signs of AI authorship, whereas the AI
messages in this corpus were more verbal and lexically more diverse than the human ones.
In Qwen3, this weighting was already present in the base model (−.53); OLMo 2, whose
answers on the chats improved slightly, weighted the properties in the opposite way after
SFT (+.37). The properties that the students had named as typical of AI text (e.g., dashes,
perfect capitalization) were barely valid in the truncated messages and barely used by
the models, and style-matched messages that had not been revised according to the
students' feedback were judged like the revised ones (change in AUC between −.03 and +.04
for every model).

## 3.3 Criterion, sensitivity, and the use of a stated base rate

Is signal detection theory an appropriate description of these judgments? Its central
assumption is that manipulations of the response bias move the criterion but leave the
sensitivity unchanged. This held approximately. Raising the stated share of AI texts from
20% to 80% moved the criterion in 48 of 72 comparisons (model × checkpoint × corpus × answer
order), in all of them toward "AI" and reliably so in 42 (mean |Δc| across all comparisons =
1.19); in the remaining 24 comparisons, c did not change, because the model gave the same
answer to nearly every text at either share. The sensitivity was not strictly invariant,
however: d′ changed by 0.14 on average, without a consistent direction, and the bootstrap
interval of the change excluded zero in 28 of the 72 comparisons. In the 14 comparisons in
which no condition was saturated, the criterion shifted reliably in all 14 and d′ in 3 (mean
|Δd′| = 0.10). The order of the answer options and the mere presence of the base-rate
sentence also acted mainly on the criterion (mean |Δc| = 0.78 and 0.42), and changed d′ by
similar amounts (0.16 and 0.13, again without a consistent direction; Appendix F). Thus,
manipulations of the response bias acted mainly on the criterion where the criterion could
move, and the criterion is a property of the model together with the prompt rather than of
the model alone. Because the accompanying changes of d′ were not negligible relative to the
effects reported below, we average d′ over both answer orders and compare checkpoints only on
identical prompts.

How strongly did the models use the stated base rate? Any observer who takes the stated
share into account should answer "AI" more readily when it is higher. A reference point
for the size of this effect is an ideal observer in a stationary environment, for whom the
stated share is the prior and changes nothing else: The log-odds of the answer then shift
by the same amount for every text, ln 16 = 2.77 nats between 20% and 80%. We use this value
as a reference, not as a norm that a sensible judge must meet (in a changing environment,
for instance, weighting a stated base rate less can be adequate), and it applies to dv only
insofar as dv is read as the model's posterior log-odds. The base
models shifted far less (mean shift of dv per model between 0.18 and 1.40 nats, i.e., 6–51%
of the benchmark), and the shift was nearly uniform across texts (between-text SD within a
condition 0.08–0.33). After post-training, five of the six models shifted more than the
benchmark (3.67–10.93 nats, i.e., 1.3–3.9 times the benchmark), and the shift differed
considerably between texts (SD up to 3.14); OLMo 3 Think was the exception (1.21 nats, as
in its base model, 1.17). These values were obtained with a base-rate sentence that named the share of AI texts
first. For the three families that fit the available GPUs, we repeated the manipulation
with the human share named first. The pattern was the same, but the size of the shift
depended on the wording in the post-trained models: The base models again shifted too
little (Gemma 3 4B: 0.16; OLMo 2: 1.35; OLMo 3: 1.07 nats), the post-trained models of
Gemma 3 4B and OLMo 2 again shifted more than the benchmark, but less than with the first
wording (5.61 vs. 10.65 and 4.37 vs. 5.39 nats), and OLMo 3 Think again stayed below it
(1.48). Post-training thus changed the use of a stated base rate from underweighting to
overweighting in five of six model families, by a factor that depended on the model and on
the wording (Table F2); the second wording was not run for Apertus, Qwen3, and Gemma 3 12B.

Note that a shift in nats also depends on the overall scale of a model's log-odds, and this
scale differed between checkpoints: Across texts of the same class, dv varied far more in
the post-trained models (within-class SD per model 0.61–2.89 nats) than in the base models
(0.21–0.53). Relative to this spread, i.e., to the differences that the models make between
texts, the stated base rate already moved the base models considerably (by 0.9–3.3 SD per
model) and the post-trained models somewhat more (2.0–5.3 SD); the relative shift increased
with post-training in five families and decreased in OLMo 3 Think (3.3 to 2.0 SD), and with
the second wording it increased only in Gemma 3 4B (0.8 to 4.2 SD), not in OLMo 2 (2.5 to
2.4 SD; base_rate_scale.py). Thus, the post-trained models gave more extreme log-odds in
general, to the base rate as well as to the texts; that the base rate gained weight relative
to the texts holds for most, but not all, models and wordings.

## 3.4 Where the change acts: model stitching and activation patching

### Model stitching

Where in the network does post-training make the difference? Because the base and the
post-trained checkpoints of a model share the same architecture and start from the same
weights, we could combine them: the lower layers from one checkpoint and the upper layers
from the other, passing the internal state at the junction on unchanged ("model
stitching"). This test requires no probe and no statistic derived from the answers; it
only measures how well the combined model discriminates. Two observations justified the
procedure. First, the representations of the texts themselves were highly similar in
the base and the post-trained checkpoints of the three models with stage-wise checkpoints
(linear CKA across layers 0.97–0.999 for OLMo 2, Apertus and OLMo 3; textbook summaries),
whereas the representations at the position where the answer is formed differed
considerably (0.44–0.99, median 0.65–0.85). Note that linear CKA without bias correction,
computed here on 238 texts in about 4,000 dimensions, is biased upward, so that values near
1 are weak evidence of identity; the contrast between the text positions and the answer
position is the more informative result. Note also that this was less so for the other
families: In Gemma 3, even the text representations changed noticeably, most at mid-depth
(0.42–0.99, median 0.80–0.83), and the answer position differed strongly (median
0.20–0.31), so that for Gemma 3 the claim that the text positions changed little rests on
the probe transfer alone (Section 3.5). Second, every stitched model produced fluent
text (perplexity 6.0–6.7 for Apertus, 8.2–9.4 for OLMo 2), and, as a check of the implementation, stitching at the very
bottom or the very top reproduced the respective parent model.

If post-training changed only the final readout stage, the upper layers should decide
the outcome. This was not the case (Figure 5A). When the first 8 of 32 layers of
Apertus came from the base model and all layers above from the SFT model, the combined
model discriminated at least as well as the base model (d′ = +0.59 [0.41, 0.80]; base
+0.43). Conversely, the first 8 layers of the SFT model sufficed to abolish the base
model's discrimination, although all 24 layers above came from the base model (−0.50
[−0.74, −0.25]). For OLMo 2, the first 16 base layers restored base-level performance
under SFT upper layers (−0.01 [−0.15, 0.14]; base −0.12, SFT −0.50), and the first 6
SFT layers were sufficient to produce the full deficit (−0.48). Finer steps showed that
the change builds up gradually over several early layers rather than in a single one
(Apertus: layers 3–8; OLMo 2: layers 1–14).

Note, however, that the hybrids did not always lie between their parents. In Apertus,
the first 8 SFT layers under base upper layers judged worse than the SFT model itself
(−0.50 vs. −0.04), with a reduced probability on the two answer words (.39 vs. .68 for
the base and .83 for the SFT model), and the first SFT layer alone under base upper
layers judged better than either parent (+0.73). Were these effects artifacts of
stitching as such? As a control, we stitched the same checkpoints on a judgment that
post-training had changed little, the truth of statements on diverse facts (counterfact;
Apertus: d′ = 1.38 base vs. 1.58 SFT; OLMo 2: 1.77 vs. 1.75). There, all hybrids stayed
within or close to the range of their parents (largest deviation from the unstitched parents 0.02 for Apertus and
0.08 for OLMo 2; none of 14 Apertus hybrids and one of 14 OLMo 2 hybrids reliably outside),
whereas on the textbook
summaries 6 of 18 Apertus hybrids and 4 of 18 OLMo 2 hybrids lay reliably outside
(largest deviation 0.46 and 0.15; outputs/stitch/stitch_summary.csv). Stitching thus
does not disturb the models' judgments in general. The control does not show, however, that
the overshoot is specific to the judgment that post-training changed: The control judgment
was far from chance, with high answer probability, whereas the textbook judgments were close
to chance, where small perturbations can reorder the texts. A judgment near chance that
post-training left unchanged would be the appropriate control, which we did not run. The
stitching results show that the early-layer weights of a checkpoint suffice to carry the
difference into an otherwise foreign network; they do not by themselves show that the
difference is computed there.

On the short chats, on which post-training lowered the sensitivity of Apertus (−0.26 to
−0.78) and raised that of OLMo 2 (−0.05 to +0.37), the early locus replicated in one
direction only. The first 8 base layers of Apertus restored the base level under SFT upper
layers (−0.27), and the first 4 base layers of OLMo 2 removed most of its gain (+0.10). The
first 8 SFT layers under base upper layers, however, did not produce the effect of SFT in
either model (Apertus: −0.31; OLMo 2: −0.07); it appeared only with 16 SFT layers (−0.70
and +0.27). On the chats, the early base layers were thus sufficient to remove the effect
of SFT, but the early SFT layers were not sufficient to produce it, in contrast to the
textbook summaries. No hybrid lay outside the range of its parents on the chats. Note also
that the effect to be localized is small for OLMo 2 on this corpus.

Both sublayer types, attention and feed-forward blocks, contributed to the effect of the
early layers (Appendix H).

### Activation patching at layers 8 and 16

The stitching results implicate the early layers, but do not yet say where in the
prompt the change acts.
In a language model, the answer can only access the text through the positions that
follow it, i.e., the question "Was this text written by a human or generated by an AI?"
and the answer cue. We therefore exchanged the internal states between base and SFT
checkpoints selectively at different positions of the prompt ("activation patching"):
at the tokens of the text, at the instruction before the text, at the question after
the text, or at the final position only.

At layer 16, the pattern was unambiguous and identical for both models (Figure 5B, which
shows the positions exchanged at this layer). Exchanging the states of the text tokens changed nothing: Apertus after SFT
remained at chance when it received the base model's text representation (−0.11; SFT
alone −0.04), and the base model kept its sensitivity when it received the SFT model's
text representation (+0.43; base alone +0.43). Exchanging only the states of the
question positions transferred the full difference in both directions: the base model's
question states restored discrimination in the SFT model (Apertus: +0.41; OLMo 2: +0.04,
from −0.50), and the SFT model's question states abolished it in the base model
(Apertus: −0.42; OLMo 2: −0.61, from −0.12). The final position alone carried part of
this effect in Apertus (−0.08 and −0.15) and in OLMo 2 most of it in one direction
(−0.06 from −0.50) and about half in the other (−0.34 from −0.12). The same held for the short chats, on which
exchanging the text states changed nothing (all changes ≤ 0.04), whereas the states after
the text transferred the difference: The SFT states made the base models judge like the
SFT models (Apertus: −0.23 → −0.75, SFT −0.73; OLMo 2: −0.07 → +0.34, SFT +0.38), and the
base states restored the base level in Apertus (−0.73 → −0.27) and partly in OLMo 2 (+0.38
→ +0.20). Within the positions after the text, the tokens of the question itself carried
little and the answer cue most of the difference (e.g., Apertus base with SFT states at the
question tokens: −0.46; at all positions after the text: −0.75); the final position alone
carried between a third and two thirds. Because post-training improved the judgments of OLMo 2 on the chats, the
change resided at the positions after the text regardless of its direction. At layer 8, the picture was mixed (Figure S1, which adds this layer): in Apertus, the text
positions still had a partial effect, and in OLMo 2 even exchanging all positions
transferred only part of the difference, consistent with the stitching result that in
OLMo 2 the change accumulates up to about layer 14. Thus, at the middle of the network,
exchanging the states at the positions after the text (question and answer cue) transferred
the difference, and exchanging the text states did not.

### Activation patching across layers

Patching at two layers can miss a contribution that is confined to other layers, as the
negated statements show (Section 3.6). We therefore repeated the patching on the textbook
summaries at 12 layers between 2 and 32 (Figure 6; outputs/patch_sweep/patch_sweep_summary.csv).
In OLMo 2, the text states did not transfer the difference at any layer, in either
direction: The base model with the SFT model's text states stayed between −0.17 and −0.05
(unpatched −0.12), and the SFT model with the base model's text states between −0.61 and
−0.44 (unpatched −0.50). The states after the text transferred the difference from layer 12
onward (base model with SFT states: −0.50 at layer 12 and −0.59 at layer 14; SFT model with
base states: −0.02 at layer 14). Below layer 12, only the exchange of all positions had a
clear effect (e.g., −0.48 for the base model at layer 6), so that in the lower layers the
difference was not carried by any single set of positions. In Apertus, in contrast, the
text states carried part of the difference in the lower half of the network: With the SFT
model's text states, the base model fell from 0.43 to 0.21–0.33 at layers 2 to 12 (22–46%
of the difference between the checkpoints), and with the base model's text states, the SFT
model rose from −0.04 to 0.10–0.26 at layers 4 to 12 (30–64%); the bootstrap interval of
the transferred share excluded zero at layers 6 to 12 in the first direction and at layers
4, 8, 10, and 12 in the second. From layer 14 onward, the text states had no effect in
either direction. The SFT model's states after the text lowered the base model from layer 4
onward, by more than the whole difference between the checkpoints (e.g., to −0.48 at layer
12, compared with −0.04 for the SFT model), which again shows that hybrids produce effects
of their own; in the other direction, the base model's states after the text restored the
SFT model partly at layers 6 to 12 (0.09–0.31) and fully from layer 14 onward (0.61). Because the text
precedes the question in this prompt, the contribution of the text states in Apertus
cannot be specific to the judgment; it reflects a change in how the lower layers encode
the text in general, in line with the stitching result that implicated layers 3 to 8 in
this model. Thus, from layer 14 onward the difference between the checkpoints was carried
by the positions after the text in both models; below, the text states contributed in
Apertus, but not in OLMo 2.

### Question placed before the text

Note that in our prompt the text precedes the question, so that its states are computed
before the question is read, and a change that is specific to the judgment can only
appear at the question positions. We therefore repeated the patching with the question
placed before the text, so that the text was read with the task known. The decline
persisted with this prompt (Apertus: +0.36 base vs. −0.19 SFT; OLMo 2: +0.09 vs. −0.42).
Table H1 (Appendix H) shows the change of d′ for each set of positions (layer 16 only). Exchanging the
text states changed d′ by at most 0.12 (3–22% of the difference between the checkpoints).
The few positions after the text ("Antwort:") transferred the whole difference in one of
the four cases (Apertus SFT with base states: −0.19 → +0.44, more than the difference
itself) and 42–49% in the other three; the question before the text transferred 18–53% in
three cases and moved the Apertus base model away from the SFT model (+0.36 → +0.55). The
effects of the single sets did not add up to the effect of exchanging all positions, so
that the states before, within, and after the text contributed jointly. Note also that a
missing effect of the text states at layer 16 does not show that they are irrelevant, because
the later positions have by then read the recipient's own text states in the layers below;
with the default prompt at layer 8, the text states did have an effect in Apertus. We
therefore conclude only that, at mid-depth, the difference between the checkpoints was not
carried by the text states alone, also when the text is read with the question known. In
summary, exchanging the text states at mid-depth transferred little of the change, whereas
exchanging the states at the positions that follow the text transferred much or all of it
with the default prompt; in this sense, post-training changed how the question takes up the
text.

## 3.5 Probes across checkpoints

If post-training leaves the
representation intact and changes how it is read out, a probe trained on the base model
should work on the post-trained model at the text positions, but less so at the position
where the answer is formed. This was the case wherever we could test it (Figure S2), with
qualifications. The comparison of the two sites exists only for the authorship corpora (all
six families); for the other judgments, the activations were stored at the last position
only, for OLMo 2 and Apertus (all three sets) and for Gemma 3 4B and Qwen3 (reviews). At the text positions of the authorship corpora, probes transferred
between the base and the final checkpoints with little loss: Averaged across models,
the transfer deficit stayed at or below 0.15 d′ at every depth, and 94% of the single
model × layer values were at or below 0.10 (largest value 0.54, for OLMo 2 on the
textbook summaries in the top layers). Because the probes at the text positions were at
their maximum in 89% of these values for the textbook summaries and in 67% for the
newspaper segments, the test can only fail where they are not, i.e., mainly for the short
chats (within-checkpoint d′ = 1.46–2.48); there, the largest deficit of any model was
0.16. At the last prompt position, the deficit grew with depth, reaching on average 0.7
to 1.9 d′ in the upper third of the network for the authorship corpora and 0.5 to 1.1 d′
for the other judgments (with one exception: for OLMo 2 on the negated statements, the
upper layers showed no deficit). Probes still transferred partially at this position
(mean d′ in the upper third up to 2.4, but near zero for Gemma 3 4B on the newspaper
segments), so the readout position was reorganized rather than replaced. Probes that were trained
and tested on the same model version (base or post-trained) continued to decode the information at the last position, but in
four of the six families less well after post-training on at least one corpus (upper third
of the network, final vs. base model: Gemma 3 4B, −0.52 on the newspaper segments and −0.90
on the textbook summaries; Gemma 3 12B, −0.57 and −0.47; OLMo 3 Think, −0.41 and −0.70;
Qwen3, −0.38 on the textbook summaries). In Apertus, the change was mixed (chats −0.26,
newspaper segments +0.43), and in OLMo 2 there was no decline
(checkpoint_transfer_last_summary.py). Post-training thus left the text positions
essentially unchanged where the test is informative, and both reorganized and, in some
models, degraded the information available at the answer position.

In summary, for authorship the information was represented, the probes at the text
positions changed little with post-training, and from the middle of the network onward the
states at the positions after the text transferred the change between checkpoints. In the
lower layers, the text states contributed to the change in Apertus, but not in OLMo 2.

## 3.6 A reproducible case with the opposite direction: negated statements

The authorship corpora cannot be shared. Is the dissociation between representation and
answer, and its modification by post-training, specific to them? We repeated the central
analyses with truth judgments on public statement sets (Section 2.8), which can be
reproduced in full (`reproduce/negation.sh`). Here, post-training acted in the opposite
direction: It repaired answers that the base models gave poorly.
Table 3 summarizes the results.

> **Table 3.** Truth judgments: sensitivity (d′) of the probes (last prompt position, two
> thirds of the network depth) and of the answers (plain prompt) for statements on diverse
> facts (counterfact) and for negated geographic statements. SFT: answers after supervised
> fine-tuning; —: no intermediate checkpoint released. The maximum d′ (AUC = 1) is 4.54 for
> the 748 statement pairs.
>
> | Judgment | Model | Probe base | Probe final | Answer base | Answer SFT | Answer final |
> |:---|:---|---:|---:|---:|---:|---:|
> | Truth | OLMo 2 7B | 2.15 | 2.05 | 1.79 | 1.77 | 1.84 |
> | (diverse facts) | Apertus 1.5 8B | 1.95 | 2.09 | 1.41 | 1.64 | 1.68 |
> | | OLMo 3 7B Think | 1.89 | 1.64 | 1.61 | 1.35 | 1.34 |
> | | Gemma 3 4B | 1.82 | 1.93 | 0.81ᵃ | — | 1.46 |
> | | Gemma 3 12B | 2.24 | 2.26 | 1.64ᵃ | — | 1.97 |
> | | Qwen3 8B | 2.28 | 2.15 | 1.94 | — | 1.77 |
> | Truth | OLMo 2 7B | 4.54 | 4.54 | 0.80 | 3.36 | 3.10 |
> | (negated statements) | Apertus 1.5 8B | 4.54 | 4.54 | −0.88 | 4.54 | 4.54 |
> | | OLMo 3 7B Think | 4.54 | 4.31 | 4.54 | 3.77 | 3.80 |
> | | Gemma 3 4B | 4.54 | 4.34 | 0.06ᵃ | — | 3.46 |
> | | Gemma 3 12B | 4.54 | 4.54 | 1.97ᵃ | — | 4.54 |
> | | Qwen3 8B | 4.54 | 4.54 | 4.54 | — | 4.54 |
>
> ᵃ The Gemma 3 base models placed only 18–32% of their probability on the two answer
> words; interpret with caution.

Table 3 lists, for all six model families, the base model and the final post-trained
(instruction-tuned) model, and for the three families with stage-wise checkpoints also the
model after supervised fine-tuning. For simple geographic statements, all of these judged
at ceiling, so that this set served only as a check. For the negated statements, however, two base
models showed a clear dissociation between representation and answer: Probes decoded
the truth of every statement (AUC = 1), but the answers of the OLMo 2 base model used
this information only weakly (d′ = 0.80), and those of the Apertus base model ordered
the statements inversely (d′ = −0.88). Both base models answered "False" almost always
(criterion c = 2.6 and 3.0). Supervised fine-tuning repaired the answers of both models, whereas the probes remained
at ceiling, and the final post-trained models kept this level. The Gemma 3 models showed
the same pattern between their base and final versions, although their base models rarely
answered with the two answer words. The base models of Qwen3 and OLMo 3, in contrast, already judged the
negated statements perfectly; post-training left this unchanged in Qwen3 and lowered it
slightly in OLMo 3.
Thus, post-training repaired the readout of negation where the base model read it
wrongly, but did not improve it further where the base model read it correctly. A probe trained on the
simple statements of the same checkpoint suggests what the base models' answers followed:
Applied to the negated statements, it separated them with the wrong sign (OLMo 2:
d′ = −1.28; Apertus: −1.43), i.e., the general truth direction of the base models
encoded whether city and country belong together and ignored the negation; whether the
answers followed this direction item by item was not tested. After SFT, the
same probe separated the negated statements in the correct direction (+0.69; +3.05). For
the statements on diverse facts, post-training changed the answers only slightly, except
in OLMo 3, where answers and probes declined together (answers 1.61 → 1.34; probes 1.89
→ 1.64).

Why did post-training change the answers for the negated statements but hardly for the
statements on diverse facts? The two sets differ in what limits the answers. The statements
on diverse facts are affirmative, and judging them requires only that the fact is known:
The probes were far from their maximum, and the answers of the base models were already
close to the probes (e.g., OLMo 2: d′ = 1.79 for the answers vs. 2.15 for the probe), so
that there was little represented information that the answers failed to use. The negated
statements require that the fact is combined with the negation. The probes show that this
combination is available at the last prompt position of every model, but the answers of
several base models did not reflect it. Post-training thus changed the answers where they
fell short of what was represented, not where they were limited by what the model knows.

Where does the repair arise? We combined the lower layers of one checkpoint with the upper
layers of the other (model stitching; Section 2.7, Figure 2C; results in Figure 7A). Stitching located the repair in the middle of the network
rather than in the early layers. In OLMo 2, the first 8 base layers combined with all
upper SFT layers still judged at the level of the SFT model (d′ = 3.76; base 0.63, SFT
3.20 in this subset of 300 pairs), whereas the first 16 base layers abolished the
repair (0.38). In Apertus, the first 8 base layers under SFT upper layers likewise left
the repair intact (4.15), the first 12 removed most of it (1.33), and the first 16 all of it
(−0.94); conversely, the first 8 SFT layers under base upper layers produced part of the
repair (1.21) and the first 12 all of it (4.15). The repair thus involved layers 9 to 16,
with a partial contribution of the layers below (outputs/stitch/stitch_neg_cities.csv). To ask where in the prompt the
repair acts, we exchanged the internal states between the checkpoints at selected positions
of the prompt (activation patching; Figure 2D). Note that in the prompt for the truth
judgments the question precedes the statement, so that the statement is read with the task
known, and the positions after the statement contain no question, only the instruction to answer
with one word and the answer cue ("Answer:"). At
layer 16, the SFT model's states at the positions after the statement repaired the base
model completely (OLMo 2: 0.63 → 3.26; Apertus: −0.87 → 4.15), and the base model's states
at these positions abolished the repair in the SFT model (3.20 → 1.07; 4.15 → −1.09). In
Apertus, the final position alone carried the full effect in one direction (base model with
the SFT model's final state: 4.15). Exchanging the states of the statement itself had no
effect in Apertus and in the SFT model of OLMo 2, but a partial one in the OLMo 2 base
model (0.63 → 1.51). At layer 8, the SFT model's states after the statement repaired the
base models only partly (Apertus: 1.61; OLMo 2: 1.07; Figure 7B), in line with the
stitching result that the repair is not complete at this depth. Exchanging the states of
the instruction and question before the statement had no effect at either layer.

Patching at two layers cannot show where the repair first appears. We therefore repeated
the patching at 12 layers between 2 and 32, with 100 statement pairs (Figure 7C;
outputs/patch_sweep/patch_sweep_summary.csv; on this subset, d′ = 0.34 for the OLMo 2 base
model and 3.09 for its SFT model, and −0.86 and 3.64, the maximum for 100 pairs, for
Apertus). We express each effect as the share of the difference between the two
checkpoints that the patch transferred. The two models differed. In OLMo 2, the repair
appeared first at the statement positions: The SFT model's statement states alone
transferred 22% of the repair at layer 10, 82% at layer 12 (d′ = 2.59), and 67% at layer
14, although all layers above and all other positions were those of the base model. The
share then fell to 33% at layer 16 and to zero from layer 20 onward. The positions after
the statement showed the complementary course (36% at layer 10, 37% at layer 12, 52% at
layer 14, 97% at layer 16), and the final position alone carried the repair only from layer
20 onward (89%). The reverse direction gave the same picture: The base model's statement
states lowered the SFT model from 3.09 to 1.38 at layer 12 and to 1.75 at layer 14, and had
little effect at layer 16 (2.92) and none above. In OLMo 2, supervised fine-tuning thus changed how the
statement itself is represented at mid-depth, in a way that the upper layers of the base
model could read correctly; the information then passed to the answer cue and to the final
position. In Apertus, in contrast, the SFT model's statement states transferred at most
23% (layers 12 and 14), whereas its states after the statement transferred 46% at layer 6
and 96% at layer 10. In the reverse direction, the base model's states after the statement
lowered the Apertus SFT model progressively from layer 10 onward, and its statement states
did not lower it at any layer; note that the Apertus SFT model was at the maximum d′, so
that a small disruption would not be visible. Hence, the two models agree in that the
repair has reached the positions after the statement by the middle of the network, but
they differ in where it arises: at the statement positions in OLMo 2, and at the answer cue
in Apertus.

To see which of the two patterns is more typical, we ran the same sweep for the two Gemma 3
models, comparing the base model with the final post-trained model at 12 layers each. Note
that these base models placed only 19% and 26% of their probability on the two answer
words, and that the post-trained Gemma 3 12B, like the Apertus SFT model, was at the maximum
d′. In both models, the states after the statement carried the repair completely from about
half of the depth onward (Gemma 3 4B, 34 layers: 75% at layer 17 and 97% at layer 20;
Gemma 3 12B, 48 layers: 100% from layer 20). Below that depth, the statement states
contributed in both models, but in different ways. In Gemma 3 4B, the post-trained model's
statement states repaired the base model only slightly (at most 19%, at layer 11), whereas
the base model's statement states abolished the repair in the post-trained model almost
completely at layers 11 and 14 (d′ = 0.09 and 0.20, from 3.42). In Gemma 3 12B, the
post-trained model's statement states transferred up to 45% of the repair (layer 12), and
the base model's statement states did not lower the post-trained model at any layer. An
isolated effect of the positions before the statement in Gemma 3 12B (layer 16: d′ = 3.47)
did not occur at the neighboring layers. Across the four models, the states after the
statement thus carried the repair from about the middle of the network onward, and in
three of them (OLMo 2 and both Gemma 3 models) the representation of the statement
contributed to it at about a third of the depth in at least one direction.

Because the question precedes the statement in this prompt, we also posed the judgment with
the statement placed before the question. The pattern of the answers was the same, and
more pronounced: The base models ordered the negated statements at or below chance (OLMo 2:
d′ = −0.14 [−0.24, −0.05]; Apertus: −2.13 [−2.29, −1.99]), and the SFT models judged them
nearly as with the default prompt (3.42 and 4.54; truth_variant_answers.py). The repair by
supervised fine-tuning thus does not depend on the question being known when the statement
is read. Patching across layers with this prompt (same 12 layers, 100 statement pairs)
gave the same localization as with the default prompt. In OLMo 2, the SFT model's statement
states alone transferred 36%, 75%, and 61% of the repair at layers 10, 12, and 14 (base
model: d′ = 0.85, 2.14, and 1.69, from −0.36; SFT model 2.99), and the base model's
statement states lowered the SFT model at the same layers (to 2.13, 1.25, and 1.71); the
states after the statement carried the repair completely from layer 16 onward (3.09), and
the question line by itself transferred nothing in either direction. In Apertus, the SFT
model's statement states did not repair the base model at any layer (at layers 8 to 12
they even lowered it, to between −3.01 and −2.64, from −2.11), whereas its states after the
statement transferred the repair partly from layer 4 and completely from layer 12 onward.
Because the statement is read before the question in this prompt, the contribution of the
statement states in OLMo 2 cannot reflect a reading of the statement in the light of the
task: Supervised fine-tuning changed how this model encodes a negated statement as such.

Steering at the answer position was consistent with a readout that uses the represented
truth direction after SFT; as explained in Appendix D, this is a consistency check rather
than an independent test. Shifting
the internal state at the answer position along the direction that separates true from
false negated statements (±2 SD, layer 21; Appendix D) moved the answers of both SFT
models toward the correct answer, more than the control directions (Apertus: +0.36 nats,
p = .02; OLMo 2: +1.07, p ≤ .04), and in these models the truth direction coincided with
the decision direction (cosine ≈ 1.0). In the base models, it did not (Apertus: −0.14,
p = .13–.33; OLMo 2: +0.09, p ≥ .54), although in both base models the positive control,
too, exceeded only the norm-matched controls. Supervised fine-tuning thus aligned the answer with
the represented truth direction; whether the Apertus base model used this direction with
the wrong sign, as its negative sensitivity would suggest, could not be established.

In summary, the truth of negated statements could be decoded at ceiling from the last
prompt position before and after post-training, the answers of several base models did not
use this information, and supervised fine-tuning repaired the answers, through weights in
the middle layers. In all four models in which we could trace the repair, it resided at the positions after
the statement from about the middle of the network onward. Where it arose differed: In
Apertus, at these positions, i.e., in how the answer cue takes up the statement; in OLMo 2,
in the representation of the statement itself; in the two Gemma 3 models, the
representation of the statement contributed as well. Note that the Gemma 3 base models
answered with low answer probability, and that two base models had nothing to repair.

What do the training data teach? The direction of the change differed between the
judgments: Post-training impaired the readout of authorship, repaired that of negated
statements, and reshaped that of the reviews. To examine whether this corresponds to
the supervised fine-tuning data, we searched the SFT mixtures of OLMo 2 and Apertus. Both
contained thousands of requests to judge statements as true or false (Tülu 3: 2,685;
Apertus, extrapolated: about 2,300), largely factual claims in Tülu 3 and mathematical
statements with explicit reasoning, often including negations, in Apertus. In contrast,
we found no request to judge whether a text had been written by a
human or by an AI: The search pattern matched five conversations in Tülu 3 and none in the
Apertus sample, and on inspection none of the five posed this judgment (they concerned,
e.g., a training script for an audio classifier and summaries of research papers; the matches
are listed in outputs/sft_check/). The matches for fake or deceptive reviews were sentiment tasks or
summaries of research papers. Because our judgments were posed in German, we repeated the
search with German patterns in the conversations written in German, which made up only
0.1% of Tülu 3 (992 conversations) and 1.7% of the Apertus sample (6,330). They
contained no request to judge authorship and no deception judgment, two mentions of
AI-generated text (Apertus), and six true/false questions. Note that a search with regular
expressions can miss requests that are phrased differently, and that the German subsets are
so small that their search has little power. The census is thus consistent with the idea that post-training improved the one judgment
for which the training data contain examples, where the base model read it wrongly, and
impaired or merely reshaped the judgments for which we found none; on its own, it cannot
establish this.

## 3.7 The training signal sets the direction of the readout

The analyses so far show that post-training changed the readout and that the direction of
the change differed between judgments; whether it depends on the training data could only
be inferred. We therefore manipulated the training data directly (Section 2.9). Table 4
summarizes the answers.

> **Table 4.** Training study: sensitivity (d′) of the answers of OLMo 2 7B after
> fine-tuning with background conversations only (C0) or with additional authorship
> judgments with correct (C1), stereotyped (C2), or random labels (C3), or with an
> unrelated topic judgment in the same layout (C4), and of the official SFT checkpoint.
> Held-out news: 400 items from the distribution of the training texts. d′, mean over
> three seeds (per-seed d′, both answer orders averaged). C0 and C4 did not learn the
> answer words of the authorship question (probability of the two answer words .65–.76, vs.
> ≈ 1.0 in C1–C3); C3 therefore serves as a further baseline (see text).
>
> | Test corpus | C0 background | C1 valid | C2 stereotype | C3 random | C4 other task | Official SFT |
> |---|---:|---:|---:|---:|---:|---:|
> | Held-out news | −0.30 | 4.28 | 0.11 | 0.13 | 0.04 | −0.14 |
> | Newspaper segments | −0.06 | 2.18 | −0.35 | −0.06 | −0.19 | −0.54 |
> | Textbook summaries | −0.12 | 1.39 | 0.32 | 0.03 | −0.03 | −0.50 |
> | Short chats | 0.04 | −0.58 | −0.29 | −0.21 | 0.04 | 0.38 |

Because the background condition C0 did not learn to answer with the two answer words,
whereas the three conditions with authorship examples did, we compared each condition with
three baselines: C0; C3, the condition with random labels, which learned the question and
the answer words but no valid mapping; and C4, which practiced an unrelated judgment in the
same layout (differences in d′ with 95% CIs from a grouped bootstrap that resamples texts and is
conditional on the three seeds per condition). Practicing the
unrelated judgment left the authorship answers where C0 left them (C4 − C0 between −0.13
and +0.10 on the three test corpora, all CIs including zero; on the held-out news, C4 was
higher than C0, +0.34 [0.26, 0.43]), so that practice with a
judgment as such did not explain the effects of the authorship labels. Note that the C4
models never answered "KI" in any run, so that their d′ is a ranking of graded preferences
under a saturated criterion, and that C3 was unstable across seeds (criterion c between
−1.10 and +2.33). Two thousand correctly labeled judgments of news texts sufficed to raise the
sensitivity of the answers far above both baselines, not only on held-out news but also
on texts from other generators and genres: the newspaper segments, written by GPT-5.4
(C1 − C3: +2.23 [1.56, 3.17]; C1 − C4: +2.37 [1.74, 3.29]), and the textbook summaries
(+1.36 [1.11, 1.64]; +1.42 [1.15, 1.73]). On the
held-out news, C1 was at the maximum for both generators and all text lengths
(d′ = 3.8–4.0 are the ceilings of the subsets), so that its success there was not due to recognizing one
generator or one length; the breakdown cannot exclude that the models detect the generation
procedure that the training and held-out texts share. The information had been represented all along; a small amount
of valid training made the answer read it. On the short chats, however, the same training
made the answers worse than chance (−0.58), also relative to random labels (C1 − C3:
−0.36 [−0.69, −0.06]) and to the unrelated judgment (C1 − C4: −0.62 [−1.01, −0.25]), because the properties that distinguished generated from human
news did not distinguish generated from human chat messages in the same way.

The stereotype acted according to its validity, but only partly reliably. Measured with
the formality score used for the labels, AI texts were less formal than human texts in
the chats and newspaper segments (AUC = .40 [.32, .49] and .39 [.28, .49]) but more
formal in the textbook summaries (.70 [.65, .75]). Relative to random labels, the
stereotype improved the answers on the textbook summaries (C2 − C3: +0.29 [0.11, 0.49];
C2 − C4: +0.35 [0.15, 0.57]) and lowered them on the newspaper segments, though not
reliably (−0.29 [−0.61, 0.07]; −0.15 [−0.57, 0.25]); on the chats, the result depended on the baseline: The stereotype did not differ from
random labels (−0.08 [−0.39, 0.24]) and was lower than the unrelated judgment, though not
reliably (C2 − C4: −0.33 [−0.77, 0.08]), because random labels themselves lowered the answers
on the chats. Note also that the
stereotype labels were not independent of other text properties: The more formal half
contained more short texts, more texts on business and domestic politics than on sports,
and more texts from Gemma 3 12B than from Qwen3 8B. Random labels lowered the answers
slightly on the chats relative to C0 (−0.26 [−0.52, 0.00]). Note that the background condition did not reproduce the effect of the
official fine-tuning. The official SFT, which updated all weights on the full mixture of
about 940,000 conversations, lowered the sensitivity on the newspaper segments and
textbook summaries (d′ about −0.5) and shifted the criterion strongly toward "human";
our background condition, with 2% of the conversations, one epoch, and low-rank adapters,
left the judgments close to those of the base model. The conditions therefore show that a
training signal can set the direction of the readout against a neutral background; they
do not yet show that the decline after the official fine-tuning has the same origin.
Fine-tuning with a larger share of the original mixture, or removing and enriching
specific parts of it (e.g., conversations in which the assistant talks about AI), could
identify which training data install the misreading.

Did the training change the representation of the texts or the way it is read out? Probes
trained within each fine-tuned model decoded authorship as well as in the base model
(e.g., newspaper segments, mean of layers 20 and 24, i.e., about two thirds of the
depth: d′ = 3.14 in the base model vs.
2.93–3.14 after fine-tuning; chats: 1.42 vs. 1.32–1.45), and probes transferred between
the base and the fine-tuned models without loss at the text positions in every condition
(transfer deficit ≤ 0.02 d′, with one exception of 0.15 in the upper layers for C1 on the
textbook summaries; note that the probes for the newspaper segments and textbook
summaries were near their maximum). At the last prompt position, in contrast, the deficit grew with
depth, most strongly after training with valid labels (upper third, newspaper segments:
C0 0.22, C1 2.09, C2 0.77, C3 0.86; textbook summaries: 0.25, 1.19, 0.72, 0.97). The
training thus left the representation of the text essentially unchanged, as far as the
probes can tell, and reorganized the position at which the answer is formed. That random labels also reorganized this position, while
barely changing the answers, shows that what mattered for the answers was which readout
was learned, not how much the answer position changed. The chats illustrate this most
clearly: After training with valid news labels, the model represented the authorship of
the chats as well as before (probe d′ = 1.32) but judged them below chance (−0.58),
consistent with the reversed transfer of news-trained probes to the chats (Section 3.1).
The dissociation that we observed after the original post-training was thus reproduced
experimentally, by training data whose cues were valid for one genre and misleading for
another.

The account that the valid news labels installed a readout along the cues of the news,
which mislead on the chats, makes a prediction that can be tested item by item. We fitted a
probe for authorship on the held-out news in the base model, at the text positions, so that
its direction is defined independently of any fine-tuned readout, and applied it unchanged
to the chats (mean of layers 20 and 24; training_study/prediction_test.py). This news probe
separated the chats with the wrong sign (d′ = −0.44 [−0.70, −0.15]). If training with valid
news labels makes the answer read this direction, the answers of the C1 models to the
individual chats should follow the probe's scores, and those of the other conditions should
not. This was the case. Across the 300 chats, the answers of the C1 models correlated with
the news-probe score (Spearman ρ = .68 [.60, .74], mean of three seeds; within the human
messages .70, within the AI messages .63), whereas those of the control conditions did not
(C0: −.11; C3: .04; C4: −.12); the difference between C1 and the random-label condition was
.63 [.51, .75]. Stereotype labels produced a weaker correlation (C2: .34 [.22, .46]). With a
news probe fitted on the 14 articles of the newspaper segments, the pattern was the same but
weaker (C1: .51; C3: .04), and this probe did not separate the chats reliably in the base
model (−0.11 [−0.36, 0.14]). At the last prompt position, the direction along which the C1
models' answers to the chats varied was aligned with the direction that separated AI from
human news in the same model (cosine .85, range across seeds .79–.90; C3: −.10; C4: .00;
random directions: 95% of the absolute cosines below .43); note that this second comparison
is partly circular, because the last position of a model that answers the news correctly
carries its decision. The statistics were fixed before this analysis was run, but the
inversion of the chats was known. Within these limits, the result supports a specific
mechanism: Fine-tuning on valid labels installed a readout along the class direction of the
training texts, and the answers to texts of another genre followed this direction, also
where it pointed the wrong way.
