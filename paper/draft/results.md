# 3 Results


We report results for three corpora of German texts that each pair human-written texts
with AI-generated counterparts: short chat messages truncated to 25 words (300 texts
from 61 human originals), segments of local newspaper articles (103 segments from 14
articles), and textbook summaries (238 texts, 119 matched human–AI pairs;
TODO: credit as Rosin & Laubrock, in preparation, pending Niklas Rosin's consent). In all
analyses, texts derived from the same human original form one group, so that a model,
a probe, or a bootstrap resample never profits from recognizing a topic. Unless stated
otherwise, sensitivity is expressed as d′ estimated from the area under the ROC curve,
with 95% confidence intervals from a grouped bootstrap; positive values indicate that AI
texts are judged as more AI-like than human texts, negative values indicate the reverse.

## 3.1 The authorship information is available

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
others to recognize AI-edited English text (EditLens; Thai et al., 2025) transferred to
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

> **Figure 3 (placeholder).** Sensitivity of probes and baselines (word and character
> n-grams, surface statistics, average input embedding, EditLens) compared with the
> sensitivity of the models' own answers, per corpus.

## 3.2 The models' answers do not use it, and post-training makes this worse

When the same models were asked directly whether a text had been written by a human or
by an AI, their answers did not reflect this information. Table 2 summarizes the
order-averaged sensitivity of the answers for the pretrained base models and for the
final post-trained models. Many cells are close to chance, and a considerable number are
clearly below it: Qwen3-8B after post-training judged the newspaper segments with
d′ = −1.01 (raw prompt) and −1.30 (chat prompt), and Gemma 3 12B judged the short chats
with d′ = −0.81 and −1.09. Note that these values were computed from the graded
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

Base models were not uniformly better, but several of them discriminated clearly above
chance where their post-trained counterparts did not (e.g., Gemma 3 12B on the textbook
summaries, d′ = +1.64; OLMo 3 on the newspaper segments, +0.60; Qwen3 on the textbook
summaries, +0.55). To test the effect of post-training without confounding it with the
prompt format, we compared each final model with its base model on the identical plain
prompt. Of these 18 comparisons, 14 showed a decrease. Because each model family
contributes three corpora and each corpus six families, the 18 comparisons are not
independent. A multilevel random-effects model with random effects for model family,
corpus, and cell yielded a mean change of Δd′ = −0.33 [−0.53, −0.13] (t(17) = −3.54,
p = .003). The most conservative test, a t-test over the six family means, gave the same
estimate (−0.34 [−0.56, −0.11], t(5) = −3.82, p = .012); all six families showed a
decrease on average, two of them only slightly (OLMo 2: −0.10; OLMo 3 Think: −0.06).
Heterogeneity between cells was considerable (I² = 82%). Restricted to the nine
comparisons in which both models placed at least half of their probability on the two
answer words, the estimate was similar but less precise (−0.31 [−0.64, 0.01], p = .057;
four families). In the three comparisons in which the
binary answers varied in both the base and the final model (Gemma 3 12B on the short
chats and the textbook summaries, Gemma 3 4B on the short chats), post-training lowered
d′ in all three cases (−0.45 to −0.95). Per corpus, the decrease was reliable for the
newspaper segments (−0.42 [−0.67, −0.16], p = .002) and the textbook summaries (−0.41
[−0.70, −0.11], p = .007), but not for the short chats (−0.19 [−0.45, 0.07], p = .15).

The decline did not hinge on the wording of the prompt. With a
paraphrased instruction and question, the mean change was −0.23 (13 of 18 comparisons
negative; t-test over the family means, t(5) = −3.63, p = .015), and the changes of the
individual cells correlated with those for the original prompt (r = .81). With the
question placed before the text, the mean change was similar (−0.30; 12 of 18 negative),
but two families changed direction (OLMo 3 Think: +0.38; Qwen3: +0.04), so that the
family-level test was not significant (p = .16; r = .63 with the original prompt). The
below-chance judgments of the short chats by Gemma 3 and Qwen3 persisted with all three
prompts.

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
examples; d′ computed per set and averaged; Figure 4B). With 16 examples, d′ rose in 9 of
11 model conditions, reliably in 8, whereas two examples hardly helped. Base and
post-trained models gained by similar amounts, so that the gap between them persisted:
After 16 examples, the post-trained model was still less sensitive than its base model in
all five comparisons (e.g., Gemma 3 4B on the short chats: +0.48 vs. −0.27; Qwen3: +0.15
vs. −0.17), and the post-trained models of Gemma 3 4B and Qwen3 remained below chance on
the chats (Appendix E). A mistaken belief about what
distinguishes AI from human texts, which examples can partly correct, is therefore likely
to be part of the explanation, but it does not account for the deficit that post-training
introduced.

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

> **Figure 4 (placeholder).** Summary across all cells (base model × corpus × prompt
> format): probe sensitivity, base-model answer, and post-trained answer; few-shot
> curves as an inset.

## 3.3 Steering: in some models, the represented direction drives the answer the wrong way

The probes and the answers could dissociate either because the answers ignore the
authorship information or because they use it with the wrong sign. To distinguish these
possibilities causally, we shifted a model's internal state at the position where the
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
(model × corpus × layer, including the truth and review judgments in Section 3.6), the
decision direction moved the answer more than the norm-matched controls (rank-based
p ≤ .05; 22 of 25 against the scale-matched controls), so that the test can detect a
direction that the answer reads.

In six of the authorship cells, the authorship direction moved the answer more than the
controls, and in all six it moved it the wrong way: Shifting a text toward "AI" made the
model answer "human" more often. This was the case for the short chats in Gemma 3 12B
(layer 31: −7.38 nats; rank-based p = .01 against both control families), Gemma 3 4B
(layer 17: −1.74; p = .04 norm-matched, .06 scale-matched), and Qwen3 (layers 24 and 30:
−1.18 and −2.06; p ≤ .03), and for the newspaper segments in Qwen3 (−0.43 and −0.85;
p ≤ .04). In these cells, the authorship direction pointed against the decision direction
(cosine −0.59 to −0.92), and the answers were clearly below chance (Table 2). In the
remaining cells, the authorship direction did not move the answer more than arbitrary
directions: Gemma 3 4B on the newspaper segments (−1.04 and −2.00 at layers 22 and 28;
p = .26–.42), Apertus and OLMo 3 on the textbook summaries before and after SFT, and Gemma
3 12B on the newspaper segments, on which its answers were at chance (+0.03; in this cell,
the positive control did not exceed the controls either). The answer thus read the
represented authorship direction, with the wrong sign, only in some models and corpora; in
the others, it followed other directions.

A complementary, descriptive comparison concerns the size of the effects. Shifting a
text by one class-mean distance along the authorship direction, i.e., from the average
human text to the average AI text, changed the answer in the same direction as the
observed difference between AI and human texts in all 20 cells in which this difference
was clearly different from zero (e.g., Qwen3 on the newspaper segments: −0.50 and −0.66
nats vs. an observed difference of −0.59). If the answer depends approximately linearly on
the internal state, this agreement is expected; it shows that the difference between the
answers to AI and human texts is carried by the state at the last prompt position, in
line with the patching results below (Section 3.5).

Steering did not, however, reveal what post-training changes for authorship. In the
Apertus base model, shifting toward "AI" moved the answer in the correct direction,
increasingly so with depth (+0.08, +0.21, and +0.35 nats at layers 16, 21, and 27), but
not more than the control directions (p ≥ .15), and after SFT the effect was close to zero
(−0.04 to +0.04). In OLMo 3 Think, neither the base nor the SFT model used the direction
(−0.01 and +0.09).[^olmo3] The change of the answers during SFT is therefore localized by
the stitching and patching analyses (Sections 3.4 and 3.5), not by steering.

[^olmo3]: An earlier comparison, in which the SFT model received its chat template and the
base model the plain prompt, had suggested a reversed use after SFT (−0.27); with the same
plain prompt for both checkpoints, this was not the case.

## 3.4 Model stitching implicates the early layers

Where in the network does post-training make the difference? Because the base and the
post-trained checkpoints of a model share the same architecture and start from the same
weights, we could combine them: the lower layers from one checkpoint and the upper layers
from the other, passing the internal state at the junction on unchanged ("model
stitching"). This test requires no probe and no statistic derived from the answers; it
only measures how well the combined model discriminates. Two observations justified the
procedure. First, the representations of the texts themselves were nearly identical in
the base and the post-trained checkpoints of the three models with stage-wise checkpoints
(linear CKA across layers 0.97–0.999 for OLMo 2, Apertus and OLMo 3; textbook summaries),
whereas the representations at the position where the answer is formed differed
considerably (0.44–0.99, median 0.65–0.85). Note that this was less so for the other
families: In Gemma 3, even the text representations changed noticeably (0.42–0.99, median
0.80–0.83), and the answer position differed strongly (median 0.20–0.31). Second, every stitched model produced fluent
text (perplexity 6.0–6.7 for Apertus, 8.2–9.4 for OLMo 2), and stitching at the very
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
post-training had left unchanged, the truth of statements on diverse facts (counterfact;
Apertus: d′ = 1.38 base vs. 1.58 SFT; OLMo 2: 1.77 vs. 1.75). There, all hybrids stayed
within or close to the range of their parents (largest deviation 0.03 for Apertus and
0.09 for OLMo 2; one of 14 hybrids per model reliably outside), whereas on the textbook
summaries 6 of 18 Apertus hybrids and 4 of 18 OLMo 2 hybrids lay reliably outside
(largest deviation 0.59). The overshoot was therefore specific to the judgment that
post-training had changed, which suggests that the change involves an interaction between
the early layers and the layers above them rather than a self-contained module. The
stitching results show that the early-layer weights of a checkpoint suffice to carry the
difference into an otherwise foreign network; they do not by themselves show that the
difference is computed there.

The early locus replicated on the short chats, on which post-training lowered the
sensitivity of Apertus (−0.26 to −0.78) and raised that of OLMo 2 (−0.05 to +0.37). The
first 8 base layers of Apertus restored the base level under SFT upper layers (−0.27),
and the first 4 base layers of OLMo 2 removed most of its gain (+0.10); on the chats,
no hybrid lay outside the range of its parents.

Both sublayer types contributed. Exchanging only the attention blocks or only the
feed-forward (MLP) blocks of the first 8 Apertus layers had partial effects in both
directions (e.g., restoring the base model's blocks under SFT: attention +0.17
[−0.06, 0.40], MLP +0.18 [0.03, 0.37], both together +0.59). In OLMo 2, the base
model's MLP blocks of the first 16 layers restored discrimination almost completely
(−0.09 [−0.29, 0.11]), whereas its attention blocks did so only partly (−0.35
[−0.55, −0.19]). Within the limits just described, the change thus appears to be
distributed over the early layers and to involve both attention and feed-forward
blocks.

## 3.5 Post-training changes how the question reads the text

The stitching results implicate the early layers, but do not yet say where in the
prompt the change acts.
In a language model, the answer can only access the text through the positions that
follow it, i.e., the question "Was this text written by a human or generated by an AI?"
and the answer cue. We therefore exchanged the internal states between base and SFT
checkpoints selectively at different positions of the prompt ("activation patching"):
at the tokens of the text, at the instruction before the text, at the question after
the text, or at the final position only.

At layer 16, the pattern was unambiguous and identical for both models (Figure 5B). Exchanging the states of the text tokens changed nothing: Apertus after SFT
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
→ +0.20). Because post-training improved the judgments of OLMo 2 on the chats, the change
resided at the question positions regardless of its direction. At layer 8, the picture was mixed (Figure S1): in Apertus, the text
positions still had a partial effect, and in OLMo 2 even exchanging all positions
transferred only part of the difference, consistent with the stitching result that in
OLMo 2 the change accumulates up to about layer 14. Thus, by the middle of the network
the difference resides entirely in the question positions.

Note that in our prompt the text precedes the question, so that its states are computed
before the question is read, and a change that is specific to the judgment can only
appear at the question positions. We therefore repeated the patching with the question
placed before the text, so that the text was read with the task known. The decline
persisted with this prompt (Apertus: +0.36 base vs. −0.19 SFT; OLMo 2: +0.09 vs. −0.42),
and the text states again carried little of it (changes ≤ 0.12). Most of the difference
now resided in the few positions after the text ("Antwort:"; e.g., Apertus SFT with the
base states at these positions: −0.19 → +0.44), and part of it in the question itself.
Thus, even when the text is read in the light of the question, post-training did not
change the text states in a way that mattered for the judgment. In summary, post-training changed
little in how the text was represented, as far as the judgment is concerned, and changed
how the question takes up this representation, through weights in the early layers and
processing at the question positions.

> **Figure 5 (placeholder, central figure).** (A) Model stitching for Apertus and
> OLMo 2 (d′ as a function of the number of layers taken from the other checkpoint, both
> directions). (B) Activation patching at layer 16 (d′ after exchanging states at
> text, pre-text, question, and final positions).

## 3.6 Beyond authorship: truth judgments and deceptive reviews

Is the dissociation between representation and answer, and its modification by
post-training, specific to authorship? We repeated the central analyses for two further
judgments with known ground truth and publicly available data (Section 2.8): judgments of
whether a statement is true, and judgments of whether a hotel review is genuine. Table 3
summarizes the results.

> **Table 3.** Sensitivity (d′) of the probes (last prompt position, two thirds of the
> network depth) and of the answers (plain prompt) for three kinds of judgment:
> authorship (was the text written by a human or by an AI? textbook summaries, repeated
> from Sections 3.1 and 3.2 for comparison), truth (is the statement true or false? either
> statements on diverse facts from the counterfact set or negated geographic statements),
> and authenticity (is the hotel review truthful or deceptive?). SFT: after supervised
> fine-tuning (answer); —: no intermediate checkpoint released. The maximum d′ (AUC = 1) is 3.73 for
> the 119 textbook pairs and 4.54 for the 748 statement pairs.
>
> | Judgment | Model | Probe base | Probe final | Answer base | SFT | Final |
> |:---|:---|---:|---:|---:|---:|---:|
> | Authorship | OLMo 2 7B | 3.73 | 3.57 | −0.12 | −0.50 | −0.44 |
> | (textbook summaries) | Apertus 1.5 8B | 3.73 | 3.73 | 0.43 | −0.04 | −0.04 |
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
> | Authenticity | OLMo 2 7B | 2.93 | 2.79 | 0.16 | 0.20 | 0.17 |
> | (hotel reviews) | Apertus 1.5 8B | 2.62 | 2.49 | 0.14 | 0.32 | 0.23 |
> | | OLMo 3 7B Think | 2.80 | 2.69 | 0.13 | −0.01 | 0.00 |
> | | Gemma 3 4B | 2.82 | 2.51 | 0.45 | — | 0.08 |
> | | Gemma 3 12B | 3.01 | 2.57 | 0.52 | — | 0.86 |
> | | Qwen3 8B | 2.56 | 2.59 | 0.06 | — | 0.50 |
>
> ᵃ The Gemma 3 base models placed only 18–32% of their probability on the two answer
> words; interpret with caution.

*Truth judgments.* For simple geographic statements, all checkpoints judged at ceiling,
so that this set served only as a check. For the negated statements, however, the base
models showed the same dissociation as for authorship, in a sharper form: Probes decoded
the truth of every statement (AUC = 1), but the answers of the OLMo 2 base model used
this information only weakly (d′ = 0.80), and those of the Apertus base model ordered
the statements inversely (d′ = −0.88). Both base models answered "False" almost always
(criterion c = 2.6 and 3.0). Supervised fine-tuning repaired the readout (OLMo 2:
d′ = 3.36; Apertus: 4.54), whereas the probes remained at ceiling. The Gemma 3 models
showed the same pattern between their base and final versions (Gemma 3 4B: 0.06 → 3.46;
Gemma 3 12B: 1.97 → 4.54), although their base models rarely answered with the two
answer words. The base models of Qwen3 and OLMo 3, in contrast, already judged the
negated statements perfectly; post-training left this unchanged in Qwen3 and lowered it
slightly in OLMo 3 (d′ = 3.80 after post-training, with a slightly lower probe, 4.31).
Thus, post-training repaired the readout of negation where the base model read it
wrongly, but did not improve it further where the base model read it correctly. A probe trained on the
simple statements of the same checkpoint revealed what the base models' answers followed:
Applied to the negated statements, it separated them with the wrong sign (OLMo 2:
d′ = −1.28; Apertus: −1.43), i.e., the general truth direction of the base models
encoded whether city and country belong together and ignored the negation. After SFT, the
same probe separated the negated statements in the correct direction (+0.69; +3.05). For
the statements on diverse facts, post-training changed the answers only slightly, except
in OLMo 3, where answers and probes declined together (answers 1.61 → 1.34; probes 1.89
→ 1.64).

Where does the repair arise? Model stitching located it in the middle of the network
rather than in the early layers. In OLMo 2, the first 8 base layers combined with all
upper SFT layers still judged at the level of the SFT model (d′ = 3.76; base 0.63, SFT
3.20 in this subset of 300 pairs), whereas the first 16 base layers abolished the
repair (0.38); in Apertus, the repair arose mainly in layers 9 to 12. Activation patching
at layer 16 showed the same locus as for authorship: The SFT model's states at the
question positions repaired the base model completely (OLMo 2: 0.63 → 3.26; Apertus:
−0.87 → 4.15), and the base model's question states abolished the repair in the SFT
model (3.20 → 1.07; 4.15 → −1.09). In Apertus, the final position alone carried the
full effect in one direction (base model with the SFT model's final state: 4.15).
Exchanging the states of the statement itself had no effect in Apertus and in the SFT
model of OLMo 2, but a partial one in the OLMo 2 base model (0.63 → 1.51).

Steering showed how the repaired readout uses the represented truth direction. Shifting
the internal state at the answer position along the direction that separates true from
false negated statements (±2 SD, layer 21; Section 3.3) moved the answers of both SFT
models toward the correct answer, more than the control directions (Apertus: +0.36 nats,
p = .02; OLMo 2: +1.07, p ≤ .04), and in these models the truth direction coincided with
the decision direction (cosine ≈ 1.0). In the base models, it did not (Apertus: −0.14,
p = .13–.33; OLMo 2: +0.09, p ≥ .54). Supervised fine-tuning thus aligned the answer with
the represented truth direction; whether the Apertus base model used this direction with
the wrong sign, as its negative sensitivity would suggest, could not be established.

*Deceptive reviews.* All models judged the reviews close to chance (d′ between 0.00 and
0.86), although probes separated truthful from deceptive reviews well (d′ = 2.5–3.0), at
least as well as a classifier on word frequencies (d′ = 2.44). Post-training changed the
probes little and the answers by small amounts in a direction that depended on the model:
Gemma 3 4B lost sensitivity (d′ 0.45 → 0.08), Qwen3 (0.06 → 0.50) and Gemma 3 12B (0.52 →
0.86) gained, and the others changed little (Table 3). What post-training changed
consistently was a shared heuristic. The base models judged positive reviews as more
truthful than negative ones, and review polarity accounted for 41–90% of the variance of
their judgments; post-training largely removed this heuristic (2–32%), except in OLMo 3
Think (76%), and all final models showed a strong bias toward "Truthful" (83–92% of the
answers). The localization and steering analyses were less conclusive than for the other
two judgments: Patching located the change at the question positions in two of four
cases, and steering did not explain it (Appendix C). The reviews are thus a case in which
post-training reshaped the readout without improving it.

*Representation and readout across checkpoints.* If post-training leaves the
representation intact and changes how it is read out, a probe trained on the base model
should work on the post-trained model at the text positions, but less so at the position
where the answer is formed. This was the case for all models and judgments (Figure S2),
with qualifications. At the text positions of the authorship corpora, probes transferred
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
segments), so the readout position was reorganized rather than replaced. Probes trained
within each checkpoint continued to decode the information at the last position, but in
some models less well after post-training (e.g., Gemma 3 4B on the newspaper segments,
upper third: d′ = 2.40 in the base and 1.88 in the final model; OLMo 3 Think on the
textbook summaries: 3.72 and 3.03). Post-training thus left the text positions
essentially unchanged where the test is informative, and both reorganized and, in some
models, degraded the information available at the answer position.

*What do the training data teach?* The direction of the change differed between the
judgments: Post-training impaired the readout of authorship, repaired that of negated
statements, and reshaped that of the reviews. To examine whether this corresponds to
the supervised fine-tuning data, we searched the SFT mixtures of OLMo 2 and Apertus. Both
contained thousands of requests to judge statements as true or false (Tülu 3: 2,685;
Apertus, extrapolated: about 2,300), largely factual claims in Tülu 3 and mathematical
statements with explicit reasoning, often including negations, in Apertus. In contrast,
neither mixture contained a single request to judge whether a text had been written by a
human or by an AI, and the matches for fake or deceptive reviews were sentiment tasks or
summaries of research papers. Because our judgments were posed in German, we repeated the
search with German patterns in the conversations written in German, which made up only
0.1% of Tülu 3 (992 conversations) and 1.7% of the Apertus sample (6,330). They
contained no request to judge authorship and no deception judgment, two mentions of
AI-generated text (Apertus), and six true/false questions. Thus, post-training improved the one judgment for which the
training data specify how to read the question, where the base model read it wrongly,
and impaired or merely reshaped the judgments for which they do not.

In summary, the pattern found for authorship generalized: In all three judgments, the
information was represented, the answers used it poorly before or after post-training,
post-training changed the readout far more than the representation of the text, and the
change resided at the question positions. Its direction, however, depended on the
judgment.

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
same layout (differences in d′ with 95% CIs from a grouped bootstrap). Practicing the
unrelated judgment left the authorship answers where C0 left them (C4 − C0 between −0.13
and +0.10 on the three test corpora, all CIs including zero), so that practice with a
judgment as such did not explain the effects of the authorship labels. Two thousand correctly labeled judgments of news texts sufficed to raise the
sensitivity of the answers far above both baselines, not only on held-out news but also
on texts from other generators and genres: the newspaper segments, written by GPT-5.4
(C1 − C3: +2.23 [1.56, 3.17]; C1 − C4: +2.37 [1.74, 3.29]), and the textbook summaries
(+1.36 [1.11, 1.64]; +1.42 [1.15, 1.73]). On the
held-out news, C1 was near its maximum for both generators and all text lengths
(d′ = 3.8–4.0 in each subset), so that its success there was not due to recognizing one
generator or one length. The information had been represented all along; a small amount
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
reliably (−0.29 [−0.61, 0.07]; −0.15 [−0.57, 0.25]); on the chats, it did not differ from
random labels (−0.08 [−0.39, 0.24]). Note also that the
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

Because training with valid labels inverted the judgments of the chats, we verified
the labels of this corpus. All 61 human messages are verbatim the students' original submissions, all 239 AI messages trace to the generation pipeline (none resembles a student message, maximum similarity .16), and independent judges recognized the chats in the labeled direction (human participants in a separate experiment: AUC = .82, Rosin & Laubrock, in preparation; Claude Opus: .69–.90). Whether individual students used AI tools to write their messages cannot be determined from the records.
