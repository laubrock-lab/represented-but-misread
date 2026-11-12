# 3 Results


We report results for three corpora of German texts that each pair human-written texts
with AI-generated counterparts: short chat messages truncated to 25 words (300 texts
from 61 human originals), segments of local newspaper articles (103 segments from 14
articles), and textbook summaries (238 texts, 119 matched human–AI pairs;
TODO: credit as Rosin & Laubrock, in prep., pending Niklas Rosin's consent). In all
analyses, texts derived from the same human original form one group, so that a model,
a probe, or a bootstrap resample never profits from recognizing a topic. Unless stated
otherwise, sensitivity is expressed as d′ estimated from the area under the ROC curve,
with 95% confidence intervals from a grouped bootstrap; positive values indicate that AI
texts are judged as more AI-like than human texts, negative values indicate the reverse.

## 3.1 The authorship information is available, and it is ordinary

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
summaries: AUC ≥ .99), and a probe on the average input embedding of the text, i.e.,
before any processing by the network, performed about as well as probes on deep layers
(short chats: d′ = 1.48). Holding out whole authors instead of single originals did not
change this picture (word frequencies: d′ = 1.79). A detector that was trained by
others to recognize AI-edited English text (EditLens; Thai et al., 2025) transferred to
our German material in a graded manner, reaching an AUC of .99 for the textbook
summaries, .73 for the newspaper segments, and .57 for the short chats (Figure 2). Thus,
the cues that distinguish AI from human texts are present and readily accessible, both
in the texts themselves and in the models' representations; they consist largely of
ordinary lexical and stylistic regularities. In terms of Brunswik's lens model, the
environment offers valid cues.

> **Figure 2 (placeholder).** Sensitivity of probes and baselines (word and character
> n-grams, surface statistics, average input embedding, EditLens) compared with the
> sensitivity of the models' own answers, per corpus.

## 3.2 The models' answers do not use it, and post-training makes this worse

When the same models were asked directly whether a text had been written by a human or
by an AI, their answers did not reflect this information. Table 1 summarizes the
order-averaged sensitivity of the answers for the pretrained base models and for the
final post-trained models. Many cells are close to chance, and a considerable number are
clearly below it: Qwen3-8B after post-training judged the newspaper segments with
d′ = −1.01 (raw prompt) and −1.30 (chat prompt), and Gemma 3 12B judged the short chats
with d′ = −0.81 and −1.09. A negative d′ cannot result from a response bias alone; it
means that the models systematically took AI texts for human texts and vice versa.

> **Table 1.** Sensitivity (d′) of the models' answers. Base models were queried with a
> plain completion prompt ("raw"); post-trained models with the same prompt and with
> their chat template ("chat"). Conditions in which a model placed less than 30% of its
> probability on the two answer words are omitted (—).
>
> | Model family | Corpus | Base (raw) | Final (raw) | Final (chat) |
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
prompt. Of these 18 comparisons, 14 showed a decrease. A random-effects summary across
all 18 comparisons yielded a mean change of Δd′ = −0.33 [−0.50, −0.17] (z = −3.95,
p < .001), with considerable heterogeneity between cells (I² = 82%). Because the same
model families contribute to several corpora, we repeated the summary after averaging
within each family; the result was the same (−0.34 [−0.49, −0.18], p < .001), and all
six families showed a decrease on average. Per corpus, the decrease was reliable for the
newspaper segments (−0.42 [−0.67, −0.16], p = .002) and the textbook summaries (−0.41
[−0.70, −0.11], p = .007), but not for the short chats (−0.19 [−0.45, 0.07], p = .15).

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
nor in a single direction across model families and text types.

For OLMo 2 and Apertus 1.5, the developers released checkpoints after each stage of
post-training, which allowed us to ask when the change occurs. In both families it
occurred at the first, supervised stage (supervised fine-tuning, SFT), in which the
model learns to imitate example answers. For OLMo 2 on the textbook summaries, d′ fell
from −0.12 (base) to −0.50 after SFT and then remained stable (−0.46 after preference
optimization, −0.44 after the final reinforcement-learning stage). For Apertus, d′ fell
from +0.43 to −0.04 after SFT and stayed near zero through the subsequent stages (0.00,
−0.03, −0.04). The later, preference-based stages thus added little to a change that
was already complete after imitation learning.

Could the post-trained models simply hold a wrong belief about what AI texts look like,
a belief that a few correctly labeled examples would correct? We tested this by placing
2, 8, or 16 labeled examples from other sources before each text (four example sets
with balanced answer order). Examples improved the judgments of almost all models. With
16 examples on the short chats, d′ rose from −0.21 to +0.66 for the base model of Gemma
3 4B, from −0.68 to +0.21 for Qwen3, and from −0.07 to +0.57 for OLMo 2, whereas the
post-trained models of Gemma 3 4B and Qwen3 remained below chance (−0.72 to −0.31 and
−0.80 to −0.17, raw prompt). Whether base models profit more from examples than
post-trained models is, however, a question about the difference between the two gains,
which we tested with a paired grouped bootstrap. Base models gained significantly more in
three of five comparisons (Gemma 3 4B, short chats: difference +0.46 [0.12, 0.85],
p = .012; OLMo 2, short chats: +0.61 [0.29, 0.90], p = .001; Gemma 3 4B, textbook
summaries: +0.47 [0.17, 0.75], p = .002), but not for Qwen3 on the short chats (+0.26
[−0.05, 0.58], p = .11), and for Qwen3 on the textbook summaries the post-trained model
gained more (−0.27 [−0.52, −0.03], p = .026), its base model being already above chance
without examples. Thus, labeled examples reduce the deficit of the post-trained models
but do not consistently remove it, and the evidence that post-training makes the
judgments less responsive to examples is mixed. A simple false belief that examples
override can therefore neither be ruled out nor be taken as the full explanation.

> **Figure 3 (placeholder).** Summary across all cells (model family × corpus × prompt
> format): probe sensitivity, base-model answer, and post-trained answer; few-shot
> curves as an inset.

## 3.3 The authorship direction drives the answer in the wrong direction

The probes and the answers could dissociate either because the answers ignore the
authorship information or because they use it with the wrong sign. To distinguish these
possibilities causally, we shifted a model's internal state at the position where the
answer is produced along the direction that separates AI from human texts, and measured
how the answer changed. The direction was estimated on held-out groups, the shift
amounted to six within-class standard deviations, and we compared it with two kinds of
control directions: random directions drawn from the same high-variance subspace as the
authorship direction, and differences between random halves of the texts within each
class. As a positive control, we also shifted along the direction that the model's own
answers follow.

In Gemma 3 4B after post-training, shifting a text toward "AI" made the model answer
"human" more often, and this effect clearly exceeded the control directions from the
middle of the network onward. On the newspaper segments, the shift changed the answer by
−3.26 nats at layer 22 and by −6.09 nats at layer 28, compared with control effects of
−0.57 ± 1.14 and +1.22 ± 1.12 (mean ± SD across random directions); on the short chats,
the effect was −5.48 nats at layer 17 (controls −0.25 ± 0.15). The positive control
behaved as expected, moving the answer toward "AI" (+11.46 and +7.98 nats at the same
layers). Only at the earliest layer tested (layer 11) could the authorship effect not
be distinguished from the controls. Gemma 3 12B showed the same reversal on the short
chats (layer 31: −17.36 nats; random directions −0.63 ± 1.31; positive control +14.48).
Note that the reversal was carried by the part of the authorship direction that points
against the model's decision direction: after this part had been removed, the remaining
authorship component moved the answer in the correct direction (newspaper segments,
layer 22: +4.16 nats; layer 28: +4.63 nats).

Steering also revealed what post-training changes. In the Apertus base model, shifting
toward "AI" moved the answer in the correct direction, increasingly so with depth
(+0.22, +0.63, and +1.07 nats at layers 16, 21, and 27), and clearly beyond the
strict control directions (at most ±0.32 nats). After SFT, this effect had vanished
(−0.08 and −0.12 nats at layers 21 and 27, within the range of the controls), whereas
the positive control continued to work in both models (base: +1.03 to +2.06; SFT: +0.94
to +1.11). In OLMo 3 Think, the base model did not use the authorship direction at layer
21 (−0.04), whereas the model after SFT used it in the wrong direction (−0.27; isotropic
random directions +0.03 ± 0.09; this pair was tested with isotropic controls only).
Thus, supervised fine-tuning turned a correct use of the authorship information into no
use (Apertus) or into a reversed use (OLMo 3).

## 3.4 The change originates in the early layers

Where in the network does post-training make the difference? Because the base and the
post-trained checkpoints of a family share the same architecture and start from the same
weights, we could combine them: the lower layers from one checkpoint and the upper layers
from the other, passing the internal state at the junction on unchanged ("model
stitching"). This test requires no probe and no statistic derived from the answers; it
only measures how well the combined model discriminates. Two observations justified the
procedure. First, the representations of the texts themselves were nearly identical in
the base and the post-trained checkpoints (linear CKA 0.97–0.998 for OLMo 2, Apertus and
OLMo 3), whereas the representations at the position where the answer is formed differed
considerably (0.5–0.9; Gemma: 0.15–0.32). Second, every stitched model produced fluent
text (perplexity 6.0–6.7 for Apertus, 8.2–9.4 for OLMo 2), and stitching at the very
bottom or the very top reproduced the respective parent model.

If post-training changed only the final readout stage, the upper layers should decide
the outcome. The opposite was the case (Figure 4, left). When the first 8 of 32 layers of
Apertus came from the base model and all layers above from the SFT model, the combined
model discriminated at least as well as the base model (d′ = +0.59 [0.41, 0.80]; base
+0.43). Conversely, the first 8 layers of the SFT model sufficed to abolish the base
model's discrimination, although all 24 layers above came from the base model (−0.50
[−0.74, −0.25]). For OLMo 2, the first 16 base layers restored base-level performance
under SFT upper layers (−0.01 [−0.15, 0.14]; base −0.12, SFT −0.50), and the first 6
SFT layers were sufficient to produce the full deficit (−0.48). Finer steps showed that
the change builds up gradually over several early layers rather than in a single one
(Apertus: layers 3–8; OLMo 2: layers 1–14).

Both sublayer types contributed. Exchanging only the attention blocks or only the
feed-forward (MLP) blocks of the first 8 Apertus layers had partial effects in both
directions (e.g., restoring the base model's blocks under SFT: attention +0.17
[−0.06, 0.40], MLP +0.18 [0.03, 0.37], both together +0.59). In OLMo 2, the base
model's MLP blocks of the first 16 layers restored discrimination almost completely
(−0.09 [−0.29, 0.11]), whereas its attention blocks did so only partly (−0.35
[−0.55, −0.19]). The damaging change is thus distributed over the early layers and
involves both how information is gathered and how it is transformed.

## 3.5 Post-training changes how the question reads the text

The stitching results locate the change in the early layers, but not yet in the text.
In a language model, the answer can only access the text through the positions that
follow it, i.e., the question "Was this text written by a human or generated by an AI?"
and the answer cue. We therefore exchanged the internal states between base and SFT
checkpoints selectively at different positions of the prompt ("activation patching"):
at the tokens of the text, at the instruction before the text, at the question after
the text, or at the final position only. As a check, patching a model with its own
states left every answer unchanged.

At layer 16, the pattern was unambiguous and identical for both families (Figure 4,
right). Exchanging the states of the text tokens changed nothing: Apertus after SFT
remained at chance when it received the base model's text representation (−0.11; SFT
alone −0.04), and the base model kept its sensitivity when it received the SFT model's
text representation (+0.43; base alone +0.43). Exchanging only the states of the
question positions transferred the full difference in both directions: the base model's
question states restored discrimination in the SFT model (Apertus: +0.41; OLMo 2: +0.04,
from −0.50), and the SFT model's question states abolished it in the base model
(Apertus: −0.42; OLMo 2: −0.61, from −0.12). The final position alone carried only part
of this effect. At layer 8, the picture was mixed (Figure S1): in Apertus, the text
positions still had a partial effect, and in OLMo 2 even exchanging all positions
transferred only part of the difference, consistent with the stitching result that in
OLMo 2 the change accumulates up to about layer 14. Thus, the difference is
produced in the early layers and has fully moved into the question positions by the
middle of the network.

In summary, post-training did not change what the models represent about a text. It
changed how the question takes up this representation, through a change that originates
in the early layers and that resides in the processing at the question positions.

> **Figure 4 (placeholder, central figure).** Left: model stitching for Apertus and
> OLMo 2 (d′ as a function of the number of layers taken from the other checkpoint, both
> directions). Right: activation patching at layer 16 (d′ after exchanging states at
> text, pre-text, question, and final positions).
