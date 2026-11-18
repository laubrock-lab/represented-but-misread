# 4 Discussion


In the present study we used two judgments with known ground truth to ask whether
post-training changes what a language model represents or how it reads this representation
out. For negated statements, the truth was represented at ceiling in all six model
families, but the answers of several base models used it poorly or ordered the statements
inversely, and supervised fine-tuning repaired the readout while the probes stayed at
ceiling. For authorship, the information that distinguishes AI from human texts was readily
available in three corpora of German texts, both in the texts themselves and in the models'
internal representations, but the answers did not use it, and often ordered the texts
inversely. Post-training lowered the sensitivity of these answers on average (Δd′ = −0.34
across families; −0.29 across three prompt wordings), with considerable variation between
models, text types, and prompts, and in the two models with stage-wise checkpoints the
change occurred during supervised fine-tuning. Post-training also changed how strongly a
stated base rate moved the answers, from less than an ideal observer would to more. Two
interventions bore on where the changes take effect. Model stitching implicated the weights
of the middle layers for negation and of the early layers for authorship, with the caveat
that stitched hybrids produce effects of their own. Activation patching showed that, at the
middle of the network, the states at the positions after the text transferred the change
between checkpoints, whereas the states of the text did not. In a training study, a small
number of labeled examples shifted the readout in the direction their labels implied. In
short, where we could test it, post-training changed little in how the models represent the
text; it changed how the question they are asked reads this representation out, in a
direction that depended on the judgment. Note that the localization rests on two model
families and single layers, and that for the deceptive reviews (Appendix C) it was
inconclusive.

## Relation to previous work

Our findings are consistent with the growing evidence that language models represent
more than they say (Burns et al., 2024; Liu et al., 2023; Orgad et al., 2025), including
for authorship (Zhou et al., 2026; Ackerman & Panickssery, 2025). They add three points.
First, the dissociation is not only a gap but frequently an inverse ordering. A miscalibrated
decision threshold, which has been proposed as the main source of probe–answer gaps
(Villuri et al., 2026), shifts the criterion in terms of signal detection theory; it
can lower accuracy but cannot produce sensitivity below chance. Reversed readouts of a
discriminative direction have been described for steering vectors (Torop et al., 2026);
here, the inverse ordering occurred in the graded preferences behind the models'
unmodified answers, and in two model families the class-mean difference at the answer
position carried this inverse ordering (Appendix D). Note, however, that the cues themselves were specific
to the corpus: A probe trained on the newspaper segments separated the chats with the
wrong sign (Section 3.2). A judgment below chance therefore need not reflect an inverted
reading of a single authorship representation; it can also reflect a fixed policy whose
cues are valid for one kind of text and invalid for another. Second, the decline is
linked to post-training. Interestingly, in the closest related study, post-training had the
opposite effect: It created rather than abolished the ability of a model to recognize
its own texts (Ackerman & Panickssery, 2025). Recognizing one's own output and judging
whether an arbitrary text was written by an AI are different tasks, and it is
conceivable that post-training sharpens the first while it distorts the second. Third,
our results bear on the superficial alignment hypothesis, according to which a model's
knowledge is acquired during pretraining, whereas post-training mainly teaches it which
formats to use (Zhou et al., 2023). This hypothesis has been supported by behavioral
evidence (Lin et al., 2023) and by comparisons of representations between base and
post-trained models (Kissane et al., 2024; Lee et al., 2024; Prakash et al., 2024;
Waldis et al., 2026), and it has been qualified by the observation that post-training
gains continue to grow with the amount of fine-tuning data (Raghavendra et al., 2024;
see also Vergara-Browne et al., 2026). To our knowledge, it has not yet been tested with
causal interventions across checkpoints. Our
patching results provide causal evidence for it in one respect: The representation of
the text was interchangeable between base and post-trained checkpoints as far as the
judgment is concerned, and probes transferred between them at the text positions
wherever this test was not limited by ceiling effects. Note, however, that the
information available at the answer position declined after post-training in some
models (Gemma 3, OLMo 3 Think), so that the hypothesis holds for the representation of
the text rather than for the model as a whole. In another respect, our results qualify
it. The change was not superficial in the sense of affecting only the final stages of
processing or the output format; it involved weights in the early layers and altered
how the question takes up the text, and it had substantive
consequences for the quality of the judgment. This is consistent with the observation
that a property tracked across the stages of post-training is shaped mainly during
supervised fine-tuning (Heidari et al., 2026).

Our localization agrees with previous comparisons of base and post-trained models in
one respect and departs from them in another. That the effect of post-training resided
in the positions of the question rather than in those of the text agrees with model
diffing and cache patching, in which much of what fine-tuning adds was concentrated at
chat-template and system-prompt tokens (Leong et al., 2025; Loos & Eberle, 2026; Minder
et al., 2026). Because our prompts used no chat template, our results suggest that the
relevant positions are those at which the task is posed, whether in template tokens or
in a question phrased in natural language. In contrast, the stitching results, which
implicated the early layers for authorship and the middle layers for negated statements,
depart from the common picture that fine-tuning mainly alters the upper layers
(Merchant et al., 2020; Zhao et al., 2026; Zhou, 2026b). It is compatible, however, with
evidence that changes in the late layers can depend on altered earlier computation
(Zhou, 2026a), that post-training retunes routing mechanisms formed in pretraining
(Gupta et al., 2026), and that task representations formed early select later
processing pathways (Bigoulaeva et al., 2026). This conclusion is tempered by the
observation that stitched hybrids can judge worse or better than both parents (Section
3.4), so that stitching shows where the weights that carry the difference reside rather
than where the difference is computed; a control judgment far from chance showed no such
overshoot, so that stitching does not disturb the models in general, but whether the
overshoot is specific to the changed judgment or to judgments near chance remains open. Note also that stitching without an adapter
at layer ℓ is equivalent to exchanging the full residual stream at all positions at
that layer, so that our stitching and patching results are two views of one family of
interventions rather than independent tests. Steering at the answer position does not
add an independent test either, because it largely restates the difference between the
classes in the answers (Appendix D). Two further cautions apply. A probe shows that information can be decoded, not that the
model uses it (Hewitt & Liang, 2019; Elazar et al., 2021); this is why we rest the claim
that post-training changed the readout on the stitching and patching results rather than on the probes.
Conversely, interventions along a subspace can change the output through a pathway other
than the one the subspace was meant to capture (Makelov et al., 2023); our steering
results therefore show which way the answer pathway reads a direction, and we compared
them with control directions drawn from the same high-variance subspace and with the
decision direction, but they do not prove that the model uses this direction under
natural conditions.

## The direction of the change depends on the training signal

Post-training repaired the readout of negated statements in the base models that misread
them, and impaired the readout of authorship on average. A common account of these outcomes
follows from the second lens in Figure 1. Post-training does not add information to the
representation; it installs a way of reading the representation out. Whether this improves
or impairs a judgment then depends on whether the training signal specifies a valid way of
reading the question. For truth judgments, the supervised fine-tuning data of both models
with stage-wise checkpoints contained thousands of statements to be judged as true or
false; for authorship, we found none. In terms of the lens model, post-training changes the
weights of the second lens, and these weights are valid only to the extent that the
training data contain valid examples. Two observations qualify this account. First, the
base models of Qwen3 and OLMo 3 already read negated statements correctly, presumably
because their pretraining data contained instruction-like material; for them,
post-training had nothing to repair. Second, the account runs the risk of circularity: If
the presence of a valid training signal were inferred only from the improvement of a
judgment, it would explain every outcome and predict none. The inspection of the training
data provides an independent measurement, but a correlational one that rests on two data
mixtures and on a pattern search with limited power; it cannot show which examples produced
the repair (the truth judgments themselves, or, for instance, the many mathematical and
reasoning tasks that teach a model to read a question precisely).

We therefore manipulated the training signal directly (Section 3.7). Adding two thousand
correctly labeled authorship judgments to otherwise identical fine-tuning data improved the
readout of texts whose cues resembled those of the training texts and impaired it where
they did not, and the probes at the text positions changed little. On the chats, the
item-level answers followed a probe for the authorship of news that was fitted in the base
model, which turns the post-hoc explanation of the inversion into a tested prediction:
Fine-tuning installed a readout along the class direction of the training texts. The
training signal can thus set the direction of the readout, as the account assumes. The
training study does not show, however, that the decline after the official fine-tuning of
OLMo 2 has the same origin: Our background condition, with a small part of the original
data and low-rank adapters, did not reproduce this decline, and identifying the parts of
real post-training data that install a misreading remains a task for future work. The
deceptive reviews (Appendix C) fit the account only loosely: Post-training removed a
polarity heuristic that the base models shared, without improving the judgments.

## Sensitivity, criterion, and the stated base rate

Signal detection theory was useful here in two ways. First, it separates what a threshold
can explain from what it cannot. Many of the models' binary answers were constant, which
by itself would suggest a miscalibrated threshold; the graded preferences behind them,
however, were ordered inversely in several models, which no placement of the criterion can
produce. Second, the theory offers a normative benchmark for a manipulation of the response
bias. An ideal observer who is told the proportion of signal trials shifts the criterion by
the change in prior log-odds and leaves the evidence for each stimulus untouched. Human
observers typically adjust less than this (Phillips & Edwards, 1966; Macmillan & Creelman,
2004), and in some tasks neglect stated base rates in favor of the individual case (Kahneman
& Tversky, 1973). The base models behaved like conservative observers: Their log-odds
shifted in the correct direction, uniformly across texts, but by 6–51% of the normative
amount. After post-training, the log-odds shifted by more than the normative amount in five
of six model families, and by different amounts for different texts, so that the stated
share no longer acted as a pure shift of the criterion. Speculatively, a post-trained model
treats a sentence about the share of AI texts less as prior information to be combined with
the evidence and more as a cue about the expected answer. Two qualifications apply. The
post-trained models gave more extreme log-odds in general, to the texts as well as to the
base rate, and relative to the spread of their own judgments the weight of the base rate
increased in most, but not in all, models and wordings (Section 3.3). Moreover, the
benchmark presupposes that the log-odds of the two answer words are read as a posterior,
which is an idealization for base models that place only about half of their probability on
these words. Within these limits, the comparison shows that post-training changed not only
which texts receive more evidence for "AI", but also how the models combine this evidence
with what the prompt states about the situation, a second aspect of the readout.

## The question reads the text

What does it mean that post-training changes how the question reads the text? In a
language model, the answer is produced at the last position of the prompt, and
information from the text reaches it only through the positions in between, i.e., the
question. At each layer, these positions retrieve information from the text through
attention and transform it through the feed-forward blocks. Post-training could
intervene at either step, by changing what is retrieved or by changing what is made of
it, and the stitching results suggest that both are involved. Speculatively, a
psychological description of such a change is a different construal of the question. A model after
post-training may understand "written by a human" differently than its base model did,
much as a task set determines which features of a stimulus an observer selects and how
they are mapped onto a response. In accounts of biased competition, a task set acts
early, at the stage of selection, and not only at a late decision stage (Desimone &
Duncan, 1995). It is instructive to see that the effect of post-training, too, involved
weights in the early layers, where the question begins to select information from the
text, and was fully located in the question positions by the middle of the network.

The layer-by-position sweep adds a further distinction. In OLMo 2, post-training changed
not only the readout but also the text representation: The SFT model's text states at
layers 12–14 transferred 82% of the repair, an effect that vanished at layers below 10 and
above 16. In Apertus, by contrast, the text states contributed at most 23% and their
replacement caused no disruption in the reverse direction. The post-text readout change
generalized across architectures; the text-representation change did not. It is plausible
that the difference reflects the post-training recipes — different training objectives or
data may distribute the gradient signal differently across positions and layers — or that
the base models differ in how readily their text representations can be read by the
post-trained readout, so that one requires an adaptation of both the reader and the text,
while the other requires only the reader. Cross-model patching, in which the text states of
one model family are combined with the readout of another through a learned adapter, might
distinguish these accounts.

The parallel to human judges is equally instructive. Human readers fail to recognize AI
text because they rely on cues that are plausible but invalid (Jakesch et al., 2023;
Rosin & Laubrock, in preparation),
which in terms of the lens model is a failure of cue utilization rather than of cue
availability (Brunswik, 1956; Karelaia & Hogarth, 2008). The models in our study fail in
a similar way: Valid cues are present in the texts and in the representations, and the
answers do not use them.

## Why would human-oriented post-training impair the detection of AI text?

Our data do not answer this question conclusively, but they constrain the possible answers.
The lens analysis of the chat messages (Appendix B) showed what most post-trained models
took as signs of AI authorship: long words, adjectives, nouns, and text that is hard to
read, i.e., a formal and nominal style. In our corpus, this was the wrong way round: The AI
messages, written by current models instructed to sound like a private group chat, were
more verbal and lexically more diverse than the students' messages. Except for Qwen3, whose
base model already weighted these properties inversely (cross-validated G = −.53, after
post-training −.61), the base models used them less. One possible explanation, which our
data cannot test directly, is an outdated stereotype of AI text: The picture of AI writing
as formal, elaborate, and impersonal fits the output of earlier assistant models and is
widespread in texts about AI, and current models that imitate a group chat no longer fit
it. If this explanation is correct, the error is a moving target that changes with each
generation of models. Two alternatives are less consistent with our data. Familiarity with
the model's own style would predict that text in a formal assistant register is judged
human, which is the opposite of what we observed, and removing the surface markers that the
students had identified as typical of AI text did not change the models' judgments. OLMo 2
does not fit the picture either: After supervised fine-tuning, it weighted the same
properties in the opposite direction and judged the chats slightly better than its base
model. Note also that the lens analysis is correlational and that many of the text
properties are correlated with each other; it shows which properties the answers follow,
not which ones cause them.

## Implications

Three implications follow, each of which requires further testing. First, the side
effects of post-training can be localized, and possibly repaired. A stitched model that
combined the first eight layers of the Apertus base model with all upper layers of the
fine-tuned model discriminated better than either parent, although such hybrids also
produce effects of their own (Section 3.5). Whether such hybrids retain
the intended effects of post-training, such as instruction following, is an open
question, and the approach is related to combining fine-tuned weights by interpolation
or averaging (Wortsman et al., 2022a, 2022b). Second, knowing where a side effect arises
makes it possible to protect the affected components during fine-tuning or to correct
them with small, targeted interventions instead of broader training; in our training
study, two thousand examples sufficed to change the readout in either direction. That such
interventions are practical has been shown, albeit with the opposite intention, for the
refusal behavior that post-training installs: Refusal is mediated by a single direction
in the residual stream, and removing this direction abolishes it (Arditi et al., 2024),
and tools that optimize such edits automatically remove refusals while changing the
model's other outputs only minimally (Weidmann, 2025). Note, however, that the same ease
of editing that makes targeted corrections possible also shows how fragile safety-related
post-training can be. Third, and in our
view most important, post-training can change how a question is read while the
knowledge that the answer would require remains in the model. The answer then no longer
reflects what the model represents. In the terms of memory research, the information is
available but not accessible (Tulving & Pearlstone, 1966); in the terms of the second
lens in Figure 1, the readout weights the internal cues as if their validity were
different from what it is, and post-training can make this misweighting worse. Whether
language models can access their own internal states at all is the question of
introspection, for which they show some, but limited and unreliable, abilities (Kadavath
et al., 2022; Binder et al., 2024; Lindsey, 2025; Gurnee et al., 2026). Our case is a
concrete and measurable instance of a model that represents something but says something
else, and it suggests that comparing the
representation at the question positions with the answer might be a useful tool for
auditing models, in the spirit of distinguishing the presence of a feature from the
behavior it supports (Templeton et al., 2024) and of examining which representations a
model can verbalize (Gurnee et al., 2026). From the perspective of self-regulation, in
which post-training is often seen as a top-down control of the dispositions acquired in
pretraining, our results add a cautionary note: A top-down task set can bias perception
and not only correct a decision.

## Limitations

Several limitations qualify these conclusions. We studied models with 4 to 12 billion
parameters, and the below-chance judgments might be weaker or absent in larger models.
We derived the sensitivity of the answers from the graded log-odds of the two answer
words; because the binary answer hardly varied in most conditions, our d′ characterizes
the evidence on which the answer is based rather than the binary decisions themselves.
The
information that the models failed to use was largely lexical and stylistic, as shown by
simple n-gram classifiers, so our results concern the readout of ordinary cues rather
than of any deep representation of authorship. The stitching and patching analyses were
restricted to two models per judgment, to two corpora for authorship, and to the plain
prompt format, because base models cannot be queried with a chat template. The two
further judgments were studied with four to six models, steering for them with two
models each and on 300 items, and the effects of post-training on the reviews were
small; steering did not explain them. Steering itself is limited in two ways: It tests
whether a direction is read more than arbitrary directions, which a partly aligned
direction can fail even if it contributes to the answer, and it did not reveal the change
during SFT for authorship, which rests on stitching and patching. The training
study used one base model, low-rank adapters, and training texts from other generators
and of another genre than the test corpora, and its background condition did not
reproduce the effect of the original fine-tuning; the stereotype labels were moreover
correlated with text length, topic, and generator. The three causal methods were applied
to partly different combinations of models and corpora; only Apertus on the textbook
summaries was examined with steering, stitching, and patching alike. The newspaper
segments stem from only 14 articles, so that bootstrap intervals that resample articles
are likely too narrow for this corpus. All judgments used German prompts; a paraphrased
prompt reproduced the decline, a prompt with the question before the text reproduced it on
average but not in every family, and we did not test English prompts. Models
from different developers also differ in their post-training recipes, including distillation from larger
models, the emphasis on reasoning, and the share of human preference data, and our
design cannot separate these factors; they are a plausible source of the heterogeneity
between models and text types. Labeled examples improved the judgments of base and post-trained models alike without
closing the gap between them; they show that a correctable belief contributes to the
failure, but not how much of it. Finally, we defined
the family of hypothesis tests for the post-training contrasts after inspecting the full
set of comparisons; we therefore report the result under a stricter correction across
all contrasts as well, which left five of six decreases significant.

## Conclusion

Language models can represent what an answer would require and still answer otherwise.
For negated statements, post-training repaired such a readout; for authorship, it impaired
the readout on average: Models that write like humans failed to recognize their own kind of
text, not because they lacked the information but because of how they read it out. In both
judgments, post-training changed little in the representation of the text where we could
test it, and the change between checkpoints was transferred by the states at the positions
after the text and carried by weights in the middle or early layers. To know what a model
knows, it is therefore not enough to look at its answers; one has to look at how the
question is read. A training study showed that the training data can set the direction of
this readout with little change at the text positions. Future studies with larger models
may show how general this mechanism is, which parts of real post-training data install a
misreading, and whether such side effects can be prevented without giving up the benefits
of post-training.
