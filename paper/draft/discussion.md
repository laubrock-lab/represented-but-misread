# 4 Discussion


In the present study we used authorship judgments as a test case for the question of
whether post-training changes what a language model represents or how it reads this
representation out. Across six open-weight base models and three corpora of German texts,
the information that distinguishes AI from human texts was readily available, both in
the texts themselves and in the models' internal representations, but the models'
answers did not use it, and often ordered the texts inversely. Post-training lowered
the sensitivity of the answers on average (Δd′ = −0.33), with considerable variation
between models and text types, and in the two models with stage-wise
checkpoints the change occurred during supervised fine-tuning. Three causal methods
bore on where this change takes effect. Steering showed that, in several post-trained
models, the direction that separates AI from human texts pushed the answer the wrong
way. Model stitching implicated the weights of the early layers, with the caveat that
stitched hybrids produce effects of their own. Activation patching showed that, by the
middle of the network, the change does not reside in the representation of the text but
in the processing at the positions of the question that follow the text. The
same pattern held for two further judgments with public data: For negated statements and
for deceptive hotel reviews, the information was represented, post-training changed the
readout far more than the representation of the text, and the change resided at the
question positions. Its direction, however, differed between the judgments, and so did
its depth. In short, post-training changed little in how the models represent the text;
it changed how the question they are asked reads this representation out, in a direction
that depended on the judgment.

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
unmodified answers, and steering showed in several models that the represented direction
pushed the answer the wrong way. Note, however, that the cues themselves were specific
to the corpus: A probe trained on the newspaper segments separated the chats with the
wrong sign (Section 3.1). A judgment below chance therefore need not reflect an inverted
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
than where the difference is computed. Note also that stitching without an adapter
at layer ℓ is equivalent to exchanging the full residual stream at all positions at
that layer, so that our stitching and patching results are two views of one family of
interventions rather than independent tests; their convergence with steering, which
does not exchange states between checkpoints, is the stronger argument. Two further cautions apply. A probe shows that information can be decoded, not that the
model uses it (Hewitt & Liang, 2019; Elazar et al., 2021); this is why we rest the claim
that the answers misread the representation on the causal tests rather than on the probes.
Conversely, interventions along a subspace can change the output through a pathway other
than the one the subspace was meant to capture (Makelov et al., 2023); our steering
results therefore show which way the answer pathway reads a direction, and we compared
them with control directions drawn from the same high-variance subspace and with the
decision direction, but they do not prove that the model uses this direction under
natural conditions.

## The direction of the change depends on the training signal

Post-training impaired the readout of authorship, repaired the readout of negated
statements in the base models that misread them, and, for the reviews, replaced a shared
polarity heuristic by a more uniform readout across most models. A common
account of these outcomes follows from the second lens in Figure 1. Post-training does
not add information to the representation; it installs a way of reading the
representation out. Whether this improves or impairs a judgment then depends on whether
the training signal specifies a valid way of reading the question. For truth judgments,
it does: The supervised fine-tuning data of both models contained thousands of
statements to be judged as true or false, and the models learned to take the logic of
the statement into account, including its negation. For authorship and deception, it
does not: The data contained no such judgments, and the readout drifted toward what
post-training made salient. For the reviews, this was a shared heuristic of all models,
namely that long and detailed reviews are genuine, a cue without validity. For
authorship, it was an inverse use of the information, which might reflect an outdated
stereotype of AI text (a speculation that we discuss below). In terms of
the lens model, post-training changes the weights of the second lens, and these weights
are valid only to the extent that the training data contain valid examples.

This account explains why the sign of the change differed between judgments, and, for
the reviews, between models: Because post-training made the models' judgments more
similar (within each review polarity), the change was negative for a model whose base version had read the reviews comparatively
well (Gemma 3 4B) and positive for a model whose base version had not (Qwen3). Two
observations qualify this account. First, the base models of Qwen3 and OLMo 3 already
read negated statements correctly, presumably because their pretraining data already
contained instruction-like material; for them, post-training had nothing to repair and
left the readout intact or lowered it slightly. Second, OLMo 3 Think, post-trained mainly
for explicit reasoning, did not join the convergence of the other models on the reviews;
a training signal aimed at reasoning traces may install a different way of reading the
question than one aimed at conversation. It also
qualifies the superficial alignment hypothesis in a second respect. A change that leaves
the representation of the text largely unchanged is superficial with respect to what the model knows, but not
with respect to its consequences, which can make a judgment considerably better or worse.
Note, however, that this account runs the risk of circularity: If the presence of a
valid training signal were inferred only from the improvement of a judgment, the account
would explain every outcome and predict none. The inspection of the training data
provides an independent measurement, and it agrees with the account: The data contained
thousands of truth judgments but no authorship or deception judgments. This comparison,
however, is correlational and rests on two data mixtures. It cannot show which examples
produced the repair of the negated statements (the truth judgments themselves, or, for
instance, the many mathematical and reasoning tasks that teach a model to read a question
precisely), and the two mixtures differed in size, languages, the share of mathematical
tasks, and the kind of their truth judgments. We therefore carried out a training study
with control conditions (Section 3.7). Adding two thousand correctly labeled authorship
judgments to otherwise identical fine-tuning data improved the readout of texts whose
cues resembled those of the training texts and impaired it where they did not; adding
stereotyped labels shifted the readout in the direction implied by the validity of the
stereotype, but reliably only where the stereotype was valid; and in all conditions the probes at the text positions changed little while
the position at which the answer is formed was reorganized. The training signal can thus
set the direction of the readout, as the account assumes. The training study does not
show, however, that the decline after the official fine-tuning of OLMo 2 has the same
origin: Our background condition, with a small part of the original data and low-rank
adapters, did not reproduce this decline, and identifying the parts of real post-training
data that install a misreading remains a task for future work.

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

The parallel to human judges is equally instructive. Human readers fail to recognize AI
text because they rely on cues that are plausible but invalid (Jakesch et al., 2023;
Rosin & Laubrock, in preparation),
which in terms of the lens model is a failure of cue utilization rather than of cue
availability (Brunswik, 1956; Karelaia & Hogarth, 2008). The models in our study fail in
a similar way: Valid cues are present in the texts and in the representations, and the
answers do not use them.

## Why would human-oriented post-training impair the detection of AI text?

Why should training that makes a model more helpful and more agreeable to humans make it
worse at telling human from AI text? Our data do not answer this question conclusively,
but they constrain the possible answers. The lens analysis of the chat messages showed
what the post-trained models took as signs of AI authorship: long words, adjectives,
nouns, and text that is hard to read, i.e., a formal and nominal style. In our corpus,
this was the wrong way round. The AI messages, written by current models instructed to
sound like a private group chat, were more verbal and lexically more diverse than the
students' messages, and the post-trained models took exactly these properties as signs
of human authorship. Except for Qwen3, whose base model already weighted these
properties inversely (G = −.52, after post-training −.58), the base models used them
less; post-training installed or strengthened the weighting, and in the two models with
stage-wise checkpoints it did so during supervised fine-tuning.

One possible explanation, which our data cannot test directly, is an outdated
stereotype of AI text. The picture of AI writing
as formal, elaborate, and impersonal fits the output of earlier assistant models and is
widespread in texts about AI, which may well be among the data that shape a model's
concept of its own kind of text during post-training. Current models that are asked to
imitate a group chat no longer fit this picture, so that a model that relies on it
judges informal AI text as human. If this explanation is correct, the error is a moving
target: As human and AI raters become more familiar with generated text, and as
generated text changes with each generation of models, the stereotype that post-training
installs may change as well, and the direction of the error with it. Two alternatives
are less consistent with our data. Familiarity with the model's own style would predict
that text in a formal assistant register is judged human, which is the opposite of what
we observed, and token-level predictability did not account for the below-chance
judgments (for Gemma 3 4B, controlling for the model's own surprisal of each text left
them unchanged
on the short chats and even enlarged it on the newspaper segments; supplementary
analysis). The idea that post-training narrows the concept of AI text to explicit
assistant markers is not supported either: Removing the markers that the students had
identified did not change the models' judgments.

OLMo 2 does not fit this picture. After supervised fine-tuning, it weighted the same
properties in the opposite direction and judged the chats slightly better than its base
model, whereas its judgments of the other two corpora became worse. We can only
speculate about the reason; its fine-tuning data contain a large number of real user
messages from chat logs, which might have shaped a more accurate picture of informal
human writing. Note also that the lens analysis is correlational and that many of the
text properties are correlated with each other; it shows which properties the answers
follow, not which ones cause them.

## Implications

Three implications follow, each of which requires further testing. First, the side
effects of post-training can be localized, and possibly repaired. A stitched model that
combined the first eight layers of the Apertus base model with all upper layers of the
fine-tuned model discriminated better than either parent, although such hybrids also
produce effects of their own (Section 3.4). Whether such hybrids retain
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
restricted to two models per judgment, to one corpus for authorship, and to the plain
prompt format, because base models cannot be queried with a chat template. The two
further judgments were studied with four to five models, steering for them with two
models each and on 300 items, and the effects of post-training on the reviews were
small; for Gemma 3 4B, steering did not reveal the source of the loss. The training
study used one base model, low-rank adapters, and training texts from other generators
and of another genre than the test corpora, and its background condition did not
reproduce the effect of the original fine-tuning; the stereotype labels were moreover
correlated with text length, topic, and generator. The three causal methods were applied
to partly different combinations of models and corpora; only Apertus on the textbook
summaries was examined with steering, stitching, and patching alike. The newspaper
segments stem from only 14 articles, so that bootstrap intervals that resample articles
are likely too narrow for this corpus. All judgments used German prompts with one
instruction wording and the question after the text; prompt variants are reported in the
supplement, and we did not test English prompts.
 Models
from different developers also differ in their post-training recipes, including distillation from larger
models, the emphasis on reasoning, and the share of human preference data, and our
design cannot separate these factors; they are a plausible source of the heterogeneity
between models and text types. The few-shot results were mixed, and they neither rule
out nor confirm that a correctable belief contributes to the failure. Finally, we defined
the family of hypothesis tests for the post-training contrasts after inspecting the full
set of comparisons; we therefore report the result under a stricter correction across
all contrasts as well, which left five of six decreases significant.

## Conclusion

Language models that write like humans fail to recognize their own kind of text, not
because they lack the information but because post-training changes how they read it
out. The same holds, with a different sign, for judgments that post-training improves:
In all three judgments we studied, post-training changed little in the representation
of the text and changed how the question reads it, at the positions of the question and
through weights in the early or middle layers. To know what a model knows, it is therefore not enough to look at
its answers; one has to look at how the question is read. A training study showed
that the training data can set the direction of this readout with little change at the
text positions. Future studies with larger models may show how general this mechanism is,
which parts of real post-training data install a misreading, and whether such side effects
can be prevented without giving up the benefits of post-training.
