# 4 Discussion


In the present study we used authorship judgments as a test case for the question of
whether post-training changes what a language model represents or how it reads this
representation out. Across eight families of open-weight models and three corpora of German texts,
the information that distinguishes AI from human texts was readily available, both in
the texts themselves and in the models' internal representations, but the models'
answers did not use it, and often used it with the wrong sign. Post-training lowered
the sensitivity of the answers on average (Δd′ = −0.33), with considerable variation
between model families and text types, and in the two families with stage-wise
checkpoints the change occurred during supervised fine-tuning. Three causal methods
converged on where this change takes effect. Steering showed that the direction that
separates AI from human texts drives the answer in the wrong direction after
post-training. Model stitching showed that the damaging change originates in the early
layers. Activation patching showed that it does not reside in the representation of the
text but in the processing at the positions of the question that follow the text. In
short, post-training did not change what the models represent about a text; it changed
how the question they are asked reads this representation out.

## Relation to previous work

Our findings are consistent with the growing evidence that language models represent
more than they say (Burns et al., 2024; Liu et al., 2023; Orgad et al., 2025), including
for authorship (Zhou et al., 2026; Ackerman & Panickssery, 2025). They add three points.
First, the dissociation is not only a gap but frequently a reversal. A miscalibrated
decision threshold, which has been proposed as the main source of probe–answer gaps
(Villuri et al., 2026), shifts the criterion in terms of signal detection theory; it
can lower accuracy but cannot produce sensitivity below chance. Reversed readouts of a
discriminative direction have been described for steering vectors (Torop et al., 2026);
here, the reversal occurred in the models' unmodified answers and was confirmed against
strict control directions and a positive control. Second, the reversal is linked to
post-training. Interestingly, in the closest related study, post-training had the
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
stitching and patching results provide causal evidence for it in one respect: The
representation of the text was interchangeable between base and post-trained
checkpoints, so post-training did not change what the models encode about a text. In
another respect, our results qualify it. The change was not superficial in the sense of
affecting only the final stages of processing or the output format; it originated in
the early layers and altered how the question takes up the text, and it had substantive
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
in a question phrased in natural language. In contrast, the early origin of the change
departs from the common picture that fine-tuning mainly alters the upper layers
(Merchant et al., 2020; Zhao et al., 2026; Zhou, 2026b). It is compatible, however, with
evidence that changes in the late layers can depend on altered earlier computation
(Zhou, 2026a), that post-training retunes routing mechanisms formed in pretraining
(Gupta et al., 2026), and that task representations formed early select later
processing pathways (Bigoulaeva et al., 2026). Note that stitching without an adapter
at layer ℓ is equivalent to exchanging the full residual stream at all positions at
that layer, so that our stitching and patching results are two views of one family of
interventions rather than independent tests; their convergence with steering, which
does not exchange states between checkpoints, is the stronger argument.

## The question reads the text

What does it mean that post-training changes how the question reads the text? In a
language model, the answer is produced at the last position of the prompt, and
information from the text reaches it only through the positions in between, i.e., the
question. At each layer, these positions retrieve information from the text through
attention and transform it through the feed-forward blocks. Post-training could
intervene at either step, by changing what is retrieved or by changing what is made of
it, and the stitching results suggest that both are involved. A psychological
description of such a change is a different construal of the question. A model after
post-training may understand "written by a human" differently than its base model did,
much as a task set determines which features of a stimulus an observer selects and how
they are mapped onto a response. In accounts of biased competition, a task set acts
early, at the stage of selection, and not only at a late decision stage (Desimone &
Duncan, 1995). It is instructive to see that the effect of post-training, too, arose in
the early layers, where the question begins to select information from the text, and
was fully located in the question positions by the middle of the network.

The parallel to human judges is equally instructive. Human readers fail to recognize AI
text because they rely on cues that are plausible but invalid (Jakesch et al., 2023),
which in terms of the lens model is a failure of cue utilization rather than of cue
availability (Brunswik, 1956; Karelaia & Hogarth, 2008). The models in our study fail in
a similar way: Valid cues are present in the texts and in the representations, and the
answers do not use them.

## Why would human-oriented post-training impair the detection of AI text?

Why should training that makes a model more helpful and more agreeable to humans make it
worse at telling human from AI text? Our data do not answer this question, but they
constrain the possible answers. A first observation is that plain AI texts, generated
without any instruction to sound human, were judged as more human than AI texts that
had been optimized to resemble human writing. On the short chats, for example, the
post-trained Gemma 3 12B separated human texts from plain AI texts with an AUC of .18
but from optimized AI texts with an AUC of .26, and the post-trained Qwen3 with AUCs of
.20 and .36. This tendency was already present in the base models and became stronger
after post-training (Gemma 3 12B: .39 before and .18 after post-training). The purest
assistant style thus appeared most human to the models. We see three explanations, which
are not mutually exclusive.

First, the reversal might reflect familiarity. During supervised fine-tuning, a model
learns to produce a particular assistant style, which then becomes its most typical and
most fluent kind of text. If the post-trained construal of the question maps "natural,
like the text I would write" onto "human", texts in the model's own register would be
judged human. Note that token-level predictability did not account for the reversal:
For Gemma 3 4B, controlling for the model's own surprisal of each text left the
reversal unchanged on the short chats and even enlarged it on the newspaper segments
(supplementary analysis). The relevant familiarity would therefore have to concern
register and style rather than the predictability of single words. Second, the
model might inherit the stereotype of its raters. Preference data reward polished,
warm, and helpful text, and the people or models who provide these ratings cannot
themselves tell AI from human text and hold invalid heuristics about it. Optimizing for
what raters prefer could thus tie "human-like" to the features of polished assistant
text, so that post-training acts as an implicit optimization against the detection of
its own output. Third, post-training might narrow the model's concept of AI text to
explicit assistant markers, such as disclaimers or bulleted lists, so that texts without
such markers default to "human". These explanations make different predictions. The
first predicts the strongest reversal for AI text whose style is closest to the model's
own fine-tuning data, the second predicts that the reversal grows with the amount of
human preference optimization, and the third predicts that AI texts with explicit
assistant markers are recognized while texts without them are not. All three are
compatible with a changed construal of the question; they differ in why the construal
changed.

## Implications

Three implications follow, each of which requires further testing. First, the side
effects of post-training can be localized, and possibly repaired. A stitched model that
combined the first eight layers of the Apertus base model with all upper layers of the
fine-tuned model discriminated better than either parent. Whether such hybrids retain
the intended effects of post-training, such as instruction following, is an open
question, and the approach is related to combining fine-tuned weights by interpolation
or averaging (Wortsman et al., 2022a, 2022b). Second, knowing where a side effect arises
makes it possible to protect the affected components during fine-tuning or to correct
them with small, targeted interventions instead of broader training. Third, and in our
view most important, post-training can change how a question is read while the
knowledge that the answer would require remains in the model. The answer then no longer
reflects what the model represents. This is a concrete and measurable instance of a
model that knows something but says something else, and it suggests that comparing the
representation at the question positions with the answer might be a useful tool for
auditing models, in the spirit of distinguishing the presence of a feature from the
behavior it supports (Templeton et al., 2024) and of examining which representations a
model can verbalize (Gurnee et al., 2026). From the perspective of self-regulation, in
which post-training is often seen as a top-down control of the dispositions acquired in
pretraining, our results add a cautionary note: A top-down task set can bias perception
and not only correct a decision.

## Limitations

Several limitations qualify these conclusions. We studied models with 4 to 12 billion
parameters, and the reversal might be weaker or absent in larger models. The
information that the models failed to use was largely lexical and stylistic, as shown by
simple n-gram classifiers, so our results concern the readout of ordinary cues rather
than of any deep representation of authorship. The stitching and patching analyses were
restricted to two model families with stage-wise checkpoints, to one corpus, and to the
plain prompt format, because base models cannot be queried with a chat template. Model
families also differ in their post-training recipes, including distillation from larger
models, the emphasis on reasoning, and the share of human preference data, and our
design cannot separate these factors; they are a plausible source of the heterogeneity
between families and text types. The few-shot results were mixed, and they neither rule
out nor confirm that a correctable belief contributes to the failure. Finally, we defined
the family of hypothesis tests for the post-training contrasts after inspecting the full
set of comparisons; we therefore report the result under a stricter correction across
all contrasts as well, which left five of six decreases significant.

## Conclusion

Language models that write like humans fail to recognize their own kind of text, not
because they lack the information but because post-training changes how they read it
out. To know what a model knows, it is therefore not enough to look at its answers; one
has to look at how the question is read. Future studies with larger models, controlled
post-training, and other judgments may show how general this mechanism is and whether
it can be prevented without giving up the benefits of post-training.
