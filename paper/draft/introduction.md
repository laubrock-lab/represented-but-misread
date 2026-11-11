# 1 Introduction


Texts written by language models have become difficult to tell apart from texts
written by humans, and the question of who wrote a text now matters in education,
journalism, and science alike. Human readers are poor at this task. In studies with
different kinds of text, ranging from stories and news to poetry and self-presentations,
participants performed close to chance (Clark et al., 2021; Köbis & Mossink, 2021;
Jakesch et al., 2023; Porter & Machery, 2024), and in a three-party Turing test a recent
language model was judged to be human more often than the human it was compared with
(Jones & Bergen, 2025). Only readers who frequently use language models for their own
writing appear to detect AI-generated text reliably (Russell et al., 2025). Human
judgments rely on heuristics that are plausible but invalid, such as taking references
to personal experience or first-person pronouns as signs of human authorship (Jakesch
et al., 2023).

One might expect language models to do better. They produce such texts themselves, and
the statistical regularities that distinguish their output from human writing are, in
principle, available to them. However, language models asked directly whether a text was
written by a human or by an AI are weak and biased judges: Their errors lean in one
direction, and the direction differs between models and model generations (Bhattacharjee
& Liu, 2024; Yuan et al., 2026). When GPT-3.5 and GPT-4 judged transcripts of Turing-test
conversations, their accuracy was even below chance (Rathi et al., 2024). In pilot work
with small open-weight models, we observed the same pattern for German texts, and in
addition that models after post-training, i.e., after the stages of supervised
fine-tuning and preference optimization that turn a pretrained model into an assistant,
often judged worse than the pretrained models from which they were derived. Why would a
model that writes like an AI fail to recognize AI-written text, and why would the
training that makes it more helpful make it worse at this?

Two classic frameworks from psychology are helpful in asking this question precisely.
Signal detection theory separates the sensitivity of a judge, i.e., the ability to
distinguish signal from noise, from the criterion, i.e., the tendency to give one answer
rather than the other (Macmillan & Creelman, 2004). A bias toward answering "human" only
shifts the criterion; it cannot make the judge systematically take AI texts for human
texts and vice versa. Sensitivity below chance therefore indicates that the information
is used with the wrong sign. Brunswik's lens model adds the distinction between the
validity of the cues that a stimulus offers and the way in which a judge utilizes them
(Brunswik, 1956; Karelaia & Hogarth, 2008). Poor judgments can result from a lack of
valid cues or from valid cues that are not used, or used in the wrong way. For human
judges, the utilization of cues can only be inferred from their judgments. For language
models, it can be examined directly, because their internal representations are
accessible.

This possibility has been used to show that language models often represent more than
they say. Simple linear classifiers ("probes") trained on a model's internal activations
can recover properties such as the truth of a statement (Burns et al., 2024; Liu et al.,
2023; Marks & Tegmark, 2024) or the correctness of the model's own answers (Orgad et
al., 2025) better than the model's answers do, and prompting a model for a judgment can
underestimate what it knows (Hu & Levy, 2023). The same dissociation has been reported
for authorship: Models encode whether they wrote a text themselves although they fail
to report it (Zhou et al., 2026), and a direction in the residual stream that tracks
self-authorship can be used to steer what a model claims to have written (Ackerman &
Panickssery, 2025; see also Panickssery et al., 2024). Interpretability research at the
scale of frontier models has made the same distinction between the presence of a
feature and the behavior it supports (Templeton et al., 2024), and has identified a
small subspace of representations that a model is able to verbalize, whose content is
shaped by post-training (Gurnee et al., 2026). Why represented information fails to
reach the answer is less clear. One recent account attributes such gaps to a
miscalibrated decision threshold (Villuri et al., 2026), i.e., to the criterion in
terms of signal detection theory. This account cannot explain sensitivity below chance.
Another line of work shows that steering along a direction that discriminates a concept
can promote the opposite behavior (Torop et al., 2026), which suggests that the readout
of represented information can itself be reversed.

What post-training does to this relation is an open question. Several studies suggest
that post-training changes how a model expresses what it knows more than what it
encodes (Zhou et al., 2023; Lin et al., 2023; Waldis et al., 2026), and the
representation of whether a model is being evaluated, tracked across the published
checkpoints of one model family, is already present in the base model and amplified
during supervised fine-tuning (Heidari et al., 2026). To our knowledge, however, it has
not been tested whether post-training changes the representation of a property or the
way in which the answer reads it out, nor where in the network such a change is
located. Model families whose developers released checkpoints after every stage of
post-training now make it possible to address this question with causal interventions
rather than correlations.

Here we set out to explain why language models fail to recognize AI-written text. We
studied eight families of open-weight models with three corpora of German texts, one of
which consists of unpublished newspaper articles that cannot have been part of any
training data. First, we tested whether the information that distinguishes AI from
human texts is available in the models' representations, and compared it with simple
text statistics. Second, we measured with signal detection theory how the models' own
answers use this information, how post-training changes it from stage to stage, and
whether labeled examples can correct it. Third, we located the effect of post-training
with three causal methods: steering the internal state along the direction that separates
AI from human texts, combining the lower and upper layers of base and post-trained
checkpoints ("model stitching"), and exchanging internal states between checkpoints at
selected positions of the prompt ("activation patching"). We find that the information
is readily available but that the models' answers do not use it, and that post-training
does not change what the models represent about a text but how the question they are
asked reads out this representation.
