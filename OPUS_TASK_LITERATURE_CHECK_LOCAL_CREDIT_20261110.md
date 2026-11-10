# One read-only job for a Claude Opus 5.5 worker: literature check on local credit through time and layer-local targets (10 November 2026, 12:00 EST)

**Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. **This job runs no experiment and edits no existing file.** It needs web access to read papers. If your session cannot open web pages, say so in your first message and stop: a review written from memory is not wanted.

**Branch rule.** Commit your one deliverable to `codex/minifly-consolidated-20261023` if your session can push there (pull with `--rebase` first, never force); otherwise to the branch your session is given, and name the branch and commit in your last message. Do not create any other branch.

## Why

On 10 November Mr. Xu asked whether delayed feedback from the stream (the next byte, or wider context arriving later) opens designs in which local learning rules need not compete with backprop head to head. The coordinating session answered from memory and said so. Several claims in that answer, and in the audit session's roadmap note (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/ROADMAP_NOTE_20261110.md`, which also asks that its literature pointers be checked), now steer experiments (E17 Phase 4, E18). They should rest on sources someone has actually read.

**Read first, in the repository:** review sections 9.60, 9.66 and 9.67 in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`; `MOD97_LOCAL_RULE_20261106/arch17.py` (its header and `forward`); `MOD97_LOCAL_RULE_20261106/exp45/E17_ALIGN_AT_BIRTH.md`; `OPUS_TASK_E18_LOCAL_HEADS_BIRTHS_20261110.md`. The learner in question has memory units with fixed, element-wise retention (no learned recurrence), written through a gate from the current input; an exact per-synapse eligibility trace exists for the write synapses; the open cost is credit for the layers above a write and for a lower bank through an upper bank's memory.

## Claims to verify (each: confirmed, corrected, or not found)

1. For recurrent units with element-wise (diagonal) recurrence, the exact online gradient within one layer costs about one trace per parameter, and published multi-layer versions approximate the credit that passes through an upper layer's memory.
2. Layer-local or block-local training objectives match backprop on small problems and fall behind as scale grows; none has been shown at language-model scale.
3. Sign-symmetric feedback stays close to backprop at large scale in feedforward networks, while feedback alignment and direct feedback alignment fall behind as depth grows.
4. In the fly mushroom body, the timing of dopamine relative to the odor sets the sign of plasticity at the output synapses, on a scale of seconds, and the dopamine neurons carry signals other than reward.

## Questions to answer

- **Q1. Online credit in stacked element-wise recurrent layers.** What is exact, what is approximated, at what memory and compute cost, and how far from backprop through time do the reported results fall? How do the methods treat parameters upstream of the recurrent unit (an input projection or a small network before the write), which is this project's case? Starting points, to be found and read, not trusted: online learning with independent recurrent modules or linear recurrent units; e-prop; online spatio-temporal learning; sparse approximations of real-time recurrent learning.
- **Q2. A target at every layer.** Methods that give each layer or block its own objective, especially a prediction of the next input or of a later representation: what they reach against backprop, at what scale, on sequences and language in particular; their known failure modes (a layer learning what is predictable rather than what the next layer needs; collapse of learned targets); and what has been tried against those. Starting points: greedy contrastive or predictive coding per block; local error signals per layer; predictive coding in time; the forward-forward family.
- **Q3. Feedback without weight transport, in depth and in time.** Sign symmetry, learned feedback weights, feedback alignment, direct feedback: reported results at scale, and any results in recurrent networks or through time.
- **Q4. Where the fly's olfactory learning pathway is plastic.** For each stage (receptor to projection neuron, projection neuron to Kenyon cell, Kenyon cell to output neuron, and recurrent or feedback loops): is experience-dependent plasticity reported in the adult, of what kind, on what timescale? What do the dopamine neurons encode besides reward and punishment? Report what is established, what is contested and what is unknown. This bears on Mr. Xu's statement that an animal has no frozen first stage; report the evidence either way without arguing a side.
- **Q5. What follows for this project.** For each of Q1–Q3, name the published method closest to (a) E17 Phase 4's arm (weight-sign feedback with credit through time at every layer) and (b) E18's local-heads arms, say what that literature reports about its limits, and list at most three concrete design changes it suggests for a two-bank learner with fixed retention rates. Mark these as suggestions; they decide nothing.

## Rules of evidence

- Every factual claim cites a source you opened and read in this session: authors, year, title, venue, a working link, and where in the paper the support is (section, table or figure). Paraphrase; quote at most a short phrase.
- A claim you could not check against a source is written under a separate heading, "From memory, not verified", or left out. Never cite a paper you did not open. Never invent a reference, a number or a link. If a search finds nothing, say so.
- Give numbers as reported, with the benchmark and the model size, and say when two sources disagree.
- Prefer primary papers to blog posts and surveys; a survey may be used to find papers.
- Keep the scale honest: say plainly when a result is on a small benchmark.

## Deliverable

One file, `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`, at most about 2,500 words plus the reference list: a half-page summary with the four verdicts first; then Q1–Q5; then the reference list; then "From memory, not verified" if anything is in it; then what you searched for and did not find. One paragraph for the review as `PROJECT_DOCUMENTS/LIT_CHECK_20261110/REVIEW_SECTION_9_72.md`. The memo is a reference for the coordinating session, Mr. Xu and Mr. Zhou; it decides nothing.
