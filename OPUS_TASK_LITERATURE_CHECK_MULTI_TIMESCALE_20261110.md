# One read-only job for a Claude Opus 5.5 worker: literature check on learning and memory at several timescales (10 November 2026, 14:15 EST)

> **Status under the process rules of 10 November (review 9.82).** Read-only; it may start without an audit of the brief. Its memo is audited like any other deliverable. **Commits:** one commit at completion on the default branch; partial receipts stay on your session's own branch or uncommitted, never on the default branch. This replaces any sentence below about pushing partial receipts.

**Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. **This job runs no experiment and edits no existing file.** It needs web access to read papers. If your session cannot open web pages, say so in your first message and stop: a review written from memory is not wanted.

**Branch rule.** Commit your deliverable to `codex/minifly-consolidated-20261023` if your session can push there (pull with `--rebase` first, never force); otherwise to the branch your session is given, and name the branch and commit in your last message. Do not create any other branch.

## Why

Mr. Zhou's question, relayed by Mr. Xu on 10 November: the "Hope" paper reminded him that brain rhythms come at several frequencies, so in principle information arriving at time T can meet information from 120 seconds earlier that is still being processed in some inner loop. Is that in the mechanism list this project surveyed, and have people worked on it?

The coordinating session's answer, given partly from memory and marked so, was that the idea splits into three different mechanisms, of which the project has one:

| Reading | In this project |
|---|---|
| **A. State at several decay rates:** a trace from long ago is still present when new input arrives | built: the E17 memory banks have five fixed retention rates (0.5, 0.8, 0.95, 0.99, 0.999 per byte) |
| **B. An inner loop** that keeps transforming old information while new arrives | absent by design: the banks have no learned recurrence |
| **C. Weights that change at several rates,** fast and slow parts of the same learner | absent: every weight learns at one rate; a replay reservoir is the only slow store |

The earlier mechanism re-screen (`output/minifly-mechanism-rescreen-20261107/MINIFLY_MECHANISM_RESCREEN_MEMO_01_20261107.md`, sections 5.2, 5.3 and 5.7) kept "learnable state over multiple timescales", "shared recurrent computation" and "longer eligibility and selective consolidation" for later discussion, and set oscillatory state aside as needing separate justification. It has no entry for reading C.

Two measured weaknesses make B and C matter here. The sign-rule learner retains nothing without replay (E16-M3, review 9.64) and loses unrehearsed knowledge faster than backprop on the same data (E17 Phase 4, review 9.70): reading C is the candidate. The two-bank learner composes nothing under any update, backprop included (review 9.66 and 9.70): reading B is the candidate. This morning's memo (`PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`) corrected three of four claims the coordinating session had made from memory, so this answer needs the same treatment before it steers an experiment.

**Read first, in the repository:** the re-screen memo sections named above; review sections 9.64, 9.67, 9.70 and 9.72 in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`; this morning's memo (Q1 and Q4; do not redo them, cite them); `MOD97_LOCAL_RULE_20261106/arch17.py` (header); `PERSISTENT_CORE_REQUIREMENTS.md` (bounded work and memory per event).

## Claims to verify (each: confirmed, corrected, or not found)

1. **The paper.** "Nested Learning: The Illusion of Deep Learning Architectures" (Behrouz, Razaviyayn, Zhong, Mirrokni; arXiv 2512.24695) builds its memory as a chain of blocks whose weights are updated at different frequencies, motivates this by multi-timescale processing in the brain, and reports continual-learning gains for its "Hope" model. Verified so far only from an abstract-level search summary.
2. **Reading A is old and works.** Recurrent networks with units or modules running at different fixed or learned rates hold long dependencies better than single-rate ones; a bank of leaky integrators at a spectrum of rates amounts to a Laplace transform of the past.
3. **Reading C exists at the synapse.** Models that give each synapse a chain of internal variables at different timescales extend memory lifetime in theory, and have been applied in trained networks with a measured retention benefit.
4. **Reading B has a known cost.** Learned recurrence that is not element-wise makes exact online credit expensive, which is why this project's banks avoid it.
5. **The fly has parallel memories at different rates** (confirmed this morning from Aso & Rubin 2016; cite, do not redo), and oscillation proper plays an established role in ordering or combining old and new information in some nervous systems.

## Questions to answer

- **Q1. The paper itself.** What exactly is updated at which frequency, and by which learning rule? Is anything in it local, or is it trained by backprop within and across levels? What is "self-modifying" in it? What are the continual-learning results, against which baselines, at what scale, with how many seeds? Is the brain-rhythm link an argument or a decoration? Any independent replication or critique.
- **Q2. Weights at several rates, for retention without rehearsal.** Synapse-level models (cascades, chains of coupled variables), fast and slow weights, consolidation by slowly tracking copies: what each stores per synapse, whether the update is local to the synapse, and what retention each has been shown to buy in a trained network, on what task, against replay and against no protection. Which of them need a per-weight importance estimate, and is that estimate local?
- **Q3. Inner loops, for multi-step computation.** Methods in which a learned transformation is applied repeatedly to held state (iteration within a step, pondering, learned recurrence between memory layers, thalamocortical-style loops): evidence that iteration buys composition or multi-step arithmetic in small networks, separated from simply spending more computation. What happens to online credit when such a loop is added: exact cost, and the published approximations and their limits (cite this morning's Q1 where it applies).
- **Q4. Oscillation proper.** Do oscillating units (complex or rotational recurrence, coupled oscillators, phase-coded gates, nested fast-in-slow rhythms) buy anything over a bank of plain decay rates for holding several items in order and binding them? Report results where the comparison is controlled; say when it is not.
- **Q5. Biology, briefly.** What is established about old and new information meeting through multi-timescale activity or rhythms (for example nested rhythms, replay events, intrinsic timescale hierarchies), what is contested, and what is known in the fly beyond this morning's Q4, including any oscillatory loop in the mushroom body. Report without arguing a side.
- **Q6. What follows for this project (suggestions; they decide nothing).** For each of A, B and C name the closest published method and its reported limits. Then at most one concrete design each for: (i) a synapse-local fast-and-slow weight test of retention at low replay share in the existing window learner; (ii) an inner loop in the two-bank learner that keeps credit forward-only or prices what is lost; (iii) whether oscillating units are worth a test at all. For each say what it would cost per synapse and per byte.

## Rules of evidence

- Every factual claim cites a source opened and read in this session: authors, year, title, venue, a working link, and where in the paper the support is (section, table or figure). Say for each central number whether you or a sub-agent read it, and spot-check the central ones yourself.
- A claim you could not check goes under "From memory, not verified", or is left out. Never cite a paper you did not open. Never invent a reference, a number or a link. If a search finds nothing, say so.
- Give numbers as reported, with benchmark, model size and number of seeds; say when sources disagree and when a result is on a small benchmark.
- Prefer primary papers to blog posts and surveys.

## Deliverable

`PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_MULTI_TIMESCALE.md`, at most about 2,500 words plus references: a half-page summary first (the five verdicts, and a three-row table saying for A, B and C what is established and what it would cost here); then Q1–Q6; the reference list; "From memory, not verified"; what you searched for and did not find. One paragraph for the review as `PROJECT_DOCUMENTS/LIT_CHECK_20261110/REVIEW_SECTION_9_74.md`. The memo is a reference for the coordinating session, Mr. Xu and Mr. Zhou; it decides nothing, and it is written so that Mr. Zhou can read the summary alone.
