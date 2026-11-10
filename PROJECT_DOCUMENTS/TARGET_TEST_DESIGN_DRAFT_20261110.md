# One target test for a continuing learner: design draft for audit

10 November 2026, coordinating session. **Design only. Nothing here is to be run.** It goes to Codex for audit first; after that its assay is qualified with no learner at all; then on references; only then on a candidate. It merges three proposals that ask for the same endpoint: Codex's promotion test (`MINIFLY_PROJECT_STATUS_AND_PROMOTION_TEST_20261110.md`), Codex's delayed-observation test (`MINIFLY_E18_DELAYED_OBSERVATION_MEMO_20261110.md`, last section), and the audit session's battery (`MINIFLY_REVIEW_20261109/ROADMAP_NOTE_20261110.md`, 4.1). Its form follows `NEXT_STEPS_NOTE_20261110.md`, 3.5. Open choices are listed at the end and are open.

## 1. The claim the test can support or refuse

One continuing learner, taught by nothing except the later bytes of its own stream, uses an earlier observation to choose its first output on cases whose recent bytes are identical and whose earlier history differs; it still does so after other learning; it changes its answer when the world changes and carries the change to cases it has not been corrected on; and what this costs is counted.

Each clause is a separate verdict. The test does not ask which learning rule is used.

## 2. The stream

Bytes only. No answer mask, no task code that names the regime, no boundary signal. The learner predicts every next byte; that is the only teaching signal. The network is not part of the design.

**Episodes.** `X` filler `C` `=` `dd` end, where `X` is a context symbol, `C` a cue symbol, the filler is 0 to k letters, and `dd` is a two-digit outcome that the stream itself writes after `=`. The outcome is the reveal. The world's rule is hidden: each context has a value v(X) and each cue a value u(C) in Z_m, and the outcome is (v(X) + u(C)) mod m. A second cue family uses another operation, so that there is an unrelated skill.

**What is in view and what must be remembered** (rule 1 of the coordinator's note). At `=` the cue and the last letters are recent bytes. The context is not, whenever the filler is at least as long as the learner's declared window; panels use only such fillers. Predicting the first outcome digit therefore needs the context from memory and both hidden values from earlier learning.

**Held combinations.** A set of (context, cue) pairs never appears in teaching. Every context and every cue appears in other pairs, so a held pair's outcome follows from what is taught and from nothing else. Exposure is counted over every source and every equivalent form (rule 13): the pair itself, any episode that reveals the same sum by another route, and replay contents.

**A wider view, as an arm and not a premise.** In one arm the stream adds, some bytes after the outcome, a line that states the two hidden values. That is the delayed observation of Mr. Zhou's idea in a form that can fail: later bytes that explain earlier ones.

## 3. The endpoint

**Twins, scored on the first output, before the reveal.** Two held episodes with the same cue, the same filler and therefore the same recent bytes at `=`, and different contexts with different outcomes. The learner writes the outcome itself; both must be right through the end symbol. A responder that sees only the recent bytes can get one of a twin and never both.

**Scored on first encounter.** A held pair is scored the first time it occurs, then the stream reveals its outcome and it stops being held. A sealed panel that is never revealed is scored alongside. The first measures a learner that keeps learning; the second measures it without that help.

Secondary endpoints, each its own row: the unrelated skill before and after; after a revision, the share of answers that still follow the old rule; whole outputs as well as final digits.

## 4. The three phases, one learner state throughout

1. **Acquire.** The stream of section 2, both cue families. Twins and the unrelated skill are scored before any change.
2. **Interfere.** A stretch in which the first family does not appear. Twins are scored again on first encounter when it returns. What is retained with a declared replay budget and what is retained with none are separate rows.
3. **Revise.** The value of one familiar cue changes, unannounced. The stream goes on revealing outcomes for some pairs with that cue. The test is on held pairs with that cue that have not occurred since the change, on twins as before, and on the other cues, which must not move.

## 5. Controls, in the order they are run

| Step | Arm | It must | If it does not |
|---|---|---|---|
| 0a | An oracle that knows the hidden values | pass every endpoint | the panels are wrong |
| 0b | A responder that sees only the recent bytes | score no twin | the twins leak |
| 0c | A recency table (the last outcome seen for the cue) | fail the twins and the held pairs after a revision | the task rewards recency and is redesigned |
| 0d | The exposure ledger and panel hashes | show every held pair unexposed by every route | the split is rebuilt |
| 1 | Backprop on an ordinary recurrent network, same bytes | acquire: twins well above 0c | diagnose the stream or the teaching before any candidate |
| 2 | The same reference through phases 2 and 3 | show what an ordinary learner retains and revises, with and without replay | that is the baseline, whatever it is |
| 3 | A learner taught on a stream that lacks the association a twin needs | fail those twins | success cannot be credited to earlier learning (the control evidence level E3 requires) |
| 4 | The wider-view arm against the same stream with those bytes shuffled within delay and frequency strata, and against a stream with as many extra ordinary bytes | beat both | later views add targets, not information |
| 5 | A candidate, against backprop on the same bytes and network | whatever is registered then | — |

Steps 0a to 0d need no learner and no roster. Nothing after step 0 starts until they pass. Nothing after step 1 starts if the reference cannot acquire.

## 6. The ledger kept in every run

External bytes delivered; bytes under the loss; replay rows and their bytes; update CPU and scoring CPU separately; state, trace, optimizer and replay memory; the declared window and sweep length. Scores are reported against delivered consequences and against CPU, not only against updates.

## 7. What this design does not yet decide

1. **Sizes:** contexts, cues, modulus, filler length, how many pairs are held.
2. **Whether state carries across episodes.** As drafted the context is inside the episode. A harder form puts it in an earlier episode, which needs a learner whose state is not reset between records; no learner on file has that.
3. **The window.** A learner with a bounded window declares it, and fillers are longer. Whether one stream can serve learners with different windows fairly.
4. **The strata for the shuffled control,** and whether the wider-view arm belongs in the first version or the second.
5. **The pass marks.** None is written here on purpose: they are registered after step 1 shows what the reference does, and before any candidate runs.
6. **What evidence level is claimed.** As drafted, first-encounter twins with step 3's control aim at E3 for the acquire phase only.
7. **Whether arithmetic values are the right hidden structure.** They make held pairs inferable and outcomes checkable; they also invite the shortcuts the E-series met. A table with no algebra would remove inference; a richer world would cost qualification.

## 8. Why these choices, from this project's own record

- No answer mask and no composition in view, because those were the two aids found late (reviews 9.76 and 9.82).
- Twins and both-correct scoring, because a pooled panel and a suffix metric let the first E20-A design answer the wrong question (9.79).
- An oracle and a recent-bytes responder before any roster, because the assay check for E20-A is what finally showed it could separate memory from its absence.
- A reference before a candidate, because two days went to learning rules for a network that could not do the task under backprop.
- Pass marks after the reference and before the candidate, because gates written by whoever runs the experiment have moved.
