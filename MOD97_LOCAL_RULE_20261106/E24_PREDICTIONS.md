# E24 predictions: is the every-byte collapse the loss or the rule? (11 November 2026, written before any new birth)

**Brief:** `OPUS_TASK_E24_EVERY_BYTE_CAUSE_20261111.md`, revision 2 (16:19 EST, 11 November), read with review 9.99.

**Sources:** the codex branch at f4c6aa8c.

**This session's release:** the coordinator released the host-check wave only. It covers:
- the admission check;
- the two host-check births, D-ans 73129 and T-ans 73129, alone in a fresh folder;
- comparison of each with its pinned original.

D-all, T-all and T-bal are not released. They wait for the coordinator after Codex has read revision 2.

## Predictions (from the brief, copied unchanged)

- **P24.1** D-all ends at a hard mean of at least 60 in at least two seeds. Confidence about 60%.
- **P24.2** T-all ends at or below 10 in all three seeds. Confidence about 80%.
- **P24.3** T-bal ends at or above 40 in at least two seeds. Confidence about 35%.

## Readings (from the brief, copied unchanged)

- P24.1 and P24.2 hold: here backprop learns from every byte from nothing and the sign sweep does not. That is an interaction between the objective and the update in the tested configuration, and the next step is to look at what the sign feedback does with targets that cannot be predicted.
- P24.1 misses (D-all under 60 in two seeds): report the level. A birth under 60 may still have learned a good deal, so a miss is read with the penalty beside it and is not called a collapse unless the scores are near the floor. The rule is then not singled out, and the next step is the stream or the loss, for both configurations.
- P24.3 holds: in this configuration the weighting restores learning. It establishes the effect of the weighting here and nothing about where such a weighting could come from without an answer mask, which becomes the next question.
- P24.3 misses with P24.2 holding: the weighting did not restore learning to 40. T-bal is still judged against T-all seed by seed, and a partial improvement below 40 is reported as such.
- P24.2 misses: it takes one T-all seed above 10 to miss, and that leaves the question of survival unresolved, not answered. The three scores are reported, and only if all three are well above the floor are the earlier collapses reread as belonging to the local heads and to the life.

## Scoring (from the brief's endpoints)

- **What is scored:** the final checkpoint, k = 12, the top head, whole answers.
- **Hard mean:** the mean of add89 and sub89.
- **Penalty per seed and configuration:** D-ans minus D-all, and T-ans minus T-all. Their difference is the interaction.
- **Baselines on file:** D-ans 91.2 / 94.5 / 90.3 and T-ans 87.5 / 90.7 / 84.7.
