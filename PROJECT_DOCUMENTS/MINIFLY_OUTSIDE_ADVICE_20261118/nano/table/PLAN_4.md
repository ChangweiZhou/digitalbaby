# The trust's step shrinks with use (roadmap second revision, tracks L and C; one change from `PLAN_3.md`) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 9346c104e; system, text, split and measure as in `PLAN_3.md`. Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

`PLAN_3.md`, reading 5: with a constant step for the trust, the smallest step allowed was chosen twice, and the loss on the last twentieth of the stream still fell as the step fell. A trust for a common kind and level is used several hundred thousand times in one reading; with a constant step it never settles. In the summing form of `PLAN_1.md` each cell's step already falls as eta over the square root of one plus the times the cell has been active.

## 2. The one change

A trust's step is eta over the square root of one plus the number of times that trust has been used. Nothing else changes (e = 0.02 as chosen in every arm of `PLAN_3.md`; every trust starts at 0.25; one reading). eta is chosen from 0.002, 0.008, 0.03, 0.1, 0.3 on the last twentieth of the stream.

*Arms.* 0, 2, 3 and 4 of `PLAN_3.md` (the table's cells; with the cells that step once a word; with the blind clock; runs of eight bytes), and one further arm, 5: runs of eight bytes together with the cells that step once a word, to see whether the two gains of `PLAN_3.md` add.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The constant step was holding the system back: with the shrinking step arms 0 and 2 each improve by at least 0.003 nats on the last tenth (to at most 1.5196 and 1.4943). *Refuted if* either improves by less; then reading 5 of `PLAN_3.md` is withdrawn.

| | Statement | Chance |
|---|---|---|
| Z1 | the chosen eta is not at an edge of the range, in every arm | 0.60 |
| Z2 | arm 0 at most 1.5196 | 0.55 |
| Z3 | arm 2 at most 1.4943 | 0.55 |
| Z4 | arm 2 stays at least 0.02 below arm 0 | 0.85 |
| Z5 | arm 2 below 1.47 | 0.15 |
| Z6 | arm 5 (eight bytes and the word cells) at least 0.005 below arm 2 | 0.55 |
| Z7 | arm 3 (blind clock) within 0.005 of arm 0 | 0.75 |

## 4. Size, cost, what is not run

As `PLAN_3.md`; the answering cell holds one more number for each trust (its count of uses). A run with its choice takes one to two CPU-minutes. Not run: a trust that depends on which other cells are active; ends found by the cells (line `nano/ends`); cells taught by the relay as one more kind (line `nano/loop`).

## 5. Outcome (entered 20 November after the runs; `res/trust_arm*_shrink1.json`)

Loss on the last tenth, nats a byte, nothing changing during it; beside each the figure of `PLAN_3.md` with a constant step.

| Arm | Shrinking step | Constant step (`PLAN_3.md`) | Named right | Chosen eta |
|---|---|---|---|---|
| 0, the table's cells, K = 6 | 1.5230 | 1.5226 | 54.3% | 0.002 (the smallest allowed) |
| 2, with the cells that step once a word | 1.4986 | 1.4973 | 55.2% | 0.002 (the smallest allowed) |
| 3, with the blind clock | 1.5209 | 1.5240 | 54.6% | 0.008 |
| 4, K = 8 | 1.5123 | 1.5139 | 54.8% | 0.002 (the smallest allowed) |
| 5, K = 8 with the cells that step once a word | **1.4940** | not run | 55.3% | 0.008 |

**The hypothesis is refuted**: with the shrinking step arm 0 is 0.0004 worse and arm 2 is 0.0013 worse than with the constant step; an improvement of 0.003 was asked. Z1 failed (the smallest eta was chosen in three arms of five). Z2 failed. Z3 failed. Z4 held (0.024). Z5 failed. Z6 failed narrowly (arm 5 is 0.0046 below arm 2; 0.005 was predicted). Z7 held (0.002). A run with its choice took 23 to 42 CPU-seconds.

Readings (mine).
1. **Reading 5 of `PLAN_3.md` is withdrawn.** The step of the trust was not holding the system back: a constant step, a ten times smaller constant step and a step that shrinks with use are within 0.004 nats of one another on the last tenth. On the last twentieth of the stream the smaller and the shrinking steps looked better by 0.003 to 0.010; the last tenth does not follow. Differences of that size between two settings are not carried from one slice of this text to another, and I shall not chase them.
2. **The two gains of `PLAN_3.md` do not add.** Runs of eight bytes gain 0.011 and the word cells 0.024 over the table's cells; together they gain 0.029, not 0.035.
3. **Where the form stands**: 1.494 to 1.499 with the ends of the words given, against 1.5165 for the reference's smoothed table and about 1.47 as nanoGPT's full setting is described. The remaining 0.025 is not in the trust's step.
4. The largest single difference measured on this line so far is not between forms: it is between a system that stops changing at the end of the stream and the same system going on (1.523 against 1.417 for arm 0; 1.494 against 1.383 for arm 5). The last tenth is a part of the text the stream did not contain, with its own names and words. The next plan therefore turns to what the system is for: what it keeps and what it gains when the text changes (`PLAN_5.md`).
