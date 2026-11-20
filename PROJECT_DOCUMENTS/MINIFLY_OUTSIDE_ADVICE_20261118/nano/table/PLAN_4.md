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
