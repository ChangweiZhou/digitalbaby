# Cells that count, and an answering cell that learns how far to trust each (roadmap second revision, tracks L and C; a step added from the evidence) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. Keep experimenting. Respect evidence and keep trying new ideas." **Source.** `claude/friendly-darwin-8mmhfm` at 2641ac38e; text, split and measure as in `PLAN_1.md`. Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

1. `PLAN_1.md`: cells that count, read through a fixed discount from the longest to the shortest, give 1.5225 nats; cells whose strengths are changed by the mismatch and summed give 1.5317.
2. `PLAN_2.md`: in the summing form, added cells that have seldom been active do harm (a clock blind to the words: 0.023 nats worse), and cells that step once a word gain only 0.008. A cell that has been active twice speaks as loudly as one active a thousand times.
3. Mr. Xu, 20 November: "The connection strength (ideally can involve many cells) should be jointly decided by the cells together." In the line `nano/joint` a strength in two parts, one at each end, scored the same as a one-sided strength on sums (84.0 against 84.2): there the two parts had the same work to do. Here each end is given a different work.

## 2. The question

**If each cell counts what follows it and says what its own counts say, and the answering cell learns from the mismatch only how far to trust a cell of each kind and experience, is the system at least as good as the fixed discount, and can it then use kinds of cells that the summing form could not?**

*The sending end.* A cell is made the first time its context occurs and from then on counts the bytes that follow it (c(v), with total n and u different bytes seen). When active it says, for every byte v, s(v) = log((c(v) + e) / (n + 65 e)), with one small number e for all cells.

*The receiving end.* The answering cell holds a trust T for each kind of cell and each level of experience (ten levels of n: 1, 2 to 3, 4 to 7, ..., 512 and more; four of u: 1, 2, 3 to 4, 5 and more). Its branches compete on the sum over the active cells of T times s(v). When the byte arrives, the trust that was used changes by eta times the sum over branches of the mismatch times s(v); then the cells count. The effect of a cell on the answer is the product of what the cell holds and what the answering cell holds for cells like it; neither end reads the other's numbers.

One reading of the stream, counts and trust together. Then the last tenth is scored with nothing changing. (A second figure, marked as a different measure, lets everything go on learning through the last tenth.)

*Arms.* (0) the table's cells, K = 6; (1) as (0) with one trust for each kind whatever the cell's experience; (2) as (0) with the cells that step once a word (A, B, C of `PLAN_2.md`); (3) as (2) with the clock that is blind to the words; (4) as (0) with K = 8.

*Settings.* eta (0.0005, 0.002, 0.008, 0.03) and e (0.005, 0.02, 0.1, 0.5) are chosen for each arm on the last twentieth of the stream; every trust starts at 0.25.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With trust, the cells that step once a word are worth having: arm 2 is at least 0.02 nats below arm 0. *Refuted if* the difference is under 0.02.

| | Statement | Chance |
|---|---|---|
| Y1 | arm 0 is at most 1.5225 (the level of the fixed discount) | 0.55 |
| Y2 | arm 1 (trust by kind alone) is at least 0.02 nats worse than arm 0 | 0.70 |
| Y3 | arm 2 is at least 0.02 nats below arm 0 | 0.50 |
| Y4 | arm 3 (blind clock) gains less than half of what arm 2 gains over arm 0 | 0.65 |
| Y5 | arm 2 is below 1.47 | 0.35 |
| Y6 | arm 4 (K = 8) is below arm 0 | 0.60 |

## 4. Size, cost, what is not run

Cells as in the earlier plans (a cell holds 65 counts, most of them zero; the numbers of cells and of counts above zero are reported). The answering cell holds 40 trusts for each kind of cell: 280 for arm 0, 400 for arm 2. A run with its choice of settings takes a few CPU-minutes. Not run here: a trust held by each cell for itself; ends of words found by the cells; cells taught by the relay beside these.

A point for the reviewers, stated here and not hidden: the answering cell's trust depends on the sending cell's experience, so a cell's message carries two things, what its counts say and how much it has counted. Whether that is within what a cell has is theirs to rule.

## 5. Addition 1 (20 November; written after arms 0 to 3 were read and before the runs it describes)

For every arm the choice of eta fell on the smallest value the plan allowed (0.0005), so the plan's range was too narrow at that end. Arms 0, 2 and 3 are run again with eta 0.00005, 0.0001 and 0.0002 added to the choice (e as before), chosen on the last twentieth of the stream as before; the first runs stay in the table. Prediction (mine): each arm improves by under 0.01 nats, and the difference between arms 2 and 0 stays above 0.02 (chance 0.8).

## 6. Addition 2 (20 November; written after addition 1 was read and before the runs it describes): the control that the trust is learned at all

In addition 1 the choice again fell on the smallest eta allowed, and the loss on the last twentieth of the stream falls steadily as eta falls. The control that this calls for: eta = 0, every trust left at 0.25, for arms 0 and 2 (e = 0.02). If that is as good as the learned trust, the readings of this plan about trust are wrong and what works is the cells' counts read at equal weight. Prediction (mine): with eta = 0 arm 0 is worse than 1.5226 by more than 0.02 (chance 0.7).
