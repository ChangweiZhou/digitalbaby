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

## 7. Outcome (entered 20 November after the runs; `res/trust_arm*.json`)

Loss on the last tenth, nats a byte, with nothing changing during it. "Plan" is the choice within the plan's range of eta; "addition 1" with the smaller values added. Beside them: the reference's smoothed table of six bytes 1.5165; the fixed discount on the table's cells 1.5225 (`PLAN_1.md`); the summing form 1.5317, and 1.5237 with the word cells (`PLAN_2.md`); nanoGPT's full setting as it is described, about 1.47.

| Arm | Loss, plan | Loss, addition 1 | Named right | Cells that have counted | Trust not learned (eta = 0) |
|---|---|---|---|---|---|
| 0, the table's cells, K = 6 | 1.5264 | 1.5226 | 54.6% | 459,129 | 1.6491 |
| 1, one trust for each kind | 1.5443 | not run | 54.4% | 459,129 | as arm 0 |
| 2, with the cells that step once a word | **1.4983** | **1.4973** | 55.4% | 1,826,213 (2,428,284 counts above zero) | 1.8028 |
| 3, with the clock blind to the words | 1.5239 | 1.5240 | 54.7% | 2,104,052 | not run |
| 4, the table's cells, K = 8 | 1.5139 | not run | 55.1% | 1,432,252 | not run |

Chosen in every arm: e = 0.02 and the smallest eta allowed (0.0005 in the plan, 0.00005 in addition 1). A run with its choice of settings took 47 to 88 CPU-seconds; one run alone 4 to 8. The compiled loop equals numpy to 4e-15 on a prefix (`trust.py check`).

**The hypothesis holds**: with trust, the cells that step once a word lower the loss by 0.028 nats (0.025 with addition 1), and 0.02 was asked. Y1 failed (1.5264 in the plan's range; 1.5226 with addition 1, against 1.5225). Y2 failed narrowly (0.018, and 0.02 was predicted). Y3 held. Y4 held (the blind clock gains 0.0025 in the plan's range and nothing with addition 1). Y5 failed (1.4973 is above 1.47). Y6 held (K = 8 is 0.0125 below K = 6). Both additions' predictions held.

Readings (mine).
1. **Cells that count, read with a learned trust, are at the level of the fixed discount** on the table's cells (1.5226 against 1.5225) **and need no chain**: a cell of any kind can be added, and the answering cell finds out how far to trust it.
2. **In this form the cells that step once a word are worth having** (0.025 to 0.028 nats), and the same cells on a clock that is blind to the words are worth nothing. In the summing form the first gained 0.008 and the second lost 0.023.
3. **Longer runs become usable**: eight bytes are better than six by 0.0125. In both earlier forms eight were worse than six.
4. **The trust has to be learned.** Left at its starting value the system is 0.13 nats worse with the table's cells and 0.31 worse with the word cells added. What the answering cell learns (the tables are in the result files): little trust in a cell of a long run that has counted once or twice (0.07 to 0.15 against 0.25 at the start), little in a cell whose followers have varied, more in a short cell that has always been followed by the same byte (up to 0.40).
5. **The step of the trust is not yet right.** The smallest value allowed was chosen twice, and the loss on the last twentieth of the stream still falls as eta falls: with a constant step a trust that is used a hundred thousand times does not settle. A step that shrinks with use is the next change (one change; `PLAN_4.md`).
6. A second figure, **a different measure**: if counts and trust go on changing through the last tenth (each byte named before it is counted), arm 0 gives 1.4136 and arm 2 gives 1.3863. No reference taught by back-propagation has been measured in that way here, so these two figures stand beside nothing yet.
7. Not shown by this plan: that the cells can find the ends of words themselves (the ends were given); that cells taught by the relay add to these.
