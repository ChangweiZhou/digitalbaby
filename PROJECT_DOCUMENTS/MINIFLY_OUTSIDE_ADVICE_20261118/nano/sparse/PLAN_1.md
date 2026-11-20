# Line `nano/sparse`, first plan (E1): counting cells keyed on a few chosen delays, on two-digit sums (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." Earlier the same day (`../../ROADMAP_SECOND_REVISION_20261120.md`, section 1): "The connection strength (ideally can involve many cells) should be jointly decided by the cells together." **Source.** `claude/friendly-darwin-8mmhfm` at fcc174ac; this line's branch is `claude/nano-sparse-8mmhfm`. Development level; no checker gate, seal or outside review is claimed. The idea is the coordinating session's; the readings marked "mine" are this session's.

**What was copied.** From `loop_small_case/cells2.py`: the symbols, `text()` and the split into shown and never-shown pairs (unchanged; `sparse.py` says so at its head). From `nano/table/trust.py`: the level of experience of a cell and the rule for the trust, with the number of byte values made a parameter (14 here).

**A disclosure.** Before this plan was written, the program was run once on development seed 1 at its default numbers (eta 0.002, e 0.02, trust born at 0.25, m = 3), as a test that it ran. It gave 56.6% of never-shown sums whole at 40,000 lines (hundreds 97.6, tens 56.6, units 100) and was flat from 5,000 lines on. The predictions below were written knowing this figure. That run should have come after the plan; I state it and do not count it as a result.

## 1. The evidence this follows

1. The relay system of `loop_small_case` answers 84.2% of never-shown two-digit sums after 400,000 lines (five seeds, median) and keeps 1.8% of addition after 200,000 lines of subtraction.
2. `nano/table`, `PLAN_5.md`: counting cells read with a learned trust lose 0.009 nats on a first text after three other books. But a cell for a whole run of bytes answers nothing it was not shown; on sums such a table answers no never-shown pair.
3. Line `nano/multi` (as reported to this session; not read here): in the relay form, connections of two sending cells jointly were worse than plain edges at equal numbers of strengths.

## 2. The question

**Do counting cells that are each keyed on the byte values at a few chosen delays, read with a learned trust, answer two-digit sums that were never shown?**

*The system.* A kind of cell is a set of at most m of the delays 1 to D (delay d: the byte d places before the byte about to be named). D = 9. For m = 3 there are 9 + 36 + 84 = 129 kinds; for m = 4 (a separate arm) 255. A cell of a kind is made the first time its combination of byte values occurs at those delays, with one connection from each of its lines (a connection of up to m ends). It counts the bytes that follow it. Before each byte every cell whose combination is present and that has counted says s(v) = log((c(v) + e) / (n + 14 e)). The answering cell's 14 branches compete on the sum over these cells of trust times s. The trust is one number for each kind and each of 40 levels of experience (how often the cell has counted, in powers of two up to 512, and how many different bytes have followed it: one, two, three or four, more). When the byte arrives, each trust that was used changes by eta times (s(arrived) less the sum of p times s), and the cells count. One stream, one byte at a time, nothing reset, no pass backward.

*What is laid down by the program:* the delay lines (as in `cells2.py`), the list of kinds (every set of at most m delays: no choice among them), the levels of experience. Nothing about digits, places, the operator or carrying.

*Text, split, score.* As `cells2.py`: lines such as `07+85=092`, 8,000 pairs shown, 500 of the 2,000 never-shown pairs scored. Two scores, both reported: **own** (after `=` the system names a byte, that byte is written in place of the line's, and so on for three bytes; the whole answer right) and **forced** (the line's own bytes stand before each answer byte; this is how `cells2.py` scores, so the relay's 84.2% is a forced figure). The claim is on the own score. Before the line stand line ends only, as in `cells2.py`. By place as well.

*Numbers chosen on development seed 1* by the mean of the last three readings of the own score at 40,000 lines: eta in {0.0005, 0.002, 0.008, 0.03}, e in {0.005, 0.02, 0.1, 0.5}, trust born at 0.02 or 0.25 (32 settings); chosen separately for m = 3 and m = 4. Then the five reported seeds 73129, 93683, 116089, 406364, 545353 once each, readings every 5,000 lines to 40,000. If cheap, 400,000 lines at the chosen setting.

## 3. Hypothesis, what would refute it, predictions (mine; written after the test run disclosed above)

*Hypothesis.* m = 3: at least 40% of never-shown sums whole (own score) and at least 95% of their units digits within 40,000 lines, at the median of the five reported seeds (mean of the last three readings). *Refuted if* the whole answer is under 20%.

Why partial (reading, mine): the units digit is fixed by two bytes and the place, which one kind of three ends holds. The tens digit needs the two tens digits and the carry; kinds of three ends hold the two tens digits with one units digit, which gives the carry only as a chance; several kinds hold the same counts, so the chance of a carry before any units digit is seen is counted several times over.

| | Statement | Chance |
|---|---|---|
| P1 | m = 3: whole answer at least 40% (median of five) | 0.85 |
| P2 | m = 3: units at least 95% | 0.97 |
| P3 | m = 3: whole answer at least 70% | 0.25 |
| P4 | m = 3: the tens digit is the weakest place on all five seeds | 0.90 |
| P5 | m = 3: the reading at 5,000 lines is within 5 points of the reading at 40,000 | 0.75 |
| P6 | m = 4 is at least 2 points below m = 3 (cells of the whole pair are always right on shown pairs, absent on never-shown ones, and take the mismatch from the kinds that generalise) | 0.50 |
| P7 | the own score and the forced score differ by less than 2 points | 0.85 |
| P8 | m = 3 is below the relay's 84.2% | 0.85 |

## 4. Size, cost, what is not run

Numbers that learn: the trusts, 129 x 40 = 5,160 for m = 3 (those used are reported), 10,200 for m = 4; and the counts, 14 for each cell (cells and counts above zero are reported; the test run made 114,958 cells). A run of 40,000 lines took 5 CPU-seconds at m = 3. The grids: about 3 CPU-minutes for m = 3, perhaps 10 for m = 4.

Not run here: keeping (E2, `PLAN_2.md` after this); any means of supplying the carry; trust by place; the relay's answer as one more kind; any reference taught by back-propagation; prose.

## 5. Addition 1 (20 November, written before its runs): the m = 4 grid ended at its edge

Development seed 1. m = 3: the 32 settings lie between 50.3 and 56.9 (`dev_m3.txt`); the best by the rule is eta 0.008, e 0.1, trust born at 0.02 (56.9), and it is taken. m = 4: 80.5 at eta 0.03, e 0.5, trust born at 0.02, which is the largest eta and the largest e of the grid (`dev_m4.txt`). Before the reported seeds, m = 4 is run on seed 1 at eta in {0.03, 0.1} and e in {0.5, 2}, trust born at 0.02 (three new settings); the best of all 35 by the same rule is taken. The m = 3 grid did not end at an edge in eta; its best e is inside the grid.
