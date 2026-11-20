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

## 6. Outcome (entered 20 November after the runs; `res/s_rep3_*`, `s_rep4_*`, `s_long3_*`, `s_long4_*`; tables in `report_e1.txt`; check in `check_output.txt`)

The extension of addition 1 gave 82.1 at eta 0.1, e 2 (again the largest of both); it was taken by the rule and the grid was not extended again, the gain of the extension being 1.6 points. m = 4's numbers are therefore at the edge of what was tried, and this is a limit on the m = 4 figures.

Never-shown additions, the system writing its own answer bytes, whole answer right, %. Five reported seeds in the order 73129, 93683, 116089, 406364, 545353. "Last three": the mean of a run's last three readings. Beside them the relay system of `loop_small_case`: 84.2 at 400,000 lines (forced score), and 0 for a table of whole lines.

| Arm | lines | each seed (last three) | median, last three | median, last reading | by place at the last reading (median): hundreds, tens, units |
|---|---|---|---|---|---|
| m = 3 (129 kinds) | 5,000 | | | 55.2 | 95.8, 55.2, 100 |
| | 10,000 | | | 56.2 | 96.2, 56.2, 100 |
| | 20,000 | | | 56.2 | 97.6, 56.2, 100 |
| | 40,000 | 56.8, 56.8, 54.1, 59.7, 55.7 | **56.8** | 57.0 | 97.6, 57.0, 100 |
| | 400,000 | 59.4, 58.2, 55.5, 61.2, 57.6 | 58.2 | 58.4 | 99.6, 58.4, 100 |
| m = 4 (255 kinds) | 5,000 | | | 74.6 | 96.4, 74.6, 100 |
| | 10,000 | | | 78.0 | 96.0, 78.0, 100 |
| | 20,000 | | | 82.4 | 98.4, 82.4, 100 |
| | 40,000 | 81.4, 84.4, 80.7, 79.9, 80.9 | **80.9** | 80.0 | 98.2, 80.0, 100 |
| | 100,000 | | | 76.4 | 98.2, 76.4, 100 |
| | 200,000 | | | 63.6 | 98.8, 63.6, 100 |
| | 400,000 | 68.4, 68.3, 62.5, 71.9, 65.9 | 68.3 | 66.8 | 98.8, 66.8, 100 |

The forced score (the scoring of `cells2.py`) equals the own score in every run and at every reading.

Size and cost (medians): m = 3: 114,949 cells that have counted, 775,425 counts above zero, 3,699 of the 5,160 trusts used, 5 CPU-seconds for 40,000 lines (52 for 400,000). m = 4: 1,213,002 cells, 5,056,570 counts above zero, 7,169 trusts used, 20 CPU-seconds (182 for 400,000). The relay system has 256 cells in between and about 40,000 strengths. So m = 3 has about 20 times and m = 4 about 130 times as many numbers that learn as the relay system; nearly all are counts.

Kinds with the most trust where each answer byte is named (seed 73129, 40,000 lines; delays counted back from the byte to be named; at the tens digit delay 7 is the first number's tens digit, 6 its units digit, 5 the operator, 4 and 3 the second number's digits, 2 the '=', 1 the hundreds digit just written). m = 3, tens digit: (2, 8), (2, 4, 7), (4, 5, 7), (7), (4, 7, 8): the two tens digits with one fixed sign. m = 4, tens digit: (1, 4, 8), (2, 8), (2, 3, 4, 7), (3, 4, 5, 7), (4, 5, 6, 7), (3, 4, 7, 8): the two tens digits, one units digit and one fixed sign. m = 4, units digit: (3, 4, 5, 7), (4, 6, 7, 8), (4, 5, 6, 7). The full lists are in the result files (`top_trust_add`).

**The hypothesis holds**: m = 3 has 56.8% of never-shown sums whole and 100% of their units digits within 40,000 lines; at least 40% and 95% were asked.

| | Statement | Outcome |
|---|---|---|
| P1 | m = 3 at least 40% | held (56.8) |
| P2 | m = 3 units at least 95% | held (100) |
| P3 | m = 3 at least 70% | failed |
| P4 | tens the weakest place on all five seeds | held |
| P5 | 5,000 lines within 5 points of 40,000 | held (55.2 against 57.0; each seed within 3.4) |
| P6 | m = 4 at least 2 points below m = 3 | failed at 40,000 lines: m = 4 is 24 points above |
| P7 | own and forced within 2 points | held (identical) |
| P8 | m = 3 below the relay's 84.2 | held |

Readings (mine).
1. **Cells keyed on a few delays answer sums never shown, which cells keyed on a whole run do not; they do it within 5,000 lines, where the relay system needs some hundred thousand.** The units and hundreds digits are right nearly always; the tens digit is the gap.
2. **m = 3 guesses the carry.** Its tens digit is right about as often as there is no carry (55%). A look inside on the development seed (my scratch program, not kept): the cells that hold both tens digits and one units digit have counts mixed with those of other places in the line where the same three delays also hold digits; alone and added they would choose the right one of the two candidates on 80%.
3. **m = 4 reaches 80.9% at 40,000 lines, within 3.3 points of the relay's figure, and then falls with further reading to 68.3% at 400,000.** I predicted m = 4 below m = 3 (P6) and was wrong at 40,000 lines: the fourth end lets a cell hold the two tens digits, one units digit and a fixed sign, which fixes the place and keeps its counts clean. The fall afterwards I have not explained. Two candidates, neither tested: the cells of the whole pair gain experience, are always right on shown pairs and absent on never-shown ones, and the trust of the kinds that generalise drifts once they are no longer needed on the lines read (the mechanism I gave for P6, arriving later than I thought); or the constant step of 0.1, chosen at 40,000 lines, is too large for a long run (the readings swing by 10 points and more between one reading and the next).
4. The number of cells is large. 115,000 and 1.2 million cells for a text of 8,000 different lines is not small beside the relay's 256 cells; most of these cells are keyed on bytes of the line before, which carry nothing. Which cells could withdraw was not tried.
