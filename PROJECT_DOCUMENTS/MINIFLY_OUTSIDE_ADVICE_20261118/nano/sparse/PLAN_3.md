# Line `nano/sparse`, third plan (E3): what supplies the carry. A second layer of counting cells keyed on what a first cell names (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." Earlier the same day (`../../ROADMAP_SECOND_REVISION_20261120.md`, section 1): "The connection strength (ideally can involve many cells) should be jointly decided by the cells together." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one. Written before any run of `sparse2.py` other than its check (`check2_output.txt`). Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. `PLAN_1.md`, five reported seeds, 40,000 lines, never-shown sums, own bytes: m = 3: 56.8% whole (hundreds 97.6, tens 57.0, units 100); m = 4: 80.9% (98.2, 80.0, 100). At 400,000 lines m = 3 is at 58.2. The tens digit is the whole gap in both.
2. A look inside m = 3 on development seed 1 (not a result; the program is in the session's scratch files and not kept): where the tens digit is named, the cell of the two tens digits and '=' has had two bytes follow it (the sum of the tens digits and that sum plus one). The two cells that also hold one units digit would, taken alone and added, pick the right one of the two on 80% of never-shown pairs; the whole system picks it on 57%, which is about how often there is no carry. Those two cells' counts are mixed with counts from other places of the line where the same delays also hold three digits.
3. Reading (mine): no byte of the line states the carry, and a counting cell that holds all four digits is a cell of the whole pair, which a never-shown pair does not have. So no first-layer cell of any m can supply the carry for a never-shown pair exactly. What is needed is a cell that holds the two units digits and says something that is the same for every pair of tens digits.

## 2. The question

**Does a second layer of counting cells, which count not the byte that arrives but which of a first cell's followers arrived, supply the carry for never-shown sums?**

*The one change.* The system of `PLAN_1.md` with m = 3 at its chosen numbers, and added to it (`sparse2.py`): for each first-layer kind X (129) and each kind Z of at most three delays (129), a second-layer cell for each combination of byte values at Z's delays that has occurred while the active cell of kind X was undecided (two, three or four different bytes have followed it). The second-layer cell counts which of the X cell's followers arrived: the one the X cell has counted most, the second, the third, the fourth (ties: the smaller byte value first), or none of them. Before a byte it says, for each byte value, the log of its smoothed share for that byte's rank in the X cell. The answering cell adds this to the rest with one trust for each pair of kinds and each level of experience of the second-layer cell, born at the same value and learned by the same rule and step. Nothing about which pair of kinds matters is laid down: all 16,641 pairs are there.

In the words of Mr. Xu's sentence (reading, mine): what such a cell contributes to a branch of the answering cell is decided by two cells together, the first naming the candidates and the second choosing among them.

*Numbers.* Those of m = 3 (eta 0.008, e 0.1, trusts born at 0.02) first on development seed 1. If one run costs under three CPU-minutes, also eta in {0.008, 0.03} with e in {0.1, 0.5} on seed 1 and the best by the rule of `PLAN_1.md`; then the five reported seeds once, 40,000 lines.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With the second layer, never-shown sums whole (own bytes, median of five seeds, 40,000 lines) are at least 20 points above m = 3 alone (at least 76.8%). *Refuted if* the gain is under 10 points (under 66.8%).

| | Statement | Chance |
|---|---|---|
| Q1 | whole at least 76.8% | 0.60 |
| Q2 | whole above m = 4's 80.9% | 0.45 |
| Q3 | whole (forced score) above the relay's 84.2% | 0.35 |
| Q4 | tens digit at least 90% | 0.35 |
| Q5 | among the eight pairs of kinds with the most trust where the tens digit is named, at least one has a first cell holding both tens digits and a second cell holding both units digits | 0.60 |
| Q6 | hundreds digit at least 99% | 0.40 |

What could make it fail (mine): the order of a first cell's two followers is by count, and 55 against 45 in a hundred needs many lines to settle, so early counts of the second layer are written against an order that later turns; 16,641 trusts for pairs, most of them for pairs that say nothing, may drown the few that matter.

## 4. Size, cost, what is not run

Numbers that learn: as m = 3, and 16,641 x 40 trusts for pairs (those used are reported) and five counts for each second-layer cell (reported). Cost: not known before the first run; my estimate is one to three CPU-minutes for 40,000 lines and several hundred MB of memory. Not run here: the second layer over m = 4; keeping with the second layer (next, if this holds); 400,000 lines unless cheap; a second layer whose cells are active for decided first cells.

# Fourth plan (E4), in the same file because it is small: two cheap single changes to `PLAN_1.md`'s system, each beside what it changed from

Written at the same time as the third plan and before any of its runs. Same authorisation and source. `sparse.py` unchanged (both options were in the program and its check from the start).

**E4a: the trust also depends on the place in the line** (`place=10`: one trust for each kind, level of experience and number of bytes since the last line end; the count of bytes since the line end is laid down by the program, which is a concession and is said). m = 3 and m = 4 at their chosen numbers, five reported seeds, 40,000 lines. *Hypothesis:* m = 3 gains at least 10 points (the two cells that hold one units digit can then be trusted at the tens place without being trusted elsewhere). *Refuted if* it gains under 3. Chances: m = 3 gains at least 10: 0.40; m = 4 gains at least 3: 0.35.

**E4b: the answer written units first** (`rev=1`, a diagnostic the coordinating session named). m = 3 and m = 4, five seeds, 40,000 lines. Reading (mine) before the run: for counting cells this should not close the gap, because the cell that would then know the carry (two tens digits, one units digit, the units digit just written) is again a cell of the whole pair. *Prediction:* neither arm gains 10 points (chance 0.70); m = 4's tens digit stays under 90% (0.70).
