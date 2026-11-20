# Keeping by replay, fifth plan: the whole system's own answer (20 November 2026)

By the same separate agent session as `PLAN_1.md` to `PLAN_4.md`; same folder, same source. Written before any run of this step; the check was run first (`check_replay4_output.txt`, extended; lives of 20,000 to 25,000 lines). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation and origin.** Asked by the coordinating session after `PLAN_4.md`, as the arm that plan named and did not run. The hypothesis and its bounds are the coordinating session's.

## 1. The question

`PLAN_4.md` measured the relay system and the store apart: at the end of the life the store alone has 84.8% of never-shown addition right and the relay system 77.0. **What does the system answer as a whole, when it answers from its store where the store holds a line and from the relay system where it does not: never-shown addition and never-shown subtraction, at the end of each phase?**

## 2. The system: no new mechanism

Arm D of `PLAN_4.md` unchanged (replay at one line for each real line, K = 8; one extra line in ten a dream, counted at full weight; add:400000 then sub:200000). The life, the strengths and the store are those of `PLAN_4.md`'s runs, made again with one more score; every earlier figure must come again.

**The whole answer.** After the six bytes of a line (`aa+bb=` or `aa-bb=`) the system gives the three bytes of the answer one at a time, reading its own earlier bytes. For each byte: if the longest counting cell of what has been read so far has counted, the byte is the one with the largest count in that cell (the store's answer); if no such cell has counted, the byte is the relay system's.

*The one number, stated and not tuned:* a cell "holds a line" for a byte when its run is the whole line up to that byte, back to the newline before the line, or the last eight bytes where the line so far is longer (K = 8, the cells of `PLAN_1.md`; the same cells measure (2) of `PLAN_4.md` used). No shorter cell is consulted.

## 3. Measures

At the end of each phase, on the 500 held addition pairs (and all 2,000 never-shown ones) and on the clean never-shown subtraction pairs (those, of all 2,000, whose line no shown pair writes; the mixed 500 beside them): the whole answer right, %; the relay system alone and the store alone on the same pairs; how many answers came wholly from the store, wholly from the relay system, or from both; and, where the store alone and the relay system alone both give an answer and differ, which was right.

## 4. Hypothesis and what would refute it (the coordinating session's)

**At the end of the life the whole system answers at least 80% of never-shown addition and at least 75% of never-shown subtraction (clean), with no line shown again by the program; median of the five reported seeds. Refuted if either is under 70.**

## 5. Predictions (mine), with chances

- W1 (0.85): never-shown addition, whole, at least 80% at the median.
- W2 (0.6): never-shown subtraction (clean), whole, at least 75% at the median.
- W3 (0.75): where the store and the relay system differ on a held addition at the end, the store is right more often than the relay system, on at least four seeds of five.
- W4 (0.6): on clean subtraction at the end the whole answer is below the relay system alone at the median (the store's subtraction lines are dreams made early in the second task, right 62% of the time).
- W5 (0.9): on held addition at the end the whole answer is within one point of the store alone (the store has a line for 498 of 500).

## 6. Cost, not run

Six runs of about 70 seconds (seed 1 as a development pass, nothing chosen on it; the five reported seeds). Not run: a rule that weighs the store against the relay system by the counts; the whole answer in arm L or with other q and w; on the harsh change of `PLAN_3.md`; further seeds unless a bound of the hypothesis is within five points.

---

## 7. A dated addition, entered 20 November after the five reported seeds were read and before its runs

On the five reported seeds (scored in section 8) the whole system has 77.2% of clean never-shown subtraction at the median (71.9, 73.8, 82.0, 77.9, 77.2), which is within five points of the hypothesis's 75 and of its bound of refutation at 70. The ten further seeds 1001 to 1010 are run and the fifteen reported together.
- W6 (0.6): on fifteen seeds the whole system has at least 75% of clean never-shown subtraction at the median.
- W7 (0.85): on fifteen seeds it has at least 80% of never-shown addition at the median.

## 8. Outcome (entered 20 November, after all runs of this plan)

Tables: `report5.txt` (made by `report5.py` from `res/d_x_D_*.json`); the check: `check_replay4_output.txt` (passed; the relay system alone and the store alone inside the new score agree with the earlier measures; with no dreams the whole answer on never-shown additions is the relay system's). The five runs give every reading of `PLAN_4.md`'s arm D again.

Never-shown pairs, whole answer right, %. Seeds 73129, 93683, 116089, 406364, 545353; median of the five; then the median of fifteen seeds (with 1001 to 1010).

| at the end of the life | the whole system | the relay system alone | the store alone |
|---|---|---|---|
| addition, the 500 held pairs | 84.0, 85.0, 87.0, 81.0, 88.4; **85.0**; fifteen: **82.6** | 77.0; fifteen 76.6 | 84.8; fifteen 82.6 |
| addition, all 2,000 never-shown pairs | 83.9, 81.6, 85.9, 83.9, 88.8; **83.9**; fifteen: 82.0 | 78.4; fifteen 76.6 | 83.6; fifteen 81.8 |
| subtraction, clean (377 to 412 pairs) | 71.9, 73.8, 82.0, 77.9, 77.2; **77.2**; fifteen: **77.0** | 80.1; fifteen 80.2 | 28.7; fifteen 28.8 |

At the end of the addition phase (five seeds): never-shown addition, whole 86.0, relay system alone 86.2, store alone 83.8; subtraction 1.1 (not yet taught).

**The hypothesis holds.** At the end of the life the whole system answers 85.0% of never-shown addition and 77.2% of clean never-shown subtraction at the median of the five reported seeds (at least 80 and 75 were required), and 82.6% and 77.0% at the median of fifteen. No line is shown again by the program. The lowest single runs of the fifteen are 72.4 (addition) and 71.9 (subtraction); none is under 70. Two of the five reported seeds are under 75 on subtraction (71.9, 73.8), so the second clause holds at the median and not on every seed.

**Where the answers come from.**

- *Addition at the end:* 99.6% of the 500 answers come wholly from the store and 0.4% from the relay system; none is mixed. Of the store's, 85.1% are right. The whole system is the store here, to within a point (W5).
- *Subtraction at the end:* 39.3% of the clean pairs are answered from the store (it holds a dreamed line for them) and 60.7% by the relay system. Of the store's answers 73.2% are right; of the relay system's 79.8%.
- *Where the store alone and the relay system alone both answer and differ.* Addition, 500 held pairs, fifteen seeds together: 1,040 pairs; the store right on 668, the relay system on 298, neither on 74. Subtraction (clean), fifteen seeds: 395 pairs; the store right on 100, the relay system on 275, neither on 20.

**What I read from it.**

1. *On the old task the store is the better witness, on the new task the relay system is.* For addition the store holds dreams written when the relay system was at its best and has not forgotten them, and the relay system has since lost about nine points. For subtraction the store's lines are dreams written while subtraction was still being learned (62% right when made), and the relay system at the end is better than they are. The rule "the store where it holds a line" therefore gains 4.8 points over the relay system alone on addition (fifteen seeds of fifteen above) and **loses 2.5 points on subtraction (fifteen seeds of fifteen below)**. The whole system is not the better of its two parts on each task; a rule that knew which part to trust would stand near 85 and 80, and this rule stands at 85 and 77.
2. *All three at once, with the price.* The system answers sums it was never shown (86.0% of never-shown addition after the first phase), keeps that through a second task (85.0%; 82.6 on fifteen seeds) and learns the second task to 77% on lines it was never shown, with no old line shown by the program. The price, in numbers: about 85,000 counting cells and 104,000 counts above zero, holding a line for nearly every one of the 10,000 sums and for two fifths of the never-shown subtractions, beside a relay system of 256 cells and 43,278 numbers; one sixth of the dreamed sums in the store are wrong and stay.
3. *What answers the never-shown sum is a dream of it.* The store answers a never-shown addition because the relay system once wrote that very line into it. The generalising was done by the relay system, once, when the line was dreamed; after that the answer is looked up. So the system "answers what it was not shown" in the sense that the program never showed it; the store has been shown it by the relay system. I state this so that the ledger's sentence is read with it.

**Predictions.** W1 held (85.0). W2 held (77.2). W3 held (five seeds of five: 61 against 17, 54 against 14, 50 against 19, 51 against 28, 33 against 15). W4 held (77.2 against 80.1; five of five below). W5 held (85.0 against 84.8). W6 held (77.0). W7 held (82.6).

**Limits.** As `PLAN_4.md`'s: dreams of a chunk are answered by the relay system of up to 500 lines earlier; q and w were chosen on one seed by a measure that plan showed to be blunt; a small domain. The held pairs were never shown as real lines, but the system chooses its own dreams from all pairs, the held ones among them, which is the point of the plan and also why the store's score on them is a score on lines the store has.

**Not run:** a rule that chooses between store and relay system by the counts or by the age of the store's line; the whole answer on `PLAN_3.md`'s harsh change; other q and w; arm L.
