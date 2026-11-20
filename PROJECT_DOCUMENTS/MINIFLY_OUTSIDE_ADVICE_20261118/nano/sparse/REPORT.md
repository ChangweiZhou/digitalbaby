# Line `nano/sparse`: counting cells keyed on a few chosen delays. Report of 20 November 2026

By a separate agent session started by the coordinating session; the frame is `../../ROADMAP_SECOND_REVISION_20261120.md`. **Authorisation:** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Status:** development level. One development seed for choosing numbers, five reported seeds (73129, 93683, 116089, 406364, 545353) for what is claimed; nothing repeated by anyone else; no checker gate, seal or outside review is claimed. The idea of the line is the coordinating session's; readings marked "mine" are this session's. Source: `claude/friendly-darwin-8mmhfm` at fcc174ac; branch `claude/nano-sparse-8mmhfm`.

Every step has a plan file committed before its runs (`PLAN_1.md` to `PLAN_8.md`; `PLAN_3.md` holds the third and fourth plans), with the outcome and every prediction scored below it. One slip, stated in `PLAN_1.md`: a single test run on the development seed at default numbers was made before the first plan was committed.

## 1. In brief

All figures: two-digit additions that were never shown (500 of the 2,000 held-out pairs, `cells2.py`'s split), the system writing its own three answer bytes, whole answer right, %, median of the five reported seeds, mean of a run's last three readings unless "last" is said. The score with the line's own bytes standing before each answer byte (`cells2.py`'s score) is equal to it in every run of the counting cells alone. Beside them, from `loop_small_case`: the relay system 84.2 at 400,000 lines, 1.8 after 200,000 lines of subtraction; a table of whole lines 0.

1. **Cells keyed on a few delays answer sums never shown, within a few thousand lines.** Kinds of at most three delays among the last nine (129 kinds): 56.8 at 40,000 lines (55.2 already at 5,000). At most four (255 kinds): 80.9 at 40,000 lines. Units digit 100 and hundreds digit 97 to 99 in both; the tens digit is the whole gap.
2. **They keep addition through subtraction with nothing rehearsed.** After subtraction, addition stands at 100% of its level with four delays (short and long life) and at 92% (short life) and 55% (long life) with three. The relay system: 1.8.
3. **No cell short of the whole pair holds the carry, and that bounds this family on the usual text.** With the answer written units first, two cells together nearly decide it: 91.0 within 40,000 lines with four delays and a shrinking step for the trust (89.7 to 93.4 over the seeds), above the relay's figure.
4. **With four delays the score falls with further reading** (80 at 40,000 lines, 59 to 68 at 400,000). It is not the trust's step, its birth value, or the cells of the whole pair. Limiting the weight a cell's counts can reach holds it on the development seed (section 6); the five seeds are in section 6.
5. **Three other routes to the tens digit**, each one change from the three-delay system (56.8): a trust that also depends on the place in the line 77.8 (the place counter is laid down); a second layer of cells that count which of a first cell's followers arrived 69.3 (67.9 to 92.3; seventy times the cells); the relay system read as one more sender (section 7).
6. **Joined to the relay system by a trust that reads its confidence, the whole generalises less than the relay system and keeps less than the counting cells** (section 7), because during the second task nothing can tell the answering cell that the relay system has lost the first.
7. **Size.** 115,000 cells and 775,000 counts above zero with three delays, 1.2 million cells and 5 million counts with four, against the relay system's 256 cells and about 40,000 strengths. Which cells could withdraw was not tried.

## 2. The system

One line for each byte value and a chain of nine delays, as in `cells2.py`. A kind of cell is a set of at most m of the delays 1 to 9; all such sets are there (no choice among them). A cell of a kind is made when its combination of byte values first occurs at those delays; it counts the bytes that follow it. Before each byte, every cell whose combination is present and that has counted says log((c(v) + e) / (n + 14 e)) for each byte value v. The answering cell's 14 branches compete on the sum of trust times what each cell says; the trust is one number for each kind and each of 40 levels of experience of the cell (the rule and the levels are those of `nano/table/trust.py`). When the byte arrives each trust that was used moves by its step times the mismatch, and the cells count. One stream, one byte at a time, nothing reset, no pass backward. Programs: `sparse.py` (the numpy reference `Ref` and the compiled kernel; `check_output.txt`), `sparse2.py` (second layer; `check2_output.txt`), `sparse3.py` (with the relay system; `check3_output.txt`). Copied files are named at the head of each program.

Numbers chosen on development seed 1 over 32 settings each: m = 3: step 0.008, e 0.1, trusts born at 0.02 (all 32 settings lie between 50 and 57); m = 4: step 0.1, e 2, born at 0.02, which is at the edge of the grid even after one extension (a limit on the m = 4 figures).

## 3. First plan: never-shown sums (`PLAN_1.md`)

| Arm | lines | each seed (last three) | median, last three | median, last | hundreds, tens, units (last, median) |
|---|---|---|---|---|---|
| m = 3 | 5,000 | | | 55.2 | 95.8, 55.2, 100 |
| | 40,000 | 56.8, 56.8, 54.1, 59.7, 55.7 | **56.8** | 57.0 | 97.6, 57.0, 100 |
| | 400,000 | 59.4, 58.2, 55.5, 61.2, 57.6 | 58.2 | 58.4 | 99.6, 58.4, 100 |
| m = 4 | 5,000 | | | 74.6 | 96.4, 74.6, 100 |
| | 40,000 | 81.4, 84.4, 80.7, 79.9, 80.9 | **80.9** | 80.0 | 98.2, 80.0, 100 |
| | 400,000 | 68.4, 68.3, 62.5, 71.9, 65.9 | 68.3 | 66.8 | 98.8, 66.8, 100 |
| relay system (`loop_small_case`) | 400,000 | | 84.2 | | 98, 92, 100 (its report, section 4, the earlier 88.0 run) |

The hypothesis (m = 3 at least 40% whole and 95% units within 40,000 lines) held. Of eight predictions six held; "m = 3 at least 70" failed, and "m = 4 below m = 3" failed (m = 4 is 24 points above).

Reading (mine). m = 3 answers the tens digit about as often as there is no carry: it guesses. The fourth end helps because a cell can then hold the two tens digits, one units digit and a fixed sign, which fixes the place and keeps the counts of that place apart from those of others; such a cell gives the carry as a chance, and two of them together choose right more often than not.

## 4. Second plan: keeping (`PLAN_2.md`)

Never-shown addition at the last reading of each phase; nothing of addition is shown during subtraction, nothing is told to the system, trusts and counts go on changing.

| Arm, life | before | after | after as % of before (seeds) | after: hundreds, tens, units |
|---|---|---|---|---|
| m = 3, add 40,000 then sub 20,000 | 57.0 | 52.4 | 91.9 (84.3 to 93.2) | 89.8, 55.8, 100 |
| m = 4, add 40,000 then sub 20,000 | 80.0 | 79.8 | 100.2 (91.0 to 102.8) | 96.0, 79.8, 100 |
| m = 3, add 400,000 then sub 200,000 | 58.4 | 32.4 | 54.5 (52.9 to 72.9) | 70.2, 51.8, 98.2 |
| m = 4, add 400,000 then sub 200,000 | 66.8 | 67.2 | 101.8 (90.8 to 120.6) | 96.2, 68.2, 100 |
| relay system, add 400,000 then sub 200,000 | 84.2 | 1.8 | 2 | |

Subtraction at the end of the short life, on the held-out pairs whose line no shown pair writes (80 to 104 pairs a seed; `loop_small_case/ERRATA_20261120.md` on the coordinating branch): m = 3 62.5 (63.8 on `cells2.py`'s score), m = 4 79.5 (88.2).

The hypothesis (m = 4, short life: at least 80% of its own level) held. Readings (mine): with three ends the cells that fix the hundreds and tens digits cannot include the operator, so subtraction's counts go into the same cells and keeping wears with the length of the second task; with four ends the kinds trusted most include the operator's place. The trusts are shared by the two tasks: m = 4's addition drops to a third within 5,000 lines of subtraction and returns to its level with no addition line read; the counts are untouched throughout.

## 5. Third and fourth plans: what supplies the carry (`PLAN_3.md`)

Each arm is one change from `PLAN_1.md`'s system; 40,000 lines.

| Arm | each seed, last three | median, last three | median, last | tens (last) | changed from |
|---|---|---|---|---|---|
| m = 3 with a second layer | 92.3, 83.1, 69.3, 68.1, 67.9 | **69.3** | 68.8 | 68.8 | 56.8 |
| m = 3, trust also by place | 79.6, 78.1, 76.9, 77.8, 74.5 | **77.8** | 77.4 | 77.4 | 56.8 |
| m = 4, trust also by place | 87.8, 86.0, 83.5, 80.9, 85.9 | **85.9** | 84.6 | 85.0 | 80.9 |
| m = 3, answer units first | 20.5, 18.8, 17.7, 21.0, 18.3 | **18.8** | 18.8 | 18.8 | 56.8 |
| m = 4, answer units first | 86.3, 36.8, 72.2, 96.7, 80.6 | **80.6** | 95.2 | 95.2 | 80.9 |
| m = 4, answer units first, shrinking step (`PLAN_5.md`) | 89.9, 89.7, 91.6, 93.4, 91.0 | **91.0** | 91.6 | 91.6 | 80.6 |

*The second layer.* For each pair of kinds (X, Z), a cell for each combination of byte values at Z's delays that has occurred while the active cell of kind X was undecided between two to four followers; it counts which of the X cell's followers arrived (the most counted, the second, ...), not the byte. All 16,641 pairs of kinds are there, with one trust for each pair and level. The gain is 12.5 points where 20 was asked and under 10 would refute; it is above m = 3 on every seed. The pairs trusted most at the tens digit are a first cell of the two tens digits with a second cell holding the units digits. Cost: 8.5 million second-layer cells, 246 CPU-seconds for 40,000 lines. Reading (mine): "the more counted follower" is not the same thing in every first cell, so a second cell shared by all first cells of a kind learns a rule that is wrong for some of them; and at this cost the form is not one to carry on as it stands. What I would keep is the kind of cell: one that counts a relation to what another cell says.

*Units first.* I predicted no gain, arguing that the cell that would know the carry is again a cell of the whole pair. That was wrong for m = 4: with the units digit written, a cell of the tens digits with the first number's units digit and a cell of the tens digits with the answer's units digit multiply to decide the carry except at a tie. m = 3 falls below guessing (18.8), unexplained.

## 6. Fifth plan: why m = 4 falls with more reading (`PLAN_5.md`)

| Arm (m = 4) | 40,000 | 100,000 | 200,000 | 400,000 | last three at the end |
|---|---|---|---|---|---|
| constant step, five seeds | 80.0 | 76.4 | 63.6 | 66.8 | 68.3 |
| shrinking step, five seeds | 77.4 | 72.4 | 63.8 | 55.8 | **59.2** |
| seed 1, unchanged | 81.2 | 79.0 | 64.6 | 55.8 | 63.4 |
| seed 1, trust born at nothing | 80.2 | 76.4 | 66.8 | 53.8 | 62.1 |
| seed 1, no cells of the whole pair | 80.2 | 76.4 | 64.8 | 55.8 | 62.1 |
| seed 1, shrinking step, a cell speaks with the weight of at most 64 counts | 76.0 | 71.8 | 70.8 | 66.4 | 68.1 |
| seed 1, shrinking step, at most 16 counts | 75.6 | 74.6 | 74.4 | 72.6 | 72.6 |
CAP_FIVE_SEEDS_ROW

The hypothesis that the fall is the constant step is refuted: with the shrinking step the swings go and the fall stays (18 points, the same on every seed). The trust's birth value and the cells of the whole pair change nothing. CAP_READING

## 7. Sixth to eighth plans: the relay system beside the counting cells (`PLAN_6.md`, `PLAN_7.md`, `PLAN_8.md`)

RELAY_SECTION

## 8. What is shown, what is not

Shown at development level: items 1 to 6 of section 1, on one text and five seeds.

Not shown, not run, or open:
- No reference taught by back-propagation was run on this line; the comparisons are with the relay system's reported figures and, in section 7, with the relay system run beside.
- The fall of m = 4 with reading has a candidate cause and one test of it (section 6); it is not established.
- m = 3 with the answer units first is below guessing; not explained.
- The second layer's unevenness has a stated cause that was not tested.
- The long keeping lives were not run again with the clean subtraction score.
- Cells that withdraw, a smaller D, prose, more than two tasks: not run.
- The place counter of the fourth plan is laid down by the program.

## 9. Commands and cost

`python3 sparse.py check`, `python3 sparse2.py check`, `python3 sparse3.py check`. A run: `python3 sparse.py <label> seed=73129 m=4 eta=0.1 e=2 T0=0.02 [plan=add:40000,sub:20000] [place=10] [rev=1] [shrink=1] [ncap=16]`; `python3 sparse2.py <label> seed=... m=3 mz=3 eta=0.008 e=0.1 T0=0.02`; `python3 sparse3.py <label> seed=... plan=add:400000,sub:200000 [xplace=1] [replay=1]`. Many at once: `grid.py`; tables: `report.py` (`report_e1.txt` to `report_e5.txt`). CPU for 40,000 lines: 5 seconds (m = 3), 20 (m = 4), 246 (second layer); with the relay system about 8 to 15 minutes for the life of 600,000 lines. Result files: `res/`.
