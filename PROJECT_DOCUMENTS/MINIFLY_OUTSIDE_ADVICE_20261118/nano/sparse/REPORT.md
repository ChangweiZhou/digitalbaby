# Line `nano/sparse`: counting cells keyed on a few chosen delays. Report of 20 November 2026

By a separate agent session started by the coordinating session; the frame is `../../ROADMAP_SECOND_REVISION_20261120.md`. **Authorisation:** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Status:** development level. One development seed for choosing numbers, five reported seeds (73129, 93683, 116089, 406364, 545353) for what is claimed; nothing repeated by anyone else; no checker gate, seal or outside review is claimed. The idea of the line is the coordinating session's; readings marked "mine" are this session's. Source: `claude/friendly-darwin-8mmhfm` at fcc174ac; branch `claude/nano-sparse-8mmhfm`.

Every step has a plan file committed before its runs (`PLAN_1.md` to `PLAN_10.md`; `PLAN_3.md` holds the third and fourth plans), with the outcome and every prediction scored below it. One slip, stated in `PLAN_1.md`: a single test run on the development seed at default numbers was made before the first plan was committed.

## 1. In brief

All figures: two-digit additions that were never shown (500 of the 2,000 held-out pairs, `cells2.py`'s split), the system writing its own three answer bytes, whole answer right, %, median of the five reported seeds, mean of a run's last three readings unless "last" is said. The score with the line's own bytes standing before each answer byte (`cells2.py`'s score) is equal to it in every run of the counting cells alone. Beside them, from `loop_small_case`: the relay system 84.2 at 400,000 lines, 1.8 after 200,000 lines of subtraction; a table of whole lines 0.

1. **Cells keyed on a few delays answer sums never shown, within a few thousand lines.** Kinds of at most three delays among the last nine (129 kinds): 56.8 at 40,000 lines (55.2 already at 5,000). At most four (255 kinds): 80.9 at 40,000 lines. Units digit 100 and hundreds digit 97 to 99 in both; the tens digit is the whole gap.
2. **They keep addition through subtraction with nothing rehearsed.** After subtraction, addition stands at 100% of its level with four delays (short and long life) and at 92% (short life) and 55% (long life) with three. The relay system: 1.8.
3. **No cell short of the whole pair holds the carry, and that bounds this family on the usual text.** With the answer written units first, two cells together nearly decide it: 91.0 within 40,000 lines with four delays and a shrinking step for the trust (89.7 to 93.4 over the seeds), above the relay's figure.
4. **With four delays the score falls with further reading** (80 at 40,000 lines, 59 to 68 at 400,000). It is not the trust's step, its birth value, or the cells of the whole pair. It is the growing weight of a cell's counts: with a cell speaking at the weight of at most 16 counts the level holds (77.0 at 40,000 lines, 75.2 at 400,000, five seeds; section 6).
5. **With look-ahead guesses that cells can be keyed on, the usual order of writing reaches 91.4 within 40,000 lines** (88.0 to 92.7; units first 93.7), and 86.8 is left after subtraction with nothing shown again (94.7% of its level; section 8). Over 400,000 lines it falls like m = 4 unless the cap is used (two seeds: 63 without, 81 with). This is the line's best system and the idea is the coordinating session's.
6. **Three other routes to the tens digit**, each one change from the three-delay system (56.8): a trust that also depends on the place in the line 77.8 (the place counter is laid down); a second layer of cells that count which of a first cell's followers arrived 69.3 (67.9 to 92.3; seventy times the cells); the relay system read as one more sender (section 7).
7. **Joined to the relay system by a trust that reads its confidence, the whole generalises less than the relay system and keeps less than the counting cells** (section 7), because during the second task nothing can tell the answering cell that the relay system has lost the first.
8. **Size.** 115,000 cells and 775,000 counts above zero with three delays, 1.2 million cells and 5 million counts with four, 3.3 million cells and 35 million counts with look-ahead, against the relay system's 256 cells and about 40,000 strengths. Which cells could withdraw was not tried.

## 2. The system

One line for each byte value and a chain of nine delays, as in `cells2.py`. A kind of cell is a set of at most m of the delays 1 to 9; all such sets are there (no choice among them). A cell of a kind is made when its combination of byte values first occurs at those delays; it counts the bytes that follow it. Before each byte, every cell whose combination is present and that has counted says log((c(v) + e) / (n + 14 e)) for each byte value v. The answering cell's 14 branches compete on the sum of trust times what each cell says; the trust is one number for each kind and each of 40 levels of experience of the cell (the rule and the levels are those of `nano/table/trust.py`). When the byte arrives each trust that was used moves by its step times the mismatch, and the cells count. One stream, one byte at a time, nothing reset, no pass backward. Programs: `sparse.py` (the numpy reference `Ref` and the compiled kernel; `check_output.txt`), `sparse2.py` (second layer; `check2_output.txt`), `sparse3.py` (with the relay system; `check3_output.txt`). Copied files are named at the head of each program.

Numbers chosen on development seed 1 over 32 settings each: m = 3: step 0.008, e 0.1, trusts born at 0.02 (all 32 settings lie between 50 and 57); m = 4: step 0.1, e 2, born at 0.02, which is at the edge of the grid even after one extension (a limit on the m = 4 figures).

*What a cell reads, and what else it is given.* A plain cell reads the byte values on the delay lines of its own kind (one to four of the nine) and the byte that arrives next; it holds 14 counts, their total and how many are above zero. The answering cell reads what the active cells say and holds the trusts; the mismatch (the byte that arrived, less its own probabilities) reaches each trust that was used, and nothing else passes back. Laid down by the program: the nine delay lines; the list of kinds (every set of delays up to m, without choice); the levels of experience; in the fourth and seventh plans the count of bytes since the line end; in the fifth plan's addition 2 the cap of 16; in the tenth plan the order three ahead, two ahead, next, and that a cell also counts the two bytes after the next. Nothing about digits, operators, places or carrying, except where a plan says so.

*Cost beside the figures.* m = 3: 115,000 cells, 775,000 counts above zero, 5 CPU-seconds for 40,000 lines. m = 4: 1.21 million cells, 5.06 million counts, 20 CPU-seconds. Second layer: a further 8.5 million cells, 246 CPU-seconds. Look-ahead (m = 4): 3.27 million cells, 34.6 million counts, about 130 CPU-seconds for 40,000 lines. Relay system: 256 cells in between, about 40,000 strengths, 400,000 lines in 16 to 34 seconds (its report).

## 3. First plan: never-shown sums (`PLAN_1.md`)

| Arm | lines | each seed (last three) | median, last three | median, last | hundreds, tens, units (last, median) |
|---|---|---|---|---|---|
| m = 3 | 5,000 | | | 55.2 | 95.8, 55.2, 100 |
| | 40,000 | 56.8, 56.8, 54.1, 59.7, 55.7 | **56.8** | 57.0 | 97.6, 57.0, 100 |
| | 400,000 | 59.4, 58.2, 55.5, 61.2, 57.6 | 58.2 | 58.4 | 99.6, 58.4, 100 |
| m = 4 | 5,000 | | | 74.6 | 96.4, 74.6, 100 |
| | 40,000 | 81.4, 84.4, 80.7, 79.9, 80.9 | **80.9** | 80.0 | 98.2, 80.0, 100 |
| | 400,000 | 68.4, 68.3, 62.5, 71.9, 65.9 | 68.3 | 66.8 | 98.8, 66.8, 100 |
| relay system, this line's runs of it (section 7) | 40,000 | | | 66.2 | 97.0, 74.0, 96.2 |
| | 400,000 | 84.2, 82.7, 86.4, 81.8, 86.7 | 84.2 | 84.8 | 95.0, 89.8, 100 |

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
| five seeds, shrinking step, at most 16 counts | 77.0 | 76.6 | 76.0 | 75.4 | **75.2** (74.0 to 77.1) |

The hypothesis that the fall is the constant step is refuted: with the shrinking step the swings go and the fall stays (18 points, the same on every seed). The trust's birth value and the cells of the whole pair change nothing. A third candidate was then tested (addition 2): a cell says log((c + e) / (n + 14 e)), so what it says of a byte that never followed it goes on falling as it counts; for a never-shown pair the cells that hold part of it have, by construction, not seen that pair. With a cell speaking at the weight of at most 16 counts, in the same proportions, the score at 400,000 lines is 1.8 points under the reading at 40,000 on five seeds, where without the cap it is 18 under. Reading (mine): the fall was the growing weight of the counts, their zeros above all. The cap is one number chosen between two values on one seed; it holds the level and does not raise it.

## 7. Sixth to eighth plans: the relay system beside the counting cells (`PLAN_6.md`, `PLAN_7.md`, `PLAN_8.md`)

The relay system is `cells2.py`'s plain system (256 cells in between, its files copied unchanged, its own compiled kernel), reading the same stream at the same time; its answers for a block of 50 lines are taken before it learns the block. Its message at a byte is the log of its branches' outputs, cut off below at 0.01 and scaled to sum to one; the answering cell of the whole holds a trust for it at ten levels of its confidence, born at nothing. The counting cells are m = 3 at their numbers. In these runs the relay system alone has 84.2, 82.7, 86.4, 81.8, 86.7 at the end of addition, which are `cells2.py`'s own figures for these seeds.

Never-shown additions, last reading of each phase, median of five seeds:

| | add 400,000 | then sub 200,000 | clean subtraction at the end |
|---|---|---|---|
| counting cells alone, m = 3 (`PLAN_2.md`) | 58.4 | 32.4 | not run on the long life |
| relay system alone | 84.8 | 1.8 | 83.7 |
| the whole, trust by the relay system's confidence (`PLAN_6.md`) | **77.6** | **12.4** | 68.3 |
| the whole, that trust also by place, no second task (`PLAN_7.md`) | 74.0 | | |
| relay system kept by replay, alone (`PLAN_8.md`, in the same runs) | 87.0 | 82.6 | 76.9 |
| the whole with the relay system kept by replay (`PLAN_8.md`) | **83.2** | **45.8** | 62.5 |

*Before the second task* the relay system lifts the whole's tens digit from 58 to 78; the relay system alone has 90. A trust by place does not close this (73.1 by the mean of the last three readings, against 76.6 without it and 84.2 for the relay system alone): refuted on every seed, though the answering cell does find the place (its trust for the relay system is large where the tens digit is named and nothing where the units digit is named).

*After the second task* the whole is worse than the counting cells alone, in every digit, the units digit included (54 against 98). Reading (mine): the relay system is right on the subtraction lines being read, so it stays trusted; on an addition line it is then confident and wrong. No trust learned from the lines being read can see this, because during subtraction no addition line is read. Joined in this way the pair generalises less than the relay system and keeps less than the counting cells.

*With the relay system kept by replay* (`nano/replay`'s run cells and replayed lines, copied; the replayed lines reach the relay system only): the relay system alone keeps 82.6 of 87.0, as that line reported. The whole has 83.2 before and 45.8 after: below the relay system with replay at every point of the life. The hypothesis put for this track (at least 84 before, at least 80% of it kept) is refuted in its second half (55% kept). Reading (mine): with the relay system kept, what pulls the whole down is the three-delay counting cells themselves, whose hundreds and tens cells hold both tasks in one count. On this evidence the two kinds joined by a learned trust do not add up; the counting cells' use to the relay system is as the store its replayed lines are drawn from. The counting cells worth joining would be those of section 8, and that was not run.

## 8. Tenth plan: look-ahead guesses that cells can be keyed on (`PLAN_10.md`; the coordinating session's idea)

One change from m = 4 with the shrinking step. Every plain cell holds three sets of counts: of the next byte, of the one after it and of the one after that. Before each byte the answering cell forms a guess for the byte three ahead, then a guess for the byte two ahead (cells keyed on at most three delays together with the first guess take part), then its answer (cells keyed on at most three delays together with either guess take part). A guess is the system's own in learning and in scoring; the trusts of the cells keyed on a guess are born at nothing; nothing about sums or places is laid down (`sparse4.py`, `check4_output.txt`).

| Arm | lines | each seed, last three | median | tens (last) |
|---|---|---|---|---|
| m = 4, shrinking step | 40,000 | | 77.4 (last) | 77.4 |
| with look-ahead, usual order of writing | 40,000 | 88.0, 91.3, 91.9, 92.7, 91.4 | **91.4** | 91.2 |
| with look-ahead, units first | 40,000 | 91.7, 92.7, 93.9, 94.6, 93.7 | **93.7** | 94.8 |
| with look-ahead, usual order, after 20,000 lines of subtraction (last reading; before: 91.2) | | 82.6, 89.0, 86.8, 89.0, 86.2 | **86.8** | 88.2 |
| with look-ahead, usual order, 400,000 lines, **two seeds only** (73129, 93683) | 400,000 | 64.5, 62.7 | 63.6 | |
| the same with a cell speaking at the weight of at most 16 counts, **two seeds only** | 400,000 | 81.4, 81.1 | 81.3 | |
| look-ahead over plain cells of at most three delays (m = 3 alone: 56.8) | 40,000 | 69.8, 72.8, 70.0, 75.5, 69.1 | **70.0** | 67.6 |

The hypothesis (at least 85% whole and 88% tens within 40,000 lines; at least 90% of its level kept) held in all parts. Where the hundreds digit is named, the guess three ahead (the units digit) is right on every never-shown pair and the guess two ahead (the tens digit) on 85.8%; the guess-keyed kinds with the most trust are the two tens digits with one more digit, together with the guessed units digit. Subtraction never written, at the end of the keeping life: 89.4. Cost: 3.3 million cells, 35 million counts above zero, 200 CPU-seconds for 60,000 lines.

Reading (mine). The carry is supplied by the counting cells themselves in the usual order of writing: the system states the units digit to itself before it writes the hundreds digit, and the two orders of writing come level. Beside it, from this line's own runs of the relay system (section 7, the same five seeds): the relay system alone has 66.2% at 40,000 lines and 84.8% at 400,000. So at 40,000 lines the look-ahead system is 25 points above the relay system, and its 40,000-line figure is 7 above the relay system's at 400,000; the two figures of that last comparison are at different numbers of lines. It is kept through a second task, where the relay system keeps 1.8. What remains wrong is the tens digit where two cells' chances balance, and the size is large. Over long reading it falls as m = 4 does (two seeds: about 90 at 40,000 lines, 63 at 400,000) unless a cell's counts are capped (two seeds: 81 at 400,000); two seeds are a reading and not a claim, and the long life with subtraction after it was not run. With plain cells of at most three delays the look-ahead is worth 13 points and not 34, at 2.0 million cells.

## 9. What is shown, what is not

Shown at development level: items 1 to 7 of section 1, on one text and five seeds.

Not shown, not run, or open:
- No reference taught by back-propagation was run on this line; the comparisons are with the relay system's reported figures and, in section 7, with the relay system run beside.
- The cap that stops m = 4's fall is one number chosen between two values on one seed; a learned limit was not tried, and keeping was not run again with the cap.
- m = 3 with the answer units first is below guessing; not explained.
- The second layer's unevenness has a stated cause that was not tested.
- The long keeping lives were not run again with the clean subtraction score.
- Look-ahead over the long life with subtraction after it, with or without the cap: NOT RUN. Look-ahead over 400,000 lines of addition: two seeds only. A guess passed on as a spread over bytes: not run. The look-ahead cells beside the relay system: not run.
- `PLAN_9.md` (units first over long reading): NOT RUN.
- With look-ahead over three delays, addition rises during subtraction with no addition line read; not explained.
- Cells that withdraw, a smaller D, prose, more than two tasks: not run.
- The place counter of the fourth plan is laid down by the program.

## 10. Commands and cost

`python3 sparse.py check`, `python3 sparse2.py check`, `python3 sparse3.py check`. A run: `python3 sparse.py <label> seed=73129 m=4 eta=0.1 e=2 T0=0.02 [plan=add:40000,sub:20000] [place=10] [rev=1] [shrink=1] [ncap=16]`; `python3 sparse2.py <label> seed=... m=3 mz=3 eta=0.008 e=0.1 T0=0.02`; `python3 sparse3.py <label> seed=... plan=add:400000,sub:200000 [xplace=1] [replay=1]`; `python3 sparse4.py <label> seed=... plan=add:40000,sub:20000 [rev=1]` (check: `python3 sparse4.py check`). Many at once: `grid.py`; tables: `report.py` (`report_e1.txt` to `report_e5.txt`). CPU for 40,000 lines: 5 seconds (m = 3), 20 (m = 4), 246 (second layer); with the relay system about 8 to 15 minutes for the life of 600,000 lines. Result files: `res/`.
