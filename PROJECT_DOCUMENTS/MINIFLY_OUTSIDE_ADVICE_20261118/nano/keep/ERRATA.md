# Keeping (K1): errata of 20 November 2026

By the agent session of line K1. Entered after the report of the line was pushed, when I read four messages of the coordinating session that had been waiting while the plans ran (two independent checks of my files are reported in them). Nothing in the plans is rewritten; this file corrects them. The tables are in `errata_tables.txt`, made by `audit.py` from the result files.

## 1. Decimals in three development tables were not read from the files (my error)

In the development tables of `PLAN_2.md`, `PLAN_3.md` and `PLAN_5.md` I typed figures from the runs' printed lines, which give whole numbers, and wrote decimals that the files do not contain. The five-seed tables of every plan, and the development tables of plans 1, 4, 6, 7 and 9, were made by `tab.py` from the files and are right. The figures as the files give them (seed 1, addition / subtraction at the end):

- `PLAN_3.md`: y 4,500: 58.2 / 76.6 and 67.6 / 85.8 (I wrote 68.0 / 86.2); y 18,000: 38.0 / 76.0 and 61.8 / 80.8 (I wrote 38.2 / 75.8, 62.0 / 81.0); y 90,000: 5.8 / 69.2 and 46.8 / 59.0 (I wrote 6.0 / 69.4, 47.2 / 59.0). Sums of the smaller scores: 125.8, 99.8, 52.6. The choice is unchanged.
- `PLAN_2.md`, one old line in 32, subtraction at the end (the addition figures are right): 128 cells at birth: exp 100,000: 4.0 (I wrote 4.2); exp 1,000,000: 59.0 (59.4); hyp 30,000: 47.4 (47.0). One cell at birth: no law 88.4 (87.6); exp 300,000: 39.0 (38.8); exp 1,000,000: 74.6 (75.2); hyp 30,000: 24.0 (24.4); hyp 100,000: 61.6 (62.0); hyp 1,000,000: 84.0 (83.6). The choice is unchanged.
- `PLAN_5.md`, no old line: theta 150: 85.8 / 23.8 (I wrote 86.0 / 24.2) and 80.8 / 0.0 (81.4 / 0.2); theta 250: 85.6 / 10.8 (86.4 / 11.4) and 81.6 / 0.2 (82.0 / 0.0); theta 400: 79.6 / 0.0 (80.2 / 0.0) and 66.4 / 2.4 (66.2 / 1.6). The choice is unchanged.

The differences are under a point and change no choice, but figures in a table must be the files' figures, and these were not.

## 2. Single last readings, and the mean of the last three

Every figure of the plans and of `REPORT.md` is the last reading of a phase. `errata_tables.txt` gives, for every five-seed arm, the last reading and the mean of the last three readings (at 50,000-line spacing) of the last phase. Where the two medians differ by more than five points:

- **Third plan, no old line, addition: 74.6 at the last reading, 42.1 as the mean of three** (by seed 42.1, 37.2, 60.3, 7.6, 81.0). The last reading flattered it; the plan's outcome did say that addition swung during the phase, but its headline figure should have been read beside this. Its subtraction with one old line in 32: 85.6 and 78.1.
- The program's choice with one old line in 32, addition: 67.0 and 61.8 (seeds 1001 to 1005: 66.0 and 59.7). The level to reach is lower by the mean of three.
- Age: subtraction 53.2 and 40.3, 56.0 and 44.4. Fourth plan, one old line in 32, subtraction: 84.8 and 77.7. Steps that shrink with use: subtraction 82.0 and 76.8, 81.6 and 75.4. Plain, other order, addition 78.8 and 73.6; plain on the text with no new line, subtraction 82.2 and 77.1; plain, seeds 1001 to 1005, one old line in 32, subtraction 85.2 and 78.0.
- For subtraction as the second task the mean of three is lower in every arm, because the task is still being learned 100,000 lines before the end.

Where the result of the line stands, the two agree. **Addition with no old line under the fourth, sixth and seventh plans is the same at the last three readings on every seed** (fourth: 83.4, 85.6, 81.2, 85.4, 86.6; sixth and seventh: 89.6, 78.8, 78.6, 85.8, 87.0; seeds 1001 to 1005: 85.0, 91.6, 77.4, 81.0, 84.8; in each case the mean of three equals the last reading to the digit). The seventh plan with one old line in 32: addition 81.6 at the last reading and 81.6 as the mean of three; subtraction 85.8 and 85.0. The sixth plan with one old line in 32: addition 71.6 and 69.7, against the program's 67.0 and 61.8.

## 3. The looks inside (`diag3.py`): the commands, and a log I did not report

The commands were not recorded with the logs. They were, from this folder, with `kage.py` as it stood at commit 13a1772 for `log/diag3_*` and at 74cb0f2 for `log/diag4_*`:

- `python3 diag3.py seed=<s> plan=add:400000,sub:200000 H=384 H0=128 young=4500 [reh=32] > log/diag3_<s>[_reh32].txt` for s = 1, 93683, 406364, and 406364 with `reh=32`;
- `python3 diag3.py seed=<s> plan=add:400000,sub:200000 H=384 H0=128 young=4500 blaw=exp btau=1e4 [reh=32] > log/diag4_<s>_reh<r>.txt` for s = 116089 (no old line) and 406364 (`reh=32`).

`log/diag3_406364_reh32.txt` was made with the others and I did not report it in `PLAN_3.md`. In it, with one old line in 32, putting the constants back gives addition 1.4 against 37.4 at the end: it makes things worse. So the sentence of `PLAN_3.md` "the whole of what is lost with no old line is in the answering cell's constants" holds for the condition it names, on the three seeds looked at, and **does not hold with one old line in 32**; there the old cells' answering strengths had changed by 0.36 of their size, the late cells spoke on addition lines (silenced, 17.2; silenced and constants put back, 49.8), and the parts had moved together. I should have shown that log beside the others.

## 4. Options in `kage.py` before their plans

`young`, `use=` and `join=sig` were in `kage.py` and `kage.c` at commit 4853fd8, before the second plan's runs, with labels of plans not yet written. I wrote the first kernel with the options I thought I might need; none of them was used in the second plan's runs (its commands are in `jobs2.txt` and `jobs2r.txt`). The form of `young` was changed before the third plan (a silenced cell is unchanged, the fading included; commit 13a1772), and `join=sig` before the fifth (the excess over a slow mean; commit 74cb0f2). The labels in the header named the plans I then expected and were corrected as the plans were written.

## 5. The subtraction score

`cells2.py`, copied unchanged, writes a subtraction line with the larger number first, so that most held-out subtraction pairs write a line that a shown pair also writes. **Every subtraction column in `PLAN_1.md` to `PLAN_10.md` and in `REPORT.md` is therefore a score on mostly shown lines**; the addition columns are not touched (no held-out addition line is written by a shown pair). `PLAN_11.md` gives the score on the lines no shown pair writes.

## 6. "128 and 128"

That 128 cells join at birth and 128 at the change of task on every seed is the arithmetic of the rule with K = 128: cells join until 128 are heard. Five seeds do not confirm it; they could not have differed. What the seeds do show is the part that could have failed: no cell joins at any other time, and the two bursts come where the old cells are silent.

And the limit in one sentence, which I take from the second check and agree with: the closing rule reads which byte values, at which delays, a cell met in its youth; the delay lines span one text line, so every window of a subtraction line holds the minus sign. The rule sorts the cells by a byte, and here the byte is the task's name.
