# Keeping, eleventh plan: the subtraction score on lines no shown pair writes; means of the last three readings. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after four messages of the coordinating session were read (they had been waiting while the earlier plans ran; I read them only after the report of the line was pushed), and before any run of this plan.

**What the coordinating session told this session** (its words are in its own files; I state what I took from them). (1) A fault in the scoring of subtraction in `cells2.py`, which this line copied unchanged: a subtraction line is written with the larger number first, so a held-out pair (a, b) and the shown pair (b, a) write the same line; about four in five of the 500 held-out subtraction lines are lines that training also writes. Addition is not touched. So every subtraction column of `PLAN_1.md` to `PLAN_10.md` and of `REPORT.md` is a score on mostly shown lines. (2) My figures are single last readings; the mean of the last three readings is to be given beside each. (3) Several errata from two checks of my files. (2) and (3) are answered in `ERRATA.md` from the files that exist; (1) needs runs.

**Question.** On the held-out subtraction pairs whose written line no shown pair writes (about 80 to 100 of 500), is subtraction answered as on all 500, in the plain system, in the program's choice, and in the candidates? For the candidates the question is sharper: cells that open only to lines met in youth might hold shown lines without having learned to subtract.

**What is run** (`clean.py`; no system is changed): the same runs as before, with the same seeds and options, scored at the end on (i) the clean pairs, (ii) the pairs a shown pair also writes, (iii) all 500, and with finer readings early in each phase (`fine=1`), which does not change the lines read. Arms, no old line and one old line in 32: plain; `reserve=128`; the fourth plan's candidate; the seventh plan's candidate (reported seeds; the first, second and fourth also on seeds 1001 to 1005); the other order and the text with no new line of the eighth plan (candidate and plain). 110 runs. Every rerun's end figures must equal the earlier run's; any that do not are reported.

**Hypothesis.** The candidates learned to subtract, not the shown lines. **Refuted if** the seventh plan's candidate's clean subtraction score is more than 10 points below its score on all 500, at the median of the reported seeds, in either condition; or more than 10 points further below than the plain system's is.

**Predictions.**
- K1 (0.65): for every arm the median clean score is within 8 points of the median score on all 500.
- K2 (0.60): the seventh plan's candidate has clean subtraction at 70% or more in both conditions (median, reported seeds).
- K3 (0.60): in the other order (subtraction the first task), the candidate's clean subtraction at the end is within 5 points of the plain system's clean subtraction after its own phase... not available from these runs for the plain system at the end of its first phase; replaced before any run by: the candidate's clean subtraction at the end is 80% or more (median).
- K4 (0.90): every rerun reproduces the earlier end figures to the digit.

**Also from these runs:** the number of lines of the second task read when its never-shown score first reaches 70% at a reading (readings at 2,000-line spacing to 20,000, 10,000 to 100,000, then 50,000), beside the plain system.

**Cost.** 110 runs of 40 to 90 CPU seconds: about 30 minutes on the clock.

**Not run.** The clean score for the arms of the first, second, fifth and ninth plans (their candidates did not keep); a split of the pairs that avoids the fault from the start (it would change every figure of `loop_small_case` and is the coordinating session's to decide).

## Outcome (entered 20 November, after all 110 runs; `table11.txt` made by `tab11.py`; `res/clean_c11_*`, `res/c2_c11_*`)

All 110 reruns give the earlier runs' end figures to the digit. Of the 500 held-out subtraction pairs, 80 to 107 write a line that no shown pair writes (reported seeds: 101, 96, 80, 104, 83; seeds 1001 to 1005: 107, 90, 102, 103, 103); no held-out addition line is written by a shown pair.

Subtraction at the end, %, five seeds and median: on all 500 pairs (the figure of the earlier plans) | **on the pairs whose line no shown pair writes**. Last column: lines of the second task read at the first reading of 70% or more on all 500 (readings every 2,000 lines to 20,000, every 10,000 to 100,000, then every 50,000; "none": not reached).

| Arm | all 500 | clean | lines to 70% |
|---|---|---|---|
| plain, no old line | 82.0, 89.8, 91.4, 86.0, 86.0; 86.0 | 79.2, 85.4, 90.0, 83.7, 80.7; **83.7** | 80,000, 70,000, 50,000, 90,000, 70,000 |
| plain, one old line in 32 | 85.6, 85.8, 90.4, 84.2, 85.8; 85.8 | 82.2, 82.3, 86.2, 87.5, 79.5; **82.3** | 70,000, 80,000, 60,000, 90,000, 70,000 |
| the program's choice, no old line | 74.0, 77.2, 78.6, 72.8, 71.6; 74.0 | 75.2, 68.8, 76.2, 68.3, 62.7; **68.8** | 150,000, 90,000, 70,000, 150,000, 150,000 |
| the program's choice, one in 32 | 78.6, 72.8, 75.4, 73.4, 77.6; 75.4 | 79.2, 63.5, 77.5, 70.2, 74.7; **74.7** | 150,000, 100,000, 90,000, 150,000, 150,000 |
| fourth plan, no old line | 85.2, 87.2, 80.0, 73.2, 78.8; 80.0 | 87.1, 87.5, 83.8, 72.1, 72.3; **83.8** | 90,000, 150,000, 90,000, 200,000, 150,000 |
| fourth plan, one in 32 | 85.0, 84.8, 83.6, 85.8, 74.6; 84.8 | 82.2, 83.3, 83.8, 81.7, 75.9; **82.2** | 80,000, 200,000, 90,000, 90,000, 100,000 |
| seventh plan (final candidate), no old line | 86.8, 85.2, 84.4, 91.6, 93.2; 86.8 | 85.1, 71.9, 83.8, 88.5, 92.8; **85.1** | 30,000, 30,000, 40,000, 30,000, 30,000 |
| seventh plan, one in 32 | 86.8, 85.8, 88.4, 85.8, 85.0; 85.8 | 86.1, 79.2, 86.2, 86.5, 79.5; **86.1** | 30,000, 40,000, 40,000, 30,000, 30,000 |
| seeds 1001 to 1005: plain, no old line; one in 32 | 83.2; 85.2 | **85.4; 82.5** | 50,000 to 150,000 |
| seeds 1001 to 1005: the program's choice | 73.4; 71.4 | **79.6; 74.8** | 150,000 to 200,000, none on one seed |
| seeds 1001 to 1005: final candidate, no old line | 84.2, 92.4, 86.2, 85.6, 88.8; 86.2 | 79.4, 91.1, 78.4, 88.3, 91.3; **88.3** | 30,000, 40,000, 30,000, 18,000, 30,000 |
| seeds 1001 to 1005: final candidate, one in 32 | 90.0, 87.6, 82.0, 88.8, 87.4; 87.6 | 88.8, 80.0, 74.5, 90.3, 91.3; **88.8** | 30,000, 30,000, 40,000, 30,000, 30,000 |

The other order (subtraction is the first task; its score at the end is what is kept): plain 2.8 and 17.8 on all 500, **1.9 and 14.5** clean; the final candidate 93.6 and 94.2 on all 500, **92.8** (94.1, 82.3, 93.8, 89.4, 92.8) **and 95.0** (98.0, 84.4, 95.0, 94.2, 95.2) clean. Lines of addition, the second task there, to 70%: candidate 90,000, 40,000, none (it ends at 66.0), 40,000, 60,000 with no old line and 50,000 to 70,000 with one in 32; plain 100,000 to 150,000.
The text with no new line: plain 83.8 and 82.2 on all 500, **85.6 and 82.7** clean; candidate 75.2 and 70.6, **76.2 and 71.2** clean.

1. **The fault did not flatter subtraction by much in any arm.** The clean score is within about five points of the score on all 500 at the median everywhere (largest differences: the program's choice with no old line, 68.8 against 74.0, and on the new seeds 79.6 against 73.4), on 80 to 107 pairs, where one pair is about one point. The systems had learned to subtract and were not holding the shown lines apart.
2. **The final candidate's subtraction stands on clean lines**: 85.1 and 86.1 (new seeds 88.3 and 88.8), beside the plain system's 83.7 and 82.3. The hypothesis is not refuted. One seed (93683, no old line) is 13 points lower on clean lines than on all (71.9 against 85.2).
3. **What the way of keeping costs the second task.** The final candidate reaches 70% of subtraction after 30,000 to 40,000 lines, the plain system after 50,000 to 90,000, the program's choice after 70,000 to 150,000; at the end the candidate is level with the plain system (85.1 and 86.1 against 83.7 and 82.3, clean) and the program's choice below it (68.8 and 74.7). So with this candidate keeping the first task does not cost the second on these measures. The fourth plan's candidate (clock) was slower than the plain system: 90,000 to 200,000 lines.

**Predictions.** K1 (0.65) held: every arm within 8 points (largest 6.2). K2 (0.60) held: 85.1 and 86.1. K3 (0.60), as replaced: held, 92.8 and 95.0. K4 (0.90) held: 110 of 110.
