# Line `nano/ahead2`, third plan: the look-ahead system over a long life with a second task after it (20 November 2026)

**Authorisation.** Mr. Xu's words of 20 November as the coordinating session quoted them to this session (`PLAN_1.md`, `PLAN_2.md`, heads). The question, the life and the cap are the coordinating session's (its messages of 19:18 UTC; a budget of one hour and a quarter stated for this plan at 19:18:53 UTC, after I had entered the request as NOT RUN for want of budget). The program and the predictions are mine. **Source.** `claude/nano-ahead2-8mmhfm` at 526324d. Development level. No checker gate, seal or outside review is claimed. Written before the cap exists in my program and before any run.

## 1. The question

The system of `PLAN_1.md` answers 91% of never-shown two-digit additions at 40,000 lines. **Does it still answer at least 75% at the end of 400,000 lines, and does it keep at least 85% of that level through 200,000 lines of subtraction with nothing of addition shown again?**

What I was told of the other line (`nano/sparse`), as reported to me and not read by me: on two seeds its look-ahead system falls with long reading (88 and 91 at 40,000 lines; 63 and 62 at 400,000), and a cap repairs most of it (82 and 81 at 400,000). Those cap figures are from two seeds; the runs below are the first on five and are not a repeat of a five-seed result.

## 2. The one change, and the arms

*The cap* (from `nano/sparse/PLAN_5.md`, addition 2, and the coordinating session's message): a set of counts with total n above N = 16 says log((c(v) N/n + e) / (N + 14 e)): its proportions at the weight of 16 counts. The counts and the level of experience are unchanged. It applies to every set of counts of every cell (plain, keyed on a guess; first, second, third). Where the description leaves a choice: I apply it to the look-ahead counts and to the cells keyed on a guess as to the plain first counts, since "a cell" is not qualified; N is not chosen by me.

Everything else as `PLAN_1.md` (eta 0.1 with the shrinking step, e = 2, births 0.02 and nothing, 255 + 130 + 130 kinds, D = 9). `ahead.py` gains the option `ncap=N` in the reference and in the kernel; the check gains a case with it, and with `ncap=0` the program must still give its earlier figures.

- **Arm 1, with the cap:** `plan=add:400000,sub:200000 every=50000 ncap=16`, seeds 73129, 93683, 116089, 406364, 545353.
- **Arm 2, without the cap:** the same with `ncap=0`; only if time remains, otherwise NOT RUN.

Reported: never-shown addition (own bytes, whole answer) at every reading; at the end of the addition phase and at the end of the life, the last reading and the mean of the last three; by place; the guesses; subtraction on the clean score at the end; cells, counts above zero, trusts used, CPU. No development seed: nothing is chosen.

## 3. What is asked, and what counts

- **Holds:** arm 1, median of five seeds, last reading at 400,000 lines at least 75%, and the last reading at the end of the life at least 85% of it.
- **Fails:** under 65% at 400,000 lines, or under 70% of its level kept. Between: stated part by part.
- The two parts are reported apart: the fall with reading (40,000-line figure of `PLAN_1.md`, 91.4, against the readings from 50,000 to 400,000) and the loss through the second task (400,000 against the end of the life).

## 4. Predictions (mine, before the cap is written)

| | Statement | Chance |
|---|---|---|
| L1 | arm 1: at least 75% at 400,000 lines (median, last reading) | 0.70 |
| L2 | arm 1: at least 85% at 400,000 lines | 0.25 |
| L3 | arm 1: the reading at 400,000 is within 5 points of the reading at 50,000 | 0.50 |
| L4 | arm 1: at the end of the life at least 85% of the 400,000 level | 0.60 |
| L5 | arm 1: both parts hold | 0.45 |
| L6 | arm 1: subtraction on the clean score at the end at least 80% | 0.55 |
| L7 | arm 1: the reading at 50,000 lines is at least 3 points below 91.4 (the cap costs something early) | 0.55 |
| L8 | arm 2 (if run): under 70% at 400,000 lines | 0.80 |
| L9 | arm 2 (if run): arm 1 is at least 10 points above it at 400,000 lines | 0.70 |

## 5. Size, cost, what is not run

As `PLAN_1.md`, with more cells for the longer life (reported). By estimate 16 to 20 CPU-minutes a run; five runs at once on four cores, about 25 minutes an arm. Not run: other values of N; a cap the answering cell learns; the half-length life; seeds 2001 to 2005; three-digit sums.

## 6. Outcome of arm 1, with the cap (entered 20 November after its runs; `res/long_cap16_*.json`, table `report_4.txt`, check `check_output_cap.txt`)

*The check.* With `ncap` the kernel agrees with the reference on two further cases (to 3e-15); without it the program gives its earlier readings (seed 73129: 74.6 and 82.8 at 5,000 and 10,000 lines).

Never-shown two-digit additions, own bytes, whole answer, %. Seeds 73129, 93683, 116089, 406364, 545353. Addition for 400,000 lines, then subtraction for 200,000 with nothing of addition shown; a reading every 50,000 lines.

| | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | sub 50k | sub 100k | sub 150k | sub 200k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median | 87.2 | 84.4 | 81.2 | 82.4 | 77.8 | 80.4 | 82.0 | **82.6** | 79.6 | 76.6 | 81.8 | **83.4** |
| 73129 | 85.4 | 82.2 | 81.0 | 80.4 | 76.4 | 80.8 | 81.2 | 82.2 | 82.2 | 79.2 | 83.6 | 83.4 |
| 93683 | 87.0 | 84.4 | 81.2 | 83.2 | 76.6 | 79.8 | 82.0 | 81.4 | 77.4 | 76.6 | 81.8 | 84.6 |
| 116089 | 87.8 | 84.6 | 81.6 | 82.4 | 77.8 | 80.4 | 82.8 | 82.8 | 79.6 | 76.6 | 80.8 | 82.6 |
| 406364 | 87.6 | 87.8 | 84.6 | 84.4 | 79.6 | 80.4 | 85.4 | 86.2 | 81.4 | 81.0 | 85.8 | 86.6 |
| 545353 | 87.2 | 83.6 | 81.2 | 82.2 | 80.0 | 79.6 | 81.4 | 82.6 | 74.2 | 72.4 | 80.2 | 81.4 |

| | each seed; median |
|---|---|
| end of addition, last reading | 82.2, 81.4, 82.8, 86.2, 82.6; **82.6** |
| end of addition, mean of the last three | 81.4, 81.1, 82.0, 84.0, 81.2; 81.4 |
| end of the life, last reading | 83.4, 84.6, 82.6, 86.6, 81.4; **83.4** |
| end of the life, mean of the last three | 82.1, 81.0, 80.0, 84.5, 78.0; 81.0 |
| end of the life as % of the end of addition | 101.5, 103.9, 99.8, 100.5, 98.5; **100.5** |
| subtraction at the end, clean score (80 to 104 pairs) | 89.1, 84.4, 75.0, 87.5, 89.2; **87.5** |

By place (hundreds, tens, units; median): 98.8, 82.6, 100 at the end of addition; 98.0, 84.4, 100 at the end of the life. Guesses at the first answer byte: three ahead 100% at both points; two ahead 86.0 and 81.6. Size (medians): 2.98 million cells and 36.4 million counts above zero at the end of addition; 3.28 million cells, 41.6 million counts, 37,429 trusts used at the end of the life; 1,318 CPU-seconds a run.

**What was asked holds, on five seeds of five**: at least 75% at the end of 400,000 lines (82.6; lowest seed 81.4) and at least 85% of that kept through the second task (100.5%; lowest seed 98.5%).

| | Statement | Outcome |
|---|---|---|
| L1 | at least 75% at 400,000 | held (82.6) |
| L2 | at least 85% at 400,000 | failed (one seed of five, 86.2) |
| L3 | 400,000 within 5 points of 50,000 | held, narrowly (4.6 under) |
| L4 | at the end at least 85% of the 400,000 level | held (100.5) |
| L5 | both parts | held |
| L6 | subtraction, clean score, at least 80% | held at the median (87.5); one seed at 75.0 |
| L7 | the reading at 50,000 at least 3 below 91.4 | held (87.2) |

Readings (mine).
1. **The two losses, told apart.** *With reading:* from 91.4 at 40,000 lines without a cap (`PLAN_1.md`) the capped system stands at 87.2 at 50,000 lines and at 82.6 at 400,000: about 4 points are the cap's price early, and about 5 more go with reading, nearly all of them by 150,000 lines; from 150,000 to 400,000 the median moves between 77.8 and 82.6 with no trend (every seed dips at 250,000 lines and comes back, which I have not explained). *Through the second task:* nothing is lost at the end (83.4 against 82.6), but the path is not flat: addition dips to 76.6 at 100,000 lines of subtraction (72.4 on one seed) and returns with no addition shown. So "kept" holds for the end of the life and for every reading at 70% or more of the level; a reading taken in the middle of the second task would show a loss of 6 points.
2. **The level is 82 to 83, not 91.** What stands at the end of the long life is 8 to 9 points under the short life's figure. The tens digit carries all of it (82.6 to 84.4; the other places 98 to 100), and the tens guess is right less often (86.0, then 81.6) than in the short life (88.6).
3. **Subtraction is learned beside it**: 87.5 on lines never written, about the short life's 89.4.
4. These are the first figures for the cap on five seeds; the other line's cap figures as reported to me (82 and 81 at 400,000 on two seeds) are close to mine (82.6), which I note and do not call a repeat. N = 16 was given, not chosen; no other N was run.
