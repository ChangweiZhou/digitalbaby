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
