# Whole answers on never-shown combinations: report (22 November 2026)

Helper session for the NEWNET line. Development level; no gate, seal or outside review is claimed. The plan
(`ANSWERS_PLAN.md`, commit `743a46a6`) was committed before any run. Source: `claude/magical-albattani-sei232` @ `15899397`; work and
results on `claude/newnet-general-sei232`. Learner: `NEWNET/net10.py`, unchanged; script `NEWNET/t21_answers.py`; raw lines
`NEWNET/answers_runs/{task}_{struct}_s{seed}.jsonl` (with 8 example answers per test). 12 runs: 3 tasks x K8, N1 x seeds 1, 2, 4 at once,
OMP_NUM_THREADS=1, about 85 min of wall time, about 5.3 CPU-hours.

**Life reached.** N1 read the full 400,000 bytes in every run (1,356-1,505 s). K8 is slower with four processes sharing the machine
(about 5 ms a byte) and was stopped by the planned 1,800 s limit: add2 at 352,380 / 366,760 bytes, copy at 376,248 / 378,224, rev at
375,072 / 383,040 (seeds 1 / 2). Its last test was made at that point. Compare the two structures at 300,000 bytes, where both have a test.

**Tests.** Learning off; 500 shown items; never-shown items: 500 for add2 but only 200 for copy and rev, because only 200 never-shown
strings exist (all 200 were used). One never-shown item is 0.005 for copy and rev and 0.002 for add2. "Whole" means the structure's
own answer, fed back byte by byte, is right in every byte including the newline. "Forced" means the right bytes are fed and the share
of answer bytes right is counted.

**Reference (not a learner).** A table answering every shown item from memory scores 1.0 on shown and 0 on never-shown items.
Chance for one place: 0.1 for a digit or a letter.

## Whole answers (shown / never-shown) at each test

| task | struct | seed | 100,000 | 200,000 | 300,000 | last (bytes) |
|---|---|---|---|---|---|---|
| add2 | K8 | 1 | 0.014 / 0.014 | 0.010 / 0.002 | 0.030 / 0.018 | 0.032 / 0.010 (352,380) |
| add2 | K8 | 2 | 0.016 / 0.008 | 0.038 / 0.004 | 0.034 / 0.010 | 0.032 / 0.002 (366,760) |
| add2 | N1 | 1 | 0.016 / 0.016 | 0.022 / 0.012 | 0.024 / 0.008 | 0.026 / 0.012 (400,000) |
| add2 | N1 | 2 | 0.018 / 0.008 | 0.034 / 0.008 | 0.024 / 0.004 | 0.036 / 0.006 (400,000) |
| copy | K8 | 1 | 0.492 / 0.005 | 0.692 / 0.005 | 0.748 / 0.010 | 0.808 / 0.020 (376,248) |
| copy | K8 | 2 | 0.402 / 0.005 | 0.608 / 0.000 | 0.710 / 0.005 | 0.766 / 0.010 (378,224) |
| copy | N1 | 1 | 0.486 / 0.000 | 0.680 / 0.010 | 0.736 / 0.020 | 0.776 / 0.020 (400,000) |
| copy | N1 | 2 | 0.502 / 0.000 | 0.596 / 0.035 | 0.718 / 0.025 | 0.772 / 0.045 (400,000) |
| rev | K8 | 1 | 0.328 / 0.000 | 0.434 / 0.010 | 0.468 / 0.005 | 0.480 / 0.000 (375,072) |
| rev | K8 | 2 | 0.332 / 0.005 | 0.430 / 0.015 | 0.542 / 0.000 | 0.546 / 0.000 (383,040) |
| rev | N1 | 1 | 0.388 / 0.020 | 0.494 / 0.000 | 0.494 / 0.005 | 0.562 / 0.005 (400,000) |
| rev | N1 | 2 | 0.392 / 0.000 | 0.470 / 0.010 | 0.546 / 0.000 | 0.546 / 0.000 (400,000) |

## By place at the last test (the mean of the two seeds)

Places: add2 hundreds / tens / units / newline; copy and rev answer letters 1 / 2 / 3 / newline. Forced = right bytes fed; free = own
bytes fed back. The forced byte share is the mean over all four places.

| task | struct | forced byte share, shown / never | forced by place, shown | forced by place, never-shown | free by place, never-shown |
|---|---|---|---|---|---|
| add2 | K8 | 0.531 / 0.506 | 0.76 / 0.23 / 0.13 / 1.00 | 0.74 / 0.18 / 0.11 / 1.00 | 0.78 / 0.12 / 0.12 / 1.00 |
| add2 | N1 | 0.548 / 0.518 | 0.79 / 0.25 / 0.15 / 1.00 | 0.76 / 0.22 / 0.09 / 1.00 | 0.79 / 0.14 / 0.13 / 1.00 |
| copy | K8 | 0.934 / 0.432 | 0.86 / 0.92 / 0.97 / 1.00 | 0.34 / 0.26 / 0.16 / 0.98 | 0.35 / 0.52 / 0.85 / 0.99 |
| copy | N1 | 0.934 / 0.541 | 0.86 / 0.93 / 0.95 / 1.00 | 0.53 / 0.45 / 0.20 / 0.99 | 0.55 / 0.50 / 0.66 / 0.96 |
| rev | K8 | 0.872 / 0.750 | 1.00 / 1.00 / 0.49 / 1.00 | 1.00 / 1.00 / 0.01 / 0.99 | 1.00 / 1.00 / 0.00 / 0.99 |
| rev | N1 | 0.888 / 0.742 | 1.00 / 1.00 / 0.56 / 1.00 | 1.00 / 0.97 / 0.01 / 1.00 | 1.00 / 0.97 / 0.01 / 0.99 |

(Every test, by place, is in the jsonl files. The by-place shares hardly change between 100,000 bytes and the end, except copy and rev
shown items, which rise.)

## Against the plan's predictions

| task | prediction (shown whole / never whole; places) | measured at the last test | verdict |
|---|---|---|---|
| add2 | 0.10-0.50 / <= 0.05; hundreds >= 0.85, tens 0.10-0.40, units 0.15-0.50 | 0.026-0.036 / 0.002-0.012; hundreds 0.72-0.79, tens 0.17-0.23, units 0.08-0.12 | never-shown as predicted; **shown far below the prediction**; hundreds and units below the predicted range |
| copy | 0.60-1.0 / <= 0.15; first letter forced (never) <= 0.5 | 0.766-0.808 / 0.010-0.045; first letter 0.28-0.39 (K8), 0.52-0.54 (N1) | as predicted, except N1's first letter slightly above 0.5 |
| rev | 0.60-1.0 / <= 0.15; first letter >= 0.95 | 0.480-0.562 / 0.000-0.005; first letter 1.00 | never-shown as predicted; **shown below the predicted 0.60** (but above the 0.30 "too short" line) |

The refutation conditions (never-shown whole >= 0.20 or 0.40, or >= half the shown whole) were not met in any run. The two
structures differed by at most 0.08 on any whole-answer share at matched tests, as predicted.

## Plain readings (only what was measured)

1. **No structure gave whole answers on never-shown combinations.** At most 4.5% (copy, N1 seed 2: 9 of 200) on never-shown items.
   On copy and rev this is against 0.48-0.81 on shown items, the reference table's pattern (1.0 / 0). The line through the life is
   flat for never-shown items and rising for shown ones.
2. **add2 was learned neither on shown nor on never-shown items** (whole 0.01-0.04 on both; forced 0.53 / 0.51). Each shown pair was
   read only about 4-5 times. The hundreds digit (0.72-0.80) is above the half that guessing between 0 and 1 gives. The tens digit is
   about 0.2 and the units digit is at chance (0.08-0.12) even when the right digits before it are fed. The units digit depends only on
   the two units digits of the question (5 and 8 bytes back); all 100 such pairs were read thousands of times, yet the place stays at
   chance.
3. **rev shows where the single places work.** The first answer letter (the byte just before "=") and the second (3 bytes back) are
   right on never-shown items (0.94-1.00). The third (the first letter of the question, 6 bytes back over the shown "=") is right on
   0.00-0.02 of never-shown items, below chance (0.1). On shown items it is right on about half (0.43-0.57). So a place carried over 1-3
   bytes generalises; a place 6 bytes back is learned only for a seen combination, and on an unseen one the structure gives some other
   letter, more often wrong than a random guess.
4. **copy:** with the right bytes fed, each answer letter (always 4 bytes back) is right on never-shown items only 0.16-0.53 of the
   time. The third falls lowest (0.16-0.20), and K8 does worse than N1 here. Unexplained: in the free run the third letter is right far
   more often (0.85 K8, 0.67 N1) than when forced (0.16-0.20). The examples show answers whose last letter is right while the first two
   are wrong (`eah=cah`, `hab=cab`, `iec=bic`). That the wrong answers are often shown strings is not informative here: 800 of the
   1,000 strings are shown. No test of a cause was run.
5. **K8 and N1 cannot be told apart on whole answers** (differences of at most 0.08 at matched tests, within the seed spread). N1 is
   better on copy's forced letters on never-shown items (0.54 against 0.43 of bytes) and is about 25% faster.

## Limits

One task size each; lives of 352,000-400,000 bytes; K8 stopped early by the planned time limit; two seeds; 200 never-shown items for
copy/rev; teach feedback only; no control learner other than the memory-table reference. The free/forced difference for copy's
third letter is reported, not explained.
