# nano/unite, fourth plan: the score after a read line, beside the score with an empty past (20 November 2026)

**Why.** The coordinating session reports a fault of the measuring instrument, its own (`loop_small_case/ERRATA_20261120.md`, section 3, on its branch at 8ae14f646; I have not read that file, and enter the figures as it gave them to me): every line, mine included, scores a never-shown sum with line ends on the lines that reach before it, a past no system meets after the first line of its life. Scored after a read line, on plain two-digit sums, the relay system reads 82.9% at 40,000 lines where the old score gave 66.2, and counting cells of at most four delays 80.1 where it gave 77.6. For a system with a clock it has not been measured. This plan is that session's request, not new words of Mr. Xu; the authorisation is the one quoted in `PLAN_1.md`. **Status.** Development level; no gate, seal or review claimed. Written before its runs. My time for this line was already spent; this is an addition asked for afterwards.

**What my old score does** (named "empty past" from here on). At the `=` of a held-out line the chain holds the line's five bytes and then three line ends. In the stream those three lines hold the end mark and the last two digits of the line before. So every cell keyed on one of the last two chain lines is, in the old score, either silent or a cell met only at the very start of the life.

**One change, to the score only.** `past=5`: before each held-out line the system is given a line drawn by chance from the lines shown (its letters drawn as in the stream), read with every entry in its likelier state; then the held-out line as before, the system writing its own answer. Mean of five draws. Nothing learns while scoring. The draws come from a random stream of their own, so the life and the empty-past figures of each run must come out equal to those already in `res/` to the digit (a check on the repeat). `check_output.txt`: ten cases PASSED after the edit.

**Runs.** The united system (`clock=trial`) on the five reported seeds, 200,000 lines, both scores at every reading (`res/past_trial_*.json`); the oracle arm on seed 73129 (`res/past_oracle_73129.json`). The finished predictions of `PLAN_1.md` are not re-scored; these figures are an addition.

**Question.** Is the fall from 93.8 at 40,000 lines to 53.7 at 200,000 still there when the sums are scored after a read line?

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| P1 | the empty-past figures repeat to the digit on all six runs | 0.90 |
| P2 | after a read line, the united system at 200,000 lines (last three, median) is at least 15 points under its reading at 40,000 lines: the fall is still there | 0.60 |
| P3 | after a read line the reading at 40,000 lines is within 5 points of the empty-past one (93.8) | 0.55 |
| P4 | after a read line the reading at 200,000 lines is higher than the empty-past one by at least 5 points | 0.45 |

I do not know which way it goes: the cells that the old score silences are keyed on the digits of an unrelated line, so they may be noise the old score spared the system, or senders it was trained to hear.

**Not run:** the keeping runs and the diagnostics re-scored; the no-clock arm; the oracle arm on the other seeds.

## Outcome (entered 20 November 2026; `res/past_trial_*.json`, `res/past_oracle_73129.json`)

Never-shown sums, own answer bytes, whole answer right, %. "Empty past": line ends before the held-out line (the score of `PLAN_1.md`). "After a read line": mean of five draws; the three gaps agree within a point and the gap of 8 is given. Reported seeds in the order 73129, 93683, 116089, 406364, 545353.

| Arm, score | 40,000 lines: each seed; median | 60,000 (median) | 100,000 (median) | 200,000: each seed; median | last three: each seed; **median** |
|---|---|---|---|---|---|
| united, empty past | 93.8, 93.6, 94.0, 91.6, 96.2; 93.8 | 92.2 | 78.4 | 58.0, 53.4, 50.6, 49.0, 52.6; 52.6 | 58.3, 54.5, 52.7, 50.3, 53.7; **53.7** |
| united, after a read line | 91.4, 91.6, 93.6, 90.8, 94.7; 91.6 | 91.6 | 81.6 | 64.0, 60.4, 58.2, 55.2, 59.6; 59.6 | 64.1, 61.9, 58.7, 56.3, 59.8; **59.8** |
| oracle, seed 73129, empty past | 89.4 | 81.2 | 83.0 | 78.6 | 79.3 |
| oracle, seed 73129, after a read line | 93.6 | 86.3 | 88.0 | 85.1 | 86.3 |

The empty-past figures and the closed entries of all six runs equal those of the first runs at every reading. **P1 held.**

**The fall is still there in the united system**: after a read line it goes from 91.6 at 40,000 lines to 59.8 at 200,000 (32 points; with an empty past 40 points). It is still the tens digit (55 to 64% right at the end; hundreds 96 to 98, units 100). **P2 held. P3 held** (91.6 against 93.8 at 40,000 lines: the old score was a little kind to the system early). **P4 held** (59.8 against 53.7 over the last three readings: the old score was unkind to it late).

**In the oracle arm the picture changes, on the one seed run**: after a read line it reads 93.6 at 40,000 lines and 86.3 over the last three readings, a fall of 7 points where the empty past showed 10 on this seed (and, at the median of five seeds with an empty past, 17: 92.0 to 75.1). So what `PLAN_1.md` and the report call "the cells' own fall with more reading" is, on this one seed, in good part smaller under the fairer score; one seed, not a median. I do not re-score the old statements. What should be read differently from here on (reading, mine):

1. The composition's fall within 200,000 lines stands under the fairer score, on five seeds.
2. Of its two located parts, the share of the clock's trials going on is the larger: after a read line the united system is 22 points under the oracle arm on seed 73129 at the end (64.1 against 86.3; with an empty past 21 under, 58.3 against 79.3), while the oracle arm itself has fallen by only 7. The diagnostic that located it (no entry tried after 30,000 lines) was scored with an empty past only, on the development seed, and has not been scored again.
3. The share of the cells' own fall is smaller than I reported, and may be small; that needs the oracle arm on the other four seeds after a read line, which is **not run**.

**Not run:** the oracle arm on four seeds; the keeping runs, the cap and the stopping of trials scored after a read line; the no-clock arm.
