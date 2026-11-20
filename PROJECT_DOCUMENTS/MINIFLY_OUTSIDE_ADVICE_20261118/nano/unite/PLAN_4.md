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
