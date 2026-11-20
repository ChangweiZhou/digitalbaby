# nano/cross2, fifth plan (E5): the forward rules on g2 at the larger step (20 November 2026)

**Authorisation.** As for the fourth plan; the coordinating session asked for this arm after E4's development outcome. **Source.** Branch `claude/nano-cross2-8mmhfm` at 5eae4fe; `hold.py` and `kern5.c` unchanged since the first commit. Written before any forward run at eta1 = 4 on this text; known to me: E2 (forward, eta1 = 1: 7.5 and 9.5% at the gap of 8 letters), E4 on seed 1 (gradient through the line: 68.6% at eta1 = 1 after a level stretch of 1,600,000 lines; 64.1% at eta1 = 4 with no level stretch), and the coordinating session's report of its one-crossing run at eta1 = 1 (level for about five million lines, 46% at 9,600,000). E4's reported seeds were running and unread. Development seed 1 only; a small case; no checker gate.

**Question.** Every forward arm on g2 was run at eta1 = 1. Is the long level stretch of the forward rules a limit of the rule or of the step?

**Learner, exactly.** As E2 (`hold.py`, `text=g2 gap=8 chain=2 ret=exact fast=1`, 256 cells, K = 16, 45,586 numbers that learn, 2,400,000 lines, read every 200,000, seed 1), with `eta1=4`; arms `cross=1` and `cross=2`. One thing changes against E2: the step.

**Measure.** As E2; and never-shown sums at the gap of 8 letters at each 200,000 lines, beside E4's gradient curves at eta1 = 4 and 1 and E2's forward curves at eta1 = 1.

**Hypothesis.** The level stretch is a limit of the step: at eta1 = 4 a forward arm leaves it within 2,400,000 lines, reaching at least 30% of never-shown sums at the gap of 8 letters (mean of the last three readings). *Refuted if* both forward arms are at or below 15%.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| F1 | cross=1 at eta1 = 4 reaches at least 30% at the gap of 8 letters | 0.35 |
| F2 | cross=2 at eta1 = 4 reaches at least 30% at the gap of 8 letters | 0.40 |
| F3 | at least one forward arm ends below its eta1 = 1 run of E2 (the step is too large for the forward rule) | 0.25 |
| F4 | both forward arms end at least 20 points below the gradient through the line at the same step (64.1%) | 0.70 |

**Cost.** About 26 and 34 CPU-minutes. **Not run.** Other steps (2, 8); the reported seeds; d2; m1 at this step.

## Outcome (entered 20 November after the two runs)

Development seed 1, g2, 2,400,000 lines, eta1 = 4 (`res/hold_e5_c1_e4_1.json`, `res/hold_e5_c2_e4_1.json`; every reading in `report_e5.txt`). Mean of the last three readings; by place at the last reading.

| | never-shown sums %, gap 0 / 4 / 8 | sums shown %, gap 0 / 4 / 8 | by place | bytes right % | CPU s |
|---|---|---|---|---|---|
| forward, cross=1, eta1 = 4 | 6.5 / 5.9 / 6.0 | 9.0 / 8.9 / 7.9 | 97, 67, 10 | 46.2 | 1,321 |
| forward, cross=2, eta1 = 4 | 6.9 / 7.6 / 7.7 | 10.4 / 9.7 / 9.3 | 98, 69, 13 | 46.6 | 2,012 |
| forward, cross=1, eta1 = 1 (E2) | 7.8 / 7.9 / 7.5 | 9.3 / 8.9 / 8.7 | 97, 76, 16 | 47.1 | 1,548 |
| forward, cross=2, eta1 = 1 (E2) | 9.8 / 10.3 / 9.5 | 12.3 / 12.3 / 11.9 | 97, 78, 18 | 47.7 | 2,024 |
| gradient through the line, eta1 = 4 (E4) | 64.4 / 64.7 / 64.1 | 68.5 / 70.3 / 70.1 | 98, 70, 98 | 53.1 | 782 |
| gradient through the line, eta1 = 1 (E4) | 69.7 / 68.4 / 68.6 | 72.8 / 73.9 / 73.9 | 99, 83, 96 | 53.6 | 767 |

Never-shown sums at the gap of 8 letters at each 200,000 lines, seed 1:

| lines, thousands | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| forward cross=1, eta1 = 4 | 6 | 5 | 5 | 5 | 5 | 5 | 7 | 7 | 7 | 6 | 7 | 5 |
| forward cross=2, eta1 = 4 | 7 | 6 | 5 | 5 | 7 | 6 | 6 | 5 | 9 | 8 | 8 | 8 |
| gradient, eta1 = 4 | 5 | 17 | 38 | 47 | 65 | 67 | 63 | 71 | 66 | 64 | 63 | 65 |
| forward cross=1, eta1 = 1 | 5 | 3 | 5 | 7 | 7 | 6 | 9 | 5 | 8 | 4 | 10 | 8 |
| forward cross=2, eta1 = 1 | 5 | 8 | 6 | 8 | 11 | 6 | 9 | 10 | 6 | 9 | 9 | 10 |
| gradient, eta1 = 1 | 8 | 11 | 12 | 8 | 13 | 13 | 11 | 15 | 48 | 66 | 61 | 79 |

**The hypothesis is refuted as registered**: at eta1 = 4 both forward arms are at or below 15% at the gap of 8 letters (6.0 and 7.7%), level through the whole run.

**Scores.** F1 failed (6.0). F2 failed (7.7). F3 held (both arms end below their eta1 = 1 runs: 6.0 against 7.5, 7.7 against 9.5; differences of the size of the movement between readings). F4 held (58 and 56 points below the gradient at the same step).

**What it shows and does not** (my reading; one seed). The larger step, which removes the level stretch for the gradient through the line, does nothing for the forward rules with one or two crossings: no rise in 2,400,000 lines, the units digit at 10 to 13% and the tens digit somewhat lower than at eta1 = 1 (67 to 69 against 76 to 78). So on this seed the gap between the forward rules and the gradient stands at both steps, and at eta1 = 4 it is not a matter of the step. Since two crossings at the last two bytes add next to nothing at either step, what the gradient uses and these rules leave out is, on my reading, credit that crosses between cells many bytes before the mismatch (the instrument's X=1 and X=2 at any time would separate "how many crossings" from "how far back"; they are numpy only and were not run at this length). Not run: other steps (2, 8, 16); the reported seeds; longer runs at eta1 = 4.
