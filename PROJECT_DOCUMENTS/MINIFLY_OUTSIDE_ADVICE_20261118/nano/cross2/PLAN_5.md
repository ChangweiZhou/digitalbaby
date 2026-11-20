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
