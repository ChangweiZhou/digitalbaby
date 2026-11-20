# nano/cross2, third plan (E3): two-digit sums with no delay line, one and two crossings (20 November 2026)

**Authorisation and source.** As for the second plan; branch `claude/nano-cross2-8mmhfm` at f684803. Written at the same time as the second plan and before any run on this text at length; known to me: the sixth plan's outcome in `loop_small_case` (the present byte only, development seed, 2,400,000 lines: 1.1% without a crossing, 5.0% with one, by place 93, 38, 10), a 40,000-line timing here, and the first plan's development-seed outcome. Development seed only; a small case; no checker gate.

**Question.** With the present byte alone on a cell's lines, are two-digit sums learned when credit passes over two connections? The reading in section 12.3 of `loop_small_case/REPORT.md` (its author's) is that a cell which tells the second byte of a number from the first would have to be taught by a holding cell that is itself taught by a reader: two crossings.

**Learner, exactly.** `hold.py` of this folder, `text=d2 chain=0 eta1=1 ret=exact fast=1`, 256 cells, K = 16, 14 byte values, 2,400,000 lines, read every 200,000; arms `cross=1` and `cross=2`, seed 1. Numbers that learn, the same in both arms: 2 × 256 × (14 + 16 + 1) = 15,872 in the cells and 2 × 14 × 256 + 14 = 7,182 at the answering cell and its return connections; 23,054 in all.

**Measure.** Whole answer right on 500 never-shown pairs, mean of the last three readings; beside it the sums shown and the answer by place at the last reading.

**Hypothesis.** Two crossings are what the text without a delay line was missing. *Refuted if* the `cross=2` arm is at or below 15% of never-shown sums.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| C1 | cross=2 reaches at least 20% of never-shown sums | 0.25 |
| C2 | cross=2 is at least 3 points above cross=1 | 0.50 |
| C3 | cross=2: the tens digit is right on at least 60% of never-shown sums at the last reading (it was 38 with one crossing) | 0.35 |
| C4 | this folder's cross=1 rerun is within 5 points of the sixth plan's 5.0% | 0.80 |

**Cost.** About 10 CPU-minutes with cross=1 and about 15 with cross=2. **Not run.** The reported seeds (not asked for in this experiment); one delay line with two crossings; longer runs; the instrument.
