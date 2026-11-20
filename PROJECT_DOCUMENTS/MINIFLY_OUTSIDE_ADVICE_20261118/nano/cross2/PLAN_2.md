# nano/cross2, second plan (E2): two-digit numbers with letters between them, one and two crossings (20 November 2026)

**Authorisation.** As for the first plan (Mr. Xu's words of 20 November in `ROADMAP_SECOND_REVISION_20261120.md`; the coordinating session's brief for track M). **Source.** Branch `claude/nano-cross2-8mmhfm` at f684803. Written before any run on this text at length; known to me when writing: the eleventh plan's outcome in `loop_small_case` (one crossing, development seed: 7.7% of never-shown sums, by place 97, 76, 16), a 40,000-line timing here (nothing learned), and the first plan's development-seed outcome on m1 (cross=2 reached 90% at 320,000 lines, cross=1 at 500,000); the first plan's reported seeds were not yet read. Development seed only unless the condition below is met; a small case; no checker gate.

**Question.** On the text g2 (`aa+<0 to 8 letters>bb=ccc`), which one crossing did not learn, does credit passed over a second connection bring the exact units digit across the letters?

**Learner, exactly.** `hold.py` of this folder, `text=g2 gap=8 chain=2 eta1=1 ret=exact fast=1`, 256 cells, K = 16, the present byte and the two before it on lines, 18 byte values, 2,400,000 lines, read every 200,000; arms `cross=1` and `cross=2`, seed 1. Numbers that learn, the same in both arms: 2 × 256 × (54 + 16 + 1) = 36,352 in the cells and 9,234 at the answering cell and its return connections; 45,586 in all.

**Measure.** Whole answer right on 500 never-shown pairs at the gap of 8 letters, mean of the last three readings; beside it the gaps of 0 and 4, the sums shown, and the answer by place at the last reading.

**Hypothesis.** What was missing for this text is credit over two connections: with `cross=2` the units digit is held exactly and never-shown sums are answered. *Refuted if* the `cross=2` arm is at or below 15% of never-shown sums at the gap of 8 letters (the eleventh plan's threshold).

**Condition for the reported seeds.** Only if the `cross=2` arm reaches 30% at the gap of 8 letters on the development seed are the five reported seeds run, in both arms, with the same settings.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| B1 | cross=2 reaches at least 30% of never-shown sums at the gap of 8 letters | 0.25 |
| B2 | cross=2 is at least 3 points above cross=1 at the gap of 8 letters | 0.55 |
| B3 | cross=2: the units digit is right on at least 30% of never-shown sums at the last reading (it was 16 with one crossing) | 0.35 |
| B4 | this folder's cross=1 rerun is within 5 points of the eleventh plan's 8% at the gap of 8 letters | 0.80 |

**Cost.** About 17 CPU-minutes with cross=1 and about 25 with cross=2; if the condition is met, ten more runs, about 3.5 CPU-hours, under one hour of wall time on four cores.

**Not run.** More lines or more cells with two crossings (the twelfth plan of `loop_small_case` takes those with one crossing); other steps; gaps beyond 8; the first family's task byte and modulus.
