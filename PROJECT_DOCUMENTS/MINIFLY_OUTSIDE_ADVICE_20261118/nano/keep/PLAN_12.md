# Keeping, twelfth plan: old lines from a store the system keeps (asked for by the coordinating session as candidate d). 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written before any run of this plan; the eleventh plan's runs were in progress and unread but for one line.

**Question** (the coordinating session's, as I understood its message). With the program's choice (`reserve=128`), the old line that is given one time in 32 during subtraction has so far been a fresh addition line made by the program from the 8,000 shown sums. If instead it is drawn from N addition lines the system kept as they arrived, how large must the store be for never-shown addition to be kept, and does a small store keep the rule or only the stored sums?

**The store.** The last N lines read before the change of task, N in {32, 256, 2,048}: 320, 2,560 and 20,480 bytes. The rule "the last N lines" knows nothing of the task; that the store stops being overwritten at the change is the program's doing, as everything in this control is. During subtraction, one line in 32 is a line drawn at random from the store.

**The learner.** `store.py`: the control `reserve=128` of `cells2.py` (384 cells, 64,910 numbers that learn), with the loop of `cells2.run` for this case written out so that the store can be kept; compiled kernel `fast=1`. With N = 0 (fresh lines) it must give the figures of `cells2.py` to the digit (67.0 at the median: 75.2, 62.8, 67.0, 64.4, 67.8), which is the check of the copy.

**Arms, five reported seeds; no setting is chosen, seed 1 is not used:** N = 0 (fresh), 32, 256, 2,048. Beside: no old line (4.4, run under `PLAN_1.md`).

**Measures.** At the end: never-shown addition (last reading and mean of the last three); addition on the stored sums themselves and on 500 shown sums; subtraction on all 500 held-out pairs and on those whose line no shown pair writes.

**Hypothesis (one).** Keeping the rule needs a store that covers the digits in their places, not all sums: a store of a few thousand lines does what fresh lines do, a store of a few tens keeps its own sums and not the rule. **Refuted if** N = 2,048 keeps never-shown addition more than 10 points below fresh lines (median), or N = 32 keeps it within 10 points of fresh lines.

**Predictions.**
- L1 (0.65): N = 2,048: never-shown addition within 5 points of fresh lines (median, last reading).
- L2 (0.65): N = 32: never-shown addition below 30% (median), with addition on the stored sums above 80%.
- L3 (0.50): N = 256: never-shown addition 40% or more (median).
- L4 (0.90): N = 0 reproduces `cells2.py`'s figures to the digit.

**Cost.** 20 runs of about 80 CPU seconds.

**Not run.** Other rules for the store (every k-th line, a random place replaced); the store with this line's own candidate, which keeps more with no old line than with some; a store that goes on being filled after the change.

## Outcome (entered 20 November, after all 20 runs; `table12.txt`, `res/c2_r12_*`)

Seeds 73129, 93683, 116089, 406364, 545353; median. Never-shown addition at the end: last reading | mean of the last three readings.

| Old line drawn from | Store, bytes | Never-shown addition | Addition on the stored sums | Addition on 500 shown sums | Subtraction: all 500 / clean |
|---|---|---|---|---|---|
| no old line (`PLAN_1.md`) | 0 | 4.4 \| 4.4 | | | 74.0 / 68.8 |
| the last 32 lines | 320 | 68.8, 49.4, 58.6, 58.8, 56.6; **58.6** \| 52.9 | 93.8, 71.9, 87.1, 87.5, 67.7; 87.1 | 60.0 | 75.0 / 75.9 |
| the last 256 lines | 2,560 | 71.6, 65.4, 69.4, 56.8, 64.6; **65.4** \| 58.1 | 69.2 | 66.6 | 77.4 / 77.1 |
| the last 2,048 lines | 20,480 | 72.4, 67.8, 70.4, 67.8, 64.4; **67.8** \| 60.7 | 67.4 | 68.4 | 76.8 / 77.2 |
| fresh lines from the 8,000 shown sums | (80,000) | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** \| 61.8 | | 64.8 | 75.4 / 74.7 |

With fresh lines `store.py` gives `cells2.py`'s figures to the digit.

1. **A store of 2,048 lines, and one of 256, do what fresh lines do** (67.8 and 65.4 against 67.0; means of three 60.7, 58.1, 61.8).
2. **A store of 32 lines, 320 bytes, keeps most of it: 58.6% of never-shown addition** (52.9 as the mean of three), where no old line keeps 4.4. It answers its own 32 sums better (87.1) than sums it does not hold, so it does hold them specially; but it keeps the rule far beyond them (58.6 on never-shown sums, 60.0 on 500 shown sums not in the store). **The hypothesis is refuted in its second half**: I had expected a small store to keep its own sums and not the rule.
3. My reading: in this control nothing addition wrote is touched; what the old lines have to do is teach the 128 new cells not to speak on addition lines. That is one thing to learn, not eight thousand, and 32 lines that carry the plus sign teach most of it. It is the same fact the third plan used from the other side: what marks an addition line is the operator, and a cell that is silent where the operator is not its own does no harm.

**Predictions.** L1 (0.65) held: 67.8 against 67.0. L2 (0.65) failed: 58.6, not below 30 (the stored sums are at 87.1, as predicted). L3 (0.50) held: 65.4. L4 (0.90) held.

**Not run**, as planned: other rules for the store; a store that goes on filling after the change; the store with this line's own candidate.
