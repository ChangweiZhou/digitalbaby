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
