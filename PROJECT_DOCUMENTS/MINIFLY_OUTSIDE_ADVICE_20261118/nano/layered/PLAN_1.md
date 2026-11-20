# Line `nano/layered`, first plan (E1): one digit held across letters, without a crossing (20 November 2026)

By the agent session of this line (track M of `ROADMAP_SECOND_REVISION_20261120.md`). **Authorisation.** Mr. Xu, 20 November 2026, as quoted in the roadmap file: "If not, can you revise the road map and keep doing some experiments? Spin off some agents to speed up the experiment process." and "Keep experimenting, we are not going to break anything."; the brief of the coordinating session for this line. **Source.** Branch `claude/nano-layered-8mmhfm` at 83c7eb0 (started from 0ba9f241): `layered.py`, `kernl.c`, `fastl.py`, `check_fastl.py` (CHECK PASSED). Written before any scored run of this system on any text; the only runs so far are the check's and three timings of 20,000 to 40,000 lines at the default steps. Development level; a small case; no checker gate, seal or review is claimed.

## 1. The question

In `loop_small_case/hold.py` every holding cell reads 16 other holding cells one byte late, so a strength's own trace is not the whole of its credit, and the rest crosses between cells. The form of the crossing that works (`cross=1 ret=exact`) passes back a value that differs from one cell read to the next (REPORT section 12.4). **Is the crossing needed at all for holding one digit across 0 to 8 letters, if the cells are of two kinds: holding cells that read only lines, and ordinary cells that read them at the same byte?** (Crosswalk `OLD_LISTS_AGAINST_THE_LOOP_20261120.md`, section 5, item 1.)

## 2. The learner, exactly

`layered.py` (its heading states the rule in full). Cells A, 128: each reads the 18 lines of the present byte; gate and candidate logistic; state = gate times state before plus (1 - gate) times candidate; gate thresholds spread at birth as in `hold.py`; one forward trace for each strength, exact because no cell A reads a cell. Cells B, 104, logistic: each reads all 128 cells A at the same byte (my choice: all, not K drawn at birth) and the 18 lines of the present byte. One least-squares answering cell with 18 branches reads the cells B. The mismatch reaches the cells B over return strengths R that they hold; each cell B changes by that sum times b (1 - b) and sends that one number on, the same to every cell A; each cell A weighs the cells B's numbers with return strengths RB that it holds (RB changes by the same product as the forward strength between the same two ends, and both fade at 1e-5 a byte) and changes each strength by the result times that strength's trace. One stream, one byte at a time, nothing passes backward over the text, no value differs from one cell read to the next.

**Numbers that learn** (forward strengths and thresholds): cells A 4,608 from lines and 256 thresholds; cells B 13,312 from cells A, 1,872 from lines, 104 thresholds; answering cell 1,890: **22,042**. `hold.py` with 256 cells on these texts (18 byte values): 9,216 + 8,192 + 512 + 4,626 = 22,546. Return strengths, counted apart: 15,184 here (R 1,872, RB 13,312); 4,608 in `hold.py`, whose crossing reuses the forward strengths.

**Texts and scoring.** `m1` (c = (a + b) mod 10) and `g1` (c = a + b), gap 8, 960,000 lines, whole answer right on the 80 sums shown at gaps of 0, 4 and 8 letters, mean over the three gaps and the last three readings (every 80,000 lines). Text, scoring and runner are copied from `hold.py` at 0ba9f241; the lines drawn for a seed are the same lines. The comparison figures are therefore **quoted** from `loop_small_case/REPORT.md` section 12.2 at 0ba9f241 and not rerun: `m1` with the crossing 91.2, 81.9, 100.0, 78.1, 100.0 (median 91.2), without 27.6, 21.2, 28.2, 40.1, 18.3 (27.6); `g1` with 99.9, 94.2, 99.0, 98.6, 97.4 (98.6), without 39.2, 45.0, 45.7, 44.4, 33.1 (44.4).

## 3. Hypothesis and what would refute it

*Hypothesis.* The crossing is not needed for this: the layered system, with no crossing, reaches at least 80% of sums shown at the median of the five reported seeds on both texts.

*Refuted if* the median is below 50% on `m1` or below 65% on `g1` (nearer to `hold.py` without the crossing than with it). Between those and 80%: not borne out, not refuted; said so.

## 4. Runs

**Development (seed 1), both texts, 960,000 lines, compiled:** steps etaA in {1/4, 1, 4} by etaB in {1/16, 1/4, 1} at nB = 8 (18 runs); then at the best of these nB = 2 and nB = 32 (4 runs). *Rule of choice, fixed now:* the setting with the highest sum of the two texts' scores; one setting for both texts. If no setting reaches 50% on either text I may try one further change of a step or of the fading and will say so; the sizes stay.

**Reported seeds** 73129, 93683, 116089, 406364, 545353, both texts: (a) the chosen setting; (b) the same with etaA = 0, the cells A never changing (a control: what the cells B and the answering cell make of holding cells as they were born).

## 5. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| L1 | `m1`, chosen setting: median of five seeds at least 80% | 0.45 |
| L2 | `g1`, chosen setting: median at least 90% | 0.50 |
| L3 | `m1`: median at least 50% (the hypothesis is not refuted on `m1`) | 0.75 |
| L4 | `m1`: the arm with cells A never changing is at least 20 points below the chosen setting at the median | 0.60 |
| L5 | `m1`, chosen setting, last reading: the gap of 8 letters is within 10 points of the gap of none at the median | 0.60 |

My reason for doubt, stated beforehand: a gate that reads only the present byte cannot tell the first digit from the second, so no cell A can take in the first digit and shut on the second; a cell can only mix the two (gate between 0 and 1 on digits), and the cells B must compute the answer from mixtures. In `hold.py` a cell reads other cells and can tell the two apart.

## 6. Cost, and what is not run

About 2 CPU-minutes a run: development 22 runs, about 45 CPU-minutes; reported 20 runs, about 40 CPU-minutes; on four cores about 25 minutes of wall time. **Not run:** `hold.py` itself (quoted); K cells A drawn at birth for each cell B; other sizes; gaps beyond 8; a comparator by back-propagation through time; the reported seeds for any setting but the chosen one and its control.
