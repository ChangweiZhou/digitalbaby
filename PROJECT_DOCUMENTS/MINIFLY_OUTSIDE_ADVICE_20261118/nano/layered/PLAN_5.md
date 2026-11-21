# Line `nano/layered`, fifth plan: the first plan's system on ten further seeds, the sum mod 10 only (20 November 2026)

(Numbered fifth because the fourth number is kept for the brief's E4; this plan may be run before it.) **Authorisation and standing** as in `PLAN_1.md`; asked for by the coordinating session after the first plan's report ("the same E1 systems on the further seeds 1001 to 1010 for the text mod 10 only, since five seeds split three ways"). **Source.** This branch at the commit that adds this file; programs unchanged since 83c7eb0. Written before any run on these seeds. Development level; no checker gate, seal or review is claimed.

**Why.** On `m1` the five reported seeds gave 51.4, 98.3, 9.3, 39.7, 99.6 for sums shown, and for the 20 sums never shown 31.7, 85.0, 3.3, 18.3, 76.7; the development seed 99.9 and 83.3. Five seeds do not say how often the text is learned, nor how firmly never-shown sums follow.

**Runs.** `layered.py`, text `m1`, gap 8, 960,000 lines, 128 + 104 cells (22,042 numbers that learn), the first plan's chosen setting unchanged (etaA = 1/4, etaB = 1, nB = 32), seeds 1001 to 1010. Nothing is chosen on these seeds. Measures, both registered here: sums shown (mean over gaps 0, 4, 8 and the last three readings); the 20 sums never shown (scored with no letters; mean of the last three readings and the count of 60). A seed "learned the text" if sums shown are at least 90%.

*Hypothesis.* Where the layered system learns the text it also answers the sums never shown: on every seed with sums shown at least 90%, never-shown sums are at least 50%. *Refuted if* on two or more such seeds never-shown sums are below 30%.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| S1 | at least 4 of the 10 seeds learn the text (shown at least 90%) | 0.50 |
| S2 | median of sums shown over the 10 seeds at least 50% | 0.45 |
| S3 | at least one seed stays at or below 15% of sums shown | 0.65 |
| S4 | on every seed that learned the text, never-shown sums at least 50% | 0.65 |
| S5 | over all 10 seeds, never-shown sums are never more than 10 points above nor more than 40 below sums shown | 0.70 |

**Cost.** 10 runs of about 2.3 CPU-minutes; about 7 minutes of wall time. **Not run:** `hold.py` on these seeds (its five reported seeds are in the first plan's addition); `g1`; the control with cells A never changing; never-shown sums at gaps of 4 and 8 letters (the copied runner scores them with no letters only).

## Outcome (entered 20 November after the runs; nothing above was changed)

`res/lay_r5_m1_*.json`; about 2.3 CPU-minutes a run. Sums shown: %, mean over gaps 0, 4, 8 and the last three readings. Never shown: %, mean of the last three readings (count of 60), no letters.

| seed | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1010 | median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sums shown | 94.2 | 8.9 | 88.9 | 11.4 | 6.0 | 87.4 | 22.2 | 94.3 | 86.5 | 76.7 | **81.6** |
| never shown | 58.3 (35) | 3.3 (2) | 61.7 (37) | 11.7 (7) | 0.0 (0) | 48.3 (29) | 8.3 (5) | 41.7 (25) | 31.7 (19) | 53.3 (32) | **36.7** |
| shown at the last reading, gaps 0/4/8 | 99/99/95 | 18/14/0 | 94/95/92 | 25/18/0 | 12/4/4 | 88/88/88 | 44/29/1 | 100/99/98 | 91/92/86 | 96/98/95 | |

Six seeds of ten end at 88% or more of sums shown at the last reading (two of them above 90 in the registered mean of three readings, three more at 86.5 to 88.9, one at 76.7 and rising: 96 at the last reading); four stay at 6 to 22%. With the first plan's five seeds and the development seed: of 16 births, 9 end above 85% at the last reading, 1 at about 50, 6 at or below 40. The seeds that learn do so after a level stretch of 80,000 to 640,000 lines; whether the others would follow later was not run.

**The hypothesis is not borne out as registered and not refuted.** Only two seeds meet the registered mark of 90%; on them never-shown sums are 58.3 and 41.7, one below the 50 I had said, neither below 30. Over the six seeds that end above 85% at the last reading, never-shown sums are 32 to 62% (median 51); over the four that did not learn, 0 to 12%. So the first plan's 77 and 85% on two seeds were the top of the range, not its middle: where the layered system learns this text it answers about half of the sums never shown, with no letters, against 0 to 12% for `hold.py` with the crossing on its five seeds (first plan's addition). The difference between the designs stands; its size is smaller than the first five seeds suggested. 20 pairs a seed; one gap.

| | Prediction | Outcome |
|---|---|---|
| S1 | at least 4 of 10 seeds with shown at least 90% (0.50) | **failed** as registered (2: 94.2, 94.3; three more at 86.5 to 88.9) |
| S2 | median of sums shown at least 50% (0.45) | held (81.6) |
| S3 | at least one seed at or below 15% (0.65) | held (three: 8.9, 11.4, 6.0) |
| S4 | on every seed that learned the text, never-shown at least 50% (0.65) | **failed** (seed 1008: 41.7) |
| S5 | never-shown never more than 10 above nor more than 40 below shown (0.70) | **failed** (1008: 52.6 below; 1009: 54.8 below; 1003, 1006 within) |

**Not run:** `hold.py` on these seeds; `g1`; never-shown sums at gaps of 4 and 8 letters; more lines for the seeds that did not learn; a look inside the cells.
