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
