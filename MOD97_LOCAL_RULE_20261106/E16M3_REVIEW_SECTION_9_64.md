**9.64 E16-M3 (cloud session, 10 November 2026): the replay share is a dose, in the opposite direction. More verbatim replay retains the hard pairs, for the rule much more than for backprop.**

`E16M3_REPORT.md`, `exp45/E16M3_TABLES.md`, `exp45/SUMMARY_E16M3.md`.

**What was run.** Fourteen drift lives with the raw reservoir of 65,536 slots: the sign rule at replay shares 0 / 0.25 / 0.75 on seeds 400009, 600013 and 300007, and backprop on the same pipeline at shares 0.25 / 0.75, plus 0.5 for 600013. Comparators are the share-0.5 learners, E16-M2's backprop life for 400009, and the drift twins.

**Results.** One of the four predictions holds.
- **P16M3.1 holds.** Share 0 forgets everything learned before the current mixture: the mod-97 ratios fall to 0.13–0.35, and the hard pairs go from 59–85 at checkpoint 3 to about 1%.
- **The predicted dose is reversed (P16M3.2 misses in both failing seeds).** The rule's hard pairs (`+-`/`-+` mean, final answer) at shares 0.25 / 0.5 / 0.75 are:

  | Seed | 0.25 | 0.5 | 0.75 |
  |---|---:|---:|---:|
  | 400009 | 43.7 | 46.3 | 73.2 |
  | 600013 | 23.3 | 44.8 | 60.2 |
  | 300007 | 59.2 | 67.7 | 75.7 |

  The hard pairs rise with the share instead of falling. They move with mod-97, not against it.
- **Backprop is not flat (P16M3.3 misses at share 0.25).** It rises the same way, by less: 78.2 / 84.7 / 88.1 and 71.9 / 81.8 / 86.7.
- **The passing seed misses too (P16M3.4).** 300007 misses the gate at 0.25 by 0.004, and its hard pairs are higher at 0.75 than at 0.25.

**What the trajectories show.** The share works through retention, not acquisition.
- Lower shares acquire the hard pairs as fast or faster.
- Every share reaches a similar peak at checkpoint 4.
- Only the high shares keep the hard pairs when the stream drifts on.

**What it means.** The reservoir is not over-dosing the rule. It is under-serving retention, which the rule depends on far more than backprop does: over the share range, the rule's hard pairs rise about 30–37 points against backprop's 10–15. At share 0.75, seed 400009 meets the family gate (0.93 / 0.87 / 0.97) and the hard-pair gate (0.82). Seed 600013 stays below on the hard pairs (0.66). The curve has not flattened, so a higher share is the next point to measure.

**Evidence level.** E2; one run per seed and share; development diagnostic. No learning-core claim is made, and the locality question in 9.60 stands.
