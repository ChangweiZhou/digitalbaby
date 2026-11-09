**9.59 E16-F6 (cloud session, 9 November 2026): under the drifting stream the gate holds on 10 of 12 seeds; 600013 fails again; the checkpoint-2 signature does not predict the tail.**

`E16F6_REPORT.md`, `exp45/SUMMARY_E16F6.md`, `exp45/E16F6_SIGNATURE.md`.

**What was run.** Three new births (800011, 900007, 1000003; 89–98% at k = 12, P16F6.1 holds) and twelve drift lives, a learner and a twin on each of six seeds, as in E16-S. The raw reservoir held 65,536 slots with online admission.

**Results.**
- **Twins (P16F6.2): hold.** Every twin is at 90 / 85 / 90 or above. Seed 500011's twin reaches 88.4 on compositions, where its twin under stages had stopped at 84.0.
- **Learners (P16F6.3): hold, five of six.** Five learners stay at or above 0.80 of their twin on all three families, with composition ratios of 0.83–0.89.
- **Seed 600013 (P16F6.4): holds.** It misses again (compositions 0.705, mod-97 operations 0.774) with the same signature as under stages: `+-` and `-+` near 42–48, max/min pairs at 89, add and sub mod 97 near 50, the twin untouched at 94. That makes its fragility a property of the seed, not of the schedule.
- **The early signature (P16F6.5): misses.** Over twelve seeds, the Spearman correlation between the checkpoint-2 mod-97 mean and the final `+-` is 0.217 (0.450 on the nine re-keyed-panel seeds alone), against 0.5. Four seeds above 80 at checkpoint 2 end at or below 65 on `+-`. The reason is that 600013 dips at checkpoint 3 (47.2), not at checkpoint 2 (79.9).

**An exploratory reading, not a prediction.** The minimum mod-97 mean over the life separates the two failing seeds (47.2, 62.0) from the ten passing ones (66.7 and above). That is fitted after the fact, so it would need a locked prediction on new seeds before it could be used. It also suggests that the E16-M2 question (rule or data) is about when compositions displace addition and subtraction, not when that happens on a fixed clock.

**Pace.** 4.7 hours of wall time against a forecast of 5.3. The births, learners and twins each came within 3% of their forecast CPU cost.
