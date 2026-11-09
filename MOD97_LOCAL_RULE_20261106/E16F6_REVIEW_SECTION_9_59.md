# 9.59 — E16-F6: the plus-minus tail on twelve drift seeds

E16-F6 ran the drift arm on six more seeds: three new births (800011, 900007, 1000003) and E16-F4's three. The arm is the raw-byte reservoir of 65,536 slots admitting each item online, with a drift twin for each seed.

- **The births pass (P16F6.1).** All three score 89.0–98.3 at k = 12.
- **Every twin passes 90 / 85 / 90 (P16F6.2).** 500011's twin, at 84.0 under stages, reaches 88.4.
- **Five of the six learners stay at or above 0.80 of their twin on all three families (P16F6.3).** 1000003 passes by 0.01 on the mod-97 operations.
- **Seed 600013 fails under drift as it did under stages (P16F6.4).** It ends at 0.71 of the twin on compositions, `+-` 41.7 and `-+` 47.8, with max-min at 89 and the twin at 94.8. Its mod-97 addition and subtraction collapsed to 28 and 25 as the compositions entered. Its reservoir is indistinguishable from the passing seeds'.
- **The pre-registered early signature fails (P16F6.5).** Over twelve drift seeds, the rank correlation between the checkpoint-2 mod-97 mean and the final `+-` pair is 0.22, against the 0.5 required. Three seeds above 80 at checkpoint 2 end with `+-` at 63–64, below 65. 600013 itself was at 79.9 at checkpoint 2 and collapsed one checkpoint later. An exploratory measure chosen after the fact, the lowest mod-97 mean over checkpoints 2–4, ranks the two failing seeds lowest but correlates only 0.40.

**On twelve drift seeds the gate holds in ten.** The two failures, 400009 and 600013, share one mechanism: interference between the plus-minus compositions and mod-97 addition and subtraction under the sign rule, which E16-M2 found the twin's update does not suffer on the same reservoir. Among the passing seeds the final `+-` lies between 63 and 73, too narrow a band for an early measure to rank.

Development result on a toy harness, one stream per seed. E16-S's three seeds ran on another machine, and sign-rule receipts are reproducible only within one machine (E16-RP). Sources: `E16F6_REPORT.md`, `E16F6_PREDICTIONS.md`, `exp45/SUMMARY_E16F6.md`, `exp45/SIGNATURE_E16F6.md`, the twelve receipts, the final weights and the three births.
