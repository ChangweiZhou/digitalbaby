# 9.45 — E16-F4: the raw-byte gate on three more fresh seeds

E16-F4 made three new births (500011, 600013, 700027) and ran the raw-byte learner (one label-free Algorithm-R reservoir of 65,536 verbatim slots) and its backprop twin on each, with no change to the harness. Only seed 700027 meets every prediction.

- **The births and acquisition hold everywhere.** The births reach 89–98% on every development operation (P16F4.1). The learner reaches 85–92 on the mod-97 operations after block 2 and 81–88 on compositions after block 4 (P16F4.5).
- **The twin misses in 500011 (P16F4.2).** Its held compositions reach only 84.0 after block 6, against a bar of 85, and were still rising. The -+ pair is at 62.5.
- **The gate misses in 600013 (P16F4.3).** The learner stands at 0.81 / 0.69 / 0.90 of the twin. This is the first family below 0.80 in nine seeds of the label-free reservoir.
- **Retention misses in 500011 and 600013 (P16F4.4).** Compositions end at 73.4 and 64.7, after losses of 13.3 and 16.5 points from block 4.
- **The failure in 600013 is in addition and subtraction.** The learner's mod-97 family fell from 85.2 to 52.1 when the compositions arrived, the deepest dip yet. It recovered only to 76.7, with add97 and sub97 near 60. The plus-minus compositions fell to about 50, while the twin held 94.
- **The reservoir is not the difference.** It looks the same as in every other seed: 1.05 MB and about 57,900 distinct strings.

The gate therefore holds in eight of nine seeds. Composition retention, already the weakest margin, fails in two of these three seeds. One seed shows that the sign rule's interference between the composition blocks and the earlier plus-minus operations can be large enough to break the 0.80 gate. Development result on a toy harness; one stream, three seeds. Sources: `E16F4_REPORT.md`, `E16F4_PREDICTIONS.md`, `exp45/SUMMARY_E16F4.md`, the births and the six receipts.
