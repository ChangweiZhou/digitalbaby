# 9.43 — E16-F3: the raw-byte reservoir at 65,536 on fresh seeds

E16-F3 ran E16-R3's verbatim arm on the three fresh seeds of E16-F (200003, 300007, 400009), using E16-F's births and twin lives and no change to the harness. The arm is one label-free Algorithm-R reservoir of 65,536 slots holding the presented bytes exactly as they appeared, replayed with no parser, re-rendering or recomputed answer. Two of the three predictions hold in every seed.

- **P16F3.1:** after block 6 the learner stands at 0.91–0.98 of the twin on the mod-97 operations, 0.87–0.88 on the held compositions and 0.96–0.98 on the mod-73 operations.
- **P16F3.3:** at block 5 the reservoir holds 1.048–1.050 MB and 57,715–58,028 distinct byte strings.
- **P16F3.2**, within 5 points of the parsed reservoir of E16-F2 on every family, holds in two seeds. It misses in 400009, where the raw arm keeps 5.5 points *more* of the compositions: 82.5 against 77.0. There the parsed arm lost 11.8 points from block 4 and the raw arm 6.7. The other eight comparisons are within 3 points.

The label-free gate therefore holds without a parser on six seeds, three standing and three fresh. On these seeds, parsing the replayed examples bought nothing that the stored bytes do not give. Composition retention, at 79.5–82.5% against the twin's 91.5–94.0%, is still the family with the least headroom. The cost forecast missed: verbatim replay was not cheaper than the parsed arm on this machine (5,330 against 5,080 CPU-s per life), unlike on the standing machine. Development result on a toy harness; one stream, three seeds. Sources: `E16F3_REPORT.md`, `E16F3_PREDICTIONS.md`, `exp45/SUMMARY_E16F3.md` and the three receipts.
