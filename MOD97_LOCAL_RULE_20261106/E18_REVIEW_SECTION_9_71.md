# 9.71 — E18: births with a next-byte target at every bank

E18 grew the two-bank windowless network of E17 from nothing in four ways, on seeds 73129, 93683 and 116089, for 24,576 updates each. The measure is the hard mean (add89 and sub89 at k = 12), set against the end-to-end backprop births on file (91.2 / 94.5 / 90.3).

| Arm | How it learns | Hard mean | Ratio to backprop |
|---|---|---|---|
| B0 | error relayed between banks and through time, weight signs, no backprop anywhere | 87.5 / 90.7 / 84.7 | 0.94–0.96 |
| B1 | no relay between banks: each bank learns only from its own next-byte head, exact backprop inside it | 80.7 / 82.2 / 77.2 | 0.85–0.88 |
| B2 | as B1, but weight signs with credit through time inside each bank | 74.8 / 69.7 / 66.3 | 0.73–0.82 |
| B3 | as B1, but weight signs with a trace on the write synapses only | about 1 on every operation | about 0.01 |

**B1.** Its first bank, on its own head, stays at the one-bank reference's level: add89 27–30 against 22–30. The second bank builds far above the brief's "held at about 55".

**B3** learned nothing: its loss stayed near 1.5 throughout.

**Verdicts.** Five of six predictions hold: P18.1, .2, .3, .4 and .6. P18.5 misses: B2 trails B1 by 5.8, 12.5 and 10.8, so it is within 10 in only one seed.

**The brief's reading lands between its two cases in every seed.**
- B1 is neither within a few points of the reference nor far below it: it is 10.5–13.1 below.
- The relay between banks is worth 6.8–8.5 points at this dose, even under weight signs.
- A next-byte target at each bank, with no error crossing between banks, carries most of the two-bank learner, but not all of it.

**Cautions.**
- B0, B1 and B2 were still rising at the last checkpoint while the reference gained only 0–4 points in its last block, so these are gaps at a fixed dose, not between converged learners.
- The references were made on the owner's machine.

**Run.** 65,118 CPU-s and 4 h 39 min on a four-core 2.10 GHz cloud Xeon, at harness commit 8d037ebb. There were no reruns, and the forecast was 5% low. `e18_check.py` matched `exp45/E18_CHECK.json` exactly.

Development result on sealed panels, evidence level E2. No life, retention or reuse is tested, and no learning-core improvement is claimed. Decisions remain Mr. Zhou's.

Sources: `E18_REPORT.md`, `E18_PREDICTIONS.md`, `exp45/E18_TABLES.md`, the twelve births and their logs.
