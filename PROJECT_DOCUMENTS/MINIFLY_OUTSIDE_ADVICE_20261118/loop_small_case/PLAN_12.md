# Twelfth plan: two-digit numbers with letters between them, longer and with more cells (roadmap second revision, M1) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "2) If not, can you revise the road map and keep doing some experiments?" and "Note you can scale the network or do whatever you like." **Source.** This branch at 62e7d5ee; `hold.py` and its kernel unchanged. Written before either run. Development seed only; a small case; no checker gate.

**Where the eleventh plan stopped.** Lines `aa+<0 to 8 letters>bb=ccc`, two delay lines, one crossing with the reader's own strengths, 256 cells, 2,400,000 lines: 7.7% of never-shown sums (by place 97, 76, 16), rising slowly. The section 12.3 of the report named three things to try one at a time; this plan takes the first two, each alone.

**Runs** (development seed 1, eta1 = 1, `chain=2 cross=1 ret=exact`, text `g2`, gap 8): (a) the same system for 9,600,000 lines, four times as long; (b) 512 cells for 2,400,000 lines.

**Hypotheses, each with what would refute it.** (a) More lines are what is missing: at least 30% of never-shown sums at the gap of 8 letters by the end; refuted if 15% or less. (b) More cells are what is missing: the same thresholds.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| U1 | (a) reaches at least 30% at the gap of 8 letters | 0.30 |
| U2 | (b) reaches at least 30% at the gap of 8 letters | 0.20 |
| U3 | in (a) the units digit is right on at least 50% of never-shown sums at the end (it was 16) | 0.30 |

**Cost.** (a) about 65 CPU-minutes, (b) about 35. Not run: a signal passed over two connections; the reported seeds.
