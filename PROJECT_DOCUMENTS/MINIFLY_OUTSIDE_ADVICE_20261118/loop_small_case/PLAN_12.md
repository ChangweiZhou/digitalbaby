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

**Outcome** (entered 20 November when both runs were in; `res/hold_d12_long.json`, `res/hold_d12_h512.json`; development seed 1).

| Run | Never-shown sums, mean of the last three readings | At the last reading, by gap 0 / 4 / 8 | By place (hundreds, tens, units) | CPU |
|---|---|---|---|---|
| (a) 256 cells, 9,600,000 lines | 35.6% | 44 / 47 / 46 | 99, 80, 57 | 5,826 s |
| (b) 512 cells, 2,400,000 lines | 9.9% | 9 / 11 / 10 | 97, 70, 16 | 2,616 s |

Run (a) over its life, never-shown sums at each 800,000 lines: 7, 6, 9, 10, 9, 8, 10, 14, 22, 23, 39, 46. It stays where the eleventh plan left it until about 5 million lines and then rises, and is still rising steeply at the end.

Hypothesis (a) holds (46% at the gap of 8 letters at the end; 30% asked). Hypothesis (b) is refuted (10%). U1 held. U2 failed. U3 held (the units digit: 57%).

Readings (mine). (1) On this text the cells that hold, taught through one crossing, were not unable; they were slow: a long level stretch and then a rise that had not ended at 9.6 million lines. More lines are what was missing; more cells are not. One seed. (2) Beside it, from the line `nano/skip` on the same text and seed, as that line reports: a delay chain with a learned clock answers 95% of never-shown sums after 1.2 million lines and 82 to 88% after 200,000. So this route is some twenty to fifty times slower than that one, and the earlier reading that it "does not learn" this text is withdrawn in favour of "learns it very slowly". (3) Not run: the reported seeds (about 100 CPU-minutes each); where the rise ends.
