# nano/loop, fourth plan (roadmap step L2 c): the number of delay lines (20 November 2026)

**Authorisation and source** as in `PLAN_1.md`. Written with `PLAN_2.md`, before any run of this plan, having seen what that plan lists. Development level; no checker gate, seal or review is claimed.

**The one variable:** D, the number of bytes present on lines: 4, 10, 16, 32 (260, 650, 1,040 and 2,080 lines). Everything else is the first plan's arm P at its setting (eta1 = 1, eta2 = 1/4, born at nothing, fading 1e-5), 1,024 cells, 8 readings, kernel `kernt.c`. The births are not changed with D, so the drive of a cell at birth, a sum of D strengths uniform in -1/2..1/2, grows as the root of D (its spread is 0.58, 0.91, 1.15 and 1.63); the settings were chosen at D = 10. Both are part of what is compared and are said here so that they are not found later.

**Numbers that learn:** 1,024 x (65 D + 66) + 65: **333,889; 733,249; 1,132,609; 2,197,569**.

**Question.** Does the system use bytes further back than the tenth, and does it need the ten?

**Hypothesis.** The loss on text never read falls as lines are added. **Refuted if**, on the mean of the seeds, neither D = 16 nor D = 32 is below D = 10 by at least 0.01 nats. (That D = 4 is worse than D = 10 is expected and refutes nothing; if it is not worse by 0.01 nats, the system makes no use of bytes 5 to 10, and that is said.)

**Runs.** Seeds 73129, 93683, 116089; D = 4, 16, 32 (nine runs of about 8 to 10 minutes); D = 10 is the first plan's claim runs (kernel `kernp.c`; the third plan's runs give the same with `kernt.c`).

**Beside every figure:** the unsmoothed tables of the last k bytes, k = 1 to 6; in particular D = 4 beside the table of the last 4 bytes, which sees the same bytes.

| | Prediction (mine, before any run of this plan) | Chance |
|---|---|---|
| D1 | D = 4 is above D = 10 by more than 0.02 nats | 0.60 |
| D2 | D = 16 is not below D = 10 by 0.01 nats or more | 0.65 |
| D3 | D = 32 is above D = 10 (worse) | 0.60 |
| D4 | D = 4 is below the unsmoothed table of the last 4 bytes in loss (1.755 nats, crude) | 0.35 |
| D5 | D = 4 names fewer bytes right than the table of the last 4 bytes (52.5%) | 0.90 |

**Cost.** About 9 x 9 = 81 core-minutes.

**Not run.** Births scaled with D; settings chosen anew for each D; other sizes; lines for bytes further back that are thinned or summed (one line for several delays); more than three seeds.

## Outcome (entered 17:04 UTC; nothing above is changed)

Arm P, born at nothing, 1,024 cells, eta1 = 1, eta2 = 1/4, 8 readings; seeds 73129, 93683, 116089; mean (`tab_output.txt`, section L2 c).

| D | lines | numbers that learn | last tenth: nats; bits; named right | the stream as it is read (reading 8), nats |
|---|---|---|---|---|
| 4 | 260 | 333,889 | 1.7962; 2.5914; 47.73% | 1.6496 |
| 10 | 650 | 733,249 | **1.7687; 2.5517; 48.61%** | 1.5940 |
| 16 | 1,040 | 1,132,609 | 1.7829; 2.5721; 48.18% | 1.5831 |
| 32 | 2,080 | 2,197,569 | 1.8261; 2.6346; 46.89% | 1.5999 |

Beside them: the unsmoothed table of the last 4 bytes, which sees what D = 4 sees: 1.755 nats (crude), 52.5% right; of the last 3 bytes: 1.796, 47.2%. Each seed is within 0.004 nats of its mean.

**The hypothesis is refuted.** Neither D = 16 nor D = 32 is below D = 10; they are above it by 0.014 and 0.057 nats. D = 4 is above D = 10 by 0.028 nats, so the system does take something from bytes 5 to 10, and little.

**My reading.** (1) Lines beyond the tenth byte cost more than they give at this size, these settings and this length: with D = 16 the stream as it is read is fitted better than with D = 10 (1.583 against 1.594) and the text never read worse, which is what taking in particulars of the stream looks like; with D = 32 both are worse. (2) The births were not scaled with D (said in the plan), so at D = 32 a cell's drive at birth is 1.8 times as wide as at D = 10 and the settings chosen at D = 10 may not suit; these runs cannot tell that from a true lack of use for far bytes. (3) With the four bytes that the table of 4 sees, the system is at the level of the unsmoothed table of 3 in loss (1.796 and 1.796) and names 47.7% where the table of 4 names 52.5%: it does not hold what a table of 4 holds, with 333,889 numbers against the table's 48,539 contexts.

**Predictions scored.** D1 held (0.028). D2 held. D3 held. D4 failed (1.796 against 1.755). D5 held. Four of five.

**Cost.** Nine runs, 5 to 10 minutes each. **Not run:** as listed in the plan; and the same at the eta1 that the second plan later found better at 1,024 cells (eta1 = 4), which may move every row.
