# The price of a bound on the store (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The question was put to this line by the coordinating session after its last table (a request, with leave to skip what is ill-posed); the coordinating session says it comes from its reading of a review of brain cells that Mr. Xu gave it (`BRAIN_CELLS_REVIEW_AGAINST_PROGRAMME_20261120.md` on its branch, which I have not read; I attribute nothing in it to Mr. Xu). **Source.** This branch at the commit that holds this file. Written before any run with the option on any stream (it has run only inside the check, on 2,500 bytes against the reference). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

Nothing in the programme makes the store smaller. `REPORT.md` of this line: the large general gains cost no cells, half of all cells have been met once (43 to 65% by stream), and birth only where surprised saves 18 to 32% for up to 0.012 nats. What the cells themselves are worth can now be asked on streams of different kinds.

## 2. The mechanism, exactly (`cap=n`, `evict=0|1`)

The store holds at most n cells. A context that has no cell, when the store is full: eight places of the store are drawn by chance (a fixed generator; a place whose cell belongs to the present byte is passed over); **the cell that has counted least and, among those, counted longest ago leaves**, and the new context takes its place with empty counts (`evict=0`). The control (`evict=1`): the first place drawn leaves. Nothing else changes. Eight is the coordinating session's "a few", as `nano/search`'s store had; not tuned.

**System:** ZA of this line (runs of 0 to 6 bytes, second cell, three recalling senders, last-byte senders, the rule at 0.03, trusts by age; the coordinating session wrote "Z": ZA is Z with the index kept on both measures, within 0.016 of Z going on, and it holds the time a cell last counted, which the rule of leaving needs). n = 50, 25 and 12% of the cells ZA has at the end of the stream with no bound (0.33 to 0.62 million); and 100% on one stream as a check that the option changes nothing while nothing leaves. Going on first; nothing changing beside it. One setting; nothing is chosen, so all six streams are run at once.

## 3. Hypothesis, what refutes it, predictions

**Hypothesis (mine).** A store of half the size, with the cell that has counted least and longest ago leaving, costs at most 0.02 nats a byte going on, on every stream, and at every size it is better than leaving by chance. **Refuted if** at half some stream is more than 0.02 worse, or leaving by chance is within 0.002 of it on three or more streams at half.

| | Statement (going on, second cell, against ZA unbounded) | Whose | Chance |
|---|---|---|---|
| K1 | at half the store no stream is more than 0.010 worse | the coordinating session's | 0.4 (its) |
| K2 | at a quarter, at least four of six within 0.02 | the coordinating session's | 0.3 (its) |
| I1 | my hypothesis holds | mine | 0.45 |
| I2 | at 12% at least one stream is more than 0.10 worse | mine | 0.55 |
| I3 | the loss at a quarter is more than twice the loss at half on at least four streams (the first cells to go are worth least) | mine | 0.65 |
| I4 | the two prose streams lose more at half than c and table do (their runs go on being new; the other streams repeat) | mine | 0.55 |

## 4. Cost, not run

7 CPU-seconds a run, 6 streams x (3 sizes x 2 rules) and one check. Not run: the bound on ZPA; more or fewer places drawn; leaving by age alone.

## 5. Outcome (entered 20 November, after the runs; `res/*__cap*_e*.json`, `report_9.txt` has both measures and the cells)

Check: with the bound at 100% on tiny Shakespeare nothing leaves and every figure is ZA's. Going on, second cell; the difference from ZA with no bound (ZA: 1.3344, 1.3894, 0.9358, 0.6908, 0.3431, 0.2846; 0.32 to 0.61 million cells at the end of the stream):

| store | who leaves | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|---|
| 50% | least counted, oldest | +0.0068 | +0.0070 | +0.0046 | +0.0035 | +0.0026 | +0.0016 |
| 50% | by chance | +0.0619 | +0.0733 | +0.0228 | -0.0142 | +0.0097 | +0.0033 |
| 25% | least counted, oldest | +0.0231 | +0.0248 | +0.0125 | +0.0093 | +0.0063 | +0.0058 |
| 25% | by chance | +0.1306 | +0.1562 | +0.0508 | -0.0106 | +0.0151 | +0.0133 |
| 12% | least counted, oldest | +0.0503 | +0.0570 | +0.0260 | +0.0165 | +0.0177 | +0.0113 |
| 12% | by chance | +0.2112 | +0.2545 | +0.0834 | -0.0106 | +0.0202 | +0.0296 |

With nothing changing the bound with the rule costs 0.001 to 0.012 at half, 0.009 to 0.036 at a quarter and 0.014 to 0.076 at 12%, and on table it gains (-0.005, -0.014, -0.022); leaving by chance costs 0.05 to 0.63.

**Half the store costs under 0.007 nats on every stream; an eighth costs 0.011 to 0.057.** Even at 12% (38,000 to 73,000 cells, after 0.5 to 1.2 million cells have left) ZA stays below xz on all six streams: by 0.36, 0.34, 0.16, 0.02, 0.02 and 0.14.

**Predictions scored.** K1 (the coordinating session's) held: the largest loss at half is 0.0070. K2 (its) held, and for all six at 0.025 or under: four streams are within 0.02 at a quarter, the two prose streams at 0.023 and 0.025. **I1 failed in its second clause**: my hypothesis said the rule is better than chance at every size on every stream; on c leaving by chance is *better than the rule and better than no bound at all* (0.677 to 0.680 against 0.691). The condition of refutation I laid down is not met (no stream over 0.02 at half; chance within 0.002 of the rule or better on two streams, c and xml, not three), but the clause is false as written and I score the prediction failed. **I2 failed**: the worst loss at 12% is 0.057, not over 0.10. I3 held (the loss at a quarter is 2.4 to 3.6 times the loss at half on all six). I4 held (prose 0.007 at half, c and table 0.003).

**My readings.** (1) Most of the store is worth little: on these streams seven cells in eight can go for 0.01 to 0.06 nats when the ones that leave are those that counted least and longest ago. The rule matters greatly on prose (a factor of four to nine in the loss against chance) and is the wrong rule on c. (2) On c, forgetting cells at random helps. The stream is files joined one after another; a cell from an earlier file that has counted much says what is no longer true, and the rule keeps exactly those. That is a fourth sign, with going on, the last byte and age, that this kind of stream wants old counts to weigh less; counts that fade, or leaving by age, are the experiment it points to (not run). (3) Leaving costs less than not being born (`PLAN_10.md`).
