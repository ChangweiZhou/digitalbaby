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
