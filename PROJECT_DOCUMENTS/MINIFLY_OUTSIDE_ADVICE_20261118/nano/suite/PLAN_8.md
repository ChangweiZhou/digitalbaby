# Is the rule's one number free of the number of senders? A test of my reading of two failures (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). This step is mine, under "verify with care". **Source.** This branch at the commit that holds this file. Written before any run with `spow` on any stream or on sums (it has run only inside the check). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

Two results of this line have one shape. `PLAN_6.md`: the since and pool cells, each kept on the base, *raise* the loss of Z on the two prose streams and py once they stand under the rule of `PLAN_4.md` (ZP against Z going on: +0.011, +0.006, +0.018), and the second cell of ZP hardly improves on its first. `PLAN_4.md`: on sums with 256 kinds the one setting is level with the per-text setting to 20,000 lines and then falls. The rule's number, 0.03, was chosen on a system in which about a dozen senders speak at a byte. In ZP 170 to 230 speak, on sums up to 256. Each trust moves by the rule as if it were alone; the branch sums of the answering cell then move by the sum over all speaking senders. My reading, untested: the rule's number is not free of the number of senders.

## 2. The change, exactly (`spow=p` with `rule=2`)

At each byte the rule's number is multiplied by **(12 / n) to the power p**, n = the senders that speak at that byte; nothing else changes. The unit 12 only fixes the scale so that the number 0.03 keeps its meaning on the systems it was chosen on (Z: 17 senders, of which about 12 speak); it could be folded into the number. p = 1 holds the summed movement of the branch sums fixed if the senders' drivers agree; p = 1/2 if they are independent. **p is chosen between 1/2 and 1 on the two prose streams only**, on ZPA (`kinds=runs+last+since+pool recall=3 gate=1 rule=2 eta=0.03 age=1`), by the sum over the two streams of the second cell's losses with nothing changing and going on, as eta was; then frozen. This is local in the programme's sense: an answering cell knows how many of its senders speak.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the chosen p: (a) going on, ZPA is below ZA on both prose streams and on py (the cells that were kept on the base no longer harm beside the rule); (b) on c, table and xml it is not above the unscaled ZPA by more than 0.005. **Refuted if** ZPA with the scale is above ZA on two or more of the three streams of (a) (then the number of senders is not the cause), or (b) fails on two or more streams (then the scale costs what the fast step gave).

| | Statement (going on, second cell, unless said) | Chance |
|---|---|---|
| H1 | (a) holds | 0.50 |
| H2 | (b) holds on all three | 0.45 |
| H3 | sums, the one setting (e 0.02, born 0.25) with the scale: within 2 points of the per-text setting's 79.4 or above (first cell, median of the five reported seeds) | 0.30 |
| H4 | sums, only the step replaced (e 2, born 0.02) with the scale: not more than 2 points under the same arm without it (81.3) | 0.60 |
| H5 | ZA with the scale is within 0.003 of ZA without it on all six streams (about twelve senders speak there, so the scale is near one) | 0.75 |
| H6 | with nothing changing, ZPA's second cell with the scale is below the unscaled on at least four streams (a slower second cell fits the stretch just read less) | 0.60 |

## 4. Runs, cost, what is not run

Development: p = 1/2 and 1 on the two prose streams, ZPA, 2 to 3 CPU-minutes each. Claim: the four other streams once at the chosen p; ZA with the scale on all six (6 CPU-seconds each); sums, two arms on five seeds (50 CPU-seconds each). Not run: other powers; a unit other than 12; the scale with the constant step.
