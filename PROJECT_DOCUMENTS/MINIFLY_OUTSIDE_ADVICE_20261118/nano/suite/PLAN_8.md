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

## 5. Outcome (entered 20 November, after the runs; `out_8_dev.txt`, `res/*__ZPA_p*.json`, `res/*__ZA_p0.5.json`, `res/sums__*_p0.5_seed*.json`, `report_8.txt`)

**The choice on the two prose streams.** Sum over tiny Shakespeare and the book of ZPA's second-cell losses with nothing changing and going on: without the scale 5.7605; p = 1/2: **5.5809**; p = 1: 5.6247. p = 1/2 was taken and frozen.

**Streams, second cell** (ZA = Z with trusts by age, no more cells than the base; ZPA = with the since and pool cells; "scaled" = the rule's number times the root of (12 / senders that speak)); differences from ZA in brackets:

| system | measure | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|---|
| ZA | going on | 1.3344 | 1.3894 | 0.9358 | 0.6908 | 0.3431 | 0.2846 |
| ZPA | going on | 1.3459 (+0.0115) | 1.3959 (+0.0065) | 0.9582 (+0.0224) | 0.5921 (-0.0987) | 0.2479 (-0.0952) | 0.2777 (-0.0069) |
| **ZPA scaled** | going on | **1.3184** (-0.0160) | **1.3660** (-0.0234) | **0.9210** (-0.0148) | **0.5590** (-0.1319) | 0.2529 (-0.0902) | **0.2483** (-0.0363) |
| ZA scaled | going on | 1.3346 | 1.3896 | 0.9368 | 0.6918 | 0.3446 | 0.2857 |
| ZA | nothing changing | 1.5454 | 1.4591 | 1.4333 | 2.1829 | 1.7527 | 0.8179 |
| ZPA | nothing changing | 1.5471 | 1.4716 | 1.5171 | 2.5265 | 2.3660 | 0.9326 |
| ZPA scaled | nothing changing | 1.4795 | 1.4170 | 1.3868 | 2.1027 | 1.8530 | 0.7356 |
| ZPA scaled, first cell | nothing changing | 1.4890 | 1.4316 | 1.3896 | 2.0636 | 1.6552 | 0.7327 |
| base, first cell | nothing changing | 1.5226 | 1.4676 | 1.4568 | 2.1222 | 1.7567 | 0.8077 |

**Two-digit sums** (40,000 lines, five reported seeds, never-shown sums named whole, median; first cell / second cell): per-text 79.4 / 88.0; the one setting 75.0 / 77.7, **scaled 72.9 / 77.1**; only the step replaced 81.3 / 91.1, **scaled 77.7 / 87.1**.

**On the streams the hypothesis held; on sums my reading is refuted.** (a) With the scale, ZPA is below ZA on both prose streams and py (by 0.016, 0.023, 0.015): the cells that were kept on the base no longer harm beside the rule. (b) On c and xml it is below the unscaled ZPA (by 0.033 and 0.029), on table 0.0050 above it. With nothing changing its second cell is below the unscaled on all six streams (by 0.05 to 0.51) and below the base on five (table: +0.096); its first cell is below the base on all six. So on streams the rule's number was not free of the number of senders, and one local quantity, how many senders speak, repairs that. **On sums the scale does not repair anything and costs 2 to 4 points**: the one setting still stands 6.5 under the per-text setting, and the arm with only the step replaced falls from 81.3 to 77.7. The fall of the one setting with reading is therefore not caused by the number of senders (it is slower with the scale, 74.3 to 72.0, but the whole curve is lower); it stays with the smoothing, as `PLAN_4.md` concluded.

**Predictions scored.** H1 held. H2 held at the bound (table +0.0050, c and xml lower). **H3 failed** (72.9 against 79.4). **H4 failed** (3.6 points under). H5 held (ZA moves by at most 0.0015 going on). H6 held (all six).

**My readings.** (1) With the scale there is, on these six streams, one system that is the best or within 0.005 of the best on every stream going on: ZPA scaled, at 75%, 76%, 82%, 77%, 66% and 57% of xz's cost. (2) The scale is general across the five kinds of stream and not across to sums: on sums 256 senders speak whose drivers largely agree on a small alphabet, and slowing them only delays learning. The claim of `PLAN_4.md` stands as it was scored there (the rule replaces both settings of the step); the scale is an addition for systems with many senders on streams, with that limit. (3) The unit 12 and the power 1/2 are two numbers; the power was chosen on prose between two values, the unit follows from where eta was chosen. A rule in which neither appears would be better; not found here.
