# One rule for the trust's step, on every stream and on two-digit sums as one stream among others (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The step was set by the coordinating session. **Source.** This branch at 154b89e. Written before any run with the rule on any stream (the kernel's option `rule=2` has been run only inside `suite.py check`, 2,500 bytes of the book against the reference). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

The programme has carried two settings of the trust's step since its first lines: on prose 0.00005, constant; on sums 0.1 divided by the root of (1 + the times the trust was used). They differ 2,000-fold at first use. With them go two smoothings (e = 0.02 and 2) and two trusts at birth (0.25 and 0.02). `PLAN_3.md`, outcome: with the prose step a recalling sender's trusts stay small and are smallest at the levels used least (prediction C8 failed), and on the two streams where the system is still above xz what it lacks is, on my reading, speed.

## 2. The rule, exactly (`rule=2` in the kernel)

Each trust (and each part of the second cell) keeps the sum A of the squares of its own driver g (g = the message at the byte that came less the expected message, the quantity its change is proportional to under any of the rules). Its change at a use is **eta x g / sqrt(A)**, A including the present g. This is the per-coordinate rule known as AdaGrad, with nothing added. It is local: a trust needs only its own past drivers. Its first change is of size eta whatever the scale of the messages; a trust used often and pulled both ways slows as the root of its uses; a trust used rarely still moves by a visible amount.

It has one number, eta. **eta is chosen on the two prose streams only** (tiny Shakespeare and the book) from the grid 0.001, 0.003, 0.01, 0.03, 0.1, by the sum over the two streams of the losses with nothing changing and going on, second cell; then frozen. Everything else stays as it is: e = 0.02, trusts of the runs born at 0.25, every other trust and part born at nothing.

**The system it is tried on:** the best general system so far that costs no cells: runs of 0 to 6 bytes, the second answering cell with parts by kind and byte, one recalling sender (`kinds=runs gate=1 recall=1`); 0.3 to 0.56 million cells, 320 trusts, at most 6,144 parts. Compared with the same system at the constant step 0.00005 (`res/*__recall1.json`, already there).

**Two-digit sums as one stream among others** (`suite.py sums`): the stream, split and scoring of `nano/merge/merge.py` and `nano/table/gate.py` (lines "ab+cd=efg" drawn from 8,000 shown pairs; 500 pairs never shown, answered digit by digit from the system's own digits, each after a read line, five draws of the line before, with nothing changing; 40,000 lines; kinds: every set of at most four of the last nine delays, 256 kinds, as the programme's sums system has; no recalling sender). Check, run before this plan: with the per-text setting (e = 2, step 0.1 shrinking, born 0.02) on the development seed 1 the first cell has 79.4 and 79.3 (last reading; mean of the last three), `gate.py`'s stored 79.44 and 79.32, with the same 1,213,119 cells. Arms on sums, five reported seeds (73129, 93683, 116089, 406364, 545353), each once:

| arm | e | born | step |
|---|---|---|---|
| per-text (the programme's) | 2 | 0.02 | 0.1 / sqrt(1 + uses) |
| the one setting | 0.02 | 0.25 | the rule, eta as chosen on prose |
| only the step replaced | 2 | 0.02 | the rule, the same eta |
| the prose setting as it was | 0.02 | 0.25 | 0.00005 constant |

## 3. Question, hypothesis, what refutes it

**Question.** Can one rule with one number replace both settings of the step without loss anywhere?

**Hypothesis.** With eta chosen on prose: (a) going on, the rule lowers the loss on at least three of the five kinds of stream and harms no stream by more than 0.005 (the rule of `PLAN_2.md`); (b) on two-digit sums the one setting is not more than 2 points under the per-text setting at the median of five seeds, first cell (mean of the last three readings). **Refuted if** (a) fails, or the one setting is more than 2 points under on sums. If only the arm "only the step replaced" holds on sums, then the step's rule is general and the smoothing is the per-text number that remains; I will say that and not call the hypothesis held.

| | Statement | Chance |
|---|---|---|
| D1 | the chosen eta lowers the going-on loss on both prose streams by at least 0.005 against the constant step | 0.50 |
| D2 | going on, the rule gains at least 0.03 on c and on table | 0.55 |
| D3 | (a) holds | 0.60 |
| D4 | with nothing changing the rule harms at least one stream by more than 0.005 | 0.50 |
| D5 | sums, the one setting: within 2 points of the per-text setting or above (so (b) holds) | 0.30 |
| D6 | sums, only the step replaced: within 2 points of the per-text setting or above | 0.50 |
| D7 | sums, the prose setting as it was: more than 10 points under the per-text setting | 0.70 |
| D8 | under the rule the recalling sender's trust rises with the length of agreement (as C8 of `PLAN_3.md`) on at least five streams | 0.55 |
| D9 | with the rule, going on, the system is below xz on c or on table | 0.30 |

## 4. Runs, cost, what is not run

Development: 5 values of eta on 2 streams, 5 CPU-seconds each. Claim: the four other streams once at the chosen eta; sums 4 arms x 5 seeds, about 50 CPU-seconds each (1.2 million cells). Not run: other rules for the step; a rule for e; eta chosen on sums or on any stream but the two prose ones; sums with classes; 400,000 lines.

## 5. Outcome (entered 20 November, after the runs; `out_4_dev.txt`, `res/*__rule2_eta*.json`, `res/sums__*.json`, `report_4.txt`, `report_4_sums.txt`)

**The choice on the two prose streams.** Sum over tiny Shakespeare and the book of the second cell's losses with nothing changing and going on: constant step 5.7107; the rule at eta 0.001, 0.003, 0.01, 0.03, 0.1: 5.7549, 5.7222, 5.6946, **5.6838**, 5.6987. eta = 0.03 was taken and frozen. From 0.01 to 0.1 the sum moves by 0.015: the rule is not sensitive to its number on these streams.

**A fault found on a reported seed of sums and repaired.** The first run of the arm "the one setting" (seed 73129) fell to 0.0% after 20,000 lines with infinite loss. Cause, found on the development seed: a driver so small that its square is zero in the machine's numbers was divided by the root of that zero and moved a trust to minus infinity. Both the kernel and the reference had it. Now a trust does not move while its sum of squares is zero; the check passes; the arm was run again on all five seeds and the faulty outputs are not kept as results. The runs on streams had no infinite value and are unaffected (the system Z of `PLAN_6.md` was run after the repair).

**Streams** (the system of section 2 at the constant step and under the rule; the difference in brackets; cells and numbers stored unchanged):

| | measure | shakespeare | book | py | c | table | xml | kinds lowered | largest harm | by the rule of PLAN_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| the rule | going on | 1.3705 (-0.0078) | 1.4006 (-0.0078) | 1.0210 (-0.0141) | 0.7952 (-0.0495) | 0.5253 (-0.0448) | 0.3547 (-0.0160) | 5 of 5 | -0.0078 | KEPT |
| the rule | nothing changing | 1.4882 (-0.0046) | 1.4245 (-0.0068) | 1.4283 (-0.0059) | 2.2840 (+0.0541) | 2.0213 (+0.1716) | 0.7908 (+0.0135) | 2 of 5 | +0.1716 | not kept |
| the rule, first cell | going on | 1.4035 (-0.0020) | 1.4361 (-0.0031) | 1.0531 (-0.0145) | 0.9321 (-0.0195) | 0.6339 (-0.0258) | 0.4030 (-0.0119) | 5 of 5 | -0.0020 | KEPT |
| the rule, first cell | nothing changing | 1.5208 (+0.0029) | 1.4593 (-0.0021) | 1.4424 (+0.0000) | 2.1577 (+0.0316) | 1.7627 (+0.0569) | 0.7831 (+0.0015) | 0 of 5 | +0.0569 | not kept |

**Two-digit sums, 40,000 lines, five seeds; never-shown sums named whole, %, after a read line** (256 kinds, 1.21 million cells, about 50 CPU-seconds a run; the second cell here has six parts by kind and byte):

| arm | cell | each seed (mean of the last three readings) | median | last reading, median | nats a byte while reading the last 5,000 lines (median) | cells | CPU s |
|---|---|---|---|---|---|---|---|
| per-text (e 2, born 0.02, step 0.1/sqrt(1+uses)) | first | 82.6, 80.0, 79.4, 78.5, 78.5 | **79.4** | 80.1 | 0.9353 | 1,213,003 | 50 |
| per-text (e 2, born 0.02, step 0.1/sqrt(1+uses)) | second | 89.4, 87.8, 88.0, 87.8, 88.7 | **88.0** | 88.2 | 0.9230 | 1,213,003 | 50 |
| the one setting (e 0.02, born 0.25, the rule at eta 0.03) | first | 77.5, 76.0, 75.0, 71.7, 74.0 | **75.0** | 72.9 | 0.9263 | 1,213,003 | 49 |
| the one setting (e 0.02, born 0.25, the rule at eta 0.03) | second | 78.6, 79.2, 76.4, 74.7, 77.7 | **77.7** | 74.6 | 0.9218 | 1,213,003 | 49 |
| only the step replaced (e 2, born 0.02, the rule at eta 0.03) | first | 83.9, 81.3, 81.9, 80.0, 81.3 | **81.3** | 82.0 | 0.9304 | 1,213,003 | 50 |
| only the step replaced (e 2, born 0.02, the rule at eta 0.03) | second | 91.7, 91.1, 89.7, 89.7, 91.3 | **91.1** | 91.4 | 0.9236 | 1,213,003 | 50 |
| the prose setting as it was (e 0.02, born 0.25, step 0.00005) | first | 65.8, 63.9, 62.0, 61.5, 62.4 | **62.4** | 65.3 | 1.2298 | 1,213,003 | 49 |
| the prose setting as it was (e 0.02, born 0.25, step 0.00005) | second | 70.3, 69.7, 66.7, 67.9, 68.7 | **68.7** | 67.9 | 0.9678 | 1,213,003 | 49 |

By lines read (first cell, median of five seeds; 5,000, 10,000, 20,000, 30,000, 35,000, 40,000): per-text 70.0, 78.4, 80.2, 78.8, 79.5, 80.1; the one setting 72.9, 79.0, 81.2, 77.1, 74.6, 72.9; only the step replaced 71.9, 80.0, 81.6, 81.1, 81.5, 82.0; the prose setting 47.3, 58.4, 64.3, 60.8, 62.2, 65.3.

**The hypothesis is refuted in its second part and holds in its first.** (a) Going on, the rule lowers all five kinds of stream (0.008 to 0.050) and harms none. (b) On sums the one setting is 4.4 points under the per-text setting at the median (75.0 against 79.4; second cell 77.7 against 88.0), outside the 2 points. As I laid down beforehand for this case: **the rule for the step is general, and the smoothing with the trust at birth is the per-text number that remains.** With e = 2 and born 0.02 kept, the rule alone is above the programme's shrinking step on every seed (81.3 against 79.4 at the median; second cell 91.1 against 88.0), so one rule with one number does replace both settings of the *step*: it beats the constant 0.00005 on all six streams going on and the shrinking 0.1 on sums. The one setting is level with the per-text setting up to 20,000 lines and then falls (81.2 to 72.9), the fall with reading that `nano/sparse` described for cells with heavy counts; the per-text smoothing does not fall.

**Predictions scored.** D1 held (0.0078 on both prose streams). D2 held (0.050 and 0.045). D3 held. D4 held: with nothing changing the rule harms c by 0.054, table by 0.172 and xml by 0.014 (a faster trust fits the stretch just read). **D5 failed**, as I thought more likely than not. D6 held. D7 held (17 points under). D8 held: under the rule the recalling sender's trust is higher over lengths of 32 and more than over lengths 8 to 15 on five streams (all but table), where under the constant step it was on two. **D9 failed**: 0.795 against xz 0.724 on c, 0.525 against 0.381 on table.

**My readings.** (1) Of the two per-text numbers the programme has carried longest, the step can go: one local rule with one number, chosen on prose, is better than either setting on its own ground going on. (2) The smoothing cannot yet: e = 0.02 with born 0.25 on sums costs 4 to 13 points under the rule (against either arm with e = 2) and 17 under the constant step. A rule for e (or for the trust at birth) that needs no number for a text is the next thing of this kind to look for; not run here. (3) With nothing changing the faster step harms on the joined streams, like every means of hearing the present (`PLAN_7.md`).
