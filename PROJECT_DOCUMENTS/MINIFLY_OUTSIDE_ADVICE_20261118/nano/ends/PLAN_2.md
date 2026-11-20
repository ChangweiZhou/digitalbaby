# A start where the answer was unsure before the byte arrived (line `nano/ends`, second plan, E2) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-ends-8mmhfm` at 90dc447. Text, split and measure as before. Written before any run of the new arm; only `ends.py check` was run (the new output against numpy; `check_output.txt`). Development level; one text; no checker gate.

## 1. The evidence this follows

`PLAN_1.md`: starts given gain 0.0231 nats over the table's cells (1.4995 against 1.5226); starts where the table-only system's surprise exceeds a threshold gain 0.0082 (1.5144), 35% of that; starts as frequent by an arbitrary function of two bytes gain 0.0003. The surprise found 0.62 of the word starts and 0.46 of its starts were word starts. My reading there: the surprise depends on which byte arrived, so the same passage is cut differently on different occasions.

## 2. The question and the one change

**Does a signal that does not depend on the byte that arrived find better starts than the surprise does?**

*The one change against arm S.* The signal for byte t is how unsure the table-only system's answer for byte t was before byte t arrived: minus the sum over the 65 branches of p log p. Byte t starts a segment where this exceeds theta. The table-only system is the same as in `PLAN_1.md` (e = 0.02, eta = 0.00005; learning through the stream, each byte named before anything changes; unchanging on the last tenth). The signal is available before byte t arrives and is first used to name byte t + 1. Everything else as arm S: the same three added kinds of contexts, the same reading with trust.

*Arm.* **U**: starts where the answer was unsure. Beside it, from `PLAN_1.md` and not run again: 0, G, R, S, H.

*Settings.* theta among 1.0, 1.25, ..., 3.25 nats (ten values) with e = 0.02, eta = 0.00005; then eta and e from the same ranges as before; all on the last twentieth of the stream; then one scoring of the last tenth.

## 3. Hypothesis, what would refute it, predictions (mine, before the run)

*Hypothesis.* U gains over arm 0 at least half of what G gains, that is at least 0.0116 nats (U at most 1.5110). *Refuted if* less.

| | Statement | Chance |
|---|---|---|
| V1 | U gains at least 0.0116 nats over arm 0 (the hypothesis) | 0.60 |
| V2 | U is below S (1.5144) | 0.75 |
| V3 | on the last tenth at least 0.6 of the starts U finds are word starts | 0.50 |
| V4 | on the last tenth U finds at least 0.8 of the word starts | 0.50 |
| V5 | U gains at least 0.8 of what G gains (at most 1.5041) | 0.25 |

Why (reading, mine): after a space or a line end the table's cells are unsure whatever letter follows, so the first letter of a word should be marked nearly every time; inside a word the answer is mostly sure. What may go wrong: the answer is also unsure after the first one or two letters of a word, so starts may come in pairs and cut words in two; and the level of unsureness falls as the cells count, so early and late cuts may differ.

## 4. Size, cost, what is not run

As arm S: 400 trusts; at most ten cells active at a byte; cells and counts reported. One arm with its choice of settings: 23 readings, about 2 CPU-minutes. Not run: a peak in place of a threshold; the signal of one short cell alone; a slower second clock; learning through the last tenth; a second text.

## 5. Outcome (entered 20 November after the run; `res/ends_armU.json`)

Loss on the last tenth, nats a byte, nothing changing; U scored there once. The other rows are from `PLAN_1.md`.

| Arm | Loss | Gain over arm 0 | Share of G's gain | Named right | Loss on the last twentieth of the stream | Cells that have counted | Segments in the last tenth | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|---|---|
| 0, the table's cells | 1.5226 | | | 54.41% | 1.5566 | 459,129 | | | |
| G, starts given | 1.4995 | 0.0231 | 1 | 55.20% | 1.5336 | 2,035,419 | 20,724 | 1 | 1 |
| S, by surprise (theta 2.5) | 1.5144 | 0.0082 | 0.35 | 54.80% | 1.5487 | 2,091,358 | 27,666 | 0.46 | 0.62 |
| **U, where the answer was unsure (theta 2.5)** | **1.5115** | 0.0111 | 0.48 | 54.97% | 1.5478 | 2,136,050 (2,737,532 counts above zero) | 24,327 | 0.78 | 0.92 |

Chosen: theta 2.5, e 0.02, eta 0.00005. Loss on the last twentieth of the stream by theta: 1.0: 1.5604; 1.25: 1.5580; 1.5: 1.5539; 1.75: 1.5507; 2.0: 1.5487; 2.25: 1.5480; 2.5: 1.5478; 2.75: 1.5515; 3.0: 1.5585; 3.25: 1.5570. On the stream U made 209,931 starts (0.77 of them word starts; 0.86 of the word starts found). 123 CPU-seconds. Check: 8e-15 against numpy for the new output.

**The hypothesis is refuted, narrowly**: U gains 0.0111 nats, 48% of G's gain; 0.0116 was asked.

Predictions: V1 failed (0.0111 against 0.0116). V2 held (1.5115 below 1.5144). V3 held (0.78). V4 held (0.92). V5 failed.

Readings (mine).
1. **The signal taken before the byte arrives finds the starts of words far better than the surprise** (0.78 and 0.92 against 0.46 and 0.62), which agrees with the reading of `PLAN_1.md` that the surprise failed by depending on the byte that came.
2. **But the loss follows only part of the way** (48% of G's gain against 35%). So agreement with the starts of words on the last tenth is not what limits the gain, or not alone.
3. Looked at afterwards, on the stream only (no score): in the first tenth of the stream 0.46 of U's starts are word starts and 0.59 of the word starts are found; in the second half 0.85 and 0.91. While the table's cells have counted little, nearly every answer is unsure, and the cuts made then are not the cuts made later. The cells that counted under the early cuts are keyed to contexts that seldom come again. In the second half the starts that are not word starts lie spread over the later letters of words (none are doubled first letters), and the missed word starts follow a space (3,839), an apostrophe (2,371) or a line end (2,362) out of 94,395.

## 6. Addition 1 (20 November; written after the outcome above and before the runs it describes): is it the early cuts?

A diagnostic, scored **only on the last twentieth of the stream** (cells taught on the rest); the last tenth is not read. Settings fixed at those chosen (theta 2.5, e 0.02, eta 0.00005). Two mixed systems that no one proposes as systems:
* **M1**: starts given (as G) in the first three tenths of the stream, starts by U after that.
* **M2**: starts by U in the first three tenths of the stream, given after that.

Beside them on the same twentieth: arm 0 1.5566, G 1.5336, U 1.5478. In M1 the scored twentieth is cut by U and in M2 as G, so each is read against the arm that cuts the scored part the same way: **M1 against U** changes only the early cuts (the scored part is cut by U in both): the difference is what the early cuts cost. **M2 against G** changes only the early cuts likewise (the scored part cut as G in both). Prediction (mine): M1 is below U by at least 0.004 (chance 0.5); M2 is above G by at least 0.004 (chance 0.5).
