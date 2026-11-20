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
