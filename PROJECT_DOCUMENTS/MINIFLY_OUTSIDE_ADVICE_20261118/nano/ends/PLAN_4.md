# A start where the answer became more unsure than at the byte before (line `nano/ends`, fourth plan, E4) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-ends-8mmhfm` at the commit before this file's. Written beside `PLAN_3.md` and before any run of either; the two are separate single changes against arm U of `PLAN_2.md` and are run side by side. Development level; one text; no checker gate.

## 1. The evidence this follows

As `PLAN_3.md`, section 1. In addition (`PLAN_2.md`, outcome, point 3): in the second half of the stream the mean unsureness is 2.77 nats at the first letter of a word, 1.38 at the second, 1.36 at the third, 1.05 at a byte that is not a letter. A threshold alone makes starts inside unfamiliar words, where the unsureness stays high for several letters.

## 2. The question and the one change

**Does asking for a rise of the unsureness, beside its level, remove wrong starts?**

*The one change against arm U.* Byte t starts a segment where the table-only system's unsureness for byte t exceeds theta **and** exceeds its unsureness for byte t - 1 by more than a margin. Both numbers are known before byte t arrives. Signal, cells and reading otherwise exactly as arm U (all the table's cells give the signal).

*Arm.* **P**. Beside it, not run again: 0, G, S, U.

*Settings.* The margin among 0, 0.5, 1.0, 1.5 nats and theta among 1.5, 2.0, 2.25, 2.5 chosen together (16 readings; e = 0.02, eta = 0.00005); then eta and e as before; all on the last twentieth of the stream; then one scoring of the last tenth.

## 3. Hypothesis, what would refute it, predictions (mine, before the run)

*Hypothesis.* P gains over arm 0 at least half of what G gains (at most 1.5110). *Refuted if* less.

| | Statement | Chance |
|---|---|---|
| X1 | P gains at least 0.0116 nats over arm 0 (the hypothesis) | 0.50 |
| X2 | P is below U (1.5115) | 0.60 |
| X3 | on the last tenth a larger share of P's starts are word starts than of U's (0.78) | 0.75 |
| X4 | P is below the arm of `PLAN_3.md` | 0.30 |

## 4. Size, cost, what is not run

As arm U. About 29 readings, 3 CPU-minutes. Not run: the two changes of `PLAN_3.md` and this plan together; a peak judged with the byte after (a start announced one byte late).
