# nano/skip, fourth plan: the trial clock where it should do nothing, where it was slow, and with its hand-set numbers moved (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at a6946a6; program and kernel unchanged. **Status.** Development level; no checker gate, seal or outside review is claimed. Written before its runs and before the outcome of `PLAN_2.md`, C, is known (those runs are in progress).

**What this follows.** `PLAN_2.md`: on `g2`, development seed, the trial clock closes the four letters within 100,000 lines and the system reaches 95% at every gap; on `m1` it had closed three letters at 200,000 lines (52.9 / 45.0 / 41.7% and rising), and with entries born half closed it closed digits as well. Three things are not known: whether the rule harms a text with nothing to pass over; whether it completes on `m1`; and how much rests on the numbers I set by hand.

**System.** As `PLAN_2.md`: 256 cells, 55,570 numbers that learn, and at the entries 72 (second form: alpha 2, p0 0.2, fading 1/3,000, P 256, floor 0.01) or 18 (first form, alpha 0.05). Nothing is retuned.

**Runs.**
A. `d2` (two-digit sums, no letters), five reported seeds, 400,000 lines: the rule (second form) beside `clock=none`.
B. `m1`, five reported seeds, 1,000,000 lines: the rule in both forms; `clock=none`; the closing by chance (alpha 0); `clock=oracle` (the check).
C. `g2`, development seed 1, 400,000 lines, second form, one hand-set number moved at a time: P 64; P 1,024; p0 0.1; p0 0.35; floor 0.05. Beside the unmoved setting's reading at 400,000 lines in `PLAN_2.md`, B (94).

**Question.** Is the rule safe where there is nothing to find, and does it finish on the text where it was slow?

**Hypothesis H4.** On `m1` the second form reaches at least 90% of the sums shown at every gap within 1,000,000 lines (median of five seeds, last three readings). **Refuted if** the median is under 90% at any gap.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| S1 | H4 holds | 0.50 |
| S2 | `d2`: the rule's median is within 3 points of `clock=none`, and on at least four of five seeds no byte value that occurs is closed at the end | 0.70 |
| S3 | `m1`: on at least one of the five seeds a digit is closed at the end (second form) | 0.30 |
| S4 | C: all five moved settings close exactly the four letters (of the bytes that occur) and have at least 85% at the last reading | 0.60 |
| S5 | `m1`: the first form's median is at least 90% at every gap | 0.35 |

**Cost.** About 35 CPU-minutes. **Not run.** `g1`; more than one number moved at a time; other texts.
