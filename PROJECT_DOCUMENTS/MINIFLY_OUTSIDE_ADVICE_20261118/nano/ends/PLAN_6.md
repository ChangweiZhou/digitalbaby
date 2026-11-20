# A start at the byte after which the answer is no longer unsure (line `nano/ends`, sixth plan, E6) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-ends-8mmhfm` at the commit before this file's. Written before any run of the new arm (one stretch of text with the new starts marked was printed to check the construction; no score). Development level; no checker gate.

## 1. The evidence this follows

`PLAN_3.md` and its addition 1: on the original text U-k (a start where the short cells' answer was unsure before the byte) keeps 60% of the gain of given starts and does as well as the rule "after a space or a line end"; nearly all of the remaining distance is that the given starts are "the first letter after a byte that is not a letter", which never starts a segment on a byte that is itself between words (the second of two line ends). `PLAN_5.md`: on the text without spaces the unsure arms make far more starts than there are words (a segment of 2.6 bytes), half of them wrong: the answer stays unsure over several bytes and each of them is made a start.

## 2. The question and the one change

**If, of a run of bytes before which the answer was unsure, only the last is made a start, are the starts better?**

*The one change against arm U-k.* Byte t starts a segment where the short cells' answer for byte t was unsure beyond theta **and** their answer for byte t + 1, formed once byte t has arrived, is not. Both are known when byte t has arrived and are first used to name byte t + 1; nothing later is read. Otherwise exactly arm U-k: the signal from the cells of runs 0 to k alone (k among 1, 2, 3), the same theta for both conditions, the same three added kinds of contexts, the same reading with trust.

*Arm.* **E**, on both texts, each beside its own arms 0, G and U-k (not run again).

*Settings.* k and theta chosen together as in `PLAN_3.md` (24 readings), then eta and e; on the last twentieth of each stream; then one scoring of each last part.

## 3. Hypotheses, what would refute them, predictions (mine, before any run)

*Hypothesis A (original text).* E is at least 0.002 nats below U-k (at most 1.5067). *Refuted if* not.
*Hypothesis B (text without spaces and line ends).* E gains over arm 0 at least half as much again as U-k does there (at least 0.0296; at most 1.8835). *Refuted if* not.

| | Statement | Chance |
|---|---|---|
| Q1 | hypothesis A holds | 0.50 |
| Q2 | on the original text E keeps at least 0.8 of G's gain (at most 1.5041) | 0.20 |
| Q3 | hypothesis B holds | 0.35 |
| Q4 | on the text without spaces at least 0.6 of E's starts in the last part are word starts (U-k: 0.52) | 0.45 |
| Q5 | on the text without spaces E is below S (1.8883), the best found arm so far | 0.45 |

## 4. Size, cost, what is not run

As arm U-k. About 37 readings a text, 3 to 4 CPU-minutes each. Not run: the same with all the table's cells giving the signal (k = 6); two thresholds in place of one; a second, slower clock.
