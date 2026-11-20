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

## 5. Outcome (entered 20 November after the runs; `res/ends_armE.json`, `res/ends_armE_textns.json`)

Loss on each last part, nats a byte, nothing changing; E scored once on each. Other rows from the earlier plans. The two texts' figures do not stand beside each other.

**Original text.**

| Arm | Loss | Gain over arm 0 | Share of G's gain | Named right | Loss on the last twentieth of the stream | Cells that have counted | Segments in the last tenth | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|---|---|
| 0, the table's cells | 1.5226 | | | 54.41% | 1.5566 | 459,129 | | | |
| G, starts given | 1.4995 | 0.0231 | 1 | 55.20% | 1.5336 | 2,035,419 | 20,724 | 1 | 1 |
| U-k (k = 1, theta 2.75) | 1.5087 | 0.0139 | 0.60 | 55.21% | 1.5412 | 1,933,562 | 22,020 | 0.92 | 0.97 |
| **E (k = 1, theta 3.0)** | **1.5019** | 0.0207 | 0.90 | 55.13% | 1.5365 | 2,143,904 (2,729,160 counts above zero) | 20,201 | 0.99 | 0.965 |

**Text without spaces and line ends.**

| Arm | Loss | Gain over arm 0 | Share of G's gain | Loss on the last twentieth of the stream | Segments in the last part | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|
| 0 | 1.9131 | | | 1.9483 | | | |
| G | 1.7896 | 0.1235 | 1 | 1.8272 | 20,724 | 1 | 1 |
| S | 1.8883 | 0.0248 | 0.20 | 1.9228 | 22,211 | 0.50 | 0.54 |
| U-k (k = 3, theta 2.5) | 1.8934 | 0.0197 | 0.16 | 1.9262 | 35,128 | 0.52 | 0.88 |
| **E (k = 3, theta 2.5)** | **1.8969** | 0.0162 | 0.13 | 1.9294 | 22,508 | 0.66 | 0.72 |

Chosen e = 0.02, eta = 0.00005 on both. On the original text the loss on the last twentieth by theta for k = 1: 2.5: 1.5532; 2.75: 1.5370; 3.0: 1.5365; 3.25: 1.5566 (the whole tables are in the result files). 176 and 149 CPU-seconds.

**Hypothesis A holds**: on the original text E is 0.0068 nats below U-k (0.002 was asked) and keeps 90% of the gain of given starts. **Hypothesis B is refuted**: on the text without spaces E gains 0.0162, less than U-k's 0.0197, where 0.0296 was asked.

Predictions: Q1 held. Q2 held (1.5019, at most 1.5041 asked; I had given it 0.20). Q3 failed. Q4 held (0.66). Q5 failed (1.8969 against 1.8883).

Readings (mine).
1. **On the original text the system now finds its starts from its own signal and keeps nine tenths of the gain of given starts** (1.5019 against 1.4995; the table's cells alone 1.5226). The rule is the system's own: a start at the byte before which the short cells were unsure and after which they are not. 0.99 of its starts are starts of words as the reference defines them, and 0.965 of those are found. Development level; one text; one scoring.
2. **The answer to this line's question is yes on this text, and it is yes because a byte marks the starts.** What the signal reads is that after a space or a line end the cell of the last byte has seen many different followers. Where no such byte exists (the derived text) the same rule keeps 13% and the best of all the signals tried 20%.
3. **On the text without spaces better agreement did not bring a lower loss.** E's starts agree better with the starts of words than U-k's (0.66 of them right against 0.52, with about as many segments as there are words) and its loss is worse (1.8969 against 1.8934). S agrees worst and scores best. So on that text agreement with the words is not what the loss follows, and I do not know what it follows; one candidate, untested, is that the starts found there mark places where the next byte is hard, which is information of its own whatever the words are.
4. **The threshold remains in a narrow place on the original text** (2.75 and 3.0 work; 2.5 and 3.25 give almost nothing). A threshold that the system sets from its own run of the signal is the plain next change there.
5. Still not shown: a second text of another kind; any of this with everything learning on; whether a slower second clock (segments of segments) adds anything.
