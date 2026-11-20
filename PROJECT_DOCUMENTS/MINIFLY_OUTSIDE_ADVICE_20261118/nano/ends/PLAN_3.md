# The signal for a start taken from the short cells alone (line `nano/ends`, third plan, E3) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-ends-8mmhfm` at the commit before this file's. Text, split and measure as before. Written before any run of the new arm; only `ends.py check` was run (`check_output.txt`). Development level; one text; no checker gate.

## 1. The evidence this follows

`PLAN_2.md`: a start where the table-only system's answer was unsure before the byte arrived gains 0.0111 nats (48% of what given starts gain). Its diagnostic: most of the distance to given starts comes from wrong cuts in unread text when the cells are mature (0.76 of the starts are word starts there), a smaller part from the cuts made early, while the cells have counted little. In unread text the long cells have often not met their context, so the whole answer is unsure inside an unfamiliar word as well as before a word.

## 2. The question and the one change

**Is the unsureness of the short cells alone a better signal for a start than that of all the table's cells?**

*The one change against arm U.* The signal for byte t is how unsure the answer of the cells of runs 0 to k alone was before byte t arrived (those cells read with their own learned trust, e = 0.02, eta = 0.00005, as a table-only system with K = k), with k chosen among 1, 2, 3. A cell of a short run has met its context many times within the first thousands of bytes, so its unsureness settles early and is the same in read and unread text. Everything else as arm U: the answering system still has all the table's cells (runs 0 to 6) and the three added kinds.

*Arm.* **U-k**. Beside it, not run again: 0, G, S, U (which is k = 6).

*Settings.* k among 1, 2, 3 and theta among 1.5, 1.75, ..., 3.25 nats chosen together (24 readings, e = 0.02, eta = 0.00005); then eta and e as before; all on the last twentieth of the stream; then one scoring of the last tenth.

## 3. Hypothesis, what would refute it, predictions (mine, before the run)

*Hypothesis.* U-k gains over arm 0 at least half of what G gains (at least 0.0116 nats; at most 1.5110). *Refuted if* less.

| | Statement | Chance |
|---|---|---|
| W1 | U-k gains at least 0.0116 nats over arm 0 (the hypothesis) | 0.75 |
| W2 | U-k is below U (1.5115) | 0.80 |
| W3 | U-k gains at least 0.8 of what G gains (at most 1.5041) | 0.40 |
| W4 | the k chosen is 1 | 0.45 |
| W5 | on the last tenth at least 0.9 of the starts found are word starts, and at least 0.9 of the word starts are found | 0.50 |

A caution, mine, written before the run: if k = 1 is chosen the signal is a property of the last byte alone, and on this text it may come to "a start after a space or a line end". That would be found by the system from its own counts and not given to it, but it would show little about texts that have no such byte; a second text of another kind would be needed for that and is not run here.

## 4. Size, cost, what is not run

As arm U: 400 trusts (the cells that give the signal are the table's own short cells; their separate trust for the signal is 80 to 160 numbers more), at most ten cells active at a byte. About 37 readings, 4 CPU-minutes. Not run: the rise of unsureness (`PLAN_4.md`, written beside this one, a separate change against U); a slower second clock; a second text.

## 5. Outcome (entered 20 November after the run; `res/ends_armU_ks1.json`)

Loss on the last tenth, nats a byte, nothing changing; the new arm scored there once. Other rows from `PLAN_1.md` and `PLAN_2.md`.

| Arm | Loss | Gain over arm 0 | Share of G's gain | Named right | Loss on the last twentieth of the stream | Cells that have counted | Segments in the last tenth | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|---|---|
| 0, the table's cells | 1.5226 | | | 54.41% | 1.5566 | 459,129 | | | |
| G, starts given | 1.4995 | 0.0231 | 1 | 55.20% | 1.5336 | 2,035,419 | 20,724 | 1 | 1 |
| S, by surprise | 1.5144 | 0.0082 | 0.35 | 54.80% | 1.5487 | 2,091,358 | 27,666 | 0.46 | 0.62 |
| U, unsure, all the table's cells | 1.5115 | 0.0111 | 0.48 | 54.97% | 1.5478 | 2,136,050 | 24,327 | 0.78 | 0.92 |
| **U-k, unsure, the cells of runs 0 and 1 alone (k = 1, theta 2.75)** | **1.5087** | 0.0139 | 0.60 | 55.21% | 1.5412 | 1,933,562 (2,541,194 counts above zero) | 22,020 | 0.92 | 0.97 |

Chosen: k = 1, theta 2.75, e 0.02, eta 0.00005. Loss on the last twentieth of the stream, best theta for each k: k = 1: 1.5412 (at 2.75; 1.5416 at 3.0; 1.5674 at 2.5); k = 2: 1.5416 (2.75); k = 3: 1.5415 (2.5); the whole table is in the result file. 189 CPU-seconds. Check against numpy 8e-15 for the short signal.

**The hypothesis holds**: U-k gains 0.0139 nats over the table's cells, 60% of what given starts gain; half was asked.

Predictions: W1 held. W2 held (1.5087 below 1.5115). W3 failed (0.60, and 0.8 was asked). W4 held (k = 1; but k = 2 and 3 are within 0.0004 on the last twentieth). W5 held (0.92 and 0.97).

Readings (mine).
1. **The system can find the starts of its segments from its own counts and keep more than half of the gain of given starts** (60% on the last tenth; 67% on the last twentieth of the stream), at development level on this one text.
2. **What it found, looked at afterwards** (stream and last tenth, no further score): in the last tenth a start follows every space (16,617), every line end (4,474), every "R" (634), every hyphen (211), every "q" (46) and "Z" (37). So the caution written before the run applies: on this text the signal of the shortest cells comes to "after a space or a line end", with a few others. The starts after "R", "q" and "Z" are an artefact of the trust: a cell that has counted little is given little trust, so the answer of the short cells is flat after a rare byte ("q" is nearly always followed by "u", yet the answer after it is the most unsure of all, 3.72 nats).
3. **The threshold sits in a narrow place.** Mean unsureness after a space is 3.15, after a line end 3.03, after "R" 2.79, "O" 2.73, "r" and "o" 2.69; with theta 2.5 the system is worse than the table's cells alone (1.5674 against 1.5566 on the last twentieth). The three values of k do equally well at their best theta.
4. **Why 60% and not more, when 0.92 and 0.97 agree?** Not settled. In the first tenth of the stream the cuts are poor (in the first fiftieth a start after nearly every byte; from a tenth on, 0.91 of starts are word starts). The system also differs from G in ways that are not errors of principle: it starts at the second of two line ends (965 of its 1,850 other starts in the last tenth) and does not start after an apostrophe.
5. Not shown: that this carries to a text without such bytes. `PLAN_5.md` tests that on this text with its spaces and line ends taken out.
