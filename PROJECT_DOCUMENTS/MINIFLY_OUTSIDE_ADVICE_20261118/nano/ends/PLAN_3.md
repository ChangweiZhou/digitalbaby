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
