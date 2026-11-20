# The same text without its spaces and line ends: can the starts still be found? (line `nano/ends`, fifth plan, E5) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-ends-8mmhfm` at 99fa2c7. Written before any run on the derived text (only its length and a stretch of it were printed, to check the construction). Development level; no checker gate.

## 1. The evidence this follows

`PLAN_3.md`: the unsureness of the shortest cells finds starts that keep 60% of the gain of given starts, but what it found is "a start after a space or a line end". A caution written there before the run: that shows little about a text in which no byte marks the starts.

## 2. The question and the one change

**When no byte marks where a word starts, do the same signals still find starts that keep at least half of the gain of given starts?**

*The one change.* The text. Every space and line end is taken out of the tiny Shakespeare text (1,115,394 bytes become 905,502); the cut between the stream and the last tenth is carried over (the stream is 815,054 bytes, the last part 90,448: the same passages as before). Punctuation and capital letters remain. Nothing else changes: the programs, the arms and the choice of settings are those of `PLAN_1.md` to `PLAN_3.md`, run afresh on this text (the last twentieth of the new stream for the settings; the last part scored once for each arm).

**This is a derived text. Its figures stand beside one another only, not beside any figure on the original text or any reference.**

*Arms.* **0**: the table's cells. **G**: starts given, carried over from the original text (the first letter after a byte that was not a letter there: 208,503 starts, 20,724 in the last part). **R**: every fifth byte. **S**: surprise beyond theta. **U**: the answer of all the table's cells unsure beyond theta. **U-k**: the same from the cells of runs 0 to k alone, k among 1, 2, 3.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The best of S, U and U-k, **named beforehand as the one that is best on the last twentieth of the stream**, gains over arm 0 at least half of what G gains. *Refuted if* less. (If G gains less than 0.01 nats the comparison is void.)

| | Statement | Chance |
|---|---|---|
| T1 | G is at least 0.02 nats below arm 0 on this text | 0.80 |
| T2 | the hypothesis holds | 0.30 |
| T3 | the arm that is best on the last twentieth is U (all the table's cells) | 0.50 |
| T4 | U-k gains less than a quarter of what G gains | 0.55 |
| T5 | on the last part at least half of U's starts are word starts | 0.50 |
| T6 | every one of S, U, U-k gains more than R | 0.75 |

Why (reading, mine): without the space the last byte alone says little about whether a word has ended, so the shortest cells should lose what made them good; the longer cells know the ends of common words ("the", "and") and should be unsure after them, but less sharply than after a space.

## 4. Size, cost, what is not run

Sizes as before (280 trusts for arm 0, 400 for the others; cells reported). Six arms, about 12 CPU-minutes in all. Not run: a text of another language or kind; the structure of contexts changed; anything learning through the last part.
