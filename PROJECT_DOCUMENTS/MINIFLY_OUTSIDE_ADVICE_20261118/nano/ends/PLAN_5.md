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

## 5. Outcome (entered 20 November after the runs; `res/ends_arm*_textns.json`)

The derived text (spaces and line ends taken out). Loss on its last part, nats a byte, nothing changing; each arm scored there once. **These figures stand beside one another only.**

| Arm | Loss | Gain over arm 0 | Share of G's gain | Named right | Loss on the last twentieth of the stream | Cells that have counted | Segments in the last part | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|---|---|
| 0, the table's cells | 1.9131 | | | 46.39% | 1.9483 | 750,287 | | | |
| G, starts given | **1.7896** | 0.1235 | 1 | 48.74% | 1.8272 | 1,985,961 | 20,724 | 1 | 1 |
| R, every fifth byte | 1.9153 | -0.0022 | | 46.43% | 1.9503 | 2,399,931 | 18,090 | 0.23 | 0.20 |
| S, surprise (theta 3.0) | **1.8883** | 0.0248 | 0.20 | 46.69% | 1.9228 | 2,246,374 | 22,211 | 0.50 | 0.54 |
| U, unsure, all the table's cells (theta 2.5) | 1.8937 | 0.0194 | 0.16 | 46.86% | 1.9291 | 1,693,210 | 34,795 | 0.52 | 0.88 |
| U-k, unsure, short cells (k = 3, theta 2.5) | 1.8934 | 0.0197 | 0.16 | 46.92% | 1.9262 | 1,564,688 | 35,128 | 0.52 | 0.88 |

Chosen in every arm e = 0.02; eta 0.00005 except G (0.0001). 32 to 149 CPU-seconds an arm.

**The hypothesis is refuted.** The arm that was best on the last twentieth of the stream is S (1.9228); it gains 0.0248 nats, 20% of what G gains. No found arm reaches a quarter.

Predictions: T1 held (0.1235). T2 failed. T3 failed (S was best there, not U). T4 held (16%). T5 held (0.52). T6 held.

Readings (mine).
1. **Without a byte that marks them, the starts found by these signals keep a sixth to a fifth of the gain of given starts.** The 60% of `PLAN_3.md` belongs to texts in which a byte marks the start; it is not a general ability to find segments.
2. **The found starts still carry something of their own**: 0.019 to 0.025 nats where the blind clock loses 0.002, which is more in nats than on the original text (0.008 to 0.014).
3. **G is a different kind of reference here.** On this text the given starts tell the system something the bytes no longer contain (where the spaces were), and that alone helps to name the next byte: G gains 0.12 nats, five times its gain on the original text. Half of that is a harder mark than half of 0.023. I keep the plan's verdict (refuted) and note that the two texts' shares are not on one scale.
4. The signals behave differently from the original text: the unsure arms make far more starts than there are words (a segment of 2.6 bytes; 0.88 of the word starts found but only half of the starts right), while the surprise makes about as many starts as there are words and gets half of them. The loss by theta is in the result files.
