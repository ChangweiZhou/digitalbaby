# Cells that step once a word, beside the table's cells (roadmap second revision, track L; a step added from the evidence) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. Keep experimenting. Respect evidence and keep trying new ideas." **Source.** `claude/friendly-darwin-8mmhfm` at 4717b491b; `table.py`, `score.py` and the text as in `PLAN_1.md`. Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

`PLAN_1.md`: cells for the runs of the last one to six bytes are at the level of the test (1.5225 nats counting, 1.5317 taught by the mismatch), and longer runs do not help (eight bytes: 1.5313 counting, 1.5483 by the mismatch): a run of eight bytes has too seldom occurred before. What lies further back than six bytes can therefore not be used by cells of this kind. Mr. Xu, 20 November: "Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc." A cell that changes what it holds once a word, and not once a byte, reaches two or three words back with a context that has occurred before.

## 2. The question

**Do cells that step once a word, added to the table's cells, lower the loss on the last tenth of the text?**

This plan gives the cells the ends of the words (a word is a run of letters; any other byte ends it). It asks only whether such a clock pays on this text. If it does, the next plan asks whether the cells can find the ends themselves (for instance where the table's own surprise rises); if it does not, that question need not be asked here.

*The cells.* At a byte, let *cur* be the letters of the word so far (empty after a byte that is not a letter), *sep* the run of other bytes before it, *prev* the word before that, and *prev2* the word before *prev*. Three kinds of cells are added; as in the table, a cell exists for each value of its context that occurs in the stream:

- **A**: (*sep*, *cur*): the word so far, however long;
- **B**: (*prev*, *sep*, *cur*);
- **C**: (*prev2*, *prev*, *sep*, *cur*).

The answering cell sums the strengths of all active cells (the table's seven and these three) and its 65 branches compete; every active cell changes by its step times the mismatch, the step being eta over the square root of one plus the times the cell has been active (`table.py`, mismatch form, unchanged).

*Arms.* (0) the table's cells alone, K = 6; (1) with A; (2) with A and B; (3) with A, B and C; (4) **the clock without the words**: as (3), but the "words" are every four bytes and the "other byte" every fifth, whatever the bytes are, so that the cells reach as far back and are blind to the words.

*Settings.* For each arm, eta (0.125, 0.25, 0.5, 1), the number of readings (1 to 4), and whether a cell of the added kinds is made at the first or at the second occurrence of its context, are chosen on the last twentieth of the stream with the cells taught on the rest; then the cells are taught on the whole stream and scored once on the last tenth.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* Arm 3 is at least 0.02 nats below arm 0. *Refuted if* the difference is under 0.02.

| | Statement | Chance |
|---|---|---|
| X1 | arm 3 is at least 0.02 nats below arm 0 | 0.60 |
| X2 | arm 3 is at least 0.05 nats below arm 0 | 0.30 |
| X3 | arm 4 (the clock without the words) gains less than half of what arm 3 gains | 0.70 |
| X4 | arm 3 is below 1.47, the level at which nanoGPT's full setting is described | 0.25 |

## 4. Size, cost, what is not run

The number of cells of each kind is reported; each has 65 strengths. A run of one arm with its choice of settings takes a few CPU-minutes. Not run here: ends of words found by the cells; cells taught by the relay beside these (with the loop line); the counting form (the added kinds do not form one chain from shorter to longer, which its discount needs).

## 5. Outcome (entered 20 November after the runs; `res/words_arm*.json`)

Loss on the last tenth, nats a byte; share of next bytes named right; settings as chosen on the last twentieth of the stream.

| Arm | Loss | Named right | Cells added (A; B; C) | eta; readings; cell made at occurrence |
|---|---|---|---|---|
| 0, the table's cells | 1.5317 | 55.20% | none (459,129 cells) | 0.5; 3; first |
| 1, with A | 1.5234 | 55.38% | 58,545 | 0.5; 2; first |
| 2, with A and B | 1.5274 | 55.67% | 58,545; 454,444 | 0.25; 4; first |
| 3, with A, B and C | **1.5237** | 55.67% | 28,111; 93,932; 62,314 | 0.25; 4; second |
| 4, the clock without the words | 1.5547 | 54.90% | 50,458; 118,990; 29,789 | 0.25; 4; second |

**The hypothesis is refuted as written**: arm 3 is 0.008 nats below arm 0, and 0.02 was asked. X1 failed. X2 failed. X3 held (arm 4 gains nothing; it loses 0.023). X4 failed. A run with its choice of settings took 31 to 72 CPU-seconds.

Readings (mine).
1. **Cells that step once a word help little in this form**: 0.008 nats and half a point of bytes named right. Most of that comes from the word so far alone (arm 1).
2. **The same cells on a clock that is blind to the words make the system worse** (0.023 nats). So the words matter (arm 3 is 0.031 below arm 4), and added cells that have seldom been active do harm: the answering cell sums the strengths of all active cells, and a cell that has been active twice speaks as loudly as one that has been active a thousand times.
3. For arms 2 to 4 the choice fell on the edge of what the plan allowed (four readings, the loss still falling by 0.008 a reading). I leave the plan as written and do not extend the readings: the second reading says the form is the limit, and the next plan changes the form.
4. **What follows.** In `PLAN_1.md` cells that count and a fixed discount did better than cells taught by the mismatch (1.5225 against 1.5317), but the discount needs the cells to form one chain from shorter to longer. The next plan keeps the counting inside each cell and lets the answering cell learn, from the mismatch, how far to trust a cell by its kind and by how much it has seen (`PLAN_3.md`).
