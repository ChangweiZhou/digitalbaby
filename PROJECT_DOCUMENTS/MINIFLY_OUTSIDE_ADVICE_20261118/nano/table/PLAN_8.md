# Cells that hear how long ago a byte last came (roadmap second revision, tracks L and C6) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at 46dc595c3; `trust.py`, `words.py` as they stand. Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

A look inside the system T2 on the last tenth (`where.py`, `res/where_T2.json`; a look, not a claim): 35% of the loss is at the first letter of a word and 10% at punctuation; and when a word ends, the choice between a space and a line end costs 0.022 nats a byte of the whole text. A line end follows a letter at places 35 to 50 of its line (median 44), a space at places 4 to 36 (median 18); the system gives a line end after a letter a probability of about 2%, because none of its cells knows how long the line already is. The table's cells see six bytes and the word cells see words; the place in the line is neither. Mr. Xu, 20 November: cells "that adjusts around different time clocks".

## 2. The question

**Do cells that hear how long ago a byte value last came lower the loss, and does the answering cell find by itself which byte value is worth timing?**

*The cells.* For a byte value v, a cell for each pair (bytes since v last came, capped at 63; the present byte). Such a cell is what a delay line of v, 63 long, gives: only the newest occurrence of v is heard. The cells count what follows and are read with trust as every other kind (`PLAN_3.md`), beside T2 (the table's cells, K = 6, and the cells that step once a word; e = 0.02, the trust's step 0.00005, one reading; nothing chosen here).

*Arms.* (0) T2 (1.4973). (A) T2 with the cells for v = the line end only (the value my look points to: given by me). (A2) as A with a second kind (bytes since the line end in fours; the present byte and the one before). (B) T2 with such cells for **every** one of the 65 byte values, 65 kinds, each with its own trust: nothing given; the answering cell must find which matter. (C) a control: as A with the count taken from a clock blind to the text (the place of the byte in the text, modulo 45).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* Arm B is at least 0.01 nats below arm 0. *Refuted if* it gains less than 0.01.

| | Statement | Chance |
|---|---|---|
| S1 | arm A gains at least 0.01 over arm 0 | 0.60 |
| S2 | arm B gains at least 0.01 over arm 0 | 0.50 |
| S3 | arm B is within 0.003 of arm A, or better | 0.45 |
| S4 | arm C (blind clock) gains less than 0.003 | 0.80 |
| S5 | arm A2 is at least 0.003 below arm A | 0.40 |
| S6 | the best arm is below 1.485 | 0.40 |

## 4. Size, cost, what is not run

At most 64 x 65 cells for a byte value (fewer occur); arm B adds 65 kinds and 2,600 trusts. A run takes one to three CPU-minutes. Not run here: these cells with the companions of `PLAN_6.md` or with runs of eight bytes; delays beyond 63; the count kept in a coarser form.
