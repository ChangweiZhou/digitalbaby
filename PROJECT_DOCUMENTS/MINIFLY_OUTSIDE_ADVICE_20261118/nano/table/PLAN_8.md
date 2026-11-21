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

## 5. Addition 1 (20 November; written after the four arms were read and before the runs it describes): a new kind of sender is born with no trust

The four arms are in (section 6 will give them): the cells that time the line end gain 0.007; the same cells for all 65 byte values make the system 0.045 **worse**. Every trust is born at 0.25 (`PLAN_3.md`), so 65 kinds that mostly have nothing to say begin with thirteen times the weight of one, and the trust's small step does not take it away within one reading. In the relay system the return strengths that were born at nothing and grew did as well as or better than those born at a size (`loop_small_case`, `nano/loop`). The one change here: **the trust in each added kind of cell is born at nothing**, and grows only if the mismatch says so. Arms A, B and C again with that change, each at the trust's step of the plan (0.00005) and at ten times it (0.0005), both reported, none chosen.

*Predictions (mine):* with trust born at nothing arm B is not worse than arm 0 by more than 0.003, at both steps (chance 0.7); arm B gains at least 0.005 at one step (chance 0.35); among the 65 kinds of arm B the kind that times the line end ends with the largest trust (chance 0.6).

## 6. Outcome (entered 20 November after all runs; `res/place_*.json`)

Loss on the last tenth, nats a byte, nothing changing; arm 0 (T2) is 1.4973. Settings were fixed beforehand; each arm was scored once.

| Arm | trust born at 0.25, step 0.00005 (the plan) | born at nothing, step 0.00005 | born at nothing, step 0.0005 |
|---|---|---|---|
| A, cells that time the line end | 1.4903 | 1.4927 | 1.4910 |
| A2, with a second kind | 1.4924 | not run | not run |
| B, cells that time every one of the 65 byte values | 1.5422 | **1.4873** | 1.5068 |
| C, a clock blind to the text | 1.4966 | 1.4959 | 1.4964 |

At a line end after a letter the loss falls from 4.01 nats (T2) to 3.77 in arm A and 3.52 in arm B born at nothing. Trust at the end of arm B born at nothing (mean over the levels used): the line end 0.058, the space 0.017, every other byte value 0.005 or less. A run took 6 to 32 CPU-seconds.

**The hypothesis is refuted as written**: with every trust born at 0.25 arm B is 0.045 nats worse than arm 0. With the trust of the added kinds born at nothing (addition 1) arm B is 0.00995 better, a hair under the 0.01 asked. S1 failed (0.0070). S2 failed. S3 failed as written, and holds with trust born at nothing (B is then 0.003 to 0.005 better than A). S4 held. S5 failed (A2 is worse than A). S6 failed (the best is 1.4873). Addition 1: its first prediction failed (at ten times the step arm B is 0.0095 worse than arm 0); its second and third held.

Readings (mine).
1. **Knowing the place in the line is worth 0.007 to 0.010 nats here**, less than half of what my look sized (0.022). A cell that knows the place raises the line end's share only a little, because the answering cell multiplies its senders' opinions and the other cells still say that a line end after a letter is nearly impossible.
2. **A new kind of sender should be born with no trust.** This is the result of the plan that reaches beyond this text. Given 65 kinds at once, each born with the trust of an old kind, the system got much worse; given the same 65 kinds born with no trust, it got better than with the one kind I had picked by hand, and its answering cell ended with by far its largest trust in the one kind that matters. Mr. Xu, 20 November: "But we need to make sure not testing everything at once and leave a mess." For the experimenter that stands. For the system it now has a counterpart: kinds of cells can be offered together if each has to earn its trust.
3. **Here the trust's step matters** (in `PLAN_4.md` it did not): at ten times the step the 65 kinds do harm again, each of the useless ones having earned a little trust by chance.
4. The clock that is blind to the text gains nothing under any setting.
