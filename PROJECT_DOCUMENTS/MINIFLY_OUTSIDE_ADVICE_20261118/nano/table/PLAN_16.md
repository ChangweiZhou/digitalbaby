# The second answering cell added to this line's own prose systems (roadmap second revision, track L; the form of `PLAN_15.md`, nothing chosen) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试。" (my translation: "Go on trying boldly."). **Source.** `claude/friendly-darwin-8mmhfm` at 1f7aaf3df; `all.py`, `two.py`, `place.py`, `relay.py`, `words.py` as they stand; the form of `PLAN_15.md`. Written before any run of this plan's program (`gate_all.py`, which exists and has not been run). Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_15.md`, the part on prose, which is complete as I write (its sums on the reported seeds are still running): on the base system of runs of 0 to 6 bytes, a second answering cell with one part of the trust for each of the last six bytes, indexed by the byte there and born at nothing, lowers the loss by 0.0200, 0.0213, 0.0233 nats on three books that no line had opened, by 0.0210, 0.0236, 0.0299 on the three worn books, and by 0.0168 on the last tenth of the test text; the control whose parts read a constant gives between 0.005 worse and 0.001 better. That base has no cells that step once a word, no timing cells, no pool, no companions. This line's best system has all of them (`PLAN_10.md`: 1.4707 on the last tenth of the test text, where the full nanoGPT network of `nano/ref` has 1.454 on one seed).

## 2. The one change, and the question

The second cell of `PLAN_15.md`, unchanged in form and numbers (six parts, the bytes at delays 1 to 6, each part born at nothing and changed by the trust's step 0.00005 times the second cell's own mismatch), added to `all.py`'s system; the given sender (the relay system's opinions) has parts like any other sender. The first cell is `all.py`'s to the last digit in the same run.

**Does the gain stand beside the other kinds of cell, or do the cells that step once a word, the timing cells and the pool already hold what the bytes at the last six delays tell?**

*Systems.* T2 (the table's cells and the word cells: `without=companions,relay,since,pool`); all kinds but the relay system (`without=relay`); all kinds together.

*Measured.* (a) The test text: its stream read once, its last tenth with nothing changing (the measure on which 1.454 was measured; scored for some fifty arms and worn), and the last tenth with everything going on (another measure; no reference). T2, all but the relay system, all kinds. (b) The life of `PLAN_15.md` with the three books opened once there (`chesterton-ball`, `austen-emma`, `melville-moby_dick`): T2 and all kinds but the relay system (the relay system's opinions exist for the test text only); each book's last tenth right after its stream, the test text's last tenth at the end of the life and on the return with everything going on. (c) The same life with the three worn books, for comparison with `PLAN_10.md` and `PLAN_14.md`. The control (parts that read a constant) for T2 in (b).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The gain stands beside the other kinds: on each of the three books of (b) the second cell is at least 0.010 nats below the first, for T2 and for all kinds but the relay system. *Refuted if* for either system fewer than two books are lower by 0.005.

| | Statement | Chance |
|---|---|---|
| X1 | T2, life (b): at least 0.010 lower on each of the three books | 0.70 |
| X2 | all kinds but the relay system, life (b): at least 0.010 lower on each | 0.55 |
| X3 | all kinds, the last tenth of the test text with nothing changing: the second cell at or under 1.4607 (0.010 below the first) | 0.60 |
| X4 | the same at or under 1.4540 (the figure of the full nanoGPT network on that tenth) | 0.35 |
| X5 | all kinds, going on through the last tenth: the second cell at least 0.010 below the first (1.3470) | 0.50 |
| X6 | the control on T2 gives less than a fifth of the gain on each book | 0.80 |
| X7 | keeping: after the three books of (b) the second cell's loss on the test text's last tenth has risen by under 0.03 (T2) | 0.45 |

If X4 holds it will be stated with its limits: one text, a last tenth scored for some fifty arms of this line, one seed of the reference, a system with 1.8 million or more cells against 10.7 million numbers, one reading against 4,000 steps of batches.

## 4. Size, cost, what is not run

The second cell holds 6 times (kinds + 1) times 40 times 65 parts: 171,600 for T2, about 1.7 million for all kinds; the number used is reported. T2 runs take under a minute; all kinds about five to ten minutes, and the relay system's opinions about ten minutes more, once. Not run: gates other than the bytes at the last six delays; a third cell; the second cell on the life of B only; the same on sums over long lives (an addition to `PLAN_15.md` if its claims hold).

## 5. Addition 1 (20 November; written after the runs of section 2 except one, and before the runs it describes): the life of (b) for all kinds, one book at a time

The life of (b) for all kinds but the relay system (the test text's stream and the three books in turn, 3.6 million bytes, 164 kinds) was killed twice for lack of memory on this machine, once beside other runs and once alone. It is NOT RUN as planned. In its place, to be read as a different measure: three lives, each the test text's stream and then **one** of the three books (`chesterton-ball`, `austen-emma`, `melville-moby_dick`), its last tenth scored right after its stream; all kinds but the relay system; the second cell. The largest of the three may also fail for memory and will then be entered as not run.

*Prediction (mine), in place of X2:* the second cell is at least 0.010 nats below the first on each book that runs (chance 0.55; on the worn books in turn it gave 0.020, 0.006, 0.025, so one book of three under 0.010 would not surprise me).
