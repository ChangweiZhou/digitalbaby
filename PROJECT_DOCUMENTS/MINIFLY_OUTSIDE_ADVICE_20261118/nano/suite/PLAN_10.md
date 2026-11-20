# A lasting cell is born only at a second meeting (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The question was put to this line by the coordinating session after its last table (a request, with leave to skip what is ill-posed); the coordinating session says it comes from its reading of a review of brain cells that Mr. Xu gave it (`BRAIN_CELLS_REVIEW_AGAINST_PROGRAMME_20261120.md` on its branch, which I have not read; I attribute nothing in it to Mr. Xu). **Source.** This branch at the commit that holds this file. Written before any run with the option on any stream (it has run only inside the check, on 2,500 bytes against the reference). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

Half of all cells of the base have been met once. The recalling senders and the last-byte senders already say, for a long context, what a cell met once would say. So the cheapest large cut of the store may be not to make a cell at a first meeting.

## 2. The mechanism, exactly (`second=1`)

A context met for the first time gets no cell. It leaves **one bit** (in a table of 134 million bits addressed by a scramble of the context; two contexts may share a bit, which lets a few cells be born at a first meeting; the reference in plain Python uses the same addresses). When a context with no cell finds its bit set, its cell is made and counts the byte that comes; the byte that followed the first meeting is not counted. The stream read is kept as the recalling senders keep it anyway; nothing else is kept of a first meeting. Said beforehand, a point where the coordinating session's picture and this system part: the last-byte senders live in the cells of the runs, so a context without a cell has no last-byte sender either; only the recalling senders (contexts of eight bytes and more) speak for a first meeting.

**System:** ZA as in `PLAN_9.md`, with and without the option; all six streams, going on first.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** The option removes at least 40% of the cells on every stream and costs under 0.02 nats a byte going on, on at least four streams. **Refuted if** it costs 0.02 or more on three or more streams.

| | Statement (going on, second cell, against ZA) | Chance |
|---|---|---|
| J1 | at least 40% fewer cells on every stream | 0.80 |
| J2 | the hypothesis holds | 0.45 |
| J3 | cost under 0.010 on at least four streams | 0.25 |
| J4 | at about the same number of cells, the bounded store of `PLAN_9.md` at half (cells leave) is better than this (cells are not born) on at least four streams: a cell met once is worth having until something better needs its place | 0.60 |

## 4. Cost, not run

7 CPU-seconds a run. Not run: the first follower recovered from the stream at the second meeting; the option on ZPA; a third meeting.

## 5. Outcome (entered 20 November, after the runs; `res/*__second.json`, `report_9.txt`)

| | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|
| cells at the end of the stream, ZA | 488,071 | 561,754 | 519,713 | 493,662 | 611,748 | 315,987 |
| the same, a cell only at a second meeting | 246,870 | 275,419 | 279,408 | 264,002 | 209,640 | 162,247 |
| fewer by | 49% | 51% | 46% | 47% | 66% | 49% |
| going on, cost against ZA | +0.0113 | +0.0097 | +0.0142 | +0.0162 | +0.0081 | +0.0035 |
| nothing changing, cost against ZA | +0.0183 | +0.0106 | +0.0159 | +0.0217 | -0.0259 | +0.0005 |
| for comparison, the bound at half with the rule of `PLAN_9.md`, going on | +0.0068 | +0.0070 | +0.0046 | +0.0035 | +0.0026 | +0.0016 |

**The hypothesis held:** 46 to 66% fewer cells on every stream for 0.004 to 0.016 nats going on, under 0.02 everywhere. **Predictions scored.** J1 held. J2 held. **J3 failed**: the cost is under 0.010 on three streams (book, table, xml), not four. J4 held on all six: with about the same number of cells, a store from which the least counted and oldest cell leaves costs a fifth to three quarters of what it costs never to make a cell at a first meeting (on table the bound at half holds more cells than this option, 306,000 against 210,000; the bound at a quarter, 153,000 cells, costs +0.0063 there, still under this option's +0.0081).

**My readings.** (1) It is a cheap large cut, as the coordinating session expected, and it needs no drawing of places and no choice of who leaves; but it is not the best cut: a cell met once is worth having until another context needs its place. (2) The two can be had together (first meetings leave a bit; the store is bounded as well); the check covers that form, it was not run on the streams. (3) The cost is not carried by the recalling senders as the proposal pictured: a first meeting of a short run leaves nothing a sender can use (the last-byte senders live in the cells), and that is where the 0.01 goes.
