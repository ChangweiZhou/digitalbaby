# Situation cells: the system learns where it is, and each place keeps its own cells and trusts (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the development life of `PLAN_1.md` (the order of `nano/suite`'s table) and its singles, while that plan's reported orders were still running and before any of them was looked at for this plan; before any run of the program of this plan. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows (`PLAN_1.md`, development order, the small system ZA)

Read one after another by one system, the six streams are not kept. Going on, at the end of the life against the stream's own stage: tiny Shakespeare 1.5035 against 1.3434, the book 1.5591 against 1.4022, Python 1.0603 against 0.9626 (C headers and the table: within 0.02). The mean over the six tests at the end is 0.9184, against 0.8357 for six systems that each read one stream. Where the loss sits, by that plan's diagnostics: with the trusts and parts of the stream's own stage put back, the excess of tiny Shakespeare falls from 0.160 to 0.051, of the book from 0.157 to 0.054, of Python from 0.098 to 0.018. So about two thirds is in the trusts, which follow the stream being read, and about one third in the counts of contexts that several streams share. A constant step in place of the rule of the step does not change the picture (the life costs 0.068 against 0.083). The system also gives no sign from which it could notice the change: Python read by the system that knows two English texts costs less a byte than those texts did.

My reading: one set of trusts and one set of short contexts cannot serve streams of different kinds, and nothing in the system says which kind it is in. What is missing is a signal of **where the system is**, learned without labels, that the cells and trusts can be kept by.

## 2. The mechanism, exactly (`society.py`)

**Situation cells.** A situation cell is a cell that counts: it holds a count for each of the 256 byte values, of the bytes read while it was awake. The cells together hold a belief (a share for each cell, summing to one). At each byte, before it is read, the cell with the largest share is **awake**. After the byte b is read, each cell's share becomes ((1 - s) x its share + s / the number of cells) x the cell's opinion of b, where its opinion is (its count of b + 1/2) / (its counts + 128), and the shares are made to sum to one; s = 1 / W. The awake cell counts the byte.

**Birth.** After every W bytes the cells look back at that block. For each cell: the sum over the block of minus the log of the opinion it had of each byte at the time. For a cell that knew nothing at the start of the block and counted every byte of it: the same sum. If the cell that knew nothing would have been cheaper than every existing cell, a new situation cell is born with the block's counts (the counts the others took from that block are taken back), and the whole belief goes to it. Nothing else decides a birth: no threshold.

**What a situation keeps.** Each situation cell has a whole system of its own (cells that count, trusts, parts: the system of `PLAN_1.md`, unchanged, here ZA). A byte is answered and counted by the system of the situation that is awake. A system born with its situation first reads the block that gave it birth. This is the extreme form, in which nothing is shared between situations; a form in which situations share the long contexts and differ in trusts and short contexts needs a new kernel and is not built here.

**One number**, W, the length of the block: it sets both how long a situation is believed to last and how much text a birth is judged on. It is chosen on the development order between 1,024 and 4,096 by the mean over the six tests of the going-on loss at the end of the life, then frozen. The 1/2 in the opinion is the usual one for a cell that counts with no knowledge (it is not varied).

**Measured as in `PLAN_1.md`**: stages, tests in forked copies, nothing changing (no cell counts, no situation is born, the belief still moves, as the clocks do) and going on; singles read through the same mechanism.

**Check before any run** (`python3 society.py check`): the compiled situation cells against a plain-Python version written out in full (shares, awake cell, births, counts); a situation's awake cell at a byte does not depend on that byte or any later one; with W larger than the stream the mechanism is one situation and gives the plain system's losses bit for bit.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With situation cells, what the system learned of a stream is kept through a life over six streams: at the end of the life, going on, each stream's test costs at most 0.02 nats a byte more than at its own stage, in every reported order; and the mean over the six tests at the end is within 0.02 of the mean of the six singles of the plain system (0.8357).

**Refuted if** the first part fails for any stream in three or more of the five reported orders, or the second in two or more.

| | Statement (ZA with situation cells, second answering cell, going on unless said; reported orders) | Chance |
|---|---|---|
| S1 | The first part of the hypothesis holds for every stream in every order | 0.45 |
| S2 | The mean over the six tests at the end is within 0.02 of the singles' mean in at least four of five orders | 0.50 |
| S3 | Against the plain life of `PLAN_1.md` in the same order, the cost of the life (mean at the end less the singles' mean) is cut by at least two thirds in every order | 0.55 |
| S4 | With nothing changing, the end is at most 0.05 above the own stage for every stream in at least four of five orders | 0.45 |
| S5 | A life ends with 4 to 12 situations in every order | 0.55 |
| S6 | The two prose streams share one situation (more than half of each body's bytes read by the same one) in at least three of five orders | 0.50 |
| S7 | A stream read alone through the mechanism is not more than 0.01 above the plain single on any of the six streams (splitting one stream into situations costs little) | 0.45 |
| S8 | On at least one stream a single through the mechanism is below the plain single by 0.005 or more (a stream that is itself made of kinds gains) | 0.40 |

## 4. Runs, size, cost, what is not run

Development: the development order at W = 1,024 and 4,096, and the six singles at the chosen W. Reported: the five orders of `PLAN_1.md` at the chosen W. Size: the sum of the systems of the situations (expected close to the plain life: 2.3 million cells). Cost: as a plain life of ZA (about 3 CPU-minutes) plus the situation cells (a few CPU-seconds). **Not run:** the large system (ZPA scaled) with situation cells (until `PLAN_1.md` has its lives); situations that share long contexts; a bound on the number of situations or on the store; a return of the same stream several times; any other form of the opinion (contexts of one or more bytes in the situation cells).

## 5. Outcome

(to be entered after the runs)
