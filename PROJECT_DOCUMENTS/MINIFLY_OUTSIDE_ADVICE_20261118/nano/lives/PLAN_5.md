# Counts in two memories: which counts spoil a life, and cells that hold what followed lately (21 November 2026; committed 00:24 UTC)

**Authorisation and steer.** As `PLAN_1.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the first part of the outcome of `PLAN_1.md`, after four of the six lives of its addition 2 had been read, and after the first development life of `PLAN_3.md`; before any run of the program of this plan. (There is no `PLAN_4.md`: the number was kept for a step on the trusts that has not been written.) Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

* `PLAN_1.md`: about a quarter of what a life costs on the streams read earlier is in the counts, not in the trusts.
* `PLAN_1.md`, addition 2 (four lives read so far): a stream read late is learned worse than by a system of its own, by 0.02 to 0.06 nats a byte, **and setting every trust back to its value at birth before the stream changes almost nothing of that** (tiny Shakespeare read fifth: +0.063, with the trusts set back +0.059). So that loss is in the counts: the store holds what other kinds of text left in contexts that the new text shares.
* `nano/suite` saw four signs on single streams that old counts should weigh less (its last report), and found a sender that says only the last byte a cell counted worth 0.005 to 0.18 nats.

What is not known is **which** counts: those of short contexts, which every stream meets; those of long contexts; or the places of the recalling senders.

## 2. First a diagnostic (the boundary is given, so it is no mechanism)

The small system's lives (development order and the five reported orders) are run three times more, and before each new body the counts of one group are emptied: (a) the cells of runs of 0 to 3 bytes; (b) the cells of runs of 4 to 6 bytes; (c) the places of the recalling senders. The trusts are left as they are. For each: the stream's own stage against its single, going on. The group whose emptying brings the later streams back to their singles is where the loss on streams read late sits.

**Predictions (mine): G1** emptying the short runs removes at least half of that loss at the median over the positions where the plain life is 0.02 or more above its single (0.55); **G2** emptying the long runs removes less than a quarter (0.60); **G3** emptying the places removes less than a quarter (0.65).

## 3. The mechanism, exactly (`mem.py`)

**A bounded kind** is a kind of cell on a run of bytes like the others, with one difference: when a cell's counts reach the bound, every count of the cell is halved (a count that falls to nothing is dropped). Such a cell holds what followed its context lately, by its own measure of lately (its own last events, not a span of bytes). It is a sender like any other: its own cells, its message by the same rule, its trust for each level born at nothing, its parts in the second answering cell. The cells that count for life are untouched, so the system without bounded kinds is the plain one.

**Three forms are tried on the development order, and one is chosen** by the mean over the six tests of the going-on loss at the end of the life (the plain answer of the second cell), then frozen: (i) a bound of 32 on runs of 0 to 3 bytes (four kinds more); (ii) bounds of 32 and of 512 on runs of 0 to 3 bytes (eight more); (iii) a bound of 32 on runs of 0 to 6 bytes (seven more). If the diagnostic of section 2 points to the long runs or the places and not to the short runs, the forms are still run as written and the outcome is read against that.

**Check before any run** (`python3 mem.py check`, `check5_output.txt`; 7 tests passed): the compiled kernel against the reference written out in full with bounded kinds (cells and counts equal; answers to 6e-6 under the rule of the step with 25 senders, to 5e-11 under the step by uses); with remembered trusts as well; no answer moves when its own byte and all later ones are replaced; with no bounded kind the losses are those of `books.py`'s kernel.

## 4. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the chosen form: (a) what is read late is learned better: at its own stage, going on, a stream's excess over its single (the single of the same system) is cut by half or more at the median over the later positions of the five reported orders in which the plain system's excess is 0.02 or more; (b) read alone, no stream is harmed by more than 0.005 and at least three of six gain 0.001 or more (the rule of `nano/suite`).

**Refuted if** (a) the median cut is under a quarter, or (b) a single is harmed by more than 0.005.

| | Statement (small system with the chosen form; second answering cell; going on; reported orders) | Chance |
|---|---|---|
| M1 | (a) holds | 0.40 |
| M2 | (b) holds | 0.55 |
| M3 | The cost of a life (mean over six tests at the end less the mean of the same system's singles) is at least a third lower than the plain system's in at least four of five orders | 0.40 |
| M4 | At the end of a life the system with bounded kinds is below the plain system on at least five of the six tests in every order | 0.50 |
| M5 | With remembered trusts as well (the N chosen in `PLAN_3.md`), the cost of a life is under 0.03 in at least three of five orders | 0.30 |
| M6 | The bounded kinds add less than 10% to the cells (form i or ii) | 0.75 |

## 5. Runs, size, cost, what is not run

Diagnostic: 18 lives of the small system (about 3 CPU-minutes each on a free machine). Development: three lives. Claim: six singles and five lives with the chosen form, and five lives with remembered trusts as well. Size: the plain system's 2.3 million cells at the end of a life and the bounded kinds' cells (expected under 0.2 million for runs of 0 to 3 bytes). **Not run:** the large system; other bounds; a bound that the system sets itself; fading by a span of bytes instead of by the cell's own events; any cell keyed on the situation.

## 5a. The choice of the form (entered 21 November, 00:50 UTC, before any single or reported life of a form was run; `res/mem__ZAM32*_dev.json`)

Development order, mean over the six tests of the going-on loss at the end of the life (the plain answer of the second cell): the plain system 0.9184; (i) a bound of 32 on runs of 0 to 3 bytes 0.8861; (ii) bounds of 32 and 512 on runs of 0 to 3 bytes **0.8656**; (iii) a bound of 32 on runs of 0 to 6 bytes 0.8847, with twice the cells. **Form (ii) is taken and frozen** (`ZAM32x512`: eight kinds more; 174,068 cells more at the end of the development life, 7.6%).

## 6. Outcome (entered 21 November, 01:18 UTC; `report_5.txt`, `res/mem__*.json`, `out/`). The lives with remembered trusts as well (M5) and half of the diagnostic lives are still running and are entered below when they are in.

**The diagnostic, so far** (development order and reported orders 73129 and 116089; six later positions in which the plain life is 0.02 or more above its single): emptying the cells of runs of 0 to 3 bytes before a new stream removes 0.88 of the loss on streams read late at the median and brings five of the six positions within 0.01 of the single; emptying the cells of runs of 4 to 6 bytes removes 0.45; emptying the places of the recalling senders 0.01; setting the trusts back (`PLAN_1.md`, addition 2) 0.14. **G1 held, G2 failed** (the long runs matter too, about half as much as the short), **G3 held.** So what spoils a stream read late is mostly what other kinds of text left in the short contexts.

**Bounded kinds, read alone** (six singles, going on; the plain single in brackets): tiny Shakespeare 1.3266 (1.3434), the book 1.3931 (1.3987), Python 0.9213 (0.9418), C headers 0.6319 (0.6952), the table 0.2977 (0.3460), markup 0.2603 (0.2889). Lower on all six, by 0.006 to 0.063 (0.4 to 14%); mean 0.8052 against 0.8357. With nothing changing they are lower on the two prose streams (0.010, 0.005) and higher on the other four (0.017, 0.071, 0.180, 0.056), as every sender of recency has been.

**Bounded kinds in a life** (five reported orders):

| order | own stage less own single, later positions: bounded kinds | the plain system, the same | cost of the life: bounded kinds | plain |
|---|---|---|---|---|
| 73129: table, C, markup, Python, Shakespeare, book | -0.049 +0.006 +0.007 +0.027 +0.014 | -0.047 +0.012 +0.029 +0.063 +0.046 | +0.028 | +0.046 |
| 93683: table, C, markup, Python, book, Shakespeare | -0.049 +0.006 +0.007 +0.028 +0.005 | -0.047 +0.012 +0.029 +0.064 +0.035 | +0.029 | +0.046 |
| 116089: Shakespeare, markup, Python, book, C, table | +0.003 +0.012 +0.014 +0.010 +0.008 | +0.005 +0.026 +0.035 +0.045 +0.012 | +0.073 | +0.093 |
| 406364: table, Shakespeare, Python, C, markup, book | +0.005 +0.010 -0.044 +0.009 +0.018 | +0.011 +0.024 -0.019 +0.019 +0.049 | +0.028 | +0.048 |
| 545353: table, book, Shakespeare, markup, C, Python | +0.004 -0.010 +0.010 -0.040 +0.009 | +0.007 +0.000 +0.016 -0.027 +0.036 | +0.035 | +0.050 |

(the cost of a life is the mean over the six tests at the end less the mean of the same system's six singles)

**The hypothesis held in both parts.** (a) The loss on streams read late is cut by 0.66 at the median over the twelve positions in which the plain system's is 0.02 or more. (b) Read alone no stream is harmed and all six gain. At the end of a life the system with bounded kinds is below the plain system on all six tests in all five orders.

**Predictions scored.** M1 held (0.66). M2 held (six of six gain; none harmed). **M3 failed**: the cost of the life is lower by 40%, 38%, 22%, 41% and 31%: a third or more in three orders, where four were asked. What remains is the loss on the streams read earlier, which sits in the trusts and which this plan does not touch. M4 held (six of six tests in every order). **M6 failed in part**: in a life the bounded kinds add 7.6 to 7.7% to the cells, but read alone they add 14 to 20% on Python, C headers and markup. M5 is not yet in.

**My reading.** Two things were found at once and should be kept apart. (1) *A general gain that has nothing to do with a life:* a context that also holds what followed it lately is a better sender on every stream, by up to 14% on source code, with nothing given for any kind of text; `PLAN_6.md` found the same on bytes that are not text (six of six streams). (2) *Most of the loss on what is read late in a life goes with it* (two thirds), because that loss was the short contexts speaking of other kinds of text, and a memory of what followed lately speaks of the present kind after a few events.
