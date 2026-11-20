# Where a segment starts, found by the system from its own surprise (line `nano/ends`, first plan, E1) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as passed to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** Branch `claude/nano-ends-8mmhfm`, started from `claude/friendly-darwin-8mmhfm` at 9346c104e. Text, split and measure as in `nano/table/PLAN_1.md` (`nano/data/README.md`). Written before any run of the arms; only `ends.py check` (the compiled loop against numpy on the first 4,000 bytes of the stream) was run before this file was committed, and its output is in `check_output.txt`. Development level; one text; no checker gate.

**Copied.** `score.py`, `table.py`, `words.py` are copied unchanged from `nano/table`. The kernel in `ends.py` is that of `nano/table/trust.py` with one more output (the probability given to the byte that arrived, formed before anything changes); the numpy reference is trust.py's.

## 1. The evidence this follows

`nano/table/PLAN_3.md`: on the last tenth of the text, cells that count what follows them, read by an answering cell that learns a trust for each kind and experience of cell, give 1.5226 nats a byte with the table's cells and 1.4973 when cells that step once a word are added. The same cells on a clock blind to the words (every fifth byte) give nothing. The ends of the words were given to the system (a word = a run of letters). Its reading 7: "Not shown by this plan: that the cells can find the ends of words themselves".

## 2. The question

**Can the system find where a segment starts from its own signal, and keep at least half of the gain that given starts bring?**

*The one change.* A new segment starts at a byte that surprised the table's cells. The surprise of byte t is minus the log of the probability that the table-only system gave to byte t when it arrived. That system is arm 0 below with e = 0.02 and eta = 0.00005 (the choice of `PLAN_3.md`): its counts and trust are learned in the same single reading of the stream, each byte named before anything changes; on the last tenth nothing changes. Byte t starts a segment if its surprise exceeds theta. The surprise of byte t is known when byte t has arrived and is first used to name byte t + 1; nothing later is read.

*The cells.* The table's cells (runs of the last 0 to 6 bytes), and three added kinds whose contexts are: (cur), the bytes from the last start to now; (prev, cur), with the segment before; (prev2, prev, cur), with the two before. A cell is made the first time its context occurs and counts what follows. All cells are read with trust exactly as in `trust.py` (40 trusts for each kind; every trust starts at 0.25).

*Arms.* **0**: the table's cells alone. **G**: starts given, the first letter after a byte that is not a letter; the same three kinds of contexts as S (so a segment is a word with what follows it up to the next word; this is not the structure of `PLAN_3.md`'s arm 2, which kept the bytes between words apart). **R**: a start every fifth byte. **S**: starts by surprise. **H**, a control I add: starts fixed by an arbitrary function of the last two bytes (each pair of byte values is given a number drawn once with a fixed seed; a start where it is below a level set so that starts are as frequent in the stream as S's). H tells whether any rule that depends on the bytes and is as frequent does as well as surprise; R tells whether mere length of context does.

*Settings.* For S, theta is chosen among 1, 1.5, 2, 2.5, 3, 3.5, 4, 5 nats with e = 0.02, eta = 0.00005. Then for every arm eta (0.00005, 0.0001, 0.0002, 0.0005) and e (0.005, 0.02, 0.1) are chosen (the range of `PLAN_3.md`'s addition 1). All choices are made on the last twentieth of the stream with the cells taught on the rest, the surprise there coming from a table-only system taught on the same rest and unchanging on the last twentieth. Then the system reads the whole stream once and the last tenth is scored once for each arm, with nothing changing. H is run after S, since it takes its frequency from S.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* S gains over arm 0 at least half of what G gains over arm 0. *Refuted if* it gains less than half. (If G gains less than 0.01 nats the comparison is void and I say so.)

| | Statement | Chance |
|---|---|---|
| Z1 | G is at least 0.02 nats below arm 0 | 0.65 |
| Z2 | S gains at least half of what G gains (the hypothesis) | 0.40 |
| Z3 | S is at least 0.005 nats below arm 0 | 0.75 |
| Z4 | R gains less than 0.005 nats over arm 0 | 0.85 |
| Z5 | H gains less than half of what S gains | 0.45 |
| Z6 | on the last tenth, at least half of the starts found by S are starts of words | 0.35 |
| Z7 | on the last tenth, S finds at least 0.6 of the starts of words | 0.60 |
| Z8 | the theta chosen is between 2 and 3.5 | 0.60 |

Why I do not expect more (reading, mine): the first letter of a word is the least expected byte of a word, so a threshold on surprise should fall on many word starts; but bytes inside rare words surprise too, and whether a byte surprises depends on what the cells have counted so far, so the same passage may be cut differently at two times, and a cell whose context is cut differently does not meet its context again.

## 4. Size, cost, what is not run

Arm 0: 280 trusts and the table's cells (459,129 had counted in `PLAN_3.md`). The other arms: 400 trusts, and three cells more are active at a byte (at most ten); the numbers of cells that have counted and of counts above zero are reported for each arm. A cell holds 65 counts, most of them zero. One reading takes 4 to 10 CPU-seconds; an arm with its choice of settings 13 readings (S: 21), so about 2 to 4 CPU-minutes an arm, under 15 CPU-minutes in all.

Not run here: a peak of surprise in place of a threshold; other signals for a start; a second, slower clock; a theta that follows the system's own level of surprise as it learns; everything going on learning through the last tenth; a second text.

Reported besides the loss: for S, G, R and H, the number of segments, the share of found starts that are starts of words and the share of starts of words that are found, on the stream and on the last tenth.

## 5. Outcome (entered 20 November after the runs; `res/ends_arm*.json`)

Loss on the last tenth, nats a byte, nothing changing during it; each arm scored there once. Beside them: `PLAN_3.md` of `nano/table` had 1.5226 for the table's cells and 1.4973 for its own word cells (the bytes between words kept apart).

| Arm | Loss | Gain over arm 0 | Named right | Loss on the last twentieth of the stream (where settings were chosen) | Cells that have counted | Counts above zero | Segments in the last tenth | Found starts that are word starts | Word starts found |
|---|---|---|---|---|---|---|---|---|---|
| 0, the table's cells | 1.5226 | | 54.41% | 1.5566 | 459,129 | 872,748 | | | |
| G, starts given | **1.4995** | 0.0231 | 55.20% | 1.5336 | 2,035,419 | 2,633,419 | 20,724 | 1 | 1 |
| R, every fifth byte | 1.5234 | -0.0008 | 54.58% | 1.5569 | 2,242,705 | 2,854,254 | 22,308 | 0.19 | 0.20 |
| S, starts by surprise (theta 2.5) | **1.5144** | 0.0082 | 54.80% | 1.5487 | 2,091,358 | 2,709,841 | 27,666 | 0.46 | 0.62 |
| H, starts by a fixed function of the last two bytes, as frequent as S's | 1.5223 | 0.0003 | 54.56% | 1.5555 | 1,999,656 | 2,600,722 | 25,151 | 0.19 | 0.23 |

Chosen in every arm: e = 0.02, eta = 0.00005 (again the smallest allowed). For S the loss on the last twentieth of the stream by theta: 1.0: 1.5525; 1.5: 1.5493; 2.0: 1.5493; 2.5: 1.5487; 3.0: 1.5492; 3.5: 1.5502; 4.0: 1.5538; 5.0: 1.5570 (arm 0 there: 1.5566). On the stream S made 237,205 starts (one in 4.23 bytes; 0.49 of them word starts; 0.61 of the 187,778 word starts found). An arm with its choice of settings took 34 to 105 CPU-seconds. The compiled loop equals numpy to 3e-15 on a prefix, the surprise included (`check_output.txt`).

**The hypothesis is refuted.** S gains 0.0082 nats over arm 0, which is 35% of what G gains (0.0231); half was asked.

Predictions: Z1 held (0.0231). Z2 failed (35%). Z3 held (0.0082). Z4 held (R is 0.0008 worse). Z5 held (H gains 0.0003, a twenty-seventh of S's gain). Z6 failed narrowly (0.46). Z7 held (0.62). Z8 held (2.5).

Readings (mine).
1. **A threshold on the system's own surprise finds starts that are worth something, and it is the surprise that does it**: starts as frequent, fixed by an arbitrary function of the last two bytes, gain nothing (0.0003), and neither does the blind clock. But the gain is a third of that of given starts, not a half.
2. **The fair reference is G, 1.4995**; the structure of contexts with the bytes between words kept apart (`PLAN_3.md`, 1.4973) is 0.002 better still and is not what S is compared with.
3. **Where the surprise fails, as far as these figures show.** It misses 38% of the word starts and more than half of its starts are not word starts. A threshold on surprise depends on which byte arrived: after a space a common first letter does not surprise and no start is made, while a rare letter inside a word makes one. So the same word is cut at one place on one occasion and not on another, and a cell whose context is cut differently does not meet its context again. This is a reading and is not tested here; the next plan tests it by taking the signal before the byte arrives (how unsure the answer was), which is the same whatever byte then comes.
4. The loss by theta is flat from 1.5 to 3.0 (within 0.0006 on the last twentieth), so the threshold itself is not the weak point.
