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
