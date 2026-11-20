# Line `nano/aheadprose`, first plan: look-ahead guesses used as lines, on prose (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session and given there as its translation from the Chinese (the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The task, its question, its hypothesis and what refutes it, the measure and the control are the coordinating session's (its message opening this session). The program, its numpy reference, the exact kinds and the predictions are this session's. **Source.** `claude/friendly-darwin-8mmhfm` at 91f62217; `score.py`, `table.py` and `words.py` are copied unchanged from `nano/table` at that commit, and the texts of the life are prepared as `nano/table/life.py` prepares them. Written before the program exists and before any run. Development level; one text and three books; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. On two-digit sums, cells that also count the bytes two and three after them, with guesses the cells can be keyed on, took never-shown sums from 77 to 91% (`nano/sparse/PLAN_10.md`), and an independent program repeated it on fresh seeds; a control whose guesses carried nothing about the present line took about 12 of the 13 to 14 points away (`nano/ahead2/REPORT.md`).
2. On prose the cells that count, with cells that step once a word (T2 of `nano/table/PLAN_3.md`, arm 2, with the trust's step 0.00005 and e = 0.02), stand at 1.4973 nats a byte on the last tenth of the test text; 35% of that loss is at the first letter of a word (`nano/table/PLAN_8.md`, section 1).
3. Kinds of cell born with no trust did no harm at any point of a life, and gained once the system had read a million bytes (`nano/table/PLAN_10.md`, sections 7 and 9).
4. The last tenth of the test text has been scored for some fifty arms and is worn as a test (`nano/table/PLAN_10.md`, section 5, reading 4b).

## 2. The question and the one change

**Do look-ahead guesses used as lines lower the loss on prose?**

A guess is a function of what the cells already see, so it brings no new fact. What it can do is let a cell be keyed on what is expected two or three bytes on, which groups contexts by their future. On sums that was the carry. The coordinating session gives it less than an even chance on prose; so do I (section 4).

*The system.* T2 exactly (the table's cells for runs of 0 to 6 bytes and the three kinds that step once a word, ten kinds; e = 0.02; trust born at 0.25, step 0.00005, forty levels of experience for each kind; one reading), with one change, in these parts. Write p for the place of the byte to be named next.

* *More counts in the same cells.* Each of T2's cells holds three sets of counts: of the next byte (as before), of the byte after it (two ahead) and of the one after that (three ahead). The text supplies each when it arrives. Each set has its own total and its own number of different bytes seen, hence its own level of experience.
* *Two guesses before each byte, in this order.* (1) T2's cells' third counts, read with their own trusts (a second and a third table of trusts for the ten kinds, born at 0.25 as the first), give the guess three ahead, g3: the byte of the largest branch. (2) Cells of two kinds keyed on g3 (below) hold counts of the next byte and of the byte two ahead; T2's cells' second counts and these cells' second counts give the guess two ahead, g2. A trust used for a guess changes by the same rule as any other when the guessed byte arrives (the step times the mismatch times what the cell said), one or two bytes later.
* *Five added kinds of cell, all born with no trust.* (a) (the last byte, g3); (b) (the last two bytes, g3); (c) (the last byte, g2); (d) (the last two bytes, g2); (e) (the last byte, g2, g3). Kinds (a) and (b) count the next byte and the byte two ahead (they are formed before g2 and help to form it); (c), (d) and (e) count the next byte. They are read as any other cell: log((c(v) + e) / (n + 65 e)) times the trust for the kind and its level of experience.
* *The answer* is the competition of the branches on the sum over T2's cells (first counts), the cells of (a) and (b) (first counts) and the cells of (c), (d), (e).
* A guess is the system's own in learning as in scoring. In a scoring nothing changes; the guesses are still formed. At the start of each stretch that is read or scored, counts and trust changes still pending for bytes two and three ahead from the stretch before are dropped (two bytes' worth).

With the added trusts at nothing the answer is T2's to the last digit; the program must give 1.4973 for T2 on the last tenth of the test text, and the three figures of `nano/table/PLAN_10.md`, section 9, for T2 on the books (1.4802, 1.3162, 1.4109).

*The one number chosen.* The step of the trusts of the five added kinds (in answering and in guessing): 0.00005 (T2's), 0.0002 or 0.0005. `nano/table/PLAN_8.md` found kinds born at nothing to want a larger step. It is chosen on the last twentieth of the stream of the test text (cells taught on the stream before it). Nothing else is chosen. All other steps stay 0.00005.

*The control.* Everything as above (the same kinds, counts, trusts and step; the guesses formed and taught as before), except that every cell keyed on a guess is keyed on the guess the system made 64 bytes earlier (nothing, that is byte value 0, for the first 64 bytes of a stretch). That guess has the same distribution and carries nothing about the present place.

## 3. The measure

Stream loss in nats a byte, nothing changing during a scoring (`score.py`). The claim is made on texts where nothing was ever chosen: the life "A, B" of `nano/table/PLAN_10.md`, addition 2: the system reads the stream of the test text (its first nine tenths), then Bryant, Burgess and Carroll in turn (first nine tenths of each), and each book's last tenth is scored right after its own stream. The last tenth of the test text is scored once for T2 (to check the program against 1.4973) and once for the system with look-ahead, in that same life, and is not scored for the control or for any setting that was not chosen.

## 4. Hypothesis (the coordinating session's), what would refute it, predictions (mine, before any run)

*Hypothesis.* The system with look-ahead kinds is at least 0.005 nats below T2 on each of the three books. *Refuted if* it is not below T2 by 0.002 on at least two. *Control:* if the control gains as much, the gain is not the guess.

| | Statement | Chance |
|---|---|---|
| Q1 | at least 0.005 below T2 on each of the three books (the hypothesis) | 0.20 |
| Q2 | at least 0.002 below T2 on at least two books (not refuted) | 0.40 |
| Q3 | not worse than T2 by more than 0.002 on any book (kinds born with no trust do no harm) | 0.80 |
| Q4 | the control is within 0.002 of T2 on all three books | 0.60 |
| Q5 | on the last tenth of the test text, at least 0.003 below T2 | 0.35 |
| Q6 | the step chosen for the added trusts is larger than T2's 0.00005 | 0.65 |
| Q7 | on the books' last tenths together, g2 names the byte two ahead rightly on at least 30% of places | 0.60 |
| Q8 | the same for g3 and the byte three ahead, at least 20% | 0.60 |
| Q9 | g2 is right less often for places where the byte two ahead is the first letter of a word than elsewhere | 0.80 |

Reading (mine) before the run. A cell of kind (d) sees two bytes and a statement that summarises, in one byte, what the cells of six bytes and of the words expect after the next byte. Where the long cells are young, such a cell has counted far more often than they have and may say something they cannot yet; where they are old, it says less than they do. On sums the guess carried a fact (the carry) that no single short cell held. I know of no such fact in prose that one byte two ahead would carry, so I expect a small gain or none.

## 5. Size, cost, what is not run

Cells: T2's (about 1.8 million that have counted after the stream of the test text, more after the books; reported), each now with three sets of 65 counts; the added kinds have at most 65 x 65 cells for (a) and (c) and 65 x 65 x 65 for (b), (d), (e) (those that have counted are reported). Trusts: 3 x 10 x 40 for T2's cells and 7 x 40 for the added kinds (two sets for (a), (b)). A life takes about one to two CPU-minutes and about three gigabytes by my estimate; the choice of the step three shorter runs; in all some ten CPU-minutes. A numpy reference of the whole loop is checked against the compiled one on a prefix before any figure is used.

If the hypothesis is refuted, I report what the guesses were worth as guesses (how often right, at the first letter of a word and inside words) and stop this line there. If step 1 gains, each of the following is one change with its own plan: guesses of the next word's first letter made at the end of a word; a guess passed on as its two most likely bytes; the same beside the kinds of `nano/table/PLAN_10.md`.

Not run here: guesses further ahead; a guess as a spread over bytes; the measure in which everything goes on changing through a last tenth; more than one order of the books; the relay system beside these cells.
