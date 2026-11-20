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

## 6. Outcome (entered 20 November after the runs; `res/dev_*.json`, `res/life_*.json`, `check_output.txt`)

The plan was committed at f8c8ea1 before `ahead.py` existed. The compiled loop equals the numpy reference to 4e-15 on four cases, with every guess equal (`check_output.txt`; the reference covers a stretch that is read, not the scoring call, which is the same loop with learning off). T2 in this program gives 1.4973 on the last tenth of the test text and 1.4802, 1.3162, 1.4109 on the books, as `nano/table` has them.

*The choice of the step* (last twentieth of the stream of the test text, cells taught on the stream before it; nats a byte): T2 1.5334; with look-ahead 1.5347 at 0.00005, 1.5379 at 0.0002, 1.5415 at 0.0005. The smallest was chosen, as the plan's rule has it. **On that twentieth the system with look-ahead is 0.0013 worse than T2 at every step tried.**

*The life A, B* (step 0.00005; each last tenth scored right after its own stream, nothing changing; nats a byte). The last tenth of the test text was scored once for T2 and once for the system with look-ahead, and not for the control.

| | test text, last tenth | Bryant | Burgess | Carroll |
|---|---|---|---|---|
| T2 | 1.4973 | 1.4802 | 1.3162 | 1.4109 |
| with look-ahead | 1.4938 | 1.4757 | 1.3095 | 1.4032 |
| gain over T2 | 0.0034 | **0.0045** | **0.0067** | **0.0077** |
| control: keyed on the guesses of 64 bytes earlier | not scored | 1.4786 | 1.3133 | 1.4073 |
| the control's gain over T2 | | 0.0016 | 0.0029 | 0.0036 |
| gain that is the present guess (system less control) | | 0.0029 | 0.0038 | 0.0041 |
| bytes named right, T2; with look-ahead (%) | 55.32; 55.31 | 56.07; 56.36 | 60.49; 60.63 | 59.99; 60.22 |

*What the guesses were worth as guesses* (share of places where the guess is the byte that came, %; a letter is a byte of a word; "first" is the first letter of a word, 19% of places; "later" a later letter, 57%; "other" any other byte, 24%).

| | all | first letter | later letter | other byte |
|---|---|---|---|---|
| the answer (the next byte), test text; three books together | 55.3; 58.3 | 17.8; 20 to 25 | 60.8; 61 to 68 | 71.0; 70 to 75 |
| the guess two ahead, test text; three books together | 38.0; 41.4 | 10.9; 14 to 19 | 38.1; 37 to 44 | 58.3; 60 to 68 |
| the guess three ahead, test text; three books together | 27.2; 30.0 | 6.8; 8 to 14 | 22.4; 21 to 28 | 53.9; 54 to 63 |

*Where the gain is* (loss by where the named byte stands; T2, then with look-ahead): Bryant, first letter 2.8995, 2.8971; later letter 1.1809, 1.1763; other 1.0633, 1.0572. Burgess 2.6910, 2.6842; 0.9941, 0.9876; 0.9363, 0.9291. Carroll 2.8178, 2.8085; 1.0437, 1.0377; 1.1858, 1.1755. The share of the loss at the first letter of a word is unchanged (37 to 41% on the books).

*Size and cost.* At the end of the life 2,419,727 of T2's cells have counted, with 3.22, 3.73 and 4.10 million counts above zero in their three sets (T2 alone: the first set). Added cells that have counted: (a) 2,166, (b) 15,157, (c) 2,013, (d) 13,505, (e) 12,419. Trusts: 1,200 for T2's cells and 280 for the added kinds. The largest trust an added kind reached: (a) 0.05, (b) 0.07, (c) 0.08, (d) 0.10, (e) 0.09, against 0.07 to 0.40 for T2's kinds. A life took 20 CPU-seconds (T2: 7).

**The hypothesis did not hold and was not refuted.** It asked for 0.005 on each of the three books; the gains are 0.0045, 0.0067, 0.0077, so one book misses by 0.0005. It would have been refuted by less than 0.002 on two books; all three are above that.

Predictions: Q1 failed (Bryant 0.0045). Q2 held. Q3 held (no book worse). Q4 failed (the control is 0.0029 and 0.0036 below T2 on two books; I had expected it level). Q5 held (0.0034). Q6 failed (the smallest step was chosen). Q7 held (41.4%). Q8 held (30.0%). Q9 held (14 to 19% at the first letter of a word against 37 to 44% at later letters).

Readings (mine).
1. **On prose the look-ahead kinds give a small gain, an order of magnitude below what they gave on sums**: 0.3 to 0.5% of the loss on the three books, where on sums they removed two thirds of the errors.
2. **About two fifths of that gain is not the guess.** The same kinds keyed on a guess from 64 bytes earlier gain 0.0016 to 0.0036. What the present guess adds is 0.003 to 0.004 nats on each book. On sums the control kept about a seventh. I have not found out what the control's cells do; a cell of (the last two bytes, a byte unrelated to the place) is a split copy of the cell of two bytes, read with its own trust by level of experience, and that alone may be a small gain.
3. **The gain is not where the loss is.** It is spread evenly: 0.002 to 0.009 at the first letter of a word, 0.005 to 0.006 at later letters, 0.006 to 0.010 at other bytes. The guesses themselves are poor exactly there: the byte two ahead is named rightly at 14 to 19% of first letters. A guess of one byte two or three places on does not carry which word comes next.
4. **The sign is not stable on the test text itself**: 0.0013 worse on the last twentieth of the stream after 950,000 bytes, 0.0034 better on the last tenth after a million. Both are single figures; I read them together as "about nothing at that age", and the books, read later in the life, as "a little". That agrees with `nano/table/PLAN_10.md` on kinds born with no trust: safe and slow.
5. The step for the added trusts wanted to be small, unlike `nano/table/PLAN_8.md`: with a larger step the loss on the development part rose.
6. Not shown: that the gain grows with a longer life; any of it on a text other than these four; the measure in which everything goes on changing.

What follows. The gain is small but above what would have refuted it, so I take the next step once, where the loss is (`PLAN_2.md`: a guess of the next word's first letter), and stop this line if that gains nothing.
