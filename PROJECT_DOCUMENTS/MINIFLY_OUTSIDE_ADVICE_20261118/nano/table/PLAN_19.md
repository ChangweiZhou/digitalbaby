# A cell is born only where the system was surprised (roadmap second revision, track C; the store) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试，小心求证。" (my translation: "Go on trying boldly, and verify with care."). The signal used here is of the kind he named earlier the same day: "Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc." **Source.** `claude/friendly-darwin-8mmhfm` at 07d109301; `gate.py` with one more argument to its kernel (the check, `check19_output.txt`, was run; nothing else was). Written before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. `growth.py` (roadmap, section 8): the number of cells that have counted grows as the bytes read to the power 0.80, and 77 to 79% of all cells have been met once. Most of the store holds no regularity.
2. Everything since has added cells: classes (five times the cells on prose), cells keyed on two classes, kinds made while reading (`nano/merge`: "accuracy bought with a larger store"). Three rules of that line for letting go of plain cells each harmed prose. Nothing in the programme makes the store smaller.
3. A cell is born today at its first meeting whatever the system knew at that byte. Where the less specific cells already named the byte, the new cell records what was not in doubt.

## 2. The one change

**A cell that has never counted is not born at a byte to which the first answering cell gave a probability of one half or more.** Once born it counts at every meeting, as now. Nothing else changes; the second answering cell of `PLAN_15.md` is there and untouched. The rule is one comparison on a signal the answering cell already forms (how surprised it was), sent to every cell alike; it knows no kind, byte or text. One half is "the byte was more likely than not"; it is stated here, not chosen by a run. Two other thresholds are run for the shape of the trade and are not claims: nine tenths (almost every cell is born) and one tenth (a cell is born only where the true byte had under one chance in ten).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The store is at least a third smaller on prose at a price under 0.010 nats: with the threshold at one half, in the life with the three worn books and in the life with the three books of `PLAN_15.md`, the cells that have counted at the end are at most two thirds of the baseline's (579,255 and 928,030) and no book's last tenth is more than 0.010 nats worse for the first cell. *Refuted if* the cells are more than four fifths of the baseline's in either life, or any book is more than 0.020 worse.

| | Statement | Chance |
|---|---|---|
| S1 | prose, threshold one half: at most two thirds of the baseline's cells, both lives | 0.50 |
| S2 | no book more than 0.010 nats worse, first cell, both lives | 0.50 |
| S3 | the second cell likewise loses no more than 0.010 on any book | 0.50 |
| S4 | sums with classes, 40,000 lines, five reported seeds, threshold one half: the second cell's median within 2 points of 97.4 | 0.60 |
| S5 | sums: at least a fifth fewer cells | 0.35 |
| S6 | threshold one tenth on prose: under half the baseline's cells, and at least one book more than 0.020 worse | 0.55 |

## 4. Runs, size, cost, what is not run

Prose: `gate.py life books=worn|new mode=none birth=0.5|0.9|0.1` (runs of 0 to 6 bytes; the baseline is `PLAN_15.md`'s runs). Sums: `gate.py sums mode=classes birth=0.5`, the five reported seeds, 40,000 lines, both cells, both scores; and without classes on two seeds. Under twenty CPU-seconds a life; eighty a sums run. The compiled loop with the birth rule equals numpy to 4e-16 and 3e-15 on two cases.

Not run: the rule on this line's system with word cells and all kinds (another kernel); cells given back after they are born; the same rule with the second cell's surprise; long reading; the count of cells along a life (the power of growth).

## 5. Outcome (entered 20 November after all runs; `res/gate_life_birth*.json`, `res/gate_sums_birth0.5_*.json`, the printed lines in `out19/`)

**Prose** (runs of 0 to 6 bytes; nats a byte on each book's last tenth right after its stream; first cell, and in brackets the second cell; cells that have counted at the end of the life).

| Threshold | Worn books: bryant, burgess, carroll | cells | The books of `PLAN_15.md`: chesterton-ball, austen-emma, melville | cells |
|---|---|---|---|---|
| none (the baseline) | 1.5023, 1.3645, 1.4456 (1.4812, 1.3408, 1.4157) | 579,255 | 1.6316, 1.2416, 1.4687 (1.6116, 1.2203, 1.4453) | 928,030 |
| nine tenths | 1.5017, 1.3635, 1.4449 (1.4810, 1.3405, 1.4152) | 552,849 (95%) | 1.6313, 1.2415, 1.4680 (1.6116, 1.2203, 1.4451) | 885,504 (95%) |
| **one half** | 1.5070, 1.3633, 1.4533 (1.4853, 1.3404, 1.4211) | 455,661 (**79%**) | 1.6382, 1.2489, 1.4715 (1.6180, 1.2260, 1.4489) | 734,613 (**79%**) |
| one tenth | 1.5813, 1.4539, 1.5319 (1.5537, 1.4198, 1.4836) | 237,893 (41%) | 1.7262, 1.3372, 1.5359 (1.6981, 1.2984, 1.5072) | 395,122 (43%) |

With the threshold at one half the first cell is 0.0047, -0.0012, 0.0077 worse on the worn books and 0.0066, 0.0073, 0.0028 worse on the others; the second cell 0.0041, -0.0004, 0.0054 and 0.0064, 0.0057, 0.0036.

**Sums** (never-shown two-digit additions, 40,000 lines, after a read line; the baseline of `PLAN_15.md`, then the threshold at one half).

| System | Seeds | First cell, last reading | Second cell, mean of the last three readings | Cells |
|---|---|---|---|---|
| without classes | 73129, 93683 | 82.0, 80.9 then 82.1, 80.6 | 87.9, 85.7 then 87.7, 84.9 | 1.21 million then 0.56 million (**46%**) |
| with classes | the five reported | 93.2, 95.2, 92.4, 89.6, 92.9 then 84.7, 93.5, 85.3, 89.5, 91.8 | 97.4, 98.0, 96.6, 96.2, 97.1 (median 97.1) then **91.8**, 96.8, **90.0**, 96.9, 98.1 (median 96.8) | 1.23 million then 0.56 million (46%) |

**The hypothesis is not shown and not refuted on prose.** The store is 79% of the baseline's in both lives, where two thirds was asked and four fifths would have refuted; no book is more than 0.008 nats worse (0.010 allowed).

| | Statement | Outcome |
|---|---|---|
| S1 | prose: at most two thirds of the baseline's cells | failed (79% in both lives) |
| S2 | no book more than 0.010 worse, first cell | held (at most 0.0077) |
| S3 | the second cell likewise | held (at most 0.0064) |
| S4 | sums with classes: the second cell's median within 2 points of 97.4 | held at the median (97.1 at the last reading; 96.8 over the last three), **but two seeds of five lose 6 and 7 points** |
| S5 | sums: at least a fifth fewer cells | held (54% fewer; I had given it 0.35) |
| S6 | one tenth on prose: under half the cells and a book more than 0.020 worse | held (41 to 43% of the cells; 0.067 to 0.096 worse) |

Readings (mine).
1. **On prose the store is not mostly waste by this measure.** A fifth of the cells are born at bytes that the system already gave better than even odds, and leaving them unborn costs 0.003 to 0.008 nats; nearly all the rest are born where the system was in doubt, and leaving those unborn is paid for in full (0.07 to 0.10 nats for the next 36%). The trade is real and it is not cheap. Almost free: the 5% of cells born at bytes that had nine chances in ten.
2. **On sums half the store is born at bytes that the layout makes certain** (the signs, the line end, and the answer once it is known), and without classes it can be left unborn at no price: 46% of the cells, the same score. I had not expected that; I had thought of the store on sums as the pairs.
3. **With classes the rule is not safe.** On two seeds of five the system loses 6 to 10 points. *Not looked into*; my guess is that the search for classes reads refinements of cells, and a cell that is unborn where its parent foretold the byte is exactly a refinement that agrees with its class.
4. The rule uses a signal that the answering cell forms anyway and sends to all cells alike, which is the form of signal Mr. Xu named. As a means of making the store grow more slowly it is a partial answer on one kind of text; the power of growth with the bytes read was not measured (NOT RUN).
