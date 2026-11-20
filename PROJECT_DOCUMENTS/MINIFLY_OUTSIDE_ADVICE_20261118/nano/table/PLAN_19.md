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
