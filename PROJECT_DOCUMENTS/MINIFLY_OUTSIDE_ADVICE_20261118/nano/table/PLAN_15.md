# A second answering cell whose trusts know what byte stands at each delay (roadmap second revision, tracks C and L; one change, the same on sums and on prose) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试。" (my translation: "Go on trying boldly."), after "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" (my translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas."). **Source.** `claude/friendly-darwin-8mmhfm` at 8ae14f646; `nano/merge/merge.py` as taken in at that line's commit e0bc883ce (its system, unchanged, is the system this plan adds to, because it is the one program that runs both sums and prose). Written before the program of this plan exists and before any run of it. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. Five times now a loss has been found in the answering cell's reading and not in what the cells hold (`PLAN_14.md`, section 1, and since then the line `nano/merge`: with only class kinds speaking the tens digit of never-shown sums is right at 98.6%, with all kinds speaking at 92%).
2. `votes.py` (roadmap, section 9): a kind of cell is a set of delays, and the same set holds different things at different places in a line. The kinds that are exact for the tens digit are also active where another byte is named, with chance followers; the kinds that are tied to the place are inexact, and they carry the wrong votes. The answering cell has one trust for a kind and a level of counts, **whatever the place**: it cannot trust the exact kind where the tens digit is named and distrust it elsewhere.
3. `where.py`: on prose 35% of the loss of this line's best system falls on the first letter of a word and 10% on punctuation: places that the last byte or two name.
4. `PLAN_14.md`: a trust that also knows one fact about the moment (how many less-counted senders speak), added as a second part born at nothing, lowered the loss on three books by about 0.01 nats in two lives. Its transfer to sums is owed and is run here as a side arm (section 4).

## 2. The one change

The system keeps its answering cell as it is. A **second answering cell** hears the same senders at the same byte, and the first cell's branches. Its branch for a byte value v is

    a2(v) = a1(v) + sum over senders k of [ sum over delays j = 1..D of G_j[kind of k][level of k][the byte at delay j] ] * s_k(v)

where a1(v) is the first cell's branch, s_k(v) is the sender's message as now, the level is the sender's level of counts as now, and D is the number of delay lines the system has (9 on sums, 6 on prose). So for each delay the second cell holds a part of the trust that depends on which byte stands at that delay. **Every part is born at nothing**: at birth the second cell answers exactly as the first. Each part that was used changes by the trust's own step rule of the system it is added to (prose: the constant step 0.00005; sums: 0.1 over the square root of one plus the number of times that entry was used) times the second cell's own mismatch for that sender, s_k(y) less the sum of p2(v) s_k(v). The first cell learns from its own mismatch as before and is not touched: its answers are the plain system's to the last digit, in the same run.

Nothing in the rule names a kind, a byte, a place or a text. No number is chosen: the count of parts is the count of delay lines, the step is the system's, the birth is nothing. Where classes are found anew (`nano/merge`), the second cell's parts for a class kind follow the first cell's trust for it (kept where the pair is kept, nothing where it is new).

*The control.* The same second cell with the same number of parts and the same steps, but every part reads a constant in place of the byte at its delay (it knows nothing about the moment). This separates what the bytes tell from what a larger total step does.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With this one form the second cell answers better than the first on both kinds of text: (i) on never-shown two-digit sums with classes (that line's frozen setting; five reported seeds, 40,000 lines; scored after a read line, `loop_small_case/ERRATA_20261120.md`, section 3) its median is at least 2 points above the first cell's; (ii) on prose (that line's prose system without classes: runs of 0 to 6 bytes) its loss is at least 0.005 nats lower on each of three books that no line has opened, read after the test text's stream. *Refuted if* the gain on sums is under 1 point, or fewer than two of the three books are lower by 0.002. Between the two: not shown.

| | Statement | Chance |
|---|---|---|
| W1 | sums with classes, after a read line: the second cell's median at least 2 points above the first cell's | 0.50 |
| W2 | sums with classes: the second cell at or above the class kinds alone (94.6 at the median) | 0.35 |
| W3 | sums without classes: the second cell at least 2 points above the first | 0.35 |
| W4 | prose: at least 0.005 lower on each of the three unopened books | 0.45 |
| W5 | prose: at least 0.002 lower on each | 0.70 |
| W6 | the control gives less than half of the second cell's gain, on sums and on prose | 0.60 |
| W7 | the last tenth of the test text, scored once after its stream: the second cell at least 0.005 lower (a worn measure; it carries little) | 0.50 |
| W8 | the side arm of section 4 (the part of `PLAN_14.md` alone, on sums with classes): under 1 point, because with 288 kinds nearly every sender has three or more less-counted senders and the count says nothing | 0.75 |

## 4. Runs, measures, size, cost, what is not run

*Sums.* `nano/merge/merge.py`'s system and text, unchanged: kinds of at most four of nine delays, e = 2.0, classes at that line's frozen setting (32 pairs, eJ = 0.02), and the same without classes. Development seed 1 first, to see that the program runs and that the first cell gives that line's figures (nothing is chosen on it); then seeds 73129, 93683, 116089, 406364, 545353, 40,000 lines, last reading and mean of the last three. Scored after a read line (five draws) and with an empty past beside it; the system's own answer bytes; first cell and second cell from the same run. Arms: the second cell; the control; the side arm (one part indexed by the number of speaking senders that have counted fewer times and at least twice, 0 to 3, as `PLAN_14.md`, in place of the parts for delays).

*Prose.* `merge.py`'s prose system without classes (seven kinds; e = 0.02; step 0.00005; trusts born at 0.25). The life: the test text's stream, then `chesterton-ball`, `austen-emma`, `melville-moby_dick` (from `minifly_v155_keys_dev_v1_20261026/data/`; no line of this programme has opened them; bytes not among the 65 values become a space, carriage returns are removed, as in `life.py`), first nine tenths of each read, each last tenth scored right after its stream with nothing changing. The last tenth of the test text scored once, after its stream. Arms: the second cell; the control. As a second reading on worn books, the life with the three books of `life.py`. With classes on prose: one life (the worn books), a reading and not a claim.

*Size.* The second cell holds D times (kinds times 40 levels times byte values) parts: 1.45 million on sums with classes, 109,200 on prose without; the number actually used is reported. *Cost.* A run on sums about two CPU-minutes with classes; thirty runs; a life on prose under a minute without classes. The compiled loop is checked against numpy on a prefix.

*Not run.* Gates other than the byte at a delay (the levels of other senders; a class as a gate; gates drawn by search); more than one further cell, or cells in layers; the word cells of this line's T2; sums over 400,000 lines; the lives of two tasks on sums; three-digit sums.

*What this would mean if it holds* (mine). An answering cell with one more part for each delay is the first step of a layer of answering cells, each of which names the next byte and learns from its own mismatch with no signal passed back: the form studied elsewhere as mixing networks with gates. If it fails on either kind of text it is entered as not transferring.

## 5. Addition 1 (20 November; written after the development runs and the first claims of section 4 had been read, and before the runs it describes): long reading

*The evidence.* Every counting system on sums loses with long reading: the line `nano/merge` reports 78.9 and 81.3 at 400,000 lines for its frozen setting on the seeds 73129 and 93683 (92 and 94 at 40,000 lines; old score), and `nano/sparse` repaired its own fall with a cap on what a long-counted cell may say, which again is the reading and not what is held. The relay system gains with reading (95.5% after a read line at 400,000 lines).

*The question.* Does the second answering cell hold its level over 400,000 lines, with nothing added and no cap? *Runs:* the system with classes and the second cell (`gate=0`), seeds 73129 and 93683, 400,000 lines, readings at that line's marks, both scores, first and second cell from the same run.

*Predictions (mine).* A1: on both seeds the second cell's mean of the last three readings, after a read line, is at least 90 (chance 0.40). A2: on both seeds the second cell is at least 10 points above the first at the end (0.60). A3: the second cell at 400,000 lines is below its own level at 40,000 lines by more than 3 points on at least one seed (0.65). If A1 fails the fall with reading is not only a matter of place, and the cap or something like it is still owed.

*Cost.* About fifteen CPU-minutes a run. Not run: the other three seeds; the system without classes.
