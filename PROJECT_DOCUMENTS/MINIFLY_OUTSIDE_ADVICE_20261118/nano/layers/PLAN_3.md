# Step 3: cells on cells, against one wide cell and against cells side by side under a reading cell (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/nano-layers-8mmhfm` at the commit that carries this file (`layers.py` as checked at 1747f29, arms added, nothing else changed; cells on cells and the reading cell are among the nine cells of the check). Written before any run of this step. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_2.md`, section 5: on prose three gates (the pair of bytes at delays 1 and 2; a trust the sending cell owns; how sure a cell below is) each add beside the delays' parts, and together in one cell they give 0.0390 nats over the first cell (the single second cell: 0.0239). On sums every arm that knows the place is at 97.3 to 97.8 on seed 1.

## 2. The one question, and the arms

**With the same families of parts, does it matter how they are arranged: in one wide cell, in cells on cells, or in cells side by side under a reading cell?** Base: the delays' parts without the level ("nl"). Families: pair, own, sure (all without the level).

- *flat*: one cell with nl + pair + own + sure (step 2's best); and the same with the fine parts of the delays added.
- *deep*: a third cell that hears the second cell's (nl) branches and has one other family: pair; own; sure of the second cell; sure of the first cell; nl again (the same family twice); the fine parts. A third cell with pair + own + sure. A chain nl > pair > own > sure (four further cells, each hearing the one below; sure reads the cell below it). The flat cell twice in depth.
- *wide under a reader*: four cells side by side, each hearing the first cell's branches with one family (pair, own, sure, nl), read by one more cell that has a trust for each of the four (born at one quarter) and learns from its own mismatch; and a further cell (nl) on the reading cell.

Nineteen further cells in one run on each text; systems, material and measures as before (sums seed 1; the life with the three worn books).

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* Depth adds beyond width: with the same families, the chain nl > pair > own > sure is at least 0.003 nats below the flat cell on the mean of the three books and not more than 1 point below it on sums. *Refuted if* the chain is not below the flat cell by 0.001 on prose. I hold this at a chance of 0.3: in this form a cell on a cell adds its parts to the same sum of trust times message, and the only thing depth changes is whose mismatch teaches each part and whose sureness a gate reads.

| | Statement | Chance |
|---|---|---|
| D1 | prose: each third cell with one family (pair, own, sure) is within 0.0015 nats of the flat cell with nl and that family (step 2: 0.0329, 0.0321, 0.0329) | 0.60 |
| D2 | prose: the chain is within 0.002 of the flat cell (0.0390), either way | 0.50 |
| D3 | prose: the same family twice (nl > nl) adds at least 0.002 over nl | 0.40 |
| D4 | prose: sure of the second cell is better than sure of the first, as the third cell's gate, by at least 0.0005 | 0.55 |
| D5 | prose: the reading cell over four cells side by side is at least 0.003 worse than the flat cell | 0.60 |
| D6 | prose: the flat cell twice in depth adds at least 0.002 over the flat cell | 0.35 |
| D7 | sums: every arm that contains nl is within 1 point of nl (97.4) | 0.65 |
| D8 | sums: the reading cell is at least 1 point below nl | 0.45 |

## 4. Size, cost, what is not run

Parts as in step 2 for each family; the reading cell has four trusts. About eight CPU-minutes on sums, one on prose. *Not run:* a reading cell with gated trusts; readers over more than four cells; depth beyond four further cells; the reported seeds; the unopened book.

## 5. Outcome (entered 20 November after the runs; `res/sums_armsstep3_seed1.json`, `res/life_armsstep3_booksworn.json`, `report_3.txt`)

Sums: seed 1, 40,000 lines, with classes, 1,231,004 cells, % after a read line, mean of the last three readings; 313 CPU-seconds for nineteen further cells. Prose: T2, 2,419,727 cells, mean gain over the first cell on the three worn books, nats; 45 CPU-seconds. First cell 92.7 and 0; the coordinator's single second cell in the runs of steps 1 and 2: 96.7 and 0.0239.

| Arrangement | Sums, % | Prose, gain over first | Prose: test text kept (after its stream; after three books) |
|---|---|---|---|
| nl (one cell) | 97.4 | 0.0292 | 1.4761; 1.5061 |
| **flat**: one cell, nl + pair + own + sure | 97.8 | **0.0390** | 1.4693; 1.5019 |
| **flat with the fine parts of the delays** | 97.2 | **0.0444** | 1.4654; 1.5019 |
| deep: nl > pair | 97.5 | 0.0320 (flat nl + pair: 0.0329) | |
| deep: nl > own | 97.5 | 0.0316 (flat: 0.0321) | |
| deep: nl > sure of the second cell / of the first | 97.5 / 97.4 | 0.0323 / 0.0323 (flat: 0.0329) | |
| deep: nl > nl (the same family twice) | 97.5 | 0.0319 | 1.4746; 1.5163 |
| deep: nl > fine | 97.1 | 0.0357 (flat second + nl: 0.0364) | |
| deep: nl > one cell with pair + own + sure | 97.8 | 0.0382 | |
| deep: chain nl > pair > own > sure | 98.0 | 0.0366 | 1.4700; 1.5044 |
| side by side under a reading cell (pair, own, sure, nl) | 97.4 | 0.0295 | |
| a cell (nl) on the reading cell | 97.6 | 0.0323 | |
| deep: the flat cell twice | 97.6 | 0.0433 | 1.4660; 1.5119 |

*Predictions.* D1 held (a third cell with one family is 0.0005 to 0.0009 nats behind the flat cell with the same families). D2 failed (the chain is 0.0024 behind the flat cell, not within 0.002). D3 held (the same family twice adds 0.0027). D4 failed (the second cell's sureness and the first cell's give the same, 0.0323). D5 held (the reading cell is 0.0095 behind the flat cell). D6 held (the flat cell twice adds 0.0043). D7 held (every arm with nl is at 97.1 to 98.0 on sums). D8 failed (the reading cell equals nl on sums, 97.4). **The hypothesis is refuted**: with the same families depth does not add beyond width; the chain of four further cells is 0.0024 nats worse than one wide cell on prose, and on sums the two are 0.2 points apart on one seed.

Readings (mine).
1. **In this form a cell on a cell is a wide cell taught less well.** A further cell adds its parts to the same sum of trust times message; put in depth, each family is taught by the mismatch of a cell that does not yet hear the families above it, and the result is a little worse than one cell that holds all of them (0.0005 to 0.0024 nats). Depth would need something this form does not have: senders of a new kind made from the cells below.
2. **The same parts twice add (0.0027 for nl > nl, 0.0043 for the flat cell twice), and what they add is paid for in keeping**: after three other books the doubled cells are 0.010 worse on the old text than the single ones. My reading is that a second copy is a larger step (the step 0.00005 was chosen for the first cell's trusts, which are taught at every byte), and a larger step follows the recent text more. I did not run the control with the step doubled; it is owed.
3. **A reading cell with one trust for each cell below returns its best input and no more** (0.0295 against 0.0292 for nl alone; 97.4 on sums). Cells side by side each repair the same first cell; their repairs do not add under one trust each. They add when they share one sum (the flat cell).
4. On sums nothing in this step is readable beyond "every arm that knows the place is at 97.1 to 98.0".
