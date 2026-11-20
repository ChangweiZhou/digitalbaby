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
