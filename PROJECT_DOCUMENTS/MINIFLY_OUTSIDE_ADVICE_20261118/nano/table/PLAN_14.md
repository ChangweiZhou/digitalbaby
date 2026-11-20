# A trust that knows whether a more specific sender is present (roadmap second revision, tracks L and C; one change from `PLAN_3.md`) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "多想想，多试试。" (my translation: "Think more, try more"), after "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。" **Source.** `claude/friendly-darwin-8mmhfm` at 7c0cdad52; `trust.py`, `words.py`, `life.py` as they stand. Written before any run. Development level; no checker gate.

## 1. The evidence this follows

Four times today a loss was found in the answering cell's reading and not in what the cells hold: keeping on sums (`loop_small_case/PLAN_13.md`), keeping on prose (`PLAN_6.md`), the fall of the sums system with long reading (repaired by a cap on a cell's weight, `nano/sparse`), and now the line `nano/merge`: classes found by the system name the tens digit of never-shown sums at 98.6% alone, and at 92.8% when read beside the other cells, because the trust of the class kinds is one number, used alike where the cell of the whole pair speaks and where it is silent. In all of these the trust of a kind depends on the kind and on how much its cell has counted, and on nothing else that is present at the byte.

## 2. The one change

For a sender at a byte, count the active senders that have counted **fewer** times than it and at least twice: senders met less often, yet with evidence, are more specific ones. The answering cell holds the trust of a kind and level separately for 0, 1, 2, and 3 or more such senders. Nothing else changes (T2: the table's cells, K = 6, and the cells that step once a word; e = 0.02; the trust's constant step 0.00005; every trust born at 0.25; one reading). The rule reads counts only; it names no kind, no byte and no text. It is the exclusion of the old discount chain (`PLAN_1.md`), for kinds that form no chain.

*Measured, on texts where nothing was chosen:* the life A, B of `PLAN_10.md`, addition 2 (the stream of the test text, then the three books; each book's last tenth scored right after its stream, nothing changing; T2 gives 1.4802, 1.3162, 1.4109) and the life of B only (T2: 1.5598, 1.2769, 1.3836). The last tenth of the test text is scored once for the changed system (T2: 1.4973); it has been scored for some fifty arms and carries little.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The change lowers the loss by at least 0.005 nats on each of the three books in the life A, B. *Refuted if* it is not lower by 0.002 on at least two.

| | Statement | Chance |
|---|---|---|
| H1 | at least 0.005 lower on each book, life A, B | 0.45 |
| H2 | at least 0.002 lower on each book, life A, B | 0.70 |
| H3 | no harm in the life of B only (a young system): each book within 0.003 of T2 or better | 0.65 |
| H4 | the trusts learned differ by situation in the expected direction: a run of two or three bytes is trusted less when three or more specific senders are present than when none is | 0.75 |

## 4. Cost, what is not run

Four times the trusts (1,600). Under a minute of CPU a life. Not run here: the same on sums (given to the line `nano/merge`, which has the system where it matters most); a trust that a cell owns; a rule free of scale.
