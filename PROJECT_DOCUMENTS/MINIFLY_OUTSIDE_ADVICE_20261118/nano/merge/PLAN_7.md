# Line `nano/merge`, seventh plan: the answering cell; a class kind is trusted by whether the member's own cell speaks, and practises the case that it does not (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run of `merge7.py` other than its check (`check7_output.txt`). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

1. On two-digit sums the class kinds alone answer never-shown sums better than all kinds together (94.6 against 92.2, five seeds; tens digit 98 to 99% against 92%). The class cells are right; the answering cell does not trust them enough where it matters.
2. Why (my reading, `PLAN_1.md` addition 1): a class cell matters exactly where the cell it stands for (the R-cell of the same place) has never been met. On a line that has been shown, that R-cell answers, the mismatch is nothing and no trust moves. The trust of a class kind for the case "the member's own cell is absent" is taught only at first meetings, which on this text are few and early.
3. Three ways of making the shown lines teach it were tried and each harmed prose: silencing the members' cells for good, silencing those the class foretells well, silencing at half the bytes. In all three the system that answers is not the system that learned, or has lost cells that say something.

## 2. The one change (to the answering cell; the classes are the frozen setting's)

**A class kind has two rows of trust**, one used where the R-cell of the same place has counted and speaks, the other where it does not. Each row is changed by the mismatch where it is used, as before. In addition, **practice**: while reading, at every byte and for every kind R that speaks and has class kinds made from it speaking beside it, the answer is formed once more with that one R-cell left out and its class kinds on their second row; the mismatch of that second answer changes those second rows and nothing else. No cell is silenced in the answer the system gives or learns its other trusts from; nothing is added to what a cell holds. No new number. `merge7.py` (a subclass; the compiled loop against numpy in its check).

The second row is the system's answer to "what would I say here had I not met this combination", asked at every byte and taught by the byte that arrives.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With the frozen setting's classes and this answering cell, never-shown two-digit sums reach at least 95% at the median of five seeds at 40,000 lines, and no book is more than 0.002 nats above the frozen setting. *Refuted if* sums are under 93, or a book is more than 0.002 above.

| | Statement | Chance |
|---|---|---|
| Y1 | two-digit sums, development seed: at least 95 | 0.45 |
| Y2 | two-digit sums, five seeds, median: at least 95 | 0.35 |
| Y3 | the same under 93 (refutation) | 0.35 |
| Y4 | prose, last twentieth of the stream: not more than 0.002 above the frozen setting's 1.5488 | 0.65 |
| Y5 | prose: at least 0.002 below it | 0.20 |
| Y6 | three-digit sums, five seeds: within 1 point of the frozen setting's 94.1 either way | 0.60 |
| Y7 | 400,000 lines, two seeds: at least 90 on both | 0.20 |

## 4. Runs, size, cost

Development material first (two-digit seed 1; prose's last twentieth). If on development material it is not below the frozen setting by more than 1 point on sums nor above it by more than 0.002 nats on prose: the five reported seeds on two-digit sums, the first life, five seeds on three-digit sums, and 400,000 lines on two seeds. Otherwise it stops there and that is the result. The second rows add 32 x 40 trusts. Cost by estimate: as the frozen setting plus a few percent. Not run: the same for the kinds keyed on two classes; a second row for plain kinds.
