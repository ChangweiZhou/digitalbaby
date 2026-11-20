# Line `nano/reuse`, first plan: are there kinds of cell that say the same? (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, since: "继续大胆假设，基于事实努力尝试。" "多想想，多试试。" "继续大胆尝试。" The coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." "Go on making bold hypotheses, and try hard on the basis of facts." "Think more, try more." "Go on trying boldly."

**Source.** `claude/friendly-darwin-8mmhfm` at cf4d01d; `nano/merge/merge.py` copied unchanged from `claude/nano-merge-8mmhfm` at e0bc883 (that line is still working; nothing later is waited for). Branch `claude/nano-reuse-8mmhfm`, folder `nano/reuse` only. The hypothesis is the coordinating session's; the form given to it, the programs and the predictions are this session's. Development level; no gate, seal or review is claimed. Before this file was committed the copied `merge.py` was run once (seed 1, 40,000 lines, its frozen setting) to see that the copy runs; its output was not read beyond that and is not kept.

## 1. The hypothesis of the line (the coordinating session's)

Kinds that say the same should be merged, as cells are. Two kinds of cell (two sets of places) are alike if their cells for the same values say the same about what follows. A system that finds this holds one store and uses it at every place where a kind turns out alike; a new place, a longer number or a new layout then needs only enough lines to recognise that its kind is alike. Nothing learned for one task has yet been used for another in this programme; this line asks whether that can be found by the system and not laid down.

## 2. This step: one question, a measurement on a system that exists

**In a system that has read two-digit sums with classes as `nano/merge` builds them, are there kinds whose cells for the same values agree beyond chance?**

*The system* is `merge.py sums` at its frozen setting (255 plain kinds, every set of at most four of nine delays; at most 32 class kinds; about 1.23 million cells; 50 CPU-seconds), development seed 1, 40,000 lines. `step1.py` reads its store at the end and changes nothing.

*What is measured.* A kind is taken as a table from the values of its key places, in an order, to the cell's most-counted follower. Only cells with at least two counts enter.
- **Pairs of plain kinds**: one kind A among the 16 of largest trust (mean over uses; the candidates `merge.py` itself uses), with at least two delays; the other, B, any plain kind with the same number of delays, A itself included; every way of matching B's places to A's (all orderings; for B = A, every ordering but the identity). Agreement: the share of common values (cells of both with at least two counts) on which the most-counted follower is the same. Chance: the share expected if the two tables were matched at random on the same common values, sum over followers v of f_A(v) f_B(v), f being the frequencies of the most-counted follower. Both, the number of common values, and the excess (agreement less chance) are reported.
- **Slices**: a class kind (class of X, values of Z) with the class held at one value is a table over the values of Z; it is compared in the same way with every plain kind that has as many delays as Z, and with the slices of the other class kinds.
- The same with the readings restricted to cells whose most-counted follower is a digit, since many kinds agree trivially where both say the byte of the layout (reported beside, not instead).

*Hypothesis of this step.* There are pairs (kind, kind, matching) that agree on at least 90% of at least 50 common values with a chance level under 50%, and they are few: under 2% of the pairs measured. *Refuted if* no pair agrees more than 20 points above its chance level on at least 50 common values (then the hypothesis of the line is dead on this task and that is the result).

## 3. Predictions (mine, before the measurement)

One thing I expect that the brief did not name: a cell is keyed on bytes at fixed places back from the byte being foretold, and in this text the two digits that are summed lie 4 and 7 places back both for the tens digit of the answer and for the units digit. So one plain kind already serves two places of one task; what a kind cannot do is serve a place whose distances differ (the first byte of the answer; a number of another length; another layout). And a kind is alike *itself* with its places exchanged wherever the task is symmetric: the cell for (3, 5) and the cell for (5, 3) of the two digits summed say the same.

| | Statement | Chance |
|---|---|---|
| S1 | the plain kind of delays (4, 7) and the slice of the class kind (X = (3, 6), Z = (4, 7)) at its class of pairs that do not carry agree on at least 90% of common values (the coordinating session's expectation) | 0.60 |
| S2 | the other slice (the class that carries) agrees with the plain kind (4, 7) on under 30% | 0.75 |
| S3 | some plain kind among the 16 agrees with itself under an exchange of places on at least 90% of at least 50 common values, digits only (the task is symmetric in its two numbers) | 0.70 |
| S4 | the kind of delays (3, 4, 6, 7) with the two numbers exchanged, places (6, 7, 3, 4), is such a case: at least 90% | 0.60 |
| S5 | among pairs of two *different* plain kinds with digit followers, under 2% of (pair, best matching) exceed their chance level by 20 points on at least 50 common values | 0.55 |
| S6 | the hypothesis of the step as stated | 0.50 |

Where I expect it to mislead if read carelessly: agreement on layout bytes (every cell after a full sum says the line end) is not reuse of a rule; hence the reading on digit followers.

## 4. Size, cost, what is not run

One run of `merge.py` (50 CPU-seconds, 1.23 million cells, 300 MB) and the comparison in numpy (my estimate: under two CPU-minutes). `step1.py check`: the agreement of the fast comparison against a plain-Python count on small tables. Not run in this step: any merging of kinds; any second task; five seeds (one development seed; the question is whether the thing exists at all). Step 2 (a second plan, written after this is read): the mechanism and the transfer experiments.
