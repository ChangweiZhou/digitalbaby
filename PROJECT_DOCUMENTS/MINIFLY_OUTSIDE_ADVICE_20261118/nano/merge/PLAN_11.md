# Line `nano/merge`, eleventh plan: a kind that is made is born with the past, where the system holds it (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run with `full=1` other than the check (`check10_output.txt`, third line). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_10.md`, addition 3: on prose the test text's last tenth at the end of the life is 1.5264 with the frozen setting and 1.5550, 1.5573, 1.5685 with kinds keyed on two classes, with made kinds, and with both. A kind born late holds only what has been read since its birth; the answering cell learns on the current text to trust it; the old text is then answered from it. The class kinds of the frozen setting do not do this, because a class cell is born with the pooled counts of its members' cells, which hold the whole past.

## 2. The one change (to setting M; nothing else)

The same rule for every kind that is made. **If some kind the system has holds all the delays of the new kind, the new kind's cells are born as the pooled counts of that kind's cells** (of the one with the fewest delays, if there are several), and count on from there. Otherwise the kind is born empty, as before. For a kind made by refining (R with one more delay) that is a kind whose delays contain R's and the new one; for a kind keyed on two classes it is a kind that holds the delays of both X's and of the rest, each of its cells being counted into the cell of (its two classes, its rest). On prose such a kind nearly always exists (a run with a gap lies inside a run); on sums seldom (a kind of five delays lies in no kind of four). Nothing new is laid down and there is no new number. `full=1` in `merge10.py`; the check counts such cells again by hand over the whole stream.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With this rule the test text's last tenth at the end of the first life is not above the frozen setting's 1.5264 by more than 0.003 nats, no book is above the frozen setting, and two-digit sums are not below M by more than 1 point at the median of five seeds. *Refuted if* the test text at the end is above 1.5306 (the system with no classes), or sums fall under 96.

| | Statement | Chance |
|---|---|---|
| K1 | test text at the end of the first life: at most 1.5294 | 0.45 |
| K2 | the same above 1.5306 (refutation) | 0.35 |
| K3 | each book of the first life below the frozen setting (1.4920, 1.3487, 1.4307) | 0.60 |
| K4 | each book of the first life within 0.003 of M (1.4883, 1.3371, 1.3921) or below | 0.25 (the gain of M on the books was recency; born full, it should go) |
| K5 | two-digit sums, five seeds, median at least 97 (M: 98.0) | 0.70 |
| K6 | prose development material: at most 1.5490 | 0.70 |

## 4. Runs, size, cost

Because of the time left in this session the runs are started together and not in two stages: development material (two-digit seed 1; prose's last twentieth), the five reported seeds on two-digit sums at 40,000 lines, the first life with the test text's last tenth scored once at the end, and the second life. Cost as M. Not run: 400,000 lines; three-digit sums; the fresh seeds.
