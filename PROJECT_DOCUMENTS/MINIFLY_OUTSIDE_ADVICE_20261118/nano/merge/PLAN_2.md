# Line `nano/merge`, second plan: the frozen setting on five seeds of sums and on the life on prose, with the controls (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file; base `claude/friendly-darwin-8mmhfm` at 71172671f. Written before any run on a reported seed, on a book, or on the last tenth of the test text. Development level; no gate, seal or review is claimed.

## 1. The system and the setting, frozen

`merge.py` as `PLAN_1.md` describes it, with the numbers chosen in step 1 on development seed 1 and on the last twentieth of the test text's stream: classes beside the cells, at most P = 32 kept pairs, e_J = 0.02; 16 kinds by trust as candidates, X of at most two delays, findings when the bytes read have doubled (first at 10,000), two sweeps. The base on each task is its comparator's (sums: 255 kinds of at most four of nine delays, step 0.1 shrinking, e = 2, born 0.02; prose: runs of 0 to 6 bytes, step 0.00005, e = 0.02, born 0.25). The same class mechanism and numbers on both tasks.

## 2. Runs

Sums, five reported seeds (73129, 93683, 116089, 406364, 545353), 40,000 lines, readings every 5,000, never-shown additions with the system's own answer bytes:
- `none`: no classes. `cls`: the frozen setting. `chance`: at every finding the classes of each kept pair are replaced by classes of the same sizes with members drawn by chance. `half`: classes found once, at 20,000 lines, their cells starting empty and counting only the second half. `rep`: replacement (not the frozen setting; reported beside it because step 1 showed it higher on sums and worse on prose).
- 400,000 lines, `none` and `cls`, on two seeds (73129, 93683) first; the other three if the cost allows. Two seeds are a reading and not a claim.

Prose, the life (the stream of the test text, then the three books; each book's last tenth scored right after its stream, nothing changing during a scoring): `none`, `cls` (with the last tenth of the test text scored once, at the end of the life), `chance`, `half` (classes found once at 500,000 bytes), `rep`. The number of cells at every 125,000 bytes.

## 3. Hypothesis, what refutes it, predictions

The hypothesis, its refutation and the predictions M1 to M11 are those of `PLAN_1.md`, section 3, unchanged, with my chances as written there before any run. After step 1 (one seed; these are new and written now): 

| | Statement | Chance |
|---|---|---|
| N1 | sums `cls`, median of five seeds, last three readings: at least 88 | 0.70 |
| N2 | sums `rep` above `cls` at the median | 0.70 |
| N3 | each book: `cls` below `none` by at least 0.005 nats (this is M7; my chance for it is now higher than the 0.20 written before step 1) | 0.50 |
| N4 | prose `chance`: no book better than `none` by more than 0.002 | 0.70 |
| N5 | prose `half`: at least half of `cls`'s gain on the mean of the three books | 0.40 |
| N6 | the last tenth of the test text at the end of the life: `cls` below `none` | 0.65 |
| N7 | number of cells on prose: with classes it grows with a power of the bytes read that is not below the base's | 0.80 |

## 4. Size and cost

Sums: about 1.2 million cells, 50 to 60 CPU-seconds a run at 40,000 lines with classes (18 without); at 400,000 lines my estimate is ten CPU-minutes. Prose: 0.45 million cells without classes and about 2.3 million with, at the end of the stream; a life about four CPU-minutes. Four cores, three or four runs at a time.

Not run: subtraction and keeping; three-digit sums; the word cells as base; more than 32 pairs; X of more than two delays; classes of classes; a change to the answering cell.
