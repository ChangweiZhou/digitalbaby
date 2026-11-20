# Line `nano/search`, third plan: twice the kinds alive and twice the kinds tried (20 November 2026)

**Authorisation and steer.** Mr. Xu's steer, as quoted to this session by the coordinating session, exactly: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas."). **Source.** `claude/nano-search-8mmhfm` at the commit before this one; the program is unchanged. Development level; no checker gate, seal or outside review is claimed. Written while the runs of `PLAN_2.md` were going: I had seen its first eleven lines of output (prose at `nrep` 16, 8, 0 and the hand-given runs; two-digit sums at 40,000 lines with search on the five seeds: 68.3, 47.4, 59.5, 43.9, 67.9; two control seeds), and nothing of the runs with `nrep` 32 or of 400,000 lines.

**Question.** `PLAN_2.md` varies the number tried at a fixed number alive (`nrep` 8, 16, 32 of 128). The other way to spend computation is on both: **K = 256 alive and `nrep` = 32**, everything else as frozen (same store, same kept bytes, same step and e). Does twice the computation make both texts better?

**Runs.** Two-digit sums, five reported seeds, 40,000 lines (tags `c3_s2_40k_k256_*`); the prose life (tag `c3_p_k256`). About 20 CPU-minutes.

**Hypothesis.** With K = 256 and `nrep` = 32 the median of two-digit sums at 40,000 lines is above that of K = 128 (59.5, last-three mean) and each of the three books is below K = 128's figure (1.5699, 1.3419, 1.5519). *Refuted if* sums are not above or two of the three books are not below.

| | Statement (mine) | Chance |
|---|---|---|
| Q1 | sums median above 59.5 | 0.65 |
| Q2 | all three books below K = 128's | 0.45 |
| Q3 | both | 0.35 |
| Q4 | sums median at least 85 | 0.10 |
| Q5 | all three books below T0 by 0.01 | 0.05 |

**Not run:** K = 512; a larger store beside the larger K (the store is already churned on prose: 55 million cells displaced at K = 128, and K = 256 will churn it more; that is part of what this run measures); 400,000 lines at K = 256.

## Outcome (entered 20 November; `res/c3_*`, `out_c3.txt`, last section of `report_2.txt`)

Two-digit sums, 40,000 lines, K = 256, `nrep` = 32: 78.7, 73.8, 85.4, 87.9, 57.3; median **78.7** (K = 128: 59.5); 2.81 M cells, 27.9 M counts above zero, 3.0 M displaced, 1,728 kinds tried, 97 CPU-seconds (medians). Prose: bryant 1.6174, burgess 1.3546, carroll 1.6038 (K = 128: 1.5699, 1.3419, 1.5519); the last tenth of tiny Shakespeare at the end 2.2031; the store full (4.10 M cells), 198 M cells displaced, 5,760 tried, 626 CPU-seconds.

**The hypothesis is refuted**: sums are above, but all three books are above K = 128's figures. Q1 held; Q2, Q3, Q4, Q5 failed.

Reading (mine): on sums twice the kinds is worth 19 points at the median and brings the searched system to the level of the hand-built kinds without look-ahead; on prose the fixed store is exhausted (47 times its size displaced in one life), so the doubling was not a clean test of more search there. A store that grows with K was named as not run.
