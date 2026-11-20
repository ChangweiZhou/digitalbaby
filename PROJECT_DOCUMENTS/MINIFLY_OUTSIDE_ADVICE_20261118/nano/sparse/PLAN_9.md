# Line `nano/sparse`, ninth plan: does the level with the answer written units first hold over long reading? (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one; `sparse.py` unchanged. Written before its runs. Development level; no checker gate, seal or outside review is claimed. Run last, if the budget allows; if it is not reached this file says NOT RUN at its end.

## 1. The evidence this follows

`PLAN_5.md`: with the answer units first and a shrinking step, m = 4 has 91.0% of never-shown sums at 40,000 lines; on the usual text m = 4 falls with reading (77.4 to 59.2) unless a cell speaks at the weight of at most 16 counts (77.0 to 75.2). Whether the 91% falls in the same way has not been run.

## 2. Arms, hypothesis, predictions (mine, before any run)

m = 4, eta 0.1, e 2, trusts born 0.02, `shrink=1 rev=1 plan=add:400000`, five reported seeds: (A) no cap; (B) `ncap=16`. Nothing chosen again.

*Hypothesis.* (B) at 400,000 lines is within 5 points of its own reading at 40,000 and at least 85%. *Refuted if* it is under 80%.

| | Statement | Chance |
|---|---|---|
| V1 | (A) falls by more than 5 points from 40,000 to 400,000 | 0.70 |
| V2 | (B) within 5 points of its 40,000 reading | 0.65 |
| V3 | (B) at 400,000 at least 85% | 0.50 |
| V4 | (B) at 40,000 within 3 points of (A) at 40,000 | 0.60 |

Cost: about 3 CPU-minutes a run. Not run: keeping with the answer units first; the cap chosen again for this text.


## NOT RUN (20 November): this plan was queued last and the session's budget went to the tenth plan's long runs instead. No run of it was started.
